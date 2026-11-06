"""Fine-tune OpenPI pi05 on LeRobot v2/v3 with episode validation and last/best."""

import functools
import json
import logging
import math
from pathlib import Path

import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import torch
import wandb

from openpi.models import pi0_config
from openpi.shared import normalize
from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import data_loader
from openpi.training import lerobot_local
from openpi.training import optimizer
from openpi.training import sharding
from openpi.training import weight_loaders

# Also works when imported by pytest, rather than executed as a script.
try:
    from scripts import train
except ModuleNotFoundError:
    import train


def build_config(settings):
    opt = settings["optimizer"]
    return config_lib.TrainConfig(
        name="pi05_lerobot",
        exp_name=settings["job_name"],
        model=pi0_config.Pi0Config(pi05=True, action_horizon=settings["policy"]["action_horizon"]),
        weight_loader=weight_loaders.CheckpointWeightLoader(settings["policy"]["pretrained_path"]),
        optimizer=optimizer.AdamW(weight_decay=opt["weight_decay"], clip_gradient_norm=opt["grad_clip_norm"]),
        lr_schedule=optimizer.CosineDecaySchedule(
            peak_lr=opt["learning_rate"],
            warmup_steps=opt["warmup_steps"],
            decay_steps=opt["decay_steps"],
            decay_lr=opt["end_learning_rate"],
        ),
        # Evaluate exactly the parameters exported for inference; avoid the extra EMA model in memory.
        ema_decay=None,
        batch_size=settings["batch_size"],
        num_workers=settings["num_workers"],
        num_train_steps=settings["num_train_steps"],
        seed=settings["seed"],
        fsdp_devices=settings["fsdp_devices"],
        resume=settings["resume"],
    )


def resolve_training_settings(settings, train_frames):
    """Resolve the total update budget using only frames in the training split."""
    settings = dict(settings)
    if settings.get("data_ratio") is not None:
        settings["num_train_steps"] = math.ceil(settings["data_ratio"] * train_frames / settings["batch_size"])
    return settings


def write_json(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def check_resume_settings(old, new):
    mutable = {"resume", "data_ratio", "num_train_steps", "num_workers", "log_freq", "save_steps", "wandb", "device"}
    for key in old.keys() | new.keys():
        if key not in mutable and old.get(key) != new.get(key):
            raise ValueError(f"Cannot change {key!r} when resuming; use a new job_name.")


def compute_stats(dataset, indices):
    stats = {key: normalize.RunningStats() for key in ("state", "actions")}
    for start in range(0, len(indices), 8192):
        batch = indices[start : start + 8192]
        stats["state"].update(dataset.states[batch])
        stats["actions"].update(dataset.actions[batch])
    return {key: value.get_statistics() for key, value in stats.items()}


def make_loader(dataset, indices, data_config, config, data_sharding, *, validation=False):
    count = len(indices)
    if validation:
        # Pad only the final batch; padded losses are excluded from the metric.
        padding = (-count) % config.batch_size
        indices = np.concatenate([indices, np.repeat(indices[-1], padding)])
    subset = torch.utils.data.Subset(dataset, indices.tolist())
    transformed = data_loader.transform_dataset(subset, data_config)
    batches = math.ceil(count / config.batch_size) if validation else None
    return data_loader.DataLoaderImpl(
        data_config,
        data_loader.TorchDataLoader(
            transformed,
            local_batch_size=config.batch_size,
            sharding=data_sharding,
            shuffle=not validation,
            num_batches=batches,
            num_workers=config.num_workers,
            seed=config.seed,
        ),
    )


def eval_step(state, rng, batch):
    model = nnx.merge(state.model_def, state.params)
    model.eval()
    loss = model.compute_loss(rng, *batch, train=False)
    return jnp.mean(loss.reshape((loss.shape[0], -1)), axis=1)


def evaluate(state, loader, count, eval_fn, seed, mesh):
    total, seen = 0.0, 0
    # Same frame subset and noise/time samples at every evaluation.
    for index, batch in enumerate(loader):
        with sharding.set_mesh(mesh):
            losses = np.asarray(eval_fn(state, jax.random.fold_in(jax.random.key(seed), index), batch))
        valid = min(len(losses), count - seen)
        total += float(losses[:valid].sum())
        seen += valid
    if seen != count:
        raise ValueError(f"Validation evaluated {seen} frames, expected {count}.")
    return total / seen


def run(settings, *, prepare_only=False):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s", force=True)
    run_dir = Path(settings["output_dir"]).expanduser().resolve() / settings["job_name"]
    manifest_path = run_dir / "run.json"
    previous = None
    if manifest_path.exists():
        previous = json.loads(manifest_path.read_text())
        if not settings["resume"] and not prepare_only:
            raise FileExistsError(f"{run_dir} exists. Use --resume or change job_name.")
        check_resume_settings(previous["settings"], settings)
    elif run_dir.exists() and any(run_dir.iterdir()):
        raise FileExistsError(f"Nonempty run directory without run.json: {run_dir}")
    elif settings["resume"]:
        raise FileNotFoundError(f"No run to resume in {run_dir}.")

    if not prepare_only:
        if jax.default_backend() != "gpu":
            raise RuntimeError("pi05 full fine-tuning requires a CUDA GPU; use --prepare-only for data checks on CPU.")
        if jax.device_count() != 1:
            raise ValueError("Expected exactly one visible GPU. Start training with scripts/train_pi05.py.")
        sharding.make_mesh(settings["fsdp_devices"])  # Validate before downloading the dataset.
    dataset_settings = dict(settings["dataset"])
    if previous and not dataset_settings.get("root"):
        dataset_settings["revision"] = previous["revision"]
    root, revision = lerobot_local.resolve_dataset(dataset_settings)
    dataset = lerobot_local.LeRobotFrames(root, dataset_settings, settings["policy"]["action_horizon"])
    train_episodes, val_episodes = lerobot_local.split_episodes(
        dataset.episodes,
        dataset_settings["eval_split"],
        dataset_settings["eval_split_seed"],
    )
    split = {"train": train_episodes, "val": val_episodes}
    if previous and (previous["split"] != split or previous["revision"] != revision):
        raise ValueError("Dataset split or revision changed since the saved run.")
    train_indices = np.flatnonzero(np.isin(dataset.episode_ids, train_episodes))
    val_indices = np.flatnonzero(np.isin(dataset.episode_ids, val_episodes))
    if len(train_indices) < settings["batch_size"]:
        raise ValueError("Training split is smaller than batch_size.")
    settings = resolve_training_settings(settings, len(train_indices))
    config = build_config(settings)
    logging.info(
        "Training budget: %d steps, batch_size=%d, effective epochs=%.6f (data_ratio=%s)",
        config.num_train_steps,
        config.batch_size,
        config.num_train_steps * config.batch_size / len(train_indices),
        settings.get("data_ratio"),
    )
    limit = settings["max_eval_samples"]
    if limit is not None and len(val_indices) > limit:
        val_indices = np.sort(
            np.random.default_rng(dataset_settings["eval_split_seed"]).choice(
                val_indices,
                limit,
                replace=False,
            )
        )
    logging.info(
        "Train: %d episodes / %d frames; val: %d episodes / %d evaluated frames",
        len(train_episodes),
        len(train_indices),
        len(val_episodes),
        len(val_indices),
    )
    run_dir.mkdir(parents=True, exist_ok=True)
    stats_dir = run_dir / "assets" / "lerobot"
    if previous:
        stats = normalize.load(stats_dir)
    else:
        stats = compute_stats(dataset, train_indices)
        normalize.save(stats_dir, stats)
        write_json(
            manifest_path,
            {
                "settings": settings,
                "revision": revision,
                "split": split,
                "train_frames": len(train_indices),
                "val_frames": len(val_indices),
            },
        )
    # Decode actual images before allocating the model, catching video/schema issues early.
    for index in (train_indices[0], val_indices[0]):
        sample = dataset[index]
        logging.info("Checked frame %d, image shapes: %s", index, {k: v.shape for k, v in sample["image"].items()})
    if prepare_only:
        logging.info("Dataset and train-only normalization prepared in %s. Start training with --resume.", run_dir)
        return

    data_config = config_lib.DataConfig(
        repo_id=dataset_settings["repo_id"],
        asset_id="lerobot",
        norm_stats=stats,
        use_quantile_norm=True,
        model_transforms=config_lib.ModelTransformFactory()(config.model),
    )
    mesh = sharding.make_mesh(config.fsdp_devices)
    data_sharding = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec(sharding.DATA_AXIS))
    replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
    train_loader = make_loader(dataset, train_indices, data_config, config, data_sharding)
    val_loader = make_loader(dataset, val_indices, data_config, config, data_sharding, validation=True)
    last_manager, resuming = checkpoints.initialize_checkpoint_dir(
        run_dir / "last",
        keep_period=None,
        overwrite=False,
        resume=settings["resume"],
    )
    best_manager, _ = checkpoints.initialize_checkpoint_dir(
        run_dir / "best",
        keep_period=None,
        overwrite=False,
        resume=settings["resume"],
    )
    best_loss = float("inf")
    if best_manager.latest_step() is not None:
        best_loss = json.loads((run_dir / "best" / str(best_manager.latest_step()) / "metrics.json").read_text())[
            "val_loss"
        ]
    wb = settings["wandb"]
    mode = wb["mode"] if wb["enable"] else "disabled"
    id_path = run_dir / "wandb_id.txt"
    run_id = id_path.read_text().strip() if id_path.exists() else wandb.util.generate_id()
    wandb.init(
        project=wb["project"],
        entity=wb.get("entity"),
        name=settings["job_name"],
        id=run_id,
        resume="allow" if mode == "online" else None,
        mode=mode,
        config=settings,
        dir=str(run_dir),
    )
    if mode != "disabled":
        id_path.write_text(run_id)
    rng, init_rng = jax.random.split(jax.random.key(config.seed))
    state, state_sharding = train.init_train_state(config, init_rng, mesh, resume=resuming)
    if resuming:
        state = checkpoints.restore_state(last_manager, state, train_loader)
    jax.block_until_ready(state)
    train_fn = jax.jit(
        functools.partial(train.train_step, config),
        in_shardings=(replicated, state_sharding, data_sharding),
        out_shardings=(state_sharding, replicated),
        donate_argnums=(1,),
    )
    eval_fn = jax.jit(eval_step, in_shardings=(state_sharding, replicated, data_sharding))
    iterator = iter(train_loader)
    start = int(state.step)
    if start >= config.num_train_steps:
        logging.info(
            "Already at step %d (target %d). Increase data_ratio or num_train_steps to continue.",
            start,
            config.num_train_steps,
        )
    losses = []
    try:
        for step in range(start + 1, config.num_train_steps + 1):
            with sharding.set_mesh(mesh):
                state, info = train_fn(rng, state, next(iterator))
            train_loss = float(info["loss"])
            if not math.isfinite(train_loss):
                raise FloatingPointError(f"Non-finite training loss at step {step}.")
            losses.append(train_loss)
            metrics = {}
            final = step == config.num_train_steps
            if step % settings["log_freq"] == 0 or final:
                metrics["train/loss"] = float(np.mean(losses))
                metrics["train/grad_norm"] = float(info["grad_norm"])
                metrics["train/learning_rate"] = float(config.lr_schedule.create()(step - 1))
                losses.clear()
            if step % settings["eval_steps"] == 0 or final:
                loss = evaluate(state, val_loader, len(val_indices), eval_fn, config.seed, mesh)
                if not math.isfinite(loss):
                    raise FloatingPointError(f"Non-finite validation loss at step {step}.")
                metrics["val/loss"] = loss
                if loss < best_loss:
                    checkpoints.save_state(best_manager, state, train_loader, step)
                    best_manager.wait_until_finished()
                    write_json(run_dir / "best" / str(step) / "metrics.json", {"step": step, "val_loss": loss})
                    best_loss = loss
                metrics["val/best_loss"] = best_loss
            if step % settings["save_steps"] == 0 or final:
                checkpoints.save_state(last_manager, state, train_loader, step)
                # Finish before donating the state's buffers to the next training step.
                last_manager.wait_until_finished()
            if metrics:
                logging.info("Step %d: %s", step, metrics)
                wandb.log(metrics, step=step)
    finally:
        last_manager.close()
        best_manager.close()
        wandb.finish()
    logging.info("Checkpoints: %s/{last,best}/<step> (params, train_state, assets)", run_dir)

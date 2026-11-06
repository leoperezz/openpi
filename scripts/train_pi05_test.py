import argparse
import dataclasses
import json
import os
from pathlib import Path

os.environ["JAX_PLATFORMS"] = "cpu"

import av
import flax.nnx as nnx
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pandas as pd
import pytest

from openpi.training import checkpoints
from openpi.training import config as config_lib
from openpi.training import lerobot_local
from openpi.training import utils
from scripts import pi05_trainer as train_pi05
from scripts import train_pi05 as launcher


@pytest.fixture
def video_dataset(tmp_path):
    (tmp_path / "meta/episodes/chunk-000").mkdir(parents=True)
    (tmp_path / "data/chunk-000").mkdir(parents=True)
    (tmp_path / "videos/camera/chunk-000").mkdir(parents=True)
    info = {
        "codebase_version": "v3.0",
        "fps": 10,
        "video_path": "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4",
        "features": {"state": {"shape": [2]}, "action": {"shape": [2]}, "camera": {"dtype": "video"}},
    }
    (tmp_path / "meta/info.json").write_text(json.dumps(info))
    pd.DataFrame({"task_index": [0]}, index=["stack disks"]).to_parquet(tmp_path / "meta/tasks.parquet")
    pd.DataFrame(
        [
            {
                "episode_index": i,
                "length": 3,
                "videos/camera/chunk_index": 0,
                "videos/camera/file_index": 0,
                "videos/camera/from_timestamp": i * 0.3,
            }
            for i in range(2)
        ]
    ).to_parquet(tmp_path / "meta/episodes/chunk-000/file-000.parquet")
    pd.DataFrame(
        [
            {
                "index": i,
                "episode_index": i // 3,
                "frame_index": i % 3,
                "timestamp": (i % 3) / 10,
                "task_index": 0,
                "state": [i, i],
                "action": [i * 10, i * 10],
            }
            for i in range(6)
        ]
    ).to_parquet(tmp_path / "data/chunk-000/file-000.parquet")
    with av.open(str(tmp_path / "videos/camera/chunk-000/file-000.mp4"), "w") as container:
        stream = container.add_stream("libx264", rate=10)
        stream.width = stream.height = 32
        stream.pix_fmt = "yuv420p"
        for i in range(6):
            frame = av.VideoFrame.from_ndarray(np.full((32, 32, 3), i * 40, dtype=np.uint8), format="rgb24")
            for packet in stream.encode(frame):
                container.mux(packet)
        for packet in stream.encode():
            container.mux(packet)
    return lerobot_local.LeRobotFrames(
        tmp_path,
        {
            "cameras": {"base_0_rgb": "camera"},
            "state_key": "state",
            "action_key": "action",
        },
        horizon=4,
    )


def test_episode_split_reproducible_and_disjoint():
    first = lerobot_local.split_episodes(range(525), 0.1, 42)
    assert first == lerobot_local.split_episodes(range(525), 0.1, 42)
    assert len(first[1]) == 52
    assert set(first[0]).isdisjoint(first[1])
    assert sorted(first[0] + first[1]) == list(range(525))


@pytest.mark.parametrize(("episodes", "fraction"), [([0], 0.1), ([0, 1], 0), ([0, 1], 1)])
def test_bad_splits(episodes, fraction):
    with pytest.raises(ValueError, match="Train/val"):
        lerobot_local.split_episodes(episodes, fraction, 42)


def test_chunks_do_not_cross_episodes(video_dataset):
    np.testing.assert_array_equal(video_dataset.action_chunk(1)[:, 0], [10, 20, 20, 20])
    np.testing.assert_array_equal(video_dataset.action_chunk(5)[:, 0], [50, 50, 50, 50])


def test_v3_video_offsets_and_masks(video_dataset):
    frame = video_dataset[3]  # timestamp zero in episode 1 is 0.3 seconds in the shared video.
    assert abs(float(frame["image"]["base_0_rgb"].mean()) - 120) < 3
    assert frame["prompt"] == "stack disks"
    assert frame["image_mask"]["base_0_rgb"]
    assert not frame["image_mask"]["left_wrist_0_rgb"]
    assert not frame["image"]["left_wrist_0_rgb"].any()


def test_norm_stats_exclude_validation(video_dataset):
    stats = train_pi05.compute_stats(video_dataset, np.array([0, 1, 2]))
    np.testing.assert_allclose(stats["state"].mean, [1, 1])
    np.testing.assert_allclose(stats["actions"].mean, [10, 10])


def test_eval_excludes_padding_and_has_repeatable_rng():
    calls = []

    def evaluate_batch(state, rng, batch):
        calls.append(np.asarray(jax.random.key_data(rng)))
        return batch

    mesh = jax.sharding.Mesh(jax.devices(), ("data",))
    batches = [np.array([1.0, 2.0]), np.array([3.0, 999.0])]
    assert train_pi05.evaluate(None, batches, 3, evaluate_batch, 42, mesh) == 2
    assert train_pi05.evaluate(None, batches, 3, evaluate_batch, 42, mesh) == 2
    np.testing.assert_array_equal(calls[:2], calls[2:])


def test_cli_overrides_and_resume_guards():
    args = argparse.Namespace(
        config=str(Path(__file__).parents[1] / "configs/train.yaml"),
        repo_id="org/data",
        set=["batch_size=4", "wandb.enable=false"],
        resume=True,
    )
    settings = launcher.load_settings(args)
    assert settings["dataset"]["repo_id"] == "org/data"
    assert settings["batch_size"] == 4
    assert settings["wandb"]["enable"] is False
    assert settings["resume"] is True
    train_pi05.check_resume_settings(settings, dict(settings, num_train_steps=10))
    with pytest.raises(ValueError, match="batch_size"):
        train_pi05.check_resume_settings(settings, dict(settings, batch_size=8))
    args.set = ["batch_size=0"]
    with pytest.raises(ValueError, match="batch_size"):
        launcher.load_settings(args)


def test_last_best_checkpoint_retention_and_restore(tmp_path):
    model = nnx.Linear(2, 2, rngs=nnx.Rngs(0))
    graph, params = nnx.split(model)
    tx = optax.adam(0.01)
    state = utils.TrainState(
        step=jnp.array(1), params=params, model_def=graph, opt_state=tx.init(params), tx=tx, ema_decay=None
    )

    class Loader:
        def data_config(self):
            return config_lib.DataConfig()

    last, _ = checkpoints.initialize_checkpoint_dir(tmp_path / "last", keep_period=None, overwrite=False, resume=False)
    best, _ = checkpoints.initialize_checkpoint_dir(tmp_path / "best", keep_period=None, overwrite=False, resume=False)
    checkpoints.save_state(last, state, Loader(), 1)
    checkpoints.save_state(best, state, Loader(), 1)
    last.wait_until_finished()
    best.wait_until_finished()
    state = dataclasses.replace(state, step=jnp.array(2))
    checkpoints.save_state(last, state, Loader(), 2)
    last.wait_until_finished()
    assert last.all_steps() == [2]
    assert best.all_steps() == [1]
    restored = checkpoints.restore_state(last, state, Loader())
    assert int(restored.step) == 2
    assert (tmp_path / "last/2/params").is_dir()
    assert (tmp_path / "best/1/train_state").is_dir()
    last.close()
    best.close()
    resumed, resuming = checkpoints.initialize_checkpoint_dir(
        tmp_path / "last", keep_period=None, overwrite=False, resume=True
    )
    assert resuming
    assert int(checkpoints.restore_state(resumed, state, Loader()).step) == 2
    resumed.close()


def test_v2_video_reader(video_dataset):
    root = video_dataset.root
    info = video_dataset.info | {
        "codebase_version": "v2.1",
        "chunks_size": 1000,
        "video_path": "videos/camera/chunk-000/file-000.mp4",
    }
    (root / "meta/info.json").write_text(json.dumps(info))
    # v2 videos start at timestamp zero for each episode.
    (root / "meta/episodes.jsonl").write_text(
        "\n".join(json.dumps({"episode_index": i, "length": 3}) for i in range(2))
    )
    (root / "meta/tasks.jsonl").write_text(json.dumps({"task_index": 0, "task": "stack disks"}))
    frames = lerobot_local.LeRobotFrames(
        root, {"cameras": {"base_0_rgb": "camera"}, "state_key": "state", "action_key": "action"}, 4
    )
    assert frames[3]["image"]["base_0_rgb"].mean() < 3
    np.testing.assert_array_equal(frames.action_chunk(2)[:, 0], [20, 20, 20, 20])


@pytest.mark.parametrize("data_ratio", [None, 1.0])
def test_training_loop_and_resume_with_tiny_model(video_dataset, tmp_path, monkeypatch, data_ratio):
    from openpi import transforms
    from openpi.models import model as model_lib

    class TinyModel(model_lib.BaseModel):
        def __init__(self):
            super().__init__(action_dim=2, action_horizon=4, max_token_len=1)
            self.linear = nnx.Linear(2, 2, rngs=nnx.Rngs(0))

        def compute_loss(self, rng, observation, actions, *, train=False):
            return jnp.mean((self.linear(observation.state)[:, None, :] - actions) ** 2, axis=-1)

        def sample_actions(self, rng, observation, **kwargs):
            return self.linear(observation.state)[:, None, :]

    def init_tiny(config, rng, mesh, *, resume):
        graph, params = nnx.split(TinyModel())
        tx = optax.adam(1e-3)
        state = utils.TrainState(
            step=jnp.array(0), params=params, model_def=graph, opt_state=tx.init(params), tx=tx, ema_decay=None
        )
        replicated = jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec())
        return jax.device_put(state, replicated), replicated

    # Only replace the GPU/model/tokenizer prerequisites; exercise the real training,
    # validation, normalization, video loader, checkpoint and resume code on CPU.
    monkeypatch.setattr(jax, "default_backend", lambda: "gpu")
    monkeypatch.setattr(train_pi05.train, "init_train_state", init_tiny)
    monkeypatch.setattr(
        config_lib.ModelTransformFactory,
        "__call__",
        lambda self, model: transforms.Group(inputs=[lambda data: {k: v for k, v in data.items() if k != "prompt"}]),
    )
    args = argparse.Namespace(
        config=str(Path(__file__).parents[1] / "configs/train.yaml"), repo_id=None, set=[], resume=False
    )
    settings = launcher.load_settings(args)
    settings.update(
        output_dir=str(tmp_path / "runs"),
        batch_size=2,
        num_workers=0,
        num_train_steps=2 if data_ratio is None else None,
        data_ratio=data_ratio,
        eval_steps=1,
        save_steps=1,
        log_freq=1,
        max_eval_samples=3,
    )
    settings["wandb"]["enable"] = False
    settings["dataset"].update(
        root=str(video_dataset.root), state_key="state", action_key="action", cameras={"base_0_rgb": "camera"}
    )
    settings["policy"]["action_horizon"] = 4
    train_pi05.run(settings)
    run_dir = Path(settings["output_dir"]) / settings["job_name"]
    assert (run_dir / "last/2/train_state").is_dir()
    best_dirs = [path for path in (run_dir / "best").iterdir() if path.name.isdigit()]
    assert len(best_dirs) == 1
    first_best = json.loads((best_dirs[0] / "metrics.json").read_text())["val_loss"]
    manifest = json.loads((run_dir / "run.json").read_text())
    assert manifest["settings"]["num_train_steps"] == 2
    assert manifest["train_frames"] == 3
    settings.update(resume=True, num_train_steps=3, data_ratio=None if data_ratio is None else 2.0)
    train_pi05.run(settings)
    assert (run_dir / "last/3/train_state").is_dir()
    assert not (run_dir / "last/2").exists()
    best_dirs = [path for path in (run_dir / "best").iterdir() if path.name.isdigit()]
    assert len(best_dirs) == 1
    assert json.loads((best_dirs[0] / "metrics.json").read_text())["val_loss"] <= first_best
    assert (best_dirs[0] / "assets/lerobot/norm_stats.json").is_file()


def test_device_selection_overrides_inherited_visibility(monkeypatch):
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2,3")
    monkeypatch.setenv("CUDA_DEVICE_ORDER", "FASTEST_FIRST")
    launcher.select_device({"device": "cuda:1", "fsdp_devices": 1})
    assert os.environ["CUDA_VISIBLE_DEVICES"] == "1"
    assert os.environ["CUDA_DEVICE_ORDER"] == "PCI_BUS_ID"


@pytest.mark.parametrize("device", ["cuda", "cpu", "cuda:-1", "cuda:0,1", None])
def test_device_selection_rejects_invalid_devices(device):
    with pytest.raises(ValueError, match="device must select one GPU"):
        launcher.validate_device({"device": device, "fsdp_devices": 1})


def test_single_gpu_rejects_fsdp():
    with pytest.raises(ValueError, match="fsdp_devices: 1"):
        launcher.validate_device({"device": "cuda:1", "fsdp_devices": 2})


def test_launcher_does_not_import_cuda_consumers():
    import subprocess
    import sys

    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from scripts import train_pi05; import sys; "
            "assert 'jax' not in sys.modules; assert 'torch' not in sys.modules; "
            "train_pi05.select_device({'device': 'cuda:1', 'fsdp_devices': 1})",
        ],
        cwd=Path(__file__).parents[1],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize("ratio", [0, -1, True, "1.5", float("nan"), float("inf")])
def test_invalid_data_ratio(tmp_path, ratio):
    import yaml

    settings = yaml.safe_load((Path(__file__).parents[1] / "configs/train.yaml").read_text())
    settings["data_ratio"] = ratio
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(settings))
    args = argparse.Namespace(config=str(config_path), repo_id=None, set=[], resume=False)
    with pytest.raises(ValueError, match="data_ratio"):
        launcher.load_settings(args)


@pytest.mark.parametrize(("ratio", "expected"), [(1.5, 19), (1, 13), (0.01, 1), (None, 42)])
def test_training_budget(ratio, expected):
    settings = {"data_ratio": ratio, "batch_size": 8, "num_train_steps": 42}
    resolved = train_pi05.resolve_training_settings(settings, train_frames=100)
    assert resolved["num_train_steps"] == expected
    assert settings["num_train_steps"] == 42


def test_legacy_config_without_data_ratio(tmp_path):
    import yaml

    settings = yaml.safe_load((Path(__file__).parents[1] / "configs/train.yaml").read_text())
    del settings["data_ratio"]
    config_path = tmp_path / "train.yaml"
    config_path.write_text(yaml.safe_dump(settings))
    args = argparse.Namespace(config=str(config_path), repo_id=None, set=[], resume=False)
    loaded = launcher.load_settings(args)
    assert loaded["data_ratio"] is None
    assert loaded["num_train_steps"] == 60000
    args.set = ["num_train_steps=null"]
    with pytest.raises(ValueError, match="num_train_steps"):
        launcher.load_settings(args)
    args.set = ["num_train_steps=null", "data_ratio=1.5"]
    assert launcher.load_settings(args)["data_ratio"] == 1.5

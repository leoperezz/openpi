"""Fine-tune pi05 on one GPU selected with device: cuda:N in the YAML."""

import argparse
import math
import os
from pathlib import Path
import re

import yaml


def load_settings(args):
    with Path(args.config).open() as stream:
        settings = yaml.safe_load(stream)
    settings.setdefault("data_ratio", None)
    settings.setdefault("num_train_steps", None)
    if args.repo_id:
        settings["dataset"]["repo_id"] = args.repo_id
    for override in args.set:
        key, separator, value = override.partition("=")
        if not separator:
            raise ValueError(f"Expected --set key=value, got {override!r}.")
        target = settings
        parts = key.split(".")
        for part in parts[:-1]:
            target = target[part]
        if parts[-1] not in target:
            raise ValueError(f"Unknown configuration key: {key}")
        target[parts[-1]] = yaml.safe_load(value)
    settings.setdefault("device", "cuda:0")
    validate_device(settings)
    if args.resume:
        settings["resume"] = True
    for key in ("batch_size", "log_freq", "eval_steps", "save_steps", "fsdp_devices"):
        if type(settings[key]) is not int or settings[key] < 1:
            raise ValueError(f"{key} must be a positive integer.")
    ratio = settings["data_ratio"]
    if ratio is not None:
        if type(ratio) not in (int, float) or not math.isfinite(ratio) or ratio <= 0:
            raise ValueError("data_ratio must be a finite positive number or null.")
    elif type(settings["num_train_steps"]) is not int or settings["num_train_steps"] < 1:
        raise ValueError("num_train_steps must be a positive integer when data_ratio is null.")
    if settings["num_workers"] < 0:
        raise ValueError("num_workers must be nonnegative.")
    if settings["max_eval_samples"] is not None and settings["max_eval_samples"] < 1:
        raise ValueError("max_eval_samples must be positive or null.")
    if settings["policy"]["action_horizon"] < 1:
        raise ValueError("action_horizon must be positive.")
    schedule = settings["optimizer"]
    if not 0 <= schedule["warmup_steps"] < schedule["decay_steps"]:
        raise ValueError("Require 0 <= warmup_steps < decay_steps.")
    if not settings["job_name"] or Path(settings["job_name"]).name != settings["job_name"]:
        raise ValueError("job_name must be a single directory name.")
    if settings["job_name"] in (".", ".."):
        raise ValueError("Invalid job_name.")
    return settings


def validate_device(settings):
    device = settings["device"]
    if not isinstance(device, str) or re.fullmatch(r"cuda:[0-9]+", device) is None:
        raise ValueError("device must select one GPU, for example cuda:0 or cuda:1.")
    if settings["fsdp_devices"] != 1:
        raise ValueError("Single-GPU training requires fsdp_devices: 1.")


def select_device(settings):
    validate_device(settings)
    # The configured ordinal refers to the host CUDA device, not an index into
    # an inherited CUDA_VISIBLE_DEVICES list. JAX sees the selected GPU as device 0.
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = str(int(settings["device"].split(":")[1]))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("repo_id", nargs="?", help="Hugging Face LeRobot dataset ID")
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument(
        "--set", action="append", default=[], metavar="KEY=VALUE", help="Override YAML using dotted keys"
    )
    parser.add_argument("--resume", action="store_true")
    parser.add_argument(
        "--prepare-only", action="store_true", help="Download/check data and compute train-only statistics"
    )
    args = parser.parse_args()
    settings = load_settings(args)
    select_device(settings)
    # Import CUDA consumers only after visibility has been configured.
    if __package__:
        from .pi05_trainer import run
    else:
        from pi05_trainer import run
    run(settings, prepare_only=args.prepare_only)


if __name__ == "__main__":
    main()

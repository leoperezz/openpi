"""LeRobot v2/v3 parquet/video reader, independent of OpenPI's pinned v2 API.

Actions remain in the recorded robot coordinates. Chunks clamp to the episode's
last action; neither action chunks nor train/validation splits cross episodes.
"""

import json
from pathlib import Path

import av
from huggingface_hub import HfApi
from huggingface_hub import snapshot_download
import numpy as np
import pandas as pd

IMAGE_SLOTS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


def split_episodes(episodes, fraction, seed):
    ids = np.asarray(sorted(episodes), dtype=np.int64)
    if len(ids) < 2 or not 0 < fraction < 1:
        raise ValueError("Train/val requires at least two episodes and 0 < eval_split < 1.")
    ids = np.random.default_rng(seed).permutation(ids)
    count = min(len(ids) - 1, max(1, round(len(ids) * fraction)))
    return sorted(ids[count:].tolist()), sorted(ids[:count].tolist())


def resolve_dataset(settings):
    if settings.get("root"):
        root = Path(settings["root"]).expanduser().resolve()
        return root, f"local:{root}"
    # Pin main to a commit before downloading so all files come from one revision.
    repo = settings["repo_id"]
    revision = HfApi().dataset_info(repo, revision=settings["revision"]).sha
    root = snapshot_download(
        repo,
        repo_type="dataset",
        revision=revision,
        allow_patterns=["meta/**", "data/**", *[f"videos/{key}/**" for key in settings["cameras"].values()]],
    )
    return Path(root), revision


def read_video_frame(path, timestamp, fps):
    """Seek to the previous keyframe, then choose the closest presentation time."""
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        container.seek(max(0, int(timestamp / stream.time_base)), stream=stream, backward=True)
        best, distance = None, float("inf")
        for frame in container.decode(stream):
            if frame.pts is None:
                continue
            frame_time = float(frame.pts * stream.time_base)
            delta = abs(frame_time - timestamp)
            if delta < distance:
                best, distance = frame, delta
            if frame_time >= timestamp:
                break
        if best is None or distance > 1 / fps + 1e-4:
            raise ValueError(f"No video frame near {timestamp:.6f}s in {path} (distance={distance}).")
        return best.to_ndarray(format="rgb24")


class LeRobotFrames:
    def __init__(self, root, settings, horizon):
        self.root = Path(root)
        self.info = json.loads((self.root / "meta/info.json").read_text())
        self.version = self.info["codebase_version"]
        self.cameras = settings["cameras"]
        self.horizon = horizon
        if not self.cameras or set(self.cameras) - set(IMAGE_SLOTS):
            raise ValueError(f"cameras must map at least one of {IMAGE_SLOTS} to dataset features.")
        features = self.info["features"]
        for key in self.cameras.values():
            if key not in features or features[key]["dtype"] != "video":
                raise ValueError(f"Camera {key!r} must be a video feature in meta/info.json.")
        for key in (settings["state_key"], settings["action_key"]):
            shape = features[key]["shape"]
            if len(shape) != 1 or not 1 <= shape[0] <= 32:
                raise ValueError(f"{key}: pi05 requires a vector with at most 32 dimensions, got {shape}.")
        if self.version.startswith("v3."):
            metadata = pd.concat(pd.read_parquet(p) for p in sorted(self.root.glob("meta/episodes/**/*.parquet")))
            self.episodes = {int(row["episode_index"]): row for row in metadata.to_dict("records")}
            tasks = pd.read_parquet(self.root / "meta/tasks.parquet")
            self.tasks = {int(row.task_index): str(name) for name, row in tasks.iterrows()}
        elif self.version.startswith("v2."):
            self.episodes = {
                row["episode_index"]: row
                for row in map(json.loads, (self.root / "meta/episodes.jsonl").read_text().splitlines())
            }
            self.tasks = {
                row["task_index"]: row["task"]
                for row in map(json.loads, (self.root / "meta/tasks.jsonl").read_text().splitlines())
            }
        else:
            raise ValueError(f"Unsupported LeRobot version: {self.version}")
        columns = [
            "index",
            "episode_index",
            "frame_index",
            "timestamp",
            "task_index",
            settings["state_key"],
            settings["action_key"],
        ]
        table = pd.concat(pd.read_parquet(p, columns=columns) for p in sorted(self.root.glob("data/**/*.parquet")))
        table = table.sort_values("index").reset_index(drop=True)
        self.states = np.stack(table[settings["state_key"]]).astype(np.float32)
        self.actions = np.stack(table[settings["action_key"]]).astype(np.float32)
        if not np.isfinite(self.states).all() or not np.isfinite(self.actions).all():
            raise ValueError("Non-finite states or actions in dataset.")
        self.episode_ids = table["episode_index"].to_numpy(dtype=np.int64)
        self.timestamps = table["timestamp"].to_numpy()
        self.task_ids = table["task_index"].to_numpy(dtype=np.int64)
        self.ends = np.empty(len(table), dtype=np.int64)
        for episode, meta in self.episodes.items():
            indices = np.flatnonzero(self.episode_ids == episode)
            if len(indices) != meta["length"] or not len(indices) or np.any(np.diff(indices) != 1):
                raise ValueError(f"Missing or non-contiguous data in episode {episode}.")
            if not np.array_equal(table.iloc[indices]["frame_index"], np.arange(len(indices))):
                raise ValueError(f"Non-contiguous frame indices in episode {episode}.")
            self.ends[indices] = indices[-1]
        if set(self.episode_ids) != set(self.episodes):
            raise ValueError("Episode metadata and frame data disagree.")

    def __len__(self):
        return len(self.states)

    def action_chunk(self, index):
        indices = np.minimum(index + np.arange(self.horizon), self.ends[index])
        return self.actions[indices]

    def __getitem__(self, index):
        episode = int(self.episode_ids[index])
        meta = self.episodes[episode]
        images = {}
        for slot, key in self.cameras.items():
            timestamp = float(self.timestamps[index])
            if self.version.startswith("v3."):
                prefix = f"videos/{key}"
                path = self.info["video_path"].format(
                    video_key=key,
                    chunk_index=int(meta[f"{prefix}/chunk_index"]),
                    file_index=int(meta[f"{prefix}/file_index"]),
                )
                timestamp += meta[f"{prefix}/from_timestamp"]
            else:
                path = self.info["video_path"].format(
                    video_key=key,
                    episode_chunk=episode // self.info["chunks_size"],
                    episode_index=episode,
                )
            images[slot] = read_video_frame(self.root / path, timestamp, self.info["fps"])
        template = next(iter(images.values()))
        return {
            "state": self.states[index].copy(),
            "actions": self.action_chunk(index).copy(),
            "image": {slot: images.get(slot, np.zeros_like(template)) for slot in IMAGE_SLOTS},
            "image_mask": {slot: np.bool_(slot in images) for slot in IMAGE_SLOTS},
            "prompt": self.tasks[int(self.task_ids[index])],
        }

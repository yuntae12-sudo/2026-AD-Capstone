"""Atomic episode-boundary checkpoints, including replay and RNG states."""

import hashlib
import json
from pathlib import Path
import shutil
import tempfile
from flax import serialization
from .replay import Replay


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def save_checkpoint(path, state, replay, versions, run_state, *, episode_boundary):
    if not episode_boundary:
        raise ValueError("checkpoint allowed only at an episode boundary")
    if replay.versions != versions:
        raise ValueError("replay/learner version mismatch")
    path = Path(path)
    if path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=path.name + "-", dir=path.parent))
    try:
        (temporary / "learner.msgpack").write_bytes(serialization.to_bytes(state))
        replay.save(temporary / "replay.npz")
        manifest = dict(
            versions=versions,
            episode_boundary=True,
            run_state=run_state,
            hashes={
                name: file_hash(temporary / name)
                for name in ("learner.msgpack", "replay.npz")
            },
        )
        (temporary / "manifest.json").write_text(
            json.dumps(manifest, indent=2, sort_keys=True) + "\n"
        )
        temporary.rename(path)
    except BaseException:
        shutil.rmtree(temporary)
        raise


def load_checkpoint(path, template, versions):
    path = Path(path)
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["versions"] != versions or not manifest["episode_boundary"]:
        raise ValueError("incompatible checkpoint")
    for name, expected in manifest["hashes"].items():
        if (
            name not in ("learner.msgpack", "replay.npz")
            or file_hash(path / name) != expected
        ):
            raise ValueError("checkpoint checksum mismatch")
    state = serialization.from_bytes(template, (path / "learner.msgpack").read_bytes())
    return state, Replay.load(path / "replay.npz", versions), manifest["run_state"]

"""Bounded array replay; no simulator objects or mutable observation dictionaries."""

from dataclasses import dataclass
import json
import zipfile
from pathlib import Path
import numpy as np
from src.contracts.observation import CORE_SHAPES, ITEM_SHAPES, FEATURE_COUNT

ITEM_COUNT = sum(int(np.prod(s)) for s in ITEM_SHAPES.values())


def pack_observation(observation):
    return (
        observation.flatten(),
        np.concatenate([observation.field_masks[k].ravel() for k in CORE_SHAPES]),
        np.concatenate([observation.item_masks[k].ravel() for k in ITEM_SHAPES]),
    )


def unpack_observation(features, fields, items):
    result = dict(features={}, field_masks={}, item_masks={})
    for name, shapes, packed in (
        ("features", CORE_SHAPES, features),
        ("field_masks", CORE_SHAPES, fields),
        ("item_masks", ITEM_SHAPES, items),
    ):
        start = 0
        for key, shape in shapes.items():
            count = int(np.prod(shape))
            result[name][key] = packed[..., start : start + count].reshape(
                (*packed.shape[:-1], *shape)
            )
            start += count
    return result


@dataclass(frozen=True)
class Transition:
    observation: object
    raw_u: np.ndarray
    reward: float
    next_observation: object
    delta_ticks: int
    bootstrap: bool
    sample_valid: bool
    teacher_accepted: bool
    episode_uid: str
    scene_id: str
    tick: int
    terminated: bool = False
    truncated: bool = False

    @property
    def discount(self):
        return 0.99**self.delta_ticks


class Replay:
    def __init__(self, capacity=50000, seed=0, versions=None, ram_limit=4 * 1024**3):
        if not isinstance(capacity, int) or capacity <= 0:
            raise ValueError("positive replay capacity required")
        layout = {
            "obs": ((FEATURE_COUNT,), np.float32),
            "next_obs": ((FEATURE_COUNT,), np.float32),
            "fields": ((FEATURE_COUNT,), bool),
            "next_fields": ((FEATURE_COUNT,), bool),
            "items": ((ITEM_COUNT,), bool),
            "next_items": ((ITEM_COUNT,), bool),
            "raw_u": ((30, 2), np.float64),
            "reward": ((), np.float64),
            "delta_ticks": ((), np.int32),
            "discount": ((), np.float64),
            "bootstrap": ((), bool),
            "teacher_accepted": ((), bool),
            "episode_uid": ((), "U36"),
            "scene_id": ((), "U64"),
            "tick": ((), np.int32),
            "terminated": ((), bool),
            "truncated": ((), bool),
        }
        required = capacity * sum(
            int(np.prod(shape)) * np.dtype(dtype).itemsize
            for shape, dtype in layout.values()
        )
        if required > ram_limit:
            raise MemoryError(
                f"replay arrays require {required} bytes, budget {ram_limit}"
            )
        self.arrays = {
            k: np.empty((capacity, *shape), dtype)
            for k, (shape, dtype) in layout.items()
        }
        self.capacity, self.size, self.cursor = capacity, 0, 0
        self.versions = dict(versions or {})
        self.rng = np.random.default_rng(seed)

    @property
    def nbytes(self):
        return sum(a.nbytes for a in self.arrays.values())

    def add(self, transition):
        t = transition
        if not t.sample_valid:
            return False
        u = np.asarray(t.raw_u)
        if (
            u.shape != (30, 2)
            or u.dtype != np.float64
            or not np.isfinite(u).all()
            or not np.isfinite(t.reward)
        ):
            raise ValueError(
                "corrupt transition: stop updates, do not repair raw actions"
            )
        if t.delta_ticks < 1 or (t.terminated or t.truncated) and t.bootstrap:
            raise ValueError("invalid transition clock/bootstrap")
        if len(t.episode_uid) > 36 or len(t.scene_id) > 64:
            raise ValueError("replay ID exceeds fixed storage")
        obs, fields, items = pack_observation(t.observation)
        next_obs, next_fields, next_items = pack_observation(t.next_observation)
        if not np.isfinite(obs).all() or not np.isfinite(next_obs).all():
            raise ValueError("nonfinite observation")
        row = dict(
            obs=obs,
            fields=fields,
            items=items,
            next_obs=next_obs,
            next_fields=next_fields,
            next_items=next_items,
            raw_u=u,
            reward=t.reward,
            delta_ticks=t.delta_ticks,
            discount=t.discount,
            bootstrap=t.bootstrap,
            teacher_accepted=t.teacher_accepted,
            episode_uid=t.episode_uid,
            scene_id=t.scene_id,
            tick=t.tick,
            terminated=t.terminated,
            truncated=t.truncated,
        )
        for k, v in row.items():
            self.arrays[k][self.cursor] = v
        self.cursor = (self.cursor + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)
        return True

    def sample(self, batch_size=256, teacher_only=False):
        eligible = (
            np.flatnonzero(self.arrays["teacher_accepted"][: self.size])
            if teacher_only
            else np.arange(self.size)
        )
        if len(eligible) == 0:
            raise ValueError("empty eligible replay")
        indices = self.rng.choice(eligible, batch_size, replace=True)
        result = {
            k: v[indices]
            for k, v in self.arrays.items()
            if k not in ("episode_uid", "scene_id")
        }
        result["observation"] = unpack_observation(
            result.pop("obs"), result.pop("fields"), result.pop("items")
        )
        result["next_observation"] = unpack_observation(
            result.pop("next_obs"), result.pop("next_fields"), result.pop("next_items")
        )
        return result

    def save(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        metadata = json.dumps(
            dict(
                capacity=self.capacity,
                size=self.size,
                cursor=self.cursor,
                versions=self.versions,
                rng=self.rng.bit_generator.state,
            )
        )
        with path.open("wb") as f:
            np.savez(
                f,
                metadata=np.array(metadata),
                **{k: v[: self.size] for k, v in self.arrays.items()},
            )

    @classmethod
    def load(cls, path, versions):
        with np.load(path, allow_pickle=False) as data:
            meta = json.loads(str(data["metadata"]))
            if meta["versions"] != versions:
                raise ValueError("replay version mismatch")
        result = cls(meta["capacity"], versions=versions)
        if (
            not 0 <= meta["size"] <= result.capacity
            or not 0 <= meta["cursor"] < result.capacity
        ):
            raise ValueError("invalid replay index metadata")
        # A full temporary obs array would add ~1.2 GB during 50k replay load.
        # Stream the stored NPY members into their final arrays in 1 MiB blocks.
        with zipfile.ZipFile(path) as archive:
            for key, destination in result.arrays.items():
                with archive.open(key + ".npy") as source:
                    version = np.lib.format.read_magic(source)
                    read_header = (
                        np.lib.format.read_array_header_1_0
                        if version == (1, 0)
                        else np.lib.format.read_array_header_2_0
                    )
                    shape, fortran, dtype = read_header(source)
                    target = destination[: meta["size"]]
                    if shape != target.shape or dtype != target.dtype or fortran:
                        raise ValueError("incompatible replay array")
                    if target.size == 0:
                        continue
                    raw = memoryview(target).cast("B")
                    offset = 0
                    while offset < len(raw):
                        count = source.readinto(raw[offset : offset + 1024 * 1024])
                        if not count:
                            raise ValueError("truncated replay array")
                        offset += count
        result.size, result.cursor = meta["size"], meta["cursor"]
        result.rng.bit_generator.state = meta["rng"]
        return result

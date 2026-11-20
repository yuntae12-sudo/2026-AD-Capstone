"""Record the exact learner passed to the first full SAC update, without changing it."""
import hashlib
import json
from pathlib import Path


def capture_first_update(update, serialize, directory, *, expected=None, provenance=None):
    directory = Path(directory)
    captured = False

    def wrapped(state, *args, **kwargs):
        nonlocal captured
        if not captured:
            if int(state.updates) != 0 or int(state.bc_updates) != 0:
                raise ValueError("initial capture requires a fresh pure SAC learner")
            raw = serialize(state)
            digest = hashlib.sha256(raw).hexdigest()
            reference = None if expected is None else Path(expected).read_bytes()
            record = dict(
                status="COMPLETE" if reference is None or reference == raw else "FAILED",
                scope="LEARNER_BEFORE_FIRST_FULL_SAC_UPDATE",
                learner_sha256=digest, updates=0, bc_updates=0,
                reference_file=None if expected is None else str(expected),
                reference_sha256=None if reference is None else hashlib.sha256(reference).hexdigest(),
                reference_bytes_equal=None if reference is None else reference == raw,
                provenance=provenance or {},
                note="Observation only; this does not prove identical subsequent GPU arithmetic or trajectories.",
            )
            with (directory / "initial-learner.msgpack").open("xb") as stream:
                stream.write(raw)
            with (directory / "initial-state-audit.json").open("x") as stream:
                json.dump(record, stream, indent=2)
                stream.write("\n")
            if record["status"] != "COMPLETE":
                raise ValueError("fresh initial learner bytes differ from the reference")
            captured = True
        return update(state, *args, **kwargs)

    return wrapped

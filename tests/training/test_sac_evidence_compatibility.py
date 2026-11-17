import pytest
from scripts.revalidate_sac_evidence import require_unchanged_dependencies


def sources():
    return {
        "src/training/sac/networks.py": b"def sample_action(): return 1\nclass Actor: pass\nclass Critic: pass\n",
        "src/training/sac/learner.py": b"def optimizer(): return 1\n",
        "src/training/sac/runtime.py": b"def versions(): return 1\ndef selected_scenes(): return 2\n",
        "src/environment/womd_control/control.py": b"def check(): return 1\n",
    }


def test_evidence_reuse_accepts_only_learner_critic_and_metadata_changes():
    old = sources()
    new = dict(old)
    new["src/training/sac/networks.py"] = old["src/training/sac/networks.py"].replace(
        b"class Critic: pass", b"class Critic: layer_norm=True"
    )
    new["src/training/sac/learner.py"] = b"def optimizer(): return 2\n"
    new["src/training/sac/runtime.py"] = old["src/training/sac/runtime.py"].replace(
        b"versions(): return 1", b"versions(): return 2"
    )
    assert len(require_unchanged_dependencies(old, new)) == 3


@pytest.mark.parametrize(
    "path,old_text,new_text",
    [
        ("src/training/sac/networks.py", b"return 1", b"return 2"),
        (
            "src/training/sac/networks.py",
            b"class Actor: pass",
            b"class Actor: changed=True",
        ),
        (
            "src/training/sac/runtime.py",
            b"selected_scenes(): return 2",
            b"selected_scenes(): return 3",
        ),
        ("src/environment/womd_control/control.py", b"return 1", b"return 0"),
    ],
)
def test_sampler_actor_data_and_checker_changes_forbid_evidence_reuse(
    path, old_text, new_text
):
    old = sources()
    new = dict(old)
    new[path] = new[path].replace(old_text, new_text)
    with pytest.raises(ValueError, match="dependency changed"):
        require_unchanged_dependencies(old, new)

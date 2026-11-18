from scripts.diagnose_sac_critic_generalization import scene_partition


def test_scene_partition_keeps_repeats_together_and_fixed_probes_in_fit():
    scenes = [f"scene-{i}" for i in range(50)]
    held = scene_partition(scenes, [])
    forced = next(scene for scene, test in zip(scenes, held) if test)
    repeated = scenes + scenes
    result = scene_partition(repeated, [forced])
    assert result[:50] == result[50:]
    assert not result[scenes.index(forced)]
    assert any(result) and not all(result)
    fit_scenes = {s for s, h in zip(repeated, result) if not h}
    held_scenes = {s for s, h in zip(repeated, result) if h}
    assert fit_scenes.isdisjoint(held_scenes)

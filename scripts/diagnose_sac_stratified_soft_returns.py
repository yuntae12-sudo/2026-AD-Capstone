#!/usr/bin/env python3
"""Read-only refresh/retention policy returns at preselected actual train states.

Condition on the model's mean first action, then use its original stochastic
Policy through the contract terminal boundary. Small MC samples estimate a
policy return; they are not exact expected Q targets or independent test scenes.
"""
import argparse,json,os,sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cuda,cpu')
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
os.environ.setdefault('TF_CPP_MIN_LOG_LEVEL','3')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def discounted_terms(reward,ell,index,alpha,delta_ticks=1):
    if delta_ticks != 1:
        raise ValueError('this KEEP/FOLLOW diagnostic requires one actual tick')
    plain=.99**index*reward
    entropy=0.0 if index==0 else -.99**index*alpha*ell
    return plain,entropy


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for n in ['diagnostic','states-from','reference-probe','manifest','output']:
        p.add_argument('--'+n,type=Path,required=True)
    p.add_argument('--replicates',type=int,default=4)
    args=p.parse_args()
    if not 1<=args.replicates<=8:p.error('replicates must be1..8')
    if args.output.exists():raise FileExistsError(args.output)
    import tensorflow as tf
    tf.config.set_visible_devices([],'GPU')
    import jax
    import jax.numpy as jnp
    import numpy as np
    from flax import serialization
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from src.contracts.observation import CoreObservation
    from src.training.sac.checkpoint import file_hash
    from src.training.sac.learner import initialize
    from src.training.sac.networks import Actor,sample_action,observation_speed
    from src.training.sac.policy import Policy,tensor_observation,proposal_from_action
    from src.training.sac.replay import Replay,pack_observation
    from src.training.sac.collector import collect_step
    from src.training.sac.runtime import versions,selected_scenes,write_report
    from src.environment.womd_control.environment import SACPlanningEnvironment
    from src.scenarios.womd20 import parse_scenario
    from scripts.sac_diagnostic_critic import diagnostic_critic,critic_factory_hash
    from scripts.sac_stratified_diagnostic_audit import audit
    from scripts.diagnose_sac_q_values import fork_environment,observation_fingerprint
    version=versions(args.manifest)
    source=json.loads((args.diagnostic/'report.json').read_text())
    origin=json.loads((args.states_from/'report.json').read_text())
    ref=json.loads(args.reference_probe.read_text())
    assert source['status']==origin['status']==ref['status']=='COMPLETE'
    assert source['versions']==origin['versions']==ref['versions']==version
    assert ref['model_sha256']==origin['arms']['online']['learner_sha256']
    assert ref['diagnostic_sha256']==file_hash(args.states_from/'report.json')
    audit(args.diagnostic,args.manifest);audit(args.states_from,args.manifest)
    rows=[json.loads(x) for x in (args.states_from/'ticks.jsonl').read_text().splitlines()]
    selected=[c for c in ref['cases'] if c['selection_role']=='accepted_predecessor']
    assert len(selected)==2 and {c['provider'] for c in selected}=={'fixed_keep','fixed_follow'}
    a=Replay.load(args.states_from/'replay.npz',version).arrays
    arm=source['arms']['online'];weights=args.diagnostic/'online-learner.msgpack'
    assert file_hash(weights)==arm['learner_sha256']
    cpu,gpu=jax.devices('cpu')[0],jax.devices('gpu')[0]
    args.output.mkdir(parents=True)
    report=dict(status='RUNNING',qualified=False,scope='FROZEN_POLICY_SAMPLED_SOFT_RETURNS',versions=version,
        script_sha256=file_hash(__file__),critic_factory_sha256=critic_factory_hash(),
        collection_audit_adapter_sha256=file_hash(Path(__file__).with_name("sac_stratified_diagnostic_audit.py")),
        source_diagnostic_sha256=file_hash(args.diagnostic/'report.json'),model_sha256=file_hash(weights),
        replay_sha256=file_hash(args.diagnostic/'replay.npz'),states_source_sha256=file_hash(args.states_from/'report.json'),
        states_replay_sha256=file_hash(args.states_from/'replay.npz'),reference_probe_sha256=file_hash(args.reference_probe),
        selected_trace_indices=[c['trace_index'] for c in selected],replicates=args.replicates,cases=[],
        note='Two previously selected train states; no Q-based case selection. Mean first action has no entropy term. Frozen original Policy Gaussian suffix includes contract ell (not raw log_pi); no critic bootstrap at actual terminal. Same replicate seeds across states/models are paired, not independent scenes. Small MC estimates, not exact expected Q. No teacher, replay writes or learner updates.')
    path=args.output/'report.json';write_report(path,report)
    try:
        with jax.default_device(gpu):
            state=initialize(tensor_observation(CoreObservation.empty()),source['seed'],
                actor=diagnostic_actor(arm,restoring=True),
                critic=diagnostic_critic(arm,restoring=True))
            state=jax.device_put(serialization.from_bytes(state,weights.read_bytes()),gpu)
            assert int(state.updates)==source['updates'] and int(state.bc_updates)==0
            mean_policy=Policy(state.actor,deterministic=True)
            alpha=float(jnp.exp(state.log_alpha));report['alpha']=alpha
            @jax.jit
            def qvalue(obs,raw):
                return jnp.minimum(state.q1.apply_fn({'params':state.q1.params},obs,raw),state.q2.apply_fn({'params':state.q2.params},obs,raw))
        scenes={(r['scene_id'],r['provider']):p for r,p in selected_scenes(args.manifest,'train')}
        for fixed in selected:
            row=rows[fixed['trace_index']]
            assert row['lineage']['kind']=='main' and row['check_valid'] and row['pre_tick']==fixed['tick']
            parent_rows=[(i,r) for i,r in enumerate(rows) if r['lineage']['kind']=='main' and r['episode_uid']==row['episode_uid'] and r['pre_tick']<=row['pre_tick']]
            assert parent_rows[0][1]['pre_tick']==0
            with jax.default_device(cpu):
                env=SACPlanningEnvironment();boundary=env.reset(parse_scenario(scenes[(row['scene_id'],row['provider'])].read_bytes()),seed=origin['seed'],provider_spec=row['provider'])
                for i,parent in parent_rows:
                    ix=5000+i;assert env.tick==parent['pre_tick']
                    packed=pack_observation(boundary.sac_observation)
                    np.testing.assert_allclose(packed[0],a['obs'][ix],atol=1e-6,rtol=0)
                    np.testing.assert_array_equal(packed[1],a['fields'][ix]);np.testing.assert_array_equal(packed[2],a['items'][ix])
                    if parent['pre_tick']==row['pre_tick']:break
                    step=collect_step(env,boundary,lambda e,t:proposal_from_action(e,t,a['raw_u'][ix]))
                    np.testing.assert_allclose(step.reward.training,a['reward'][ix],atol=1e-8,rtol=0)
                    assert step.transition is not None and step.transition.bootstrap
                    boundary=step.boundary
            before=observation_fingerprint(boundary.sac_observation)
            original_state,original_memory,original_token=env._state,env.memory,env.token
            with jax.default_device(gpu):
                mean=mean_policy.action(boundary.sac_observation)
                q=float(np.asarray(qvalue(tensor_observation(boundary.sac_observation),jnp.asarray(mean[None])))[0])
            case=dict(scene_id=row['scene_id'],provider=row['provider'],tick=env.tick,source_trace_index=fixed['trace_index'],observation_sha256=before,
                      mean_raw_u=mean.tolist(),mean_first_action_q=q,rollouts=[])
            report['cases'].append(case);write_report(path,report)
            for rep in range(args.replicates):
                seed=88019+rep
                with jax.default_device(gpu):policy=Policy(state.actor,seed=seed)
                branch=fork_environment(env);current=boundary
                plain=entropy=0.;rejected=braking=stopped=0
                trace=args.output/f"{row['provider']}-rep{rep}.jsonl"
                with trace.open('x') as f:
                    for index in range(200):
                        ell=0.
                        if index==0:raw=mean
                        else:
                            with jax.default_device(gpu):
                                obs=tensor_observation(current.sac_observation)
                                mu,rho=policy.forward({'params':state.actor.params},obs)
                                _,key=jax.random.split(policy.rng)
                                raw=policy.action(current.sac_observation)
                                u,l,_=sample_action(mu,rho,observation_speed(obs),key)
                                np.testing.assert_array_equal(raw,np.asarray(u[0]))
                                ell=float(np.asarray(l)[0])
                        with jax.default_device(cpu):
                            step=collect_step(branch,current,lambda e,t:proposal_from_action(e,t,raw))
                        if step.transition is None:raise ValueError('invalid actual transition')
                        rp,en=discounted_terms(step.reward.training,ell,index,alpha,step.transition.delta_ticks)
                        plain+=rp;entropy+=en
                        rejected+=int(not step.result.check_result.valid)
                        braking+=int(step.result.source.value in ('BRAKE','EMERGENCY'))
                        stopped+=int(branch.ego.speed<.1)
                        f.write(json.dumps(dict(index=index,pre_tick=step.result.pre_token.sim_tick,raw_u=np.asarray(raw).tolist(),ell=ell,
                            entropy_included=index>0,reward=step.reward.training,discounted_reward=rp,discounted_entropy_bonus=en,
                            check_valid=step.result.check_result.valid,check_reason=step.result.check_result.reason,source=step.result.source.value,
                            bootstrap=step.transition.bootstrap,outcome=step.result.terminal_reason.value if step.result.terminal_reason else None,
                            next_observation_sha256=observation_fingerprint(step.boundary.sac_observation)))+'\n')
                        current=step.boundary
                        if not step.transition.bootstrap:break
                    else:raise ValueError('actual contract terminal not reached in200steps')
                assert env._state is original_state and env.memory==original_memory and env.token==original_token
                assert observation_fingerprint(env._boundary().sac_observation)==before
                value=dict(replicate=rep,seed=seed,ticks=index+1,outcome=step.result.terminal_reason.value,
                    discounted_reward=plain,discounted_entropy_bonus=entropy,soft_return=plain+entropy,
                    q_minus_sampled_soft_return=q-plain-entropy,rejected=rejected,braking=braking,stationary=stopped,
                    trace_file=trace.name,trace_sha256=file_hash(trace))
                case['rollouts'].append(value);write_report(path,report);print(json.dumps(dict(provider=row['provider'],**value)),flush=True)
            values=np.array([x['soft_return'] for x in case['rollouts']])
            case['summary']=dict(mean_soft_return=float(values.mean()),min_soft_return=float(values.min()),max_soft_return=float(values.max()),
                sample_std=None if len(values)<2 else float(values.std(ddof=1)),q_minus_mean_sampled_soft_return=q-float(values.mean()))
            write_report(path,report)
        assert versions(args.manifest)==version and file_hash(weights)==report['model_sha256']
        assert file_hash(args.diagnostic/'replay.npz')==report['replay_sha256']
        assert file_hash(args.states_from/'replay.npz')==report['states_replay_sha256']
        assert serialization.to_bytes(state)==weights.read_bytes()
        report['status']='COMPLETE'
    except BaseException as exc:
        report.update(status='FAILED',error=f'{type(exc).__name__}: {exc}');raise
    finally:write_report(path,report)


if __name__=='__main__':main()

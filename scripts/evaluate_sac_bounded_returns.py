#!/usr/bin/env python3
"""Bound realized soft returns without claiming exact actor reconstruction.

Uses stored actual actions/rewards and recorded Gaussian RNG. Sigma contract
bounds provide an interval for the unrecorded entropy term. Critic arithmetic
is highest precision, independently compared with CPU at every stored action.
"""
import argparse, hashlib, json, os, sys
from pathlib import Path
os.environ.setdefault('JAX_PLATFORMS','cuda,cpu')
os.environ.setdefault('JAX_ENABLE_X64','true')
os.environ.setdefault('XLA_PYTHON_CLIENT_PREALLOCATE','false')
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('diagnostic','parent','source','actual-values','fixed-cohort-values','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();assert not args.output.exists()
    import jax, jax.numpy as jnp, numpy as np
    from flax import serialization
    from scripts.audit_sac_fixed_policy_continuation import audit
    from scripts.sac_soft_return_bounds import entropy_bounds, conditional_return_bounds, ELL_ROUNDING_MARGIN
    from scripts.sac_diagnostic_actor import diagnostic_actor
    from scripts.sac_diagnostic_critic import diagnostic_critic
    from src.contracts.observation import CoreObservation
    from src.training.sac.learner import initialize
    from src.training.sac.policy import tensor_observation
    from src.training.sac.replay import unpack_observation
    read=lambda p:json.loads(p.read_text())
    sha=lambda p:hashlib.sha256(p.read_bytes()).hexdigest()
    checked=audit(args.diagnostic,args.parent,args.source)
    report=read(args.diagnostic/'report.json');source=read(args.source/'report.json')
    known=read(args.actual_values);fixed=read(args.fixed_cohort_values)
    assert report['critic_steps']==5000 and report['plan']['original_replay_size']==6000
    assert fixed['status']=='COMPLETE' and fixed['scope']=='HIGH_PRECISION_FIXED_TRAIN_COHORT_REANALYSIS'
    assert fixed['critic_matmul_precision']=='highest' and fixed['all_critic_rows_checked_on_cpu']
    assert known['status']=='COMPLETE' and known['actual_transitions']==1000
    for prior in (known,fixed):
        for name,h in prior['input_sha256'].items():assert sha(Path(name))==h,name
    lookups={stage:{r['collection_id']:r for r in next(s for s in fixed['snapshots'] if s['critic_steps']==count)['rows']} for stage,count in (('before',0),('after',5000))}
    assert len(lookups['before'])==len(lookups['after'])==947
    known_rows={r['collection_id']:r for r in known['rows']}
    paths=[(stage,args.source/'online-learner.msgpack' if stage=='before' else args.diagnostic/'critic-evaluation-state.msgpack') for stage in ('before','after')]
    inputs=dict(checked['source_sha256'],**{str(p):sha(p) for p in [args.actual_values,args.fixed_cohort_values]+[p for _,p in paths]})
    source_paths=[Path(__file__),Path('scripts/sac_soft_return_bounds.py'),Path('src/training/sac/networks.py'),Path('src/training/sac/policy.py')]
    sources={str(p):sha(p) for p in source_paths}
    ticks=[json.loads(line) for line in (args.diagnostic/'ticks.jsonl').read_text().splitlines()]
    with np.load(args.diagnostic/'diagnostic-replay.npz',allow_pickle=False) as z:
        arrays={k:z[k][6000:].copy() for k in ('obs','fields','items','raw_u','reward','bootstrap','delta_ticks')}
    assert len(ticks)==len(arrays['raw_u'])==5000
    assert jax.__version__==report['versions']['dependencies']['jax']
    assert jax.config.jax_default_prng_impl=='threefry2x32'
    gpu,cpu=jax.devices('gpu')[0],jax.devices('cpu')[0]
    rows=[];groups={};max_device=max_reference=max_noise=0.
    with jax.default_device(gpu),jax.default_matmul_precision('highest'):
        template=initialize(tensor_observation(CoreObservation.empty()),source['seed'],actor=diagnostic_actor(source['arms']['online'],restoring=True),critic=diagnostic_critic(source['arms']['online'],restoring=True))
        states={stage:jax.device_put(serialization.from_bytes(template,p.read_bytes()),gpu) for stage,p in paths}
        for stage,p in paths:assert serialization.to_bytes(states[stage])==p.read_bytes()
        before,after=states['before'],states['after']
        for attr in ('actor','log_alpha','alpha_optimizer'):assert serialization.to_bytes(getattr(before,attr))==serialization.to_bytes(getattr(after,attr)),attr
        assert int(after.updates)==int(before.updates)+5000 and int(before.bc_updates)==int(after.bc_updates)==0
        alpha=float(jnp.exp(before.log_alpha));assert alpha==report['alpha']==known['alpha']
        cpu_states={name:jax.device_put(state,cpu) for name,state in states.items()}
        apply1,apply2=before.q1.apply_fn,before.q2.apply_fn
        @jax.jit
        def value(p1,p2,obs,raw):
            return jnp.minimum(apply1({'params':p1},obs,raw),apply2({'params':p2},obs,raw))
        rng=None
        for i,tick in enumerate(ticks):
            if tick['branch_step']==0:rng=jax.random.PRNGKey(tick['policy_seed'])
            assert rng is not None
            rng,key=jax.random.split(rng)
            eps=np.asarray(jax.random.normal(key,(1,30,2),dtype=jnp.float64))
            with jax.default_device(cpu):
                cpu_eps=np.asarray(jax.random.normal(jax.device_put(key,cpu),(1,30,2),dtype=jnp.float64))
                cpu_obs=unpack_observation(*(jnp.asarray(arrays[k][i:i+1]) for k in ('obs','fields','items')))
                cpu_raw=jnp.asarray(arrays['raw_u'][i:i+1])
            noise_delta=abs(float(np.sum(eps**2))-float(np.sum(cpu_eps**2)))
            assert noise_delta<=1e-12,(i,'epsilon norm',noise_delta)
            max_noise=max(max_noise,noise_delta)
            ell_low,ell_high=(float(x[0]) for x in entropy_bounds(eps))
            if tick['collection_id']<=1000:
                assert ell_low<=known_rows[tick['collection_id']]['ell']<=ell_high
            assert arrays['reward'][i]==tick['reward'] and arrays['bootstrap'][i]==tick['bootstrap']
            assert arrays['delta_ticks'][i]==tick['delta_ticks']==1
            np.testing.assert_array_equal(arrays['raw_u'][i],tick['raw_proposal']['raw_u'])
            obs=unpack_observation(*(jnp.asarray(arrays[k][i:i+1]) for k in ('obs','fields','items')))
            raw=jnp.asarray(arrays['raw_u'][i:i+1])
            row={k:tick[k] for k in ('collection_id','rollout','root_index','policy_seed','branch_step','scene_id','provider','pre_tick','bootstrap','outcome','reward')}
            row.update(cohort='inherited1000' if tick['collection_id']<=1000 else 'new4000',epsilon_norm_squared=float(np.sum(eps**2)),epsilon_cpu_gpu_norm_difference=noise_delta,ell_lower=ell_low,ell_upper=ell_high)
            for stage,state in states.items():
                q=float(np.asarray(value(state.q1.params,state.q2.params,obs,raw))[0])
                target=float(np.asarray(value(state.target1,state.target2,obs,raw))[0])
                c=cpu_states[stage]
                with jax.default_device(cpu):
                    cq=float(np.asarray(value(c.q1.params,c.q2.params,cpu_obs,cpu_raw))[0])
                    ct=float(np.asarray(value(c.target1,c.target2,cpu_obs,cpu_raw))[0])
                delta=max(abs(q-cq),abs(target-ct));assert delta<=1e-3,(i,stage,'critic cpu/gpu',delta)
                max_device=max(max_device,delta)
                if row['collection_id'] in lookups[stage]:
                    previous=lookups[stage][row['collection_id']]
                    delta=max(abs(q-previous['q']),abs(target-previous['target_q']));assert delta<=1e-3,(i,stage,'precise reference',delta)
                    max_reference=max(max_reference,delta)
                row[stage]=dict(q=q,target_q=target,cpu_q=cq,cpu_target_q=ct)
            rows.append(row);groups.setdefault(tick['rollout'],[]).append(row)
            if (i+1)%100==0:print(json.dumps(dict(evaluated=i+1,total=5000)),flush=True)
    complete=[];episodes=[];excluded=[]
    for rollout,group in groups.items():
        if group[-1]['bootstrap']:
            excluded.append(dict(rollout=rollout,ticks=len(group),reason='DIAGNOSTIC_BUDGET'));continue
        reward_only,lower,upper=conditional_return_bounds([r['reward'] for r in group],[r['ell_lower'] for r in group],[r['ell_upper'] for r in group],[r['bootstrap'] for r in group],alpha)
        for i,row in enumerate(group):
            row.update(steps_before_terminal=len(group)-1-i,terminal_outcome=group[-1]['outcome'],reward_only_return=float(reward_only[i]),soft_return_lower=float(lower[i]),soft_return_upper=float(upper[i]))
            if row['collection_id'] in lookups['before']:
                exact=lookups['before'][row['collection_id']]['observed_soft_return']
                assert lower[i]-1e-10<=exact<=upper[i]+1e-10,(row['collection_id'],'known return outside interval')
            for stage in states:
                q=row[stage]['q'];row[stage].update(q_minus_return_lower=q-upper[i],q_minus_return_upper=q-lower[i],absolute_error_lower_bound=max(lower[i]-q,q-upper[i],0.))
        episodes.append(dict(rollout=rollout,provider=group[0]['provider'],cohort=group[0]['cohort'],ticks=len(group),outcome=group[-1]['outcome'],undiscounted_return=sum(r['reward'] for r in group),reward_only_return=float(reward_only[0]),soft_return_lower=float(lower[0]),soft_return_upper=float(upper[0])))
        complete.extend(group)
    def summarize(selected):
        answer=dict(ticks=len(selected))
        if selected:
            answer['mean_interval_width']=float(np.mean([r['soft_return_upper']-r['soft_return_lower'] for r in selected]))
            for stage in states:
                answer[stage]=dict(mae_lower_bound=float(np.mean([r[stage]['absolute_error_lower_bound'] for r in selected])),mean_signed_gap_lower=float(np.mean([r[stage]['q_minus_return_lower'] for r in selected])),mean_signed_gap_upper=float(np.mean([r[stage]['q_minus_return_upper'] for r in selected])),above_interval=sum(r[stage]['q']>r['soft_return_upper'] for r in selected),below_interval=sum(r[stage]['q']<r['soft_return_lower'] for r in selected))
        return answer
    summaries={}
    for cohort in ('all','inherited1000','new4000'):
        summaries[cohort]={}
        for provider in ('all','fixed_keep','fixed_follow'):
            selected=[r for r in complete if (cohort=='all' or r['cohort']==cohort) and (provider=='all' or r['provider']==provider)]
            summaries[cohort][provider]=dict(all_ticks=summarize(selected),first_stochastic_action=summarize([r for r in selected if r['branch_step']==0]),terminal=summarize([r for r in selected if not r['bootstrap']]),by_terminal_outcome={outcome:summarize([r for r in selected if r['terminal_outcome']==outcome]) for outcome in sorted({r['terminal_outcome'] for r in selected})})
    for group in (inputs,sources):
        for path,h in group.items():assert sha(Path(path))==h,path
    result=dict(status='COMPLETE',qualified=False,scope='REALIZED_SOFT_RETURN_INTERVAL_REANALYSIS',actual_transitions=5000,complete_rollouts=len(episodes),complete_ticks=len(complete),budget_truncated_excluded=excluded,alpha=alpha,critic_matmul_precision='highest',cross_device_tolerance=1e-3,cross_device_max_difference=max_device,precise_reference_max_difference=max_reference,epsilon_norm_tolerance=1e-12,epsilon_norm_max_difference=max_noise,sigma_bounds=[.005,.03],ell_rounding_margin=ELL_ROUNDING_MARGIN,exact_actor_reconstruction=False,exact_soft_return_claim=False,actor_and_temperature_unchanged=True,known_entropy_rows_contained=1000,known_complete_return_rows_contained=947,summaries=summaries,episodes=episodes,rows=rows,input_sha256=inputs,source_sha256=sources,note='Saved actual actions and rewards are unchanged. Entropy interval uses recorded RNG and contract sigma bounds, not reconstructed mu/rho or inverted covariance. Conditional returns exclude current-action entropy; incomplete tails excluded. Intervals bound sampled-path returns, not expected-Q uncertainty or generalization. No historical numeric-reproduction or policy-improvement claim.')
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps(dict(status='COMPLETE',complete_ticks=len(complete),summaries=summaries['all'])),flush=True)


if __name__=='__main__':
    main()

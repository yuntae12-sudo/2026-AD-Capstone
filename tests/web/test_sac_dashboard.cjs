const test = require('node:test');
const assert = require('node:assert/strict');
const {completeEvaluation, sameCohort, deriveSummary, visibleEvaluations} = require('../../web/sac_dashboard/app.js');
function evaluation(label, goals=0, count=4, state='COMPLETE') {
  return {label, state, summary:{episodes:count, outcomes:{ROUTE_GOAL:goals,COLLISION:count-goals}},
    episodes:Array.from({length:count},(_,i)=>({scene_id:'scene'+Math.floor(i/2),provider:i%2?'fixed_follow':'fixed_keep'}))};
}
function data() {
  return {evaluation_target:4, state:'RUNNING', process_alive:true,
    arms:[{arm:'candidate',state:'COMPLETE',job_state:'COMPLETE',updates:10000,total:10000},
      {arm:'control',state:'RUNNING',job_state:'RUNNING',updates:7000,total:10000}],
    evaluations:[evaluation('candidate'),evaluation('control',0,0,'PARTIAL / WAITING'),evaluation('candidate1000'),evaluation('rule_reference',2)],workers:[]};
}
test('training completed with collisions is a failed driving result',()=>{
  const s=deriveSummary(data());assert.match(s.title,/목표 도달 0 \/ 4회/);assert.equal(s.severity,'failure');assert.match(s.nextTitle,/대조군 학습 중/);
});
test('unassessed policies remain pending, never zero success',()=>{
  const d=data();d.evaluations=d.evaluations.filter(e=>e.label!=='candidate');
  assert.equal(deriveSummary(d).severity,'pending');assert.match(deriveSummary(d).title,/기다리고/);
});
test('partial report marked COMPLETE is not a full evaluation',()=>{
  const d=data();const e=evaluation('candidate',0,2);assert.equal(completeEvaluation(d,e),false);
});
test('duplicate scene/provider pairs cannot qualify as full evaluation',()=>{
  const d=data(), e=evaluation('candidate');e.episodes[3]=e.episodes[0];assert.equal(completeEvaluation(d,e),false);
});
test('100 percent goal success does not automatically claim stable policy',()=>{
  const d=data();d.evaluations[0]=evaluation('candidate',4);const s=deriveSummary(d);
  assert.equal(s.severity,'review');assert.match(s.body,/판단하지 않습니다/);
});
test('interrupted process changes the next action even if artifacts look complete',()=>{
  const d=data();d.process_alive=false;assert.match(deriveSummary(d).nextTitle,/실행 상태와 로그/);
});
test('same-cohort deltas reject mismatched scene identities',()=>{
  const d=data(),a=evaluation('candidate'),b=evaluation('candidate1000');assert.equal(sameCohort(d,a,b),true);
  b.episodes[0].scene_id='different';assert.equal(sameCohort(d,a,b),false);
});
test('history is opt-in and Rule is an explicitly retained reference',()=>{
  assert.deepEqual(visibleEvaluations(data(),false).map(e=>e.label),['candidate','control','rule_reference']);
  assert.deepEqual(visibleEvaluations(data(),true).map(e=>e.label),['candidate','control','rule_reference','candidate1000']);
});
test('single-arm manual experiments do not await an absent control',()=>{
  const d=data();d.arms=d.arms.slice(0,1);d.state='COMPLETE_PENDING_REVIEW';d.process_alive=false;
  assert.equal(deriveSummary(d).waiting.length,0);assert.match(deriveSummary(d).nextTitle,/완료된 결과/);
});

test('live parent cannot hide a missing training child',()=>{
  const d=data();d.arms[1].process_alive=false;assert.match(deriveSummary(d).nextTitle,/실행 상태와 로그/);
});

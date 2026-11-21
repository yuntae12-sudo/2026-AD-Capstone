'use strict';
const $ = id => document.getElementById(id);
const labels = {candidate:'새 수집 SAC · 후보',control:'기존 수집 SAC · 대조군',candidate1000:'후보 · 이전 1,000회',control1000:'대조군 · 이전 1,000회',rule_reference:'Rule-based · 기존 평가'};
const states = {RUNNING:'진행 중',WAITING:'대기',RESTORING:'기존 상태 복원 중',COMPLETE:'실행 완료',FAILED:'오류',STOPPED:'사용자 중단',COMPLETE_PENDING_REVIEW:'실행 완료 · 결과 검토 필요'};
const num = n => Number(n).toLocaleString('ko-KR');
const pct = n => n == null ? '—' : (100*n).toFixed(1)+'%';
const el = (tag, text, cls) => {const n=document.createElement(tag);if(text!=null)n.textContent=text;if(cls)n.className=cls;return n;};
let current=null, fetching=false;
function bar(value,total){const track=el('div',null,'bar');const fill=el('div',null,'bar-fill');fill.style.width=pct(total?Math.min(1,Math.max(0,value/total)):0);track.append(fill);track.setAttribute('role','progressbar');track.setAttribute('aria-valuemin','0');track.setAttribute('aria-valuemax',String(total));track.setAttribute('aria-valuenow',String(value));return track;}
async function get(url){const r=await fetch(url,{cache:'no-store'});if(!r.ok)throw new Error('결과 파일을 읽지 못했습니다 ('+r.status+').');return r.json();}

// Summary text is computed from complete evaluation records, never training loss.
function completeEvaluation(d, e) {
  return Boolean(e && e.state === 'COMPLETE' && d.evaluation_target > 0 &&
    e.summary.episodes === d.evaluation_target && e.episodes.length === d.evaluation_target &&
    new Set(e.episodes.map(r => r.scene_id + '/' + r.provider)).size === d.evaluation_target);
}
function sameCohort(d, a, b) {
  if (!completeEvaluation(d, a) || !completeEvaluation(d, b)) return false;
  const keys = e => e.episodes.map(r => r.scene_id + '/' + r.provider).sort().join('|');
  return keys(a) === keys(b);
}
function deriveSummary(d) {
  const completed = d.arms.map(a => d.evaluations.find(e => e.label === a.arm))
    .filter(e => completeEvaluation(d, e));
  const candidate = completed.find(e => e.label === 'candidate') || completed[0];
  const waiting = d.arms.filter(a => !completed.some(e => e.label === a.arm));
  let title = '주행 평가를 기다리고 있습니다';
  let body = '학습이 진행되어도 실제 주행 평가는 별도로 확인해야 합니다.';
  let severity = 'pending';
  if (candidate) {
    const n = candidate.summary.episodes, goals = candidate.summary.outcomes.ROUTE_GOAL || 0;
    const collisions = candidate.summary.outcomes.COLLISION || 0;
    title = goals === 0 ? `${labels[candidate.label]}: 목표 도달 ${goals} / ${n}회` : `${labels[candidate.label]}: ${n}회 중 ${goals}회 목표 도달`;
    body = `개발 평가에서 ${collisions}회가 충돌로 종료됐습니다. ` +
      (goals === 0 ? '이번 정책으로 목표 주행을 달성하지 못했습니다.' : '일부 목표 도달만으로 안정적인 정책이라고 판단하지 않습니다.');
    severity = goals === 0 ? 'failure' : 'review';
  }
  let nextTitle, nextBody;
  const running = d.arms.find(a => a.job_state === 'RUNNING');
  const activeWorker = d.workers.find(w => w.alive);
  if (d.state === 'FAILED' || d.state === 'STOPPED' || (d.state === 'RUNNING' && !d.process_alive) ||
      d.arms.some(a => a.job_state === 'RUNNING' && a.process_alive === false) ||
      (d.jobs || []).some(j => j.state === 'RUNNING' && j.alive === false)) {
    nextTitle = '실행 상태와 로그를 확인하세요';
    nextBody = '실험이 중단됐거나 기록과 실제 프로세스 상태가 다릅니다. 진행률만으로 실행 중이라고 판단하지 않습니다.';
  } else if (running) {
    nextTitle = `${labels[running.arm]} 학습 중`;
    nextBody = `${num(running.updates)} / ${num(running.total)}회 업데이트. 이후 저장 검증과 주행 평가를 진행합니다. 지금 새 학습을 중복 실행할 필요는 없습니다.`;
  } else if (waiting.length) {
    nextTitle = '남은 주행 평가를 확인하세요';
    nextBody = waiting.map(a => labels[a.arm]).join(', ') + '의 최종 평가가 아직 완료되지 않았습니다. 같은 예산의 최종 비교는 보류합니다.';
  } else if (activeWorker) {
    nextTitle = '후속 분석 결과를 확인하세요';
    nextBody = '학습과 평가는 끝났습니다. 실패 원인·Q 예측·주행 영상의 후속 작업 상태를 확인합니다.';
  } else {
    nextTitle = '완료된 결과를 함께 검토하세요';
    nextBody = '성공·충돌·거절·제동·정지·보상을 함께 확인합니다. 실행 완료만으로 정책을 채택하거나 학습 예산을 늘리지 않습니다.';
  }
  return {title, body, severity, candidate, waiting, nextTitle, nextBody};
}
function jobName(name) {
  const workers = {'failure-analysis-job':'실패 원인 분석', visualization:'주행 영상 생성',
    'policy-diagnostics':'Q 예측과 실제 미래 보상 검사', 'capture-metrics-job':'충돌 이벤트·추종 오차 집계'};
  if (workers[name]) return workers[name];
  const match = /^(candidate|control)-(train\d+|audit\d+|validation)$/.exec(name);
  if (match) return labels[match[1]] + ' · ' + (match[2].startsWith('train') ? '학습' : match[2].startsWith('audit') ? '저장 검증' : '주행 평가');
  return name === 'review-full' ? '전체 비교 결과 검산' : name;
}
function renderOverview(d) {
  const summary = deriveSummary(d);
  $('verdict').className = 'verdict ' + summary.severity;
  $('verdict-title').textContent = summary.title;
  $('verdict-body').textContent = summary.body;
  $('next-title').textContent = summary.nextTitle;
  $('next-body').textContent = summary.nextBody;
  $('verdict-stats').replaceChildren();
  if (summary.candidate) {
    const a = summary.candidate.summary;
    for (const [label, value] of [['목표 도달', `${a.outcomes.ROUTE_GOAL || 0} / ${a.episodes}회`],
      ['충돌 종료', `${a.outcomes.COLLISION || 0} / ${a.episodes}회`]]) {
      const box = el('div'); box.append(el('span', label), el('strong', value)); $('verdict-stats').append(box);
    }
  }
  $('run-state').textContent = states[d.state] || d.state;
  if (d.state === 'RUNNING' && !d.process_alive) $('run-state').textContent = '실행 프로세스 없음';
  const line = el('div', null, 'total-line');
  line.append(el('span', '계획한 추가 학습량 · ' + num(d.total_extra) + ' / ' + num(d.total_target)),
    el('strong', pct(d.total_target ? d.total_extra / d.total_target : 0)));
  $('total').replaceChildren(line, bar(d.total_extra, d.total_target),
    el('p', `시작 모델의 ${num(d.arms[0].inherited)}회는 추가 학습 분모에서 제외합니다. 복원·평가·영상 생성 시간도 제외합니다.`, 'footnote'));
  $('arms').replaceChildren(...d.arms.map(a => {
    const c = el('article', null, 'card'), top = el('div', null, 'card-top');
    top.append(el('h3', labels[a.arm]), el('span', a.state === 'COMPLETE' ? '학습 예산 완료' : states[a.state] || a.state, 'tag'));
    const n = el('div', num(a.updates), 'number'); n.append(el('small', ' / ' + num(a.total) + '회'));
    c.append(top, n, bar(a.extra, a.target), el('p', `새 학습 ${num(a.extra)} / ${num(a.target)}회 · ${pct(a.extra/a.target)}`));
    const phases = el('div', null, 'phase-list');
    for (const [suffix, text] of [[`train${a.total}`, '학습'], [`audit${a.total}`, '저장 검증'], ['validation', '주행 평가']]) {
      const job = d.jobs.find(j => j.name === a.arm + '-' + suffix);
      const stage = el('div', null, 'phase'); stage.append(el('span', text), el('b', job ? (states[job.state] || job.state) : '대기'));
      phases.append(stage);
    }
    c.append(phases, el('p', `주행 평가 ${a.evaluation_count} / ${d.evaluation_target}회 · 학습 완료와 주행 성공은 다릅니다.`));
    if (a.job_state === 'RUNNING' && !a.process_alive) c.append(el('p', '학습 프로세스를 찾지 못했습니다. 로그 확인 필요.', 'bad'));
    if (a.state === 'RESTORING') c.append(el('p', `기존 상태 복원 ${a.restoration.restored || 0} / ${a.restoration.total || a.inherited}`));
    return c;
  }));
  $('jobs').replaceChildren(...[...d.jobs, ...d.workers].map(j => {
    const n = el('div', jobName(j.name), 'job');
    n.append(el('span', (j.state === 'WAITING_FOR_COMPARISON' ? '전체 비교 완료 대기' : states[j.state] || j.state) + (j.alive ? ' · 프로세스 실행 중' : '')));return n;
  }));
  $('experiment-explanation').querySelector('h3').textContent = d.arms.length === 2 ? '이 실험은 무엇을 비교하나요?' : `이번 실행: ${labels[d.arms[0].arm]}만 실행`;
}
function visibleEvaluations(d, history) {
  const order = [...d.arms.map(a => a.arm), 'rule_reference'];
  if (history) order.push(...d.arms.map(a => a.arm + '1000'));
  return order.map(name => d.evaluations.find(e => e.label === name)).filter(Boolean);
}
function renderResults() {
  if (!current) return;
  const d = current, provider = $('provider').value;
  const allDone = d.arms.every(a => completeEvaluation(d, d.evaluations.find(e => e.label === a.arm)));
  $('comparison-note').textContent = (allDone ? '각 SAC의 평가 기록이 완료됐습니다. ' : '최종 비교 대기: 아직 평가하지 않은 SAC의 성능은 알 수 없습니다. ') +
    '개발 장면을 반복 사용한 평가이며 독립 test가 아닙니다. Rule-based 행은 같은 조건의 기존 평가를 재사용합니다.';
  const subset = e => provider === 'all' ? e.summary : e.by_provider[provider];
  $('performance-cards').replaceChildren(...visibleEvaluations(d, false).map(e => {
    const s = subset(e), c = el('article', null, 'performance-card');
    c.append(el('h3', labels[e.label]));
    if (!s.episodes) { c.append(el('strong', '평가 대기', 'score pending-text'), el('p', '아직 주행 결과가 없습니다. 0% 성공이라는 뜻이 아닙니다.')); }
    else {
      c.append(el('span', e.reused ? '기존 평가 재사용' : completeEvaluation(d, e) ? '이번 평가 결과' : '일부 평가 · 최종 결과 아님', 'tag'));
      c.append(el('strong', `${s.outcomes.ROUTE_GOAL || 0} / ${s.episodes}회`, 'score'), el('p', '목표 도달'),
        el('div', `충돌 종료 ${s.outcomes.COLLISION || 0}회 · 기타 종료 ${s.episodes - (s.outcomes.ROUTE_GOAL || 0) - (s.outcomes.COLLISION || 0)}회`, s.outcomes.COLLISION ? 'bad' : ''));
    } return c;
  }));
  const fresh = d.evaluations.find(e => e.label === 'candidate'), old = d.evaluations.find(e => e.label === 'candidate1000');
  $('change-note').replaceChildren();
  if (sameCohort(d, fresh, old)) {
    const a = subset(fresh), b = subset(old);
    if (a.episodes && b.episodes) {
      $('change-note').append(el('b', '같은 후보의 이전 1,000회 평가와 비교'));
      for (const [key, name] of [['rejected', '경로 거절'], ['braking', '제동 개입'], ['stationary', '정지 비중']]) {
        const delta = 100 * (a[key] - b[key]);
        $('change-note').append(el('span', `${name} ${pct(b[key])} → ${pct(a[key])} (${delta>0?'+':''}${delta.toFixed(1)}%p)`));
      }
      $('change-note').append(el('small', '방문 상태와 전체 주행 길이가 달라질 수 있습니다. 수치 변화가 원인 분석이나 대조군과의 최종 비교를 대신하지 않습니다.'));
    }
  }
  const visible = visibleEvaluations(d, $('show-history').checked);
  $('metrics').replaceChildren(...visible.map(e => {
    const s = subset(e), row = el('tr');
    const state = !s.episodes ? '평가 대기' : completeEvaluation(d,e) ? '평가 완료' : '부분 평가';
    const values = [labels[e.label], state + (e.reused ? ' · 재사용' : ''), s.scenes+' / '+s.episodes,
      pct(s.success_rate),pct(s.collision_rate),pct(s.rejected),pct(s.braking),pct(s.stationary),s.mean_return==null?'—':s.mean_return.toFixed(3)];
    values.forEach((v,i) => row.append(el('td',v,i===4&&s.collision_rate>0?'bad':'')));return row;
  }));
  const reasonsMap = {KINEMATIC_LIMIT:'운동 한계 위반',PREDICTED_OFFROAD:'도로 이탈 예상',PREDICTED_COLLISION:'충돌 예상',UNRESOLVED:'검사 미해결',TRACKING_ERROR:'추종 오차'};
  const rows=[];
  for (const e of visible) for (const episode of e.episodes) {
    if (provider!=='all'&&episode.provider!==provider) continue;
    const tr=el('tr'); const reasons=Object.entries(episode.reasons||{}).filter(([k])=>k!=='OK').map(([k,v])=>(reasonsMap[k]||k)+' '+v).join(', ')||'없음';
    [labels[e.label],episode.scene_id,episode.provider.replace('fixed_','').toUpperCase(),
      {COLLISION:'충돌',ROUTE_GOAL:'목표 도달',TASK_LOST:'과제 유지 실패'}[episode.outcome]||episode.outcome,episode.ticks,reasons]
      .forEach(v=>tr.append(el('td',String(v))));rows.push(tr);
  }
  $('episodes').replaceChildren(...rows);
  $('reports').replaceChildren(...d.links.map(l=>{const a=el('a',l.label);a.href='/media?path='+encodeURIComponent(l.path);a.target='_blank';a.rel='noopener';return a;}));
}
function draw(canvas,points,key){const box=canvas.getBoundingClientRect(),scale=window.devicePixelRatio||1;canvas.width=Math.max(1,box.width*scale);canvas.height=210*scale;const c=canvas.getContext('2d');c.scale(scale,scale);const w=box.width,h=210,L=60,R=12,T=15,B=30;const valid=points.filter(p=>Number.isFinite(p[key]));if(!valid.length){c.fillStyle='#9eafa5';c.fillText('학습 기록 대기 중',L,90);return;}
  const xs=valid.map(p=>p.update),ys=valid.map(p=>p[key]);let lo=Math.min(...ys),hi=Math.max(...ys);if(hi===lo){hi+=Math.abs(hi)*.1||1;lo-=Math.abs(lo)*.1||1;}const x0=Math.min(...xs),x1=Math.max(...xs);const x=v=>L+(v-x0)/(x1-x0||1)*(w-L-R),y=v=>h-B-(v-lo)/(hi-lo)*(h-T-B);c.font='10px monospace';c.strokeStyle='#344035';c.fillStyle='#9eafa5';for(let i=0;i<4;i++){const value=lo+(hi-lo)*i/3,yy=y(value);c.beginPath();c.moveTo(L,yy);c.lineTo(w-R,yy);c.stroke();c.fillText(Math.abs(value)<.01?value.toExponential(1):value.toFixed(2),2,yy+3);}c.fillText(num(x0),L,h-7);c.fillText(num(x1),Math.max(L,w-62),h-7);c.strokeStyle='#c6eb87';c.lineWidth=1.7;c.beginPath();valid.forEach((p,i)=>i?c.lineTo(x(p.update),y(p[key])):c.moveTo(x(p.update),y(p[key])));c.stroke();if(valid.length===1){c.beginPath();c.arc(x(valid[0].update),y(valid[0][key]),3,0,Math.PI*2);c.fillStyle='#c6eb87';c.fill();}}
function renderCharts(){if(!current)return;const key=$('metric').value;const help={q1_loss:'Q1 loss는 예상 가치와 학습 목표의 차이입니다. 작아져도 실제 충돌이 줄었다는 뜻은 아닙니다.',q2_loss:'Q2는 두 번째 가치 추정기입니다. 실제 주행 결과와 별도로 읽어야 합니다.',actor_loss:'Actor loss는 정책을 업데이트하는 목적값입니다. 성공률이나 주행 점수가 아닙니다.',alpha:'Alpha는 탐색과 보상의 균형을 조절하는 값입니다. 높고 낮음만으로 정책 성능을 판단할 수 없습니다.',actor_q1_abs_max:'정책 행동에 대한 Q1 절댓값의 최댓값입니다. 실제 미래 보상과의 차이는 별도 검증이 필요합니다.',target_abs_max:'TD target은 가치 학습의 목표값입니다. 큰 값의 변화가 주행 개선을 뜻하지는 않습니다.'};$('metric-help').textContent=help[key];$('charts').replaceChildren(...current.arms.map(a=>{const card=el('div',null,'card');const canvas=el('canvas',null,'chart');canvas.setAttribute('role','img');canvas.setAttribute('aria-label',(labels[a.arm]||a.arm)+' '+key+' 학습 추이');const vals=a.curve.points.filter(p=>p[key]!=null);const last=vals.length?vals[vals.length-1][key]:null;card.append(el('h3',labels[a.arm]||a.arm),canvas,el('p','기록 '+num(a.curve.rows)+'회 · '+a.curve.bucket_size+'개씩 평균 · 마지막 구간 '+(last==null?'—':last.toPrecision(5)),'chart-caption'));requestAnimationFrame(()=>draw(canvas,a.curve.points,key));return card;}));}

function renderMedia(d) {
  const history = $('video-period').value === 'history';
  const available = d.media.filter(m => history ? m.label !== '이번 실험' : m.label === '이번 실험');
  const previous = $('video-choice').value;
  $('video-choice').replaceChildren(...available.map(m => {
    const o = el('option', m.label.replace('candidate','후보 SAC').replace('control','대조군 SAC')+' · '+m.name);o.value=m.path;return o;
  }));
  if (available.some(m => m.path === previous)) $('video-choice').value=previous;
  $('video-choice').disabled = !available.length;
  if (!available.length) {
    $('video-choice').append(el('option', '영상 없음'));
    const worker = d.workers.find(w => w.name === 'visualization');
    $('video-status').textContent = history ? '저장된 이전 영상이 없습니다.' : worker ?
      '이번 실험의 영상은 아직 없습니다. 영상 작업: '+(worker.state==='WAITING_FOR_COMPARISON'?'전체 비교 완료 대기':states[worker.state]||worker.state)+'.' :
      '이번 실행에는 저장된 영상이 없습니다. 수동 실행 도구는 GIF 생성을 자동 연결하지 않습니다.';
    $('media').replaceChildren(el('div', history ? '이전 영상 없음' : '이번 모델의 주행 영상 대기', 'empty'));
    return;
  }
  $('video-status').textContent = history ? '이전 1,000회 모델의 영상입니다. 이번 최종 모델의 성능 근거로 사용하지 마세요.' : '이번 실험에서 저장된 영상입니다. 전체 평가 결과와 함께 확인하세요.';
  const m = available.find(x=>x.path===$('video-choice').value)||available[0];
  // Preserve playback across polling when the selected file has not changed.
  if ($('media').dataset.path === m.path && $('media').querySelector('img')) return;
  $('media').dataset.path=m.path;
  const f=el('figure',null,'media-card'),img=el('img');img.src='/media?path='+encodeURIComponent(m.path);
  img.alt=m.label+' '+m.name;const cap=el('figcaption',m.label.replace('candidate','후보 SAC').replace('control','대조군 SAC')+' · '+m.name+' ');
  const a=el('a','원본 열기');a.href=img.src;a.target='_blank';a.rel='noopener';cap.append(a);f.append(img,cap);$('media').replaceChildren(f);
}
function shell(s){return "'"+s.replaceAll("'","'\\''")+"'";}
function renderCommands(d){const output='outputs/manual_sac_'+new Date().toISOString().slice(0,10).replaceAll('-','')+'_01';const arm=d.arms.length===2?'both':d.arms[0].arm;const base=' --preset '+shell('outputs/'+d.name)+' \\\n  --arm '+arm+' --updates 10000 \\\n  --output '+shell(output);$('plan-command').textContent='python scripts/sac_experiment.py plan'+base;$('run-command').textContent='python scripts/sac_experiment.py run'+base;$('log-command').textContent='tail -f '+shell(output+'/'+d.arms[0].arm+'-train10000.log');$('monitor-command').textContent=d.scope==='MANUAL_REFRESH_CONTINUATION'?'watch -n 5 '+shell('cat '+shell('outputs/'+d.name+'/status.json')):'python scripts/monitor_sac_continuation.py --output '+shell('outputs/'+d.name);}
function renderLogs(d) {
  const open = new Set([...$('logs').querySelectorAll('details[open] summary')].map(n => n.textContent));
  $('logs').replaceChildren(...d.logs.map(log => {
    const details = el('details'); details.open = open.has(log.name);
    details.append(el('summary', log.name), el('pre', log.tail, 'log')); return details;
  }));
}
async function refresh(){if(fetching||!$('run').value)return;fetching=true;try{const id=$('run').value;const d=await get('/api/run?id='+encodeURIComponent(id));if(id!==$('run').value)return;current=d;$('error').hidden=true;$('connection').textContent='● 연결됨 · 5초 갱신';$('updated').textContent='확인 '+new Date(d.updated_at).toLocaleTimeString('ko-KR');renderOverview(d);renderResults();renderCharts();renderMedia(d);renderCommands(d);renderLogs(d);}catch(e){$('error').hidden=false;$('error').textContent=e.message+' 마지막 화면은 이전 데이터입니다. 웹 서버 터미널을 확인하세요.';$('connection').textContent='연결 끊김 · 재시도 중';}finally{fetching=false;}}
async function discover(){try{const d=await get('/api/runs');const before=$('run').value;const saved=localStorage.getItem('sac-run');const preferred=d.runs.includes(before)?before:d.runs.includes(saved)?saved:d.runs[0];$('run').replaceChildren(...d.runs.map(n=>{const o=el('option',n);o.value=n;return o;}));if(preferred)$('run').value=preferred;else{$('error').hidden=false;$('error').textContent='지원하는 실험 기록이 없습니다. 현재 matched continuation 또는 수동 실행 결과 폴더를 확인하세요.';}await refresh();}catch(e){$('error').hidden=false;$('error').textContent=e.message;}}
if (typeof document !== 'undefined') {
$('run').addEventListener('change',()=>{localStorage.setItem('sac-run',$('run').value);refresh();});$('refresh').addEventListener('click',discover);$('provider').addEventListener('change',renderResults);$('metric').addEventListener('change',renderCharts);window.addEventListener('resize',renderCharts);
for(const button of document.querySelectorAll('[data-copy]'))button.addEventListener('click',async()=>{try{await navigator.clipboard.writeText($(button.dataset.copy).textContent);button.textContent='복사됨';setTimeout(()=>button.textContent='복사',1500);}catch{button.textContent='텍스트를 선택해 복사';}});
discover();setInterval(refresh,5000);setInterval(discover,30000);

$('show-history').addEventListener('change', renderResults);
$('video-period').addEventListener('change', () => current && renderMedia(current));
$('video-choice').addEventListener('change', () => current && renderMedia(current));
$('learning-details').addEventListener('toggle', () => {if ($('learning-details').open) renderCharts();});
document.querySelector('nav a[href="#learning"]').addEventListener('click', () => {$('learning-details').open = true;});
}
if (typeof module !== 'undefined') module.exports = {completeEvaluation, sameCohort, deriveSummary, visibleEvaluations};

# SAC 개발 진행 기록

사용자 요청 범위: 환경 보완 → teacher/보상/collector/replay → 규약 SAC/BC → gate 통과 후 학습·비교 평가.
기준은 v1.2/schema3이며 원본 해시는 [착수 점검](sac-readiness-20261116.md)에 기록되어 있다.
구현·규약 대조·회귀·실제 실행·실패 분석을 반복하며, loss 감소를 정책 성공으로 판정하지 않는다.

## 현재 상태

| 단계 | 현재 증거 | 남은 작업 |
|---|---|---|
| 1. 환경·PD·속도 | 손상 truncation, coverage 분류, 적용 입력·동시 사건 기록, 고정 경로·검사 동등성 검증 | FOLLOW 목표 간격의 hard 거절 의미 확인 |
| 2. 학습 경험 생성 | 기존 Frenet 다항식 연결, 실제 보상·raw action·next_obs replay, train 분리 수집 실행 | teacher 취약 상태 보완, 20,000전이 수집 및 실제 수집 처리량 |
| 3. SAC·BC | 규약 encoder/sampler/twin-Q/BC/SAC/alpha/target/checkpoint 구현 및 CPU/GPU 검증 | 본 BC 5,000 minibatch, 충분한 데이터와 3 seed 실행 |
| 4. 수락·학습·비교 | 행동별 gate 집계와 read-only 실제 상태 확률 probe 구현 | 합성 4,096×초기/상단 분산, 실제 validation mean/256표본 gate, 온라인 학습·평가 실행기, BC-only 대비 비교 |

**전체 학습은 미완료다.** 현재 checkpoint는 연결 시험용 BC 2회 업데이트이며 `qualified=false`다.
SAC 1회 GPU 수식 연결 시험도 5,000 replay 기준 미달을 명시하고 본학습으로 집계하지 않았다.
20k 수집·5000 BC·본 SAC를 백그라운드에서 실행 중이라는 뜻이 아니다.

## 자가 검토에서 발견하고 처리한 문제

1. DATA_CORRUPTION이 terminated로 처리되던 규약 불일치 수정. 실패 final_obs를 reset_obs와 섞지 않는다.
2. scalar 지도/객체 계산이 검사 시간을 지배했다. 원래 표본/한계를 유지한 배열화·캐시 이후 실제 3장면 18입력을 이전 검사기와 대조했다.
3. 끝점에서만 정지하는 시험 경로가 Hermite 끝점에서 역방향 속도를 만들었다. 실패 입력은 보존하고 정지 suffix를 가진 정상 사례와 거절 회귀를 구분했다.
4. 기존 Frenet의 scalar 속도·초기 가속도0 투영이 실제 vx/vy·가속도 벡터와 맞지 않았다. 새 어댑터는 현재 지도 C2 frame에서 실제 위치·속도·가속도를 역변환하고 기존 KEEP/FOLLOW 다항식 생성기를 재사용한다. 기존 PPO/Frenet/MPC 코드는 수정하지 않았다.
5. 원래 다항식 jerk가 knot 유한 차분·Hermite에서 증폭됐다. teacher 생성의 편안한 가속도 목표를 1m/s²로 제한하되 더 큰 실제 초기 가속도는 보존한다. 공통 검사 상한 6m/s²·20m/s³는 유지한다. 이 설정과 원본 planner 이식 차이는 teacher 버전에 기록했다.
6. encoder의 Python dict 순서가 JIT에서 달라질 가능성을 제거했다. 고정 규약 순서를 사용하며 eager/JIT 대조를 검증한다.
7. replay 복원 시 큰 임시 배열로 메모리가 증가하지 않도록 NPY member를 1MiB씩 직접 최종 배열에 읽는다. 50k 배열 자체는 2.900GiB다. 전체 50k 충전 상태의 실제 peak RSS 검증은 아직 남았다.

## 확인이 필요한 규약 해석

기준서 §3: FOLLOW는 지정 대상 유지·추월/AVOID 전환 금지.
§11: `g_des=minimum_gap+time_gap*speed`, 실제 gap 오차에 보상 벌점.
현재 환경 `_check`는 `gap < g_des`를 전 구간 hard COMMAND_VIOLATION으로 처리한다.
문서가 이 목표 간격을 hard safety threshold로도 요구하는지는 명시적이지 않다.

사용자가 검사 기준 유지를 요구했으므로 이 조건을 임의로 제거하지 않았다.
목표 간격은 보상으로 유도하고 충돌·추월을 hard 검사할지, 현재 hard gap을 유지할지 확인을 요청했다.
이 결정에 의존하는 대량 수집·학습은 보류한다. 단순히 쉬운 장면만 골라 수락률을 높이지 않는다.

현재 validation 8장면 중 `83c142d159860844`는 초기 gap 11.231m, 목표 13.978m로
출발 시점부터 모든 제안이 hard gap 조건을 위반한다. 초기 상태만 평가하는 경우 FOLLOW mean
수락률 상한은 7/8=87.5%다. 다른 시점까지 포함한 전체 gate의 상한이라고 확대 해석하지 않는다.

## 실제 실행 결과

- 환경: [고정 경로·판정 동등성·속도 기록](sac-environment-gate-20261116.md).
- 최종 CPU 환경 회귀 106개(27.55초)와 학습 연결 회귀 13개(18.46초), 합계 119개 통과. teacher vector anchor·미래 로그 차단·raw u 보존·빈 replay 복원·gate 집계를 포함한다.
- GPU learner 회귀 3개 통과(63.20초): 밀도/gradient, BC std 고정, twin Q/target/alpha, bootstrap0 actor 미호출, 체크포인트 다음 업데이트 동일성.
- 실제 train 초기 112개(scene/provider) teacher 검사: KEEP 44/60, FOLLOW 14/52 통과. 전체 이유 OK58, COMMAND_VIOLATION27, KINEMATIC_LIMIT21, PREDICTED_COLLISION4, UNRESOLVED_SWEEP2. 이는 전체 rollout 성공률이나 BC 유효율이 아니다.
- 별도 train 연결 시험: seed0의 순서대로 두 장면 각10tick, actual replay20개, BC 적격10개. 실패10개도 원래 raw u와 실제 fallback 결과로 replay에 남았다.
- GPU BC batch256×2: loss 0.003725→0.003019, 컴파일 포함11.63초. 기기 peak bytes in use 약174MiB, process peak RSS 약2.13GiB. 정책 성능은 미검증.
- 실제 replay에서 GPU SAC batch256×1 수식 연결 시험: 모든 loss/gradient/alpha 유한. 컴파일 포함26.58초, device peak bytes in use 약395MiB, process peak RSS 약2.40GiB. replay20개이므로 본학습 기준에 미달하며 테스트 전용 entry point로 실행했다.

로컬 산출물:
- `outputs/sac_environment_gate/teacher_initial_train_v3.json`
- `outputs/sac_environment_gate/validation_anchor_feasibility.json`
- `outputs/sac_pipeline_smoke_v3/collection/{report.json,ticks.jsonl,replay.npz}`
- `outputs/sac_pipeline_smoke_v3/bc/`와 `sac_update_probe.json`
- 최종 trace/discount/버전 메타데이터를 포함한 재실행: `outputs/sac_pipeline_smoke_final/{collection,bc}/`

산출물마다 당시 source/dependency/manifest 버전을 기록한다. 이후 코드 변경 시 기존 replay와
checkpoint의 자동 재사용을 거절한다. 소스 변경 뒤에는 새 output에서 연결 시험을 재생성한다.

## 실행 도구

```bash
# 빠른 연결 시험: 본학습/gate 통과로 취급하지 않는다.
python scripts/run_sac_pipeline.py collect --smoke \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest.json \
  --output outputs/new-sac-smoke/collection
JAX_PLATFORMS=cuda python scripts/run_sac_pipeline.py bc --smoke \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest.json \
  --data outputs/new-sac-smoke/collection --output outputs/new-sac-smoke/bc
```

`--smoke`를 생략하면 collection20k/BC5000 설정이며, BC는 qualified collection을 요구한다.
현재는 위 FOLLOW 해석 확인 후 대량 실행한다. test는 열지 않으며,
scenario_id 대체 그룹의 주행 구간 중복 독립성은 여전히 확인되지 않았다.

환경 보완은 기존 #5에 누적하고, 학습 연결은 별도 기능 브랜치 하나에서 완성한다.
추가 PR을 세부 모듈마다 만들지 않는다. 자동 main 병합이나 학습 성능 보장은 하지 않는다.

# SAC 개발 진행 기록

현재 사용자 요청 범위: **Rule-based 경로 생성 vs 순수 SAC 경로 생성 비교**.
2026-11-17 결정에 따른 [실험 개정](../architecture/rule-based-vs-pure-sac.md)이
아래 과거 기록의 BC 필수 선행 조건보다 우선한다. 기존 BC 산출물은 보조 실험으로 보존한다.
기준은 v1.2/schema3이며 원본 해시는 [착수 점검](sac-readiness-20261116.md)에 기록되어 있다.
구현·규약 대조·회귀·실제 실행·실패 분석을 반복하며, loss 감소를 정책 성공으로 판정하지 않는다.

## 현재 위치 — 2026-11-18

학습·평가 기반과 첫3-seed 실험을 완료했고, **학습한 경로의 실제 주행 실패를
분리·보완하는 단계**다. 아래 과거 시점의 “미구현/미착수”는 당시 기록이다.
현재 범위는 KEEP/FOLLOW이며, 5개 행동 전체나 PPO/ASMC까지 완료했다는 뜻이 아니다.

| 처음 계획한 작업 | 현재 확인된 결과 | 남은 완료 조건 |
|---|---|---|
| 규약·속도·장면 점검 | 규약 대조, 실행 비용 측정, train59/validation8개 확보 | 확대 평가 장면의 별도 확보·분리 |
| 환경·PD·reward·collector·replay | 구현, 회귀, 실제 Waymax 연결·저장 검증 | 반복 재계획의 실패 원인 보완·재검증 |
| SAC 모델과 학습 실행 | 순수 SAC 구현, seed0/1/2 첫 예산 학습·평가 완료 | 실제 주행 가능한 학습 정책 확보 |
| Rule-based vs SAC 비교 | 첫 실패 비교·GIF·원인 보고서 완료 | 안정화 후 동일 조건 재평가, 별도 장면 일반화, 최종 보고 |

사용자 결정에 따라 BC는 주 실험에서 제외했다. 과거 BC 수집·학습 결과를 순수 SAC
학습에 섞지 않는다. 첫 순수 SAC의 validation은 각 seed16회 모두 충돌했다.
B 평균 좌표의 별도 비교도 reset 안정성만 개선했고 실제 주행 회귀가 남아 채택하지 않았다.
구현 항목 수를 세어 “전체 목표의 몇 % 완료”라고 표현하지 않는다. 아직 안정적인
SAC 주행 성능을 확보하지 못한 것이 일정의 핵심 불확실성이다.

### 2~3주 일정의 현재 운영 기준

11월16일 착수 점검을 일정 시작으로 **가정하면** 목표 범위는11월30일~12월7일이다.
이는 사용자가 확정한 특정 마감일이나 성능 달성 보장이 아니다.

1. 우선 같은 중간 상태에서 정책 경로·평균0 경로·Rule-based 경로를 검사하고,
   경로 자체의 실패와 가상/실제 PD 추종을 분리한다. 검사 한계는 유지한다.
2. 원인이 확인된 변경만 짧은 실제 주행으로 검증한다. 초기 상태 통과나 loss 감소만으로
   장시간 학습을 재개하지 않는다. seed0 짧은 실행 → 나머지 seed 재검증 순서를 유지한다.
3. 짧은 주행 회귀를 없앤 설정으로 본 실험을 실행하고, 동일 조건의 Rule-based와
   성공·충돌·거절·제동·정지 비중·보상을 함께 비교한다. 설정과 장면을 실행 중 바꾸지 않는다.
4. 개발에 반복 사용한 validation8개와 별도 평가를 구분해 최종 표·GIF·한계를 정리한다.
   예산 내 개선이 입증되지 않으면 실패 결과와 원인을 그대로 보고한다.

세부 근거: [착수 점검](sac-readiness-20261116.md),
[첫 학습 실패 분석](sac-final-failure-20261118.md),
[초기 업데이트 비교](sac-short-update-20261118.md),
[동일 상태 재계획·PD 분리](sac-replanning-20261118.md).

후속 분리 진단은 실제6episode·363tick을 재현했다.105개 상태·행동·이력 조합에서
비교한315개 제안 중 거절95개는 모두 가상 PD 이전의 reference 검사에서 실패했다.
빠른 학습 주행의 실제 선택 경로 추종 오차 최대는2.37mm였다(EMERGENCY 제외).
PD 게인 변경보다 상태별 경로 생성·학습을 우선 보완할 근거이며, 모든 PD 문제를
배제하거나 학습 성능을 확보했다는 뜻은 아니다.

학습 중 병행할 [시각화 → 실패 분석 → 평가 기준 정리 작업 목록](sac-analysis-workplan.md)을
별도 기록한다. 시각화는 저장된 checkpoint로 독립적인 CPU Waymax rollout을 만들며,
실행 중인 고정 학습 worktree와 replay를 수정하지 않는다.
실패 분석 보고서의 초기 버전도 실제 WARMUP 로그에 적용했고, KEEP/FOLLOW 대표 충돌을
저장 경로로 재현해 확인했다. 세부 수치·실행 명령·판정 한계는 위 작업 목록에 기록한다.

순수 SAC 첫 실행은809회 업데이트 후 수치 발산으로 중단됐다.
[안정화 진단·개정](sac-stability-20261117.md)에 원인 재현, 동일 데이터 대조 시험,
critic 변경과 자체 warmup만 재사용하는 새 실험 절차를 기록한다.

## 2026-11-17 실험 변경 시점의 상태

- 이전 teacher 수집20,082전이, accepted16,113개, BC seed0 5,000회 학습까지 완료했다.
- BC validation 평균 경로 통과율은 KEEP14.87%, FOLLOW29.98%로 gate 실패했다.
  BC seed1/2 및 BC 기반 온라인 SAC는 시작하지 않았다.
- Rule-based validation16episode는 목표7·충돌8·TASK_LOST1이었다. 기록된 코드·manifest가
  같은 경우만 순수 SAC 주 실험의 baseline으로 재사용한다.
- 순수 SAC 실행기는 빈 replay/새 learner에서 시작하고 BC 자료를 불러오지 않는다.
  warmup5,000개 뒤 업데이트를 시작하며, 비교 보고서는 Rule-based vs PURE_SAC를 요구한다.
  코드 구현·검증과 장시간 학습 완료를 구별한다. 실제 상태는 각 job/report를 확인한다.
- 전환 검증: 관련 CPU 테스트25개 통과, 추가 보완을 포함한 순수 SAC 규약 테스트8개를
  GPU 환경에서도 통과했다. 초기 actor의 유한 SAC 업데이트를 확인했으나 작은 합성 batch의
  test-only 수식 검증이며 실제5,000전이 이후의 본학습을 대신하지 않는다.
  기존 합성73,728표본 및 Rule-based16episode의 runtime·manifest·전체 coverage를 대조했고
  현재 core code hash가 같음을 확인했다. 물리 검사/PD/학습 수식 소스는 이번 전환에서 바꾸지 않았다.

## 이전 개발 착수 시점의 기록

| 단계 | 현재 증거 | 남은 작업 |
|---|---|---|
| 1. 환경·PD·속도 | 손상 truncation, coverage 분류, 적용 입력·동시 사건 기록, 고정 경로·검사 동등성 검증 | 장면별 긴 rollout과 실제 사건 지속 점검 |
| 2. 학습 경험 생성 | 기존 Frenet 다항식 연결, 실제 보상·raw action·next_obs replay, train 분리 수집 실행 | 20,000전이 수집 및 실제 수집 처리량 |
| 3. SAC·BC | 규약 encoder/sampler/twin-Q/BC/SAC/alpha/target/checkpoint 구현 및 CPU/GPU 검증 | 본 BC 5,000 minibatch, 충분한 데이터와 3 seed 실행 |
| 4. 수락·학습·비교 | 행동별 gate 집계와 read-only 실제 상태 확률 probe 구현 | 합성·실제 validation gate 실행, 온라인 학습·복구·비교 완료 검증 |

**전체 학습은 미완료다.** 현재 checkpoint는 연결 시험용 BC 2회 업데이트이며 `qualified=false`다.
SAC 1회 GPU 수식 연결 시험도 5,000 replay 기준 미달을 명시하고 본학습으로 집계하지 않았다.
사전학습 실행기는 20k 수집 → BC5000 → validation gate를 연결하고, 후속 실험 감독 실행기는
gate 통과 후 온라인 SAC와 비교 평가까지 연결한다. 실제 장시간 실행 완료는 아직 남아 있다.
실행 상태는 로컬 job/report를 확인하며, 완료 전 결과를 선반영하지 않는다.

## 자가 검토에서 발견하고 처리한 문제

1. DATA_CORRUPTION이 terminated로 처리되던 규약 불일치 수정. 실패 final_obs를 reset_obs와 섞지 않는다.
2. scalar 지도/객체 계산이 검사 시간을 지배했다. 원래 표본/한계를 유지한 배열화·캐시 이후 실제 3장면 18입력을 이전 검사기와 대조했다.
3. 끝점에서만 정지하는 시험 경로가 Hermite 끝점에서 역방향 속도를 만들었다. 실패 입력은 보존하고 정지 suffix를 가진 정상 사례와 거절 회귀를 구분했다.
4. 기존 Frenet의 scalar 속도·초기 가속도0 투영이 실제 vx/vy·가속도 벡터와 맞지 않았다. 새 어댑터는 현재 지도 C2 frame에서 실제 위치·속도·가속도를 역변환하고 기존 KEEP/FOLLOW 다항식 생성기를 재사용한다. 기존 PPO/Frenet/MPC 코드는 수정하지 않았다.
5. 원래 다항식 jerk가 knot 유한 차분·Hermite에서 증폭됐다. teacher 생성의 편안한 가속도 목표를 1m/s²로 제한하되 더 큰 실제 초기 가속도는 보존한다. 공통 검사 상한 6m/s²·20m/s³는 유지한다. 이 설정과 원본 planner 이식 차이는 teacher 버전에 기록했다.
6. encoder의 Python dict 순서가 JIT에서 달라질 가능성을 제거했다. 고정 규약 순서를 사용하며 eager/JIT 대조를 검증한다.
7. replay 복원 시 큰 임시 배열로 메모리가 증가하지 않도록 NPY member를 1MiB씩 직접 최종 배열에 읽는다. 50k 배열 자체는 2.900GiB다. 전체 50k 충전 상태의 실제 peak RSS 검증은 아직 남았다.

## FOLLOW 승인 반영 및 후속 검증

2026-11-16 사용자 결정: **목표 간격은 보상으로 유도하고 충돌·추월 금지는 엄격히 검사**.
기존 `gap < g_des` 일괄 거절을 제거했다. 지정 대상 ID, dropout 유예, 충돌 여유,
운동 한계는 유지한다. 기준 경로와 가상 PD 모두 같은 adaptive sweep에서 물리 OBB의
route 종방향 순서를 검사한다. 실제 로그 대상의 예상 밖 움직임은 post-step 사건과
명령 위반 벌점으로 기록하며 안전 성공으로 숨기지 않는다.

초기 상태 점검에서 FOLLOW teacher 수락은 14/52 → 41/52로 바뀌었다. 이는 저속 보완 전
비교이며 전체 rollout 또는 학습 모델 수락률이 아니다. 이후 teacher v4는 2m/s 미만에서
현재 차선 내 측방 위치를 유지하고, 거의 정지한 상태에서는 hold 후보를 사용한다.
KEEP/FOLLOW 명령·보상은 유지하므로 정지했다고 목표 성공을 주지 않는다.
실제 vx/vy와 가속도는 보존하고, 0.01m/s 미만의 미세 drift로 차체 OBB가 회전하지 않도록
경로 heading은 마지막 유효 heading 또는 실제 anchor yaw를 유지한다.

후속 validation 초기 8장면×2행동 teacher 경로 **16/16 수락**. 본학습 성공이나
BC stochastic gate 통과가 아니다. 항상 정지 편법은 평가의 stationary 비율·이동 거리·
return·종료 사유에 남기며, 기준 정책 비교는 별도로 남아 있다.

- 변경된 계약·WOMD·SAC 관련 회귀 90개 통과(46.16초). FOLLOW 충돌·추월·가상 PD·실제 사건, 속도 벡터 보존, 수집 복구를 포함한다.
- 고정 경로 PD 합성 13개 및 기존 실제 3초 경로 2개 재검증 통과. 실제 위치 RMS는 0.0552/0.0433m, 실제 도로·충돌 실패0이었다. 추가 입력 `baac7893b025167a`의 개발용 route reference는 jerk 한계에서 초기 거절됐다. 원본 실패도 보존하며 전체 장면 통과로 해석하지 않는다.
- 승인 후 새 train smoke: replay20개, BC 적격18개. 거절2개도 원래 raw action과 실제 fallback 결과를 보존했다.
- 새 GPU BC batch256×2 및 실제 validation 실행기 연결 완료. smoke 출력은 수락 gate를 승인하지 않는다.
- 합성 탐색 실행기는 production 보간·adaptive grid·극값·운동 검사 함수를 공유한다. 예비 16개/조건은 4,096개 기준을 대체하지 않는다.
- episode 경계에 replay/RNG/장면 순서/trace byte 위치를 저장한다. 복원 시 미확정 trace는 별도 파일에 보존하고 마지막 완료 경계로 돌아간다. 두 세대 checkpoint를 유지한다.

후속 산출물은 `outputs/sac_follow_gap_approved/`에 있다. 이전 checkpoint는 코드 해시가
다르므로 새 코드에서 학습 재사용하지 않는다.

## 이전 foundation 실행 결과

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
FOLLOW 승인은 반영했다. test는 열지 않으며,
scenario_id 대체 그룹의 주행 구간 중복 독립성은 여전히 확인되지 않았다.

환경 보완은 기존 #5에 누적하고, 학습 연결은 별도 기능 브랜치 하나에서 완성한다.
추가 PR을 세부 모듈마다 만들지 않는다. 자동 main 병합이나 학습 성능 보장은 하지 않는다.


## 대량 사전학습 실행

```bash
python scripts/run_sac_pretraining_job.py \
  --manifest /absolute/path/to/inventory/eligible_manifest.json \
  --output /absolute/path/to/new-job
```

고정 revision의 별도 worktree에서 실행한다. `job.json`은 PID·작업 단계·오류를,
`logs/`는 자식 프로세스 출력을, `collection/report.json`은 수집량·수락량·episode 결과를 남긴다.
`--resume`은 기존 수집의 마지막 episode 경계에서 이어가며, 같은 output의 동시 실행은 잠근다.
20k는 반복 수집을 포함한 전이 수이고 서로 다른 장면 20k개라는 뜻이 아니다.

합성 9사례×2분산×4096 검사와 수집 후 GPU BC 5000 minibatch를 실행한다.
BC는 seed0/1/2를 순서대로 사용하되 첫 실패 gate에서 진단을 위해 중단한다.
validation의 모든 scene/provider에서 결정적 mean rollout을 평가하고 행동별 90%를 먼저
요구한다. 이후 사전에 고정한 tick0/10/30의 도달 가능한 상태에서 mean 유효 상태당256개
표본을 검사한다. 도달하지 못한 tick과 mean 실패 비율도 공개한다. 이는 초기 연결 gate이며
모든 실제 상태의 확률적 안전을 인증하지 않는다. validation은 개발용이며 scenario 대체
그룹의 주행 구간 독립성은 미확인이다.

이 실행기는 gate 결과를 판정하고 실패 이유를 저장한다. 백그라운드 프로세스가 스스로
소스 코드를 수정하는 기능은 아니며, 후속 온라인 SAC·기준 정책·BC 대비 비교 평가는
아래 실험 감독 실행기로 연결한다. 실제 학습 성능은 검증 전이며 보장하지 않는다.


## 초기 ODD 재점검 (2026-11-16 후속 작업)

첫 실제 충돌 3개를 저장된 제안 경로로 재실행해 실제 위치·속도·판정·사건이 일치함을
확인했다. 세 경우 모두 정지한 ego를 뒤에서 접근한 로그 차량이 충돌했다. 주변 차량은
ego에 반응하는 에이전트가 아니므로 긴급제동만으로 모든 충돌이 예방되는 것은 아니다.
이 사건을 성공으로 재분류하거나 replay에서 제거하지 않는다.

초기 68개 고유 장면의 측정 앵커를 별도 점검한 결과 train의 `8e7e1e79d2e00130`은
속도 0.01482m/s, route 방향 속도 -0.01482m/s, 측정 곡률 약68.92/m였다.
기존 reset은 속력·가속도 크기만 확인해 이러한 초기 역방향/곡률 위반을 놓쳤다.
기준서 §9는 초기 측정 상태가 ODD/실행 한계 밖이면 사유를 기록해 제외하도록 요구한다.
따라서 production 검사와 동일하게 moving threshold0.01m/s, lateral acceleration4m/s²,
curvature0.3/m, route velocity -0.01m/s 한계를 reset에도 적용했다. 측정값을 덮어쓰거나
제약을 완화한 것이 아니다. 같은 규칙으로 모든 train/validation 후보를 다시 점검한다.

이전 `outputs/sac_pretraining_20261116/` 작업은 중단하고 진단 자료로 보존한다.
해당 데이터는 수정된 런타임과 버전이 다르므로 본 BC/SAC에 재사용하지 않는다.
새 manifest는 원본과 같은 inventory 디렉터리에 별도 파일로 쓰며 원본 체크섬·제외 이유를
유지한다. 추가 회귀를 포함한 관련 테스트98개 통과(45.76초).


## 온라인 SAC·비교 실험 실행기

`run_sac_online.py`는 같은 runtime/manifest의 qualified BC5000 checkpoint와 실제
validation gate를 요구한다. gate의 모든 scene/provider·도달 가능한 tick0/10/30 probe,
행동별 mean90%·stochastic90%·합성95%를 확인한다. smoke BC로 시작하려는 실제 CLI
실행은 output 생성 전에 거절됐다.

- CPU Waymax / GPU 정책·learner를 명시적으로 분리한다. 실제 validation의 GPU 정책·CPU
  시뮬레이터 연결 시험을 수행했으며 결과에 `TFRT_CPU_0`를 기록한다. 연결 시험은 학습 성능이 아니다.
- 실제 유효 전이마다 batch256 업데이트1회. 손상 전이는 업데이트하지 않는다. validation
  전이는 replay에 넣지 않으며 actor의 결정적 mean으로 평가한다.
- 온라인 전이5000마다 평가, 10000을 지난 첫 episode 경계에 실험 checkpoint.
  추가로 약1000전이마다 episode 경계 복구본2세대를 보존한다. 학습기·replay·장면순서 RNG·
  정책 RNG·trace 위치·runner hash를 저장하며 미확정 trace를 별도 보존한다.
- 처음은 seed0/1/2 각각10000전이의 초기 실험이다. budget 도달 뒤 진행 중 episode는 끝낸다.
  장기 수렴이나 성능 향상을 보장하는 예산이 아니다.
- 동일 validation 전체에서 BC·SAC를 평가하고 teacher·정지 제안·제동 제안·항상 거절되는
  제안·충돌 방향 제안을 기준 정책으로 비교한다. 모든 제안은 같은 공통 검사를 거친다.
  고정 FOLLOW 대상의 로그 유효성은 정책이 바꿀 수 없어 대상 ID 제거를 정책 행동으로
  가장하지 않는다. 실제 TASK_LOST 결과와 기존 dropout/종료 회귀를 확인한다.
- 성공·충돌·offroad·거절·제동·정지 비중·return·episode 길이·추종 오차를 함께 보고한다.
  비교는 seed별 paired delta와 평균·표본 표준편차를 JSON/Markdown으로 남긴다.

`run_sac_full_experiment.py`는 수집과 함께 기준 정책 평가를 먼저 수행한다. 사전학습
gate 통과 뒤 수집 편중 감사, BC 평가, 온라인 SAC, SAC 평가, 최종 비교를 연결한다. 의존 작업 실패·버전 불일치·
평가 실패 시 증거를 보존하고 중단한다. 백그라운드 실행기가 스스로 코드를 수정하지는 않는다.
새 실행기 전체의 장시간 학습·복구·비교 완료는 실제 qualified BC가 나온 뒤 검증해야 하며,
함수/연결 시험을 그 결과로 대신하지 않는다.

정지 편중을 줄이려는 별도 출발 경로 후보는 일부 실제 상태에서 curvature_rate 검사에
실패했다. 실패 후보는 `stationary_launch_probe.json`에 보존하고 teacher에는 적용하지 않았다.
실제 BC 결과를 보고 데이터 편중·근사 오차·경로 검사 실패를 구분해 후속 수정한다.


후속 연결 검증: 기준 정책5개×KEEP/FOLLOW 각각2tick 및 GPU BC 정책×2행동 각각2tick을
실제로 실행했다. 모두 smoke이며 정책 성능을 판정하지 않는다. GPU teacher mean probe의
256개 표본은 공통 CPU 검사에서256개 통과했고, 실제 environment token/memory/observation이
변하지 않는 것을 확인했다. 이는 단일 상태의 probe 구현 검증이며 BC gate 통과가 아니다.
실제 BC gate와 온라인 정책이 같은 GPU/CPU 배치를 쓰도록 기기를 기록하고 진입 시 대조한다.

초기 ODD 보완 후 수집은 `outputs/sac_pretraining_reset_v2_20261116/`에서 진행한다.
새 manifest의 후보126개는 train59개 고유 장면과 validation8개 고유 장면의 KEEP/FOLLOW
쌍이며, FOLLOW가 없는 train 장면은 KEEP만 포함한다. 합성73,728표본 검사는 다시 통과했다.
source·data·모델 의미를 변경하지 않는 실행기 확장은 동일 runtime hash의 episode 경계
복구본에서 재개할 수 있다. script/runner revision은 별도로 기록한다.

GPU 정책 → CPU Waymax → 실제 train 전이 replay → GPU 업데이트 연결 시험에서도
loss/gradient/alpha가 유한했다. replay23개·batch256·업데이트1회이며 test-only entry를
사용했다. production의 replay5000 및 BC gate 요건을 우회한 학습으로 집계하지 않는다.
본 온라인 CLI에는 이 우회 옵션이 없다.

## 2026-11-18 순수 SAC 최종 평가와 실패 진단

`pure_sac_stable_20261117`의 seed0/1/2 학습·평가·비교가 모두 완료됐다.
최종 validation은 각 seed16회 모두 충돌, 제안 경로 전부 `UNRESOLVED_SWEEP` 거절이다.
학습 시작 직후부터 train 경로 통과율도 붕괴했다. train/validation reset96개 조합의
평균 경로와 표본192개를 추가 조사했으며 실제 Waymax 비교 GIF2개를 저장했다.
자세한 수치·접촉 근거·다음 개발 순서는
[최종 실패 분석](sac-final-failure-20261118.md)을 따른다. 이번 진단에서는 학습 규약이나
검사 한계를 변경하지 않았으며 새 장시간 학습은 시작하지 않았다.

## 2026-11-18 초기 업데이트 보완 비교

직접 출력 gain·Q 초기화와, 사용자 승인하에 별도 실험으로 허용된 B 평균 좌표를
0/1/10/100회 업데이트에서 비교했다. seed0의12개 설정 중 B/gain0.001/Q0/actor
LayerNorm 후보가 처음 통과하던 reset 경로를 유지했고, seed1/2에서도 유지했다.
그러나 실제 반복 재계획에서는 목표에 도달하던 장면이 충돌로 바뀌었다.
고정 replay와 새 경험을 넣는 짧은 온라인 진단을 구분해 기록하며, 초기 검사 통과를
주행 성공으로 취급하지 않는다. 기본 모델·주 실험 설정을 바꾸거나 장시간 학습을
재시작하지 않았다. 수치와 채택 판단은
[초기 업데이트 비교 결과](sac-short-update-20261118.md)를 따른다.

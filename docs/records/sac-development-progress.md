# SAC 개발 진행 기록

사용자 요청 범위: 환경 보완 → teacher/보상/collector/replay → 규약 SAC/BC → gate 통과 후 학습·비교 평가.
기준은 v1.2/schema3이며 원본 해시는 [착수 점검](sac-readiness-20261116.md)에 기록되어 있다.
구현·규약 대조·회귀·실제 실행·실패 분석을 반복하며, loss 감소를 정책 성공으로 판정하지 않는다.

## 현재 상태

| 단계 | 현재 증거 | 남은 작업 |
|---|---|---|
| 1. 환경·PD·속도 | 손상 truncation, coverage 분류, 적용 입력·동시 사건 기록, 고정 경로·검사 동등성 검증 | 장면별 긴 rollout과 실제 사건 지속 점검 |
| 2. 학습 경험 생성 | 기존 Frenet 다항식 연결, 실제 보상·raw action·next_obs replay, train 분리 수집 실행 | 20,000전이 수집 및 실제 수집 처리량 |
| 3. SAC·BC | 규약 encoder/sampler/twin-Q/BC/SAC/alpha/target/checkpoint 구현 및 CPU/GPU 검증 | 본 BC 5,000 minibatch, 충분한 데이터와 3 seed 실행 |
| 4. 수락·학습·비교 | 행동별 gate 집계와 read-only 실제 상태 확률 probe 구현 | 합성·실제 validation gate 실행, 온라인 학습·비교 평가 실행기 |

**전체 학습은 미완료다.** 현재 checkpoint는 연결 시험용 BC 2회 업데이트이며 `qualified=false`다.
SAC 1회 GPU 수식 연결 시험도 5,000 replay 기준 미달을 명시하고 본학습으로 집계하지 않았다.
아래 후속 실행기는 20k 수집 → BC5000 → validation gate까지 연결한다. 온라인 SAC 실행기는 아직 남아 있다.
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
소스 코드를 수정하는 기능은 아니며, gate 통과 후 온라인 SAC·기준 정책·BC 대비 비교 평가
실행기를 연결하는 작업은 남아 있다. 학습 성능은 보장하지 않는다.


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

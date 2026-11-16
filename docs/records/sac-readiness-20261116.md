# SAC 학습 착수 점검 — 2026-11-16

기준 코드: `b558ba8`의 공통 계약·WOMD 제어 환경. 이 점검은 학습을 실행하거나
수락 기준을 변경하지 않고 누락 구현, 실행 비용, 개발용 장면 가용성을 확인한다.
기존 87개 회귀 통과는 아래 전체 학습 gate 통과를 뜻하지 않는다.

## 기준과 판정

기계 계약은 [planning_v3.json](../../configs/contracts/planning_v3.json)의 v1.2/schema3이다.
같은 버전의 소유자 기준선 `PPO_SAC_PD_구현기준선_v1_2.txt` 원본과 대조했다.
JSON만으로는 encoder 구조, reward 전체 수식, replay 용량, 학습 순서와 상세 gate를
모두 복원할 수 없으므로 학습 모듈 구현 시 아래 조항도 함께 적용해야 한다.

- JSON SHA256: `e754f282200b5cc4ce4097fbdcbc8e0b510b63ee2bb759388806bdd5bb84a4fc`
- 텍스트 SHA256: `10b2a882e5dfb3f31b4b300793a67a3d10aa19886421a3782e7b9a58394b4a64`
- `기반 있음`은 해당 일부 구현이 있다는 뜻이며 해당 gate의 통과 판정이 아니다.
- KEEP→FOLLOW부터 진행하고 후속 행동을 mask하는 것은 원본 §3/§12의 개발 순서다.
  원본 §18의 전체 연결 완료 조건까지 충족했다고 주장하지 않는다.

| 규약 범위 | 현재 증거 | 판정 / 필요한 일 |
|---|---|---|
| G0·G1 좌표/시간/ID/shape/mask/codec | `src/contracts/`, `tests/contracts/` | 기반 있음. DecisionTransition/VersionBundle의 학습 경계와 전체 encoder·sampler는 없음 |
| §4 encoder | `CoreObservation` 및 현재/과거 관측 생성기 | 미구현. mask를 포함한 MLP[64,64] mean/max pooling, 객체 ID별 3시각 결합, 점→polyline pooling, actor/Q1/Q2 encoder 분리 필요 |
| §5 상관 sampler | JSON에 식·버전 있음, production sampler 없음 | 미구현. 선형 mu[30,2], rho[2], 상관행렬 B, 속도별 g, float64 샘플/raw_u, 규약 ell·gradient 검증 필요. tanh/clip/독립 noise로 대체 금지 |
| G2 실제 장면 환경 | reset/route/corridor/current observation/Waymax step | 기반 있음. 가용 장면 수·제외 분모와 FOLLOW 실제 실행 범위 확인 필요 |
| G3 고정 경로 PD | 모델 대조·일부 합성 회귀, 실제 한 tick 오차 | 미통과. 직선·가감속·곡선·저속·정지·초기 오차·교체의 고정 경로 시험 전체가 필요 |
| G4 공통 검사·fallback | NEW/REUSED/BRAKE/EMERGENCY, 만료/dropout/실제 사고 회귀 | 부분 검증. 아래 종료 의미 불일치 보완 및 collector의 실제 next_obs 대조 필요 |
| §10 teacher·BC | 기존 `src/planning/frenet_planner.py`; 현재 개발용 경로 2종 | 새 환경용 teacher adapter/데이터/BC 모두 없음. teacher도 30점→raw_u→동일 step으로 실행 |
| §11 shared_tick_reward_v3 | 새 StepResult에는 reward 없음; 기존 merge reward는 별도 | 미구현. 실제 적용 명령 snapshot, raw reject·fallback 비용, 실제 운동·사건, once-only 보상 필요 |
| §10~11 collector·replay | 진단 JSON만 있음 | 미구현. raw action과 실제 fallback 구분, final_obs/reset_obs, 손상 제외, bootstrap/discount, RAM 상한 검증 필요 |
| G5 learner·checkpoint | 기존 training 코드는 legacy PPO | 새 SAC 없음. twin Q/target, alpha, optimizer/RNG/replay/version을 포함한 episode 경계 checkpoint 필요 |
| G6 소규모 학습 | BC/SAC checkpoint·학습 경험 없음 | 미착수. seed 0/1/2, BC-only 대비 BC+SAC 평가 필요 |
| G7·G8 행동 확장/PPO/ASMC | 인터페이스 일부만 있음 | 후속 단계. 초기 KEEP/FOLLOW 결과를 5행동·전체 ASMC 비교로 확대 해석하지 않음 |

### 확인된 규약 불일치

`environment._advance()`는 DATA_CORRUPTION도 다른 실패처럼 `terminated=true`로
반환한다. 원본 §11은 `truncated=true, sample_valid=false`와 replay 제외를 요구한다.
새로 등장한 객체의 width=0인 합성 사례로 다음 값을 재현했다.

```text
actual:   DATA_CORRUPTION, terminated=true,  truncated=false, sample_valid=false
required: DATA_CORRUPTION, terminated=false, truncated=true,  sample_valid=false
```

현재 손상 표본 제외 표시는 맞지만 종료 의미는 학습 collector 연결 전에 수정해야 한다.
또한 enum에 있는 MAP_COVERAGE_EXIT를 현재 환경은 별도로 반환하지 않고 corridor 위반으로
처리한다. 새 reward/event 구현에서 원본 §11의 지도 범위 이탈 실패 분류를 구분해야 한다.
이번 점검에서 runtime 동작이나 기존 회귀 기대값을 바꾸지는 않았다.

### 학습 연결에 적용할 고정 조건

- teacher 시작 목표 20,000전이, BC 5,000 minibatch, BC 동안 분산 head 초기값 유지.
  통과 teacher의 `MSE(mu,u_teacher)`를 쓰고 실제 실행 전이를 동일 raw_u에 연결한다.
- G3: 위치 RMS≤0.3m/최대≤0.75m, 속도 RMS≤1m/s/최대≤2m/s,
  방향 최대≤15°, 해당 시험의 실제 도로/충돌 실패 0. 현재 반복 재앵커 오차로 대체 불가.
- sampler 합성 사례별 4,096개, 초기/상단 sigma에서 운동 검사 통과≥95%.
  실제 validation BC mean은 행동별 전체 검사≥90%. 수락된 mean 상태별 확률 샘플
  256개로 KEEP/FOLLOW 각각 전체 검사≥90%, mean 실패율과 상태별 분포도 별도 공개.
- 학습률 3e-4, batch256, gamma .99/tick, tau .005, gradient norm clip10,
  alpha 초기 .001/범위 [1e-6,.01], 전이당 update 1회, 유효 replay 5k 이후 업데이트.
- replay 50k/RAM 4GiB 상한. 관측·mask·action을 packed 배열로 보관하고 실제 메모리 측정.
  신경망은 float32, sampler/저장 raw_u는 float64, 새 JAX 학습 프로세스는 x64 활성화.
- episode 종료(기록 끝/200tick 포함)는 bootstrap=0, 수집 batch 경계는 bootstrap=1.
  bootstrap=0 행은 target actor/log-prob를 호출하지 않는다. 손상 전이는 제외한다.
- 평가 매5k전이/checkpoint 매10k전이를 기준으로 하되 checkpoint는 episode 경계에 저장.
  seed 0/1/2. provider/PD/검사/reward 버전을 run 안에서 고정한다.

## 재현 도구

```bash
export WOMD20_ROOT="<WOMD training_20s 디렉터리>"
export READINESS_OUTPUT="outputs/sac_readiness_20261116/inventory"
JAX_PLATFORMS=cpu PYTHONPATH=. python scripts/screen_womd_training.py screen \
  --root "$WOMD20_ROOT" --output "$READINESS_OUTPUT" --shards 200
JAX_PLATFORMS=cpu PYTHONPATH=. python scripts/screen_womd_training.py verify \
  --output "$READINESS_OUTPUT"

# 다른 데이터 스캔/학습 작업과 동시에 실행하지 않는다.
JAX_PLATFORMS=cpu PYTHONPATH=. python scripts/benchmark_womd_control.py \
  --proto "$READINESS_OUTPUT/candidates/00005-00044.pb" \
  --steps 20 --trials 2 --output outputs/sac_readiness_20261116/timing/keep-route-cpu.json
```

스크리너는 CRC 검증된 record에서 scene ID 해시 split을 먼저 정한다.
`group_id=scenario_id` 대체값과 `overlap_unverified`를 명시한다. test는 지도/정책
검증을 수행하지 않고 보류한다. production route builder의 사전 필터를 통과한 장면만
원본 proto를 로컬 캐시하고, 이어 실제 환경 reset을 KEEP/FOLLOW 각각 적용한다.
프로토바이트 SHA256, shard 크기/mtime, 코드 해시, 실제 lane/boundary ID와 제외 사유를 남긴다.
동일 scene ID 중복은 eligible manifest에서 격리한다. source/runtime 변경 시 새 출력 경로를 쓴다.

`eligible_manifest.json`은 **reset 수락 목록**이다. teacher 유효 경로·학습 성공·독립 test를
인증하지 않는다. 기록의 남은 tick 합은 목표/실패 조기 종료를 무시한 수집량 상한이다.
동일 장면 KEEP/FOLLOW 실행을 서로 독립적인 장면으로 세지 않는다.

벤치마크는 원본 로딩·그림·JSON 저장을 제외하고 경로 생성, 전체 검사, 실제 dynamics,
다음 관측을 측정한다. 디바이스 완료를 기다리며 reset 시간/첫 tick/그 이후를 구분한다.
동일 장면을 두 번 reset하므로 두 번째 trial은 일부 JIT 캐시가 재사용된다.
실제 teacher/learner는 없으므로 이 수치를 학습 처리량으로 표현하지 않는다.

## 실행 비용과 장비

RTX 4060 8GiB, 시스템 RAM 약15GiB, WSL2, Python 3.10.21,
JAX/JAXLIB 0.6.2, Flax 0.10.7, Optax 0.2.8, NumPy 2.2.6,
TensorFlow 2.21.0, waymo-waymax 0.1.0에서 측정했다.
CUDA 장치 인식과 float64 소규모 행렬 연산은 확인했으며, 아직 없는 SAC 모델의
batch256 VRAM 사용량이나 replay 실제 resident memory를 측정했다고 주장하지 않는다.
50k replay의 obs/next_obs·field/item mask·raw_u만 packed 배열로 계산하면 약2.88GiB다.
metadata/reward/복사 버퍼/할당 오버헤드까지 4GiB 이내인지 구현 시 추가 측정해야 한다.

train 장면 `d675f8a070ebe2e4`에서 route_reference/KEEP를 동일 초기 상태로 반복했다.
데이터 스캔과 실행 시간을 겹치지 않았다. GPU/CPU 모두 같은 공통 검사를 사용한다.

| 실행 | trial당 tick | 평균 초/tick (trial 0 / 1) | reset 초 (trial 0 / 1) |
|---|---:|---:|---:|
| CPU, 기존 x64=false 프로세스 | 20 | 2.103 / 2.112 | 1.466 / 0.606 |
| CPU, 학습 예정 x64=true 프로세스 | 10 | 2.179 / 2.152 | 1.310 / 0.601 |
| GPU, x64=true 프로세스 | 10 | 5.103 / 5.043 | 4.309 / 3.185 |

x64=true의 두 번째 CPU trial은 첫 tick을 제외해도 평균2.150초, p95 2.178초다.
따라서 느린 속도를 첫 JIT 컴파일만으로 설명할 수 없다. 모든 tick에서 NEW가 선택됐다.
GPU의 두 번째 trial은 CPU보다 약2.34배 느리다. 현재 Python/NumPy 검사와 JAX 배열의
작은 단위 접근이 섞인 환경을 그대로 GPU로 옮기는 방법은 이 workload에서 도움이 되지 않았다.
이 결과는 미래 신경망 learner를 CPU에 두라는 결론이 아니다. 환경과 GPU learner 간
배치 전송 또는 환경의 배열화/벡터화를 별도로 설계·측정해야 한다.

다른 실행 출처와 FOLLOW도 CPU/x64=true에서 별도로 측정했다.

| 장면 / 입력 | trial당 tick | 평균 초/tick (trial 0 / 1) | 각 trial의 실행 출처 |
|---|---:|---:|---|
| train `baac7893b025167a`, constant/KEEP | 30 | 2.896 / 2.845 | NEW7 / REUSED7 / BRAKE10 / EMERGENCY6 |
| validation `8b4c1e7e09f5f4c3`, route/FOLLOW | 10 | 1.355 / 1.337 | NEW10 |

후자의 FOLLOW 경로는 실제10 tick 모두 수락됐다. 두 번 같은 초기 상태로 실행한
짧은 timing workload이며 독립 성능 표본2개나 teacher 품질 검증으로 세지 않는다.
fallback workload에서는 첫 tick을 제외한 warm p95도3.855초였다.
따라서 한 장면의 평균2.15초를 모든 상태의 처리 비용으로 일반화할 수 없다.

별도 cProfile 실행 10 tick에서는 `step_waypoints` 누적28.613초 중
`_check`가24.138초(약84%), `_scene_check`가18.514초,
`Route.contains`가12.705초였다. 뒤의 함수들은 앞 함수 내부에 포함되므로 합산하지 않는다.
`build_observation`도 reset 포함11회에3.206초가 들었다.
`_scene_check`는10 tick에서20,002회 호출됐다. 현재 검사의 미래 경로 및 가상 PD
전체 sweep에서 도로 polygon/차체 검사가 반복되는 것이 이 사례의 주 병목이다.
cProfile 자체의 오버헤드가 있으므로 수집 시간 추정에는 위 비계측 wall clock을 사용한다.

## 확보한 개발용 장면

로컬에는 원본 1,000 shard, 약92.52GB가 있다. 이번에는 앞200개 shard의14,166개
record를 확인했다. 이 중 train11,320 / validation1,429 / test1,417로 hash 분리됐다.
test1,417개는 정책·지도 수락 검사를 하지 않고 보류했다.

나머지12,749개 중 사전 필터 거절12,659개, 구조 오류21개, 도로 후보69개였다.
69개 모두 production reset을 두 provider로 검사했다. KEEP68개와 FOLLOW60개가
수락됐고, 초기 제한속도 초과1개 장면은 두 provider 모두 거절됐다.
KEEP 수락 장면 중8개는 선행 대상이 없어 FOLLOW reset이 거절됐다.

| 개발 split | 고유 장면 수 | KEEP 수락 | FOLLOW 수락 | KEEP 남은 tick 상한 | FOLLOW 남은 tick 상한 |
|---|---:|---:|---:|---:|---:|
| train | 60 | 60 | 52 | 11,723 | 10,158 |
| validation | 8 | 8 | 8 | 1,561 | 1,561 |

장면 ID 중복은 없었고 train/validation group ID 집합은 서로 겹치지 않았다.
다만 scenario ID 대체 그룹이므로 실제 겹치는 주행 구간의 독립성은 미확인이다.
KEEP 기준 초기 속도 분포는 train 정지(<0.2m/s)6 / 저속(0.2~2)4 / 이동(≥2)50,
validation 정지2 / 저속1 / 이동5다. 전체 ODD의 대표 표본이라는 주장은 하지 않는다.

사전 필터의 주요 거절 사유는 현재 유효 corridor 없음5,933, 합류/교차로 제외1,816,
초기 상태 범위 밖1,569, 경계 coverage 부족1,327, source 시간 간격1,030,
분기/짧은 route808 등이었다. 구조 오류21개는 track 길이와 timestamp 수의 불일치였다.
이 수치는 현재의 보수적인 초기 ODD/지도 구현에서의 제외이며 원본 데이터 전체의
불량률로 해석하지 않는다. test는 구조 검사도 하지 않았으므로 오류율 분모에 섞지 않는다.

스캔 합산 약505초, 실제 reset 호출 합산 약145초였다(프로세스 시작·cache 정리 제외).
후보 캐시와 manifest는 아래 로컬 산출물에 보존했다.

- `outputs/sac_readiness_20261116/inventory/eligible_manifest.json`
- 같은 디렉터리의 `screen-*.json`, `reset-*.json`, `summary.json`, `candidates/*.pb`
- manifest SHA256: `185d511bfc78e4590f0d69ab6b51a6391609fa1a46567b64c8b2399c2a2ff879`

모든 수락 행의 cached proto checksum과 train/validation 분리 여부를 다시 확인했다.
이 목록으로 KEEP/FOLLOW pilot 수집을 시작할 장면을 선택할 수 있다.
하지만 teacher 전이는 아직0개이며, 조기 목표/실패 종료로 실제 수집량은 표의 상한보다
작다. 두 행동의 같은 장면을 합쳐 독립 장면으로 부르거나, 반복 reset의 중복 경험만으로
데이터 다양성이 확보됐다고 주장하지 않는다. teacher 연결 후 유효 표본 수가 부족하면
나머지 shard를 같은 규칙으로 확장하며 validation을 train으로 옮기지 않는다.

## 수집·평가 예산

수락 경로를 실행한 CPU warm trial의 약2.15초/tick을 임시 기준으로 계산한 예다.
teacher 탐색/여러 후보 검사, learner 업데이트, reset/I/O, 실패로 제외되는 BC 표본은
포함하지 않았으며 전체 데이터의 평균 속도나 확정 완료시간이 아니다.

| 가정한 작업량 | 제어 루프에만 필요한 시간 |
|---|---:|
| 실제 전이20,000개 | 약11.9시간 |
| 실제 전이100,000개 | 약59.7시간 |
| 100,000개씩 seed3개 | 약179시간(7.5일) |
| validation8장면×2행동×평균80tick | 약46분/평가 |
| 위 평가를5k마다100k까지20회 | 약15.3시간/seed |

평균80tick은 예산 예시이지 실측된 validation episode 길이가 아니다.
BC용20,000개 수집 시간은 **통과한 teacher 전이/초**를 따로 측정해야 한다.
SAC에서는 유한 거절 action도 실제 실패 전이로 저장하므로 NEW 수락률만큼
전이 수를 줄여 계산하면 안 된다. 확률 gate의 상태별256후보 검사는 추가 비용이다.
현재 속도에 대량 학습과 반복 평가를 그대로 얹으면2~3주 일정의 여유가 작다.
fallback workload의 약2.85초/tick을 적용하면100k 수집만 약79시간/seed가 된다.
100k는 예산 비교 예시이며 규약이 요구하는 최소 학습량으로 새로 정한 값이 아니다.
같은 수락 개발 경로의 `_check`만5회 반복 측정한 평균도1.952초였다(CPU/x64=true,
매회1,028개 검사 시각,5회 모두수락). 이 비용으로 validation 상태100개×256후보를
검사한다고 가정하면 약13.9시간이다. 상태100개는 비용 예시이며 gate의 표본 수를
새로 정한 것이 아니다. 실제 BC mean/확률 샘플의 분포에 따른 측정이 추가로 필요하다.

## 점검 산출물 검증

`JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest -q tests/scripts/test_womd_readiness.py`
결과 **8 passed (2.81초)**. split_v2 고정 사례, test geometry/캐시 접근 방지,
중복 격리, 사전 필터와 실제 reset 앵커/route 대조, 시간 통계를 검증했다.
변경 문서 상대 링크5개와 `git diff --check`도 확인했다.
기존 runtime 소스는 변경하지 않았다.

### 다음 구현 우선순위

1. collector 이전에 DATA_CORRUPTION 종료 의미와 MAP_COVERAGE_EXIT 사건 분류를 보완한다.
   G3 고정 경로 시험과 G4 next_obs/실제 action·fallback 대조도 학습 전에 마무리한다.
2. static 지도/polygon/투영 계산의 캐시와 배열 단위 차체 검사를 먼저 최적화한다.
   이전 구현과 동일 장면·고정 입력에서 수락/거절, 경계 접촉, fallback, 실제 제어 결과를 대조한다.
   표본 간격·검사 대상·threshold·float64 계산 정확도를 줄여 속도를 얻지 않는다.
3. 규약 encoder/sampler/보상/collector/replay/teacher adapter를 연결하고, 하나의 전이가
   실제 입력→실제 결과→보상→next_obs까지 일치하는지 확인한다. 기존 PPO 실행과 새 SAC 실행을 구분한다.
4. 통과 teacher 전이 수와 초/유효전이를 측정한 후 BC와 확률 수락 gate, SAC 예산을 확정한다.
   후보 여러 개를 검사하는 teacher 비용과 학습 업데이트/평가 비용은 지금 벤치마크에 없다.

처리량 개선 목표를 정할 때는 단일 장면의 최저 지연 대신 KEEP/FOLLOW·수락/거절·
fallback을 포함한 고정 workload 전체를 사용한다. 전처리/캐시와 환경 여러 개의 병렬 실행도
각각 측정해야 하며 아직 달성하지 않은 배속을 일정에 확정값으로 넣지 않는다.

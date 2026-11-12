# 20초 WOMD 제어 환경

이 개발 단계는 [schema 3 공통 계약](../contracts/planning.md)을 실제 Waymax 제어 step에
연결한다. 기존 `MergeEnvironment`, 4행동 PPO, Frenet+MPC 실행 경로는 변경하지 않는다.

## 연결한 흐름

```text
Scenario proto → reset/지도·초기 상태 검증 → 현재·과거 관측
  → World XY 30점 제안 → 보간·운동·차체·추종 검사
  → NEW / REUSED / BRAKE / EMERGENCY 선택 → PD 또는 보호 입력
  → Waymax 0.1초 → 실제 사건·다음 관측
```

구현은 [womd_control](../../src/environment/womd_control/)에 있다.

| 경계 | 현재 동작 |
|---|---|
| `reset(scene_key, seed, provider_spec)` | proto 또는 `(shard_path, record_index)`를 받아 새로운 episode와 Boundary 생성 |
| `submit_behavior(request, token)` | 물리 시간 없이 명령 수락·거절, revision 갱신, 동일 요청 재전송 처리 |
| `planning_anchor(token)` | 현재 자차 위치·속도·가속도·yaw의 읽기 전용 snapshot |
| `step_waypoints(proposal, token)` | 제안을 검사하고 실제 물리를 정확히 한 tick 진행 |
| `step_forced(token)` | 현재 KEEP/FOLLOW 단계에는 강제 호출 경계가 없으므로 오류 반환 |

고정 KEEP/FOLLOW provider를 지원한다. 새로운 의미의 명령만 ID를 증가시키고,
같은 의미의 요청과 5 tick heartbeat는 ID·생성 시각·PD 기억을 유지한다.
token의 episode/scene/tick/revision/schema가 다르면 물리를 진행하지 않는다.
새 명령 수락은 경로와 PD 오차 기억을 초기화한다.

자차 XY·속도·yaw는 `InvertibleBicycleModel`의 가속도/곡률 입력으로 갱신한다.
주변 객체는 원본 로그를 재생한다. 새로 등장한 객체의 크기가 이전 invalid 프레임의
0으로 남지 않도록, step 후 현재 source frame의 크기·시간을 복원한다.
이 과정에서 자차의 제어된 XY·속도·yaw를 로그로 덮어쓰지 않는다.

## 데이터와 지도 수락 기준

- `current_time_index` 이후 연속 3개 유효 프레임을 확보하고, 기록이 최소 5초 남아야 한다.
- source 간격은 0.1±0.005초이며 모델 tick은 0.1초다. 누적 시계 차이를 보고한다.
- 초기 속도 0~15m/s, 유효한 차량 크기·가속도·제한속도·차로 소속을 요구한다.
- 실제 `left/right_boundaries`의 lane index 범위와 `road_line/road_edge` feature를 사용한다.
  현재 차체를 포함하는 연속 유효 구간을 선택하며, 누락 구간을 임의 폭으로 채우지 않는다.
- 같은 방향의 유일한 후속 차로를 연결한다. 분기·합류·보간 차로·신호·정지표지 또는
  필요한 길이의 지도 부족은 초기 실행 범위에서 제외한다.
- 목표는 요청 속도 기준 30~80m 앞에 고정하고, 목표 뒤 계획용 지도도 확보한다.
  초기 거리 상계로 명백히 도달 불가능한 목표를 제외한다.

geometry 방식은 원본 경계 feature에 대한 투영과 연속 polygon 검사다.
서로 충돌하는 경계 연결, 내부 누락, 잘못된 좌우 관계, 좁은 폭, 자기 교차는 제외한다.
이 보수적인 초기 경로 구성기는 전체 WOMD의 모든 정상 도로를 처리한다고 주장하지 않는다.
`ResetRejected`는 정책 실패와 분리해서 보고한다.

## 관측과 검사

관측은 실제 sim history, 현재 유효 객체, 현재 지도와 명령·실행 상태로 구성한다.
미래 자차 로그는 정책에 공개하지 않는다. 전체 feature shape와 field/item mask는
공통 계약을 따르며, 비활성·누락 필드는 0+false다. 이 단계에서는 신호화된 route를
제외하고 signal tensor를 비활성으로 둔다. 초기 provider는 PPO 후보 관측을 발급하지 않는다.

현재 상태를 앵커로 5차 Hermite 보간을 만들고, 정지 suffix의 제한된 정렬을 기록한다.
검사에는 속도·가속도·곡률·횡가속도·jerk·곡률 변화율, 역주행, 경계 polygon의 차체 포함,
현재 객체의 등속 예측 충돌과 PD 가상 추종을 사용한다. 가상 동역학은 설치된 Waymax와
독립 vx/vy·yaw가 다른 경우까지 대조한다.

검사 간격은 최대 0.02초이며 상대 이동과 회전에 따라 세분화한다. 다항식 내부 속도 극값과
knot 양쪽도 포함한다. 0.002초보다 작은 간격이 필요하면 `UNRESOLVED_SWEEP`로 거절한다.
유한 표본 검사이므로 연속 시간 안전 보증은 아니다.

유효 새 경로가 우선이고, 거절 시 같은 명령의 기존 경로를 생성 시각 기준 최대 2 tick까지만
재검사·재사용한다. 그 다음은 현재 상태에서 만든 제동 후보를 같은 검사로 확인하고,
통과하지 못하면 비상 제동을 적용한다. 기존 경로의 나이는 갱신하지 않는다.
비상 제동 역시 항상 충돌을 피한다는 보증은 아니다.
제동 후보의 가속도 변화는 보간 후 jerk의 여유를 확보하도록 tick당 0.15m/s²로
제한한다. 최종 경로의 공통 jerk 상한 20m/s³와 나머지 검사 기준은 그대로다.
정지 전환·큰 초기 가속도 등에서는 이 후보도 거절될 수 있다.

FOLLOW는 원래 대상 ID를 유지하고 간격을 검사한다. 대상 누락은 마지막 실제 관측으로
최대 2 tick 추정하며, 3 tick째에는 `TASK_LOST`로 종료한다. 다른 차량으로 바꾸지 않는다.
실제 충돌·offroad·corridor 위반은 목표 성공보다 먼저 처리한다. Waymax의 원본 overlap과
offroad도 기록하며, 실제 사건 판정은 0.1초 단위다.
source 기록 끝 또는 200 tick에서 truncation하고 장면을 자동 연결하지 않는다.
손상된 현재 객체 geometry는 `DATA_CORRUPTION`과 `sample_valid=false`로 처리한다.

## 실행

```bash
conda activate its-merge

# 사용자의 데이터 위치를 지정한다.
export WOMD20_ROOT="<WOMD training_20s 디렉터리>"

PYTHONPATH=. python scripts/smoke_womd_control.py \
  --shard "$WOMD20_ROOT/training_20s.tfrecord-00005-of-01000" \
  --record-index 44 --steps 5

# 종료 또는 최대 200 tick까지 실행하는 도로 기반 제어 진단
PYTHONPATH=. python scripts/smoke_womd_control.py \
  --shard "$WOMD20_ROOT/training_20s.tfrecord-00005-of-01000" \
  --record-index 44 --steps 200 --planner route_reference
```

`--provider fixed_follow`로 고정 FOLLOW를 선택할 수 있다. 적합한 선행 차량이 없으면
reset을 거절한다. 기본 `--planner constant_acceleration`은 현재 속도·가속도를
연장한다. `route_reference`는 현재 지도와 명령을 이용해 3초 뒤 경로 지점으로
5차 다항식을 만들며, 현재 상태의 위치·속도·가속도를 앵커로 사용한다.
FOLLOW에서는 동일 대상의 현재 관측과 간격으로 목표 속도를 제한한다.
두 입력 모두 미래 로그를 읽지 않고 같은 검사를 받는 개발 도구이며,
ASMC 이식이나 학습된 SAC가 아니다. 경로의 수락이나 충돌 회피를 보장하지 않는다.

결과는 기본 `outputs/womd_control/`에 저장한다. source frame, tick, 실제 제어 입력,
검사 사유, 실행 출처, 실제 종료 상태, 지도 feature ID와 시계 차이를 확인한다.
보고서 schema 2는 매 tick 원자적으로 저장하고, 실행 오류에도 완료한 tick을 남긴다.
파일명에는 record/provider/planner/요청 tick 수가 포함된다.
`--progress-every 10`으로 진행을 출력하며 0이면 출력을 끈다.

- `status`: `EPISODE_ENDED`, `STEP_LIMIT_REACHED`, `RESET_REJECTED`, `RUNTIME_ERROR`.
- `outcome`: 목표 도달, 기록 끝, 시간 제한, 실패, 실행 상한, reset 제외, 실행 오류를 구분한다.
  `--steps 200`은 상한이고 실제 종료가 먼저 발생하면 즉시 멈춘다.
- `attempts`: 새 제안·재사용·제동 후보별 검사 결과와 운동 제한의 세부 원인.
  `check`는 원래 새 제안의 결과이며 선택된 후보의 결과와 다를 수 있다.
- `tracking_error`: 실제 step 후 상태와 **선택된 경로**의 같은 시각을 비교한 위치·속도·방향 오차.
  재사용 경로는 원래 생성 시각을 유지한다. 비상 입력에는 경로가 없으므로 `null`이며
  RMS에 0으로 넣지 않는다. 정지 상태의 방향 오차도 `null`이다.
- `motion`: 실제 속도·yaw 차분과 입력 jerk/곡률 변화율. 최초 tick의 이전 제어가
  없으면 입력 변화율은 `null`이다. `tracking_by_source`는 실행 출처별 표본 수와 RMS/max를 담는다.

reset 제외는 exit code 2, 실행 예외는 비정상 exit로 반환한다. 정상적으로 보고서를
완성하면 충돌 사례도 exit code 0이므로, 판정에는 반드시 `outcome`과 `terminal_reason`을 사용한다.

## 검증과 남은 범위

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest -q \
  tests/contracts tests/environment/test_womd_control.py \
  tests/environment/test_womd_control_validation.py \
  tests/environment/test_behavior_action.py \
  tests/scenarios/test_womd20.py \
  tests/scenarios/test_physical_merge_validation.py \
  tests/scenarios/test_physical_merge_validation_real_womd.py
```

회귀는 제어 자차/로그 주변차 분리, 미래 로그 차단, 실제 모델 대조, stale 요청,
명령 ID·revision·heartbeat, 새 객체 크기, 경로 재사용 만료, 제동, 대상 소실,
지도 제외, 손상 데이터, 사고 우선순위와 기록 끝까지의 진행을 다룬다.
[장기 실행 검증 기록](../records/womd-control-validation.md)에 실제 장면의 성공·실패,
제동 보완 전후 비교와 수치의 해석 범위를 기록했다.

이 단계에는 PPO 후보 생성/호출 스케줄러, 학습 encoder/sampler, STOP/MERGE/AVOID의
실행 상태 전이, reward/collector/replay, BC/SAC 학습이 없다. StepResult는 제어 검증용이고
학습 reward·가짜 action을 생성하지 않는다. 합성 회귀와 일부 실제 장면 실행은 전체
G3/G4 수락, 전체 데이터 검증, GPU 실행 검증이나 정책 성능 평가를 대신하지 않는다.

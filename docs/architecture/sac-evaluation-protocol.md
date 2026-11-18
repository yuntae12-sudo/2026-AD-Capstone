# Rule-based vs 순수 SAC 평가 운영 기준

2026-11-18. [주 실험 개정](rule-based-vs-pure-sac.md)의 지표를 함께 판단하고,
개발 중 진단과 최종 평가를 구분한다. 물리 검사·보상·학습 진입 규약을 완화하거나
BC 전용 gate를 다시 추가하는 문서가 아니다.

## 비교 단위와 고정 조건

- 현재 범위는 KEEP/FOLLOW다. 동일 scene/provider/초기 상태/대상 ID에서 두 정책을 실행한다.
- 고정한 checkpoint, actor/Q 구조, manifest, contract, reward, 검사·PD, simulator와
  실행 script 해시를 기록한다. 진행 중인 run의 설정·장면을 바꾸지 않는다.
- Rule-based는 현재 Frenet 어댑터다. SAC는 결정적 평균 경로로 주 평가하고,
  확률 표본 검사는 별도로 표시한다. BC/teacher replay는0이어야 한다.
- 같은 장면의 KEEP/FOLLOW와 seed0/1/2는 반복 측정이다. 이를 독립 장면6개로 세지 않는다.
- 미완료 진단, 예산 중단, 손상 데이터와 policy 실패를 구분한다. 손상으로 전체 평가가
  완료되지 않으면 완료된 공식 비교로 표시하지 않는다. 충돌/거절 장면은 임의 제외하지 않는다.

## 지표의 분모와 해석

현재 구현 `scripts/sac_experiment_runtime.py::summarize_episodes`와 같은 정의를 쓴다.

| 지표 | 분모와 정의 | 같이 확인할 항목 |
|---|---|---|
| 성공률 | ROUTE_GOAL episode / 평가 episode | RECORD_END·TIME_LIMIT·TASK_LOST는 성공이 아님 |
| 충돌률 | 실제 COLLISION event가 있는 episode / 평가 episode | 로그 주변 차량의 비반응성과 충돌 전 이력 |
| 도로 이탈률 | 실제 OFFROAD event가 있는 episode / 평가 episode | corridor/map coverage 실패도 outcomes/events에 따로 공개 |
| 경로 거절률 | 거절된 신규 제안 tick / 전체 실행 tick | REUSED로 움직여도 신규 제안 거절은 유지 |
| 제동률 | BRAKE 또는 EMERGENCY tick / 전체 실행 tick | 두 출처의 개별 횟수와 최초 개입 시각 |
| 정지 비중 | 실제 post-step 속도<0.1m/s tick / 전체 실행 tick | 초기 속도와 episode 길이, 조기 충돌 |
| 평균 보상 | episode별 training reward 합의 평균 | 성공·충돌·정지 비중을 동시에 보고 |
| 추종 위치 RMS | 실행된 경로가 있는 tick의 오차 제곱 평균 제곱근 | 분모 tracking_samples 공개; EMERGENCY를0오차로 넣지 않음 |

사고 event는 같은 episode에서 여러 종류가 함께 있을 수 있으므로 비율의 합이100%가
될 필요가 없다. tick 지표는 episode 길이로 가중되므로 episode별 값과 길이도 보존한다.
이 값들을 고정3초 PD 시험의 gate와 혼동하지 않는다.

## 개선 판정과 짧은 실행의 역할

보상 상승 하나로 개선을 선언하지 않는다. 성공률이 낮아지거나 충돌이 늘면 원인을
설명해야 하며, 정지·제동 증가로 얻은 보상 개선은 따로 표시한다. 여러 지표가 상충하면
그 trade-off를 그대로 보고하고 임의의 가중 합으로 숨기지 않는다.

짧은 진단에서는 먼저 수치 유한성·raw action/계약 보존을 확인하고, reset 경로 검사,
같은 중간 상태 검사, 실제 반복 재계획 주행을 순서대로 확인한다. 이전에 성공하던
같은 장면의 회귀를 확인하지 않고 큰 학습 예산을 쓰지 않는다. 이는 개발 비용을 줄이는
운영 기준이며, 몇 개 장면 통과를 공식 성능 인증이나 새로운90% gate로 취급하지 않는다.

최종 표는 KEEP/FOLLOW를 분리하고 SAC 각 seed와 Rule-based의 paired 차이,
seed 평균·표본 표준편차를 모두 제공한다. 성공/실패 장면 목록과 초기 속도별 분석도
함께 보존한다. seed 중 가장 좋은 결과만 대표값으로 고르지 않는다.

## 추가 평가 장면 확보 계획

현재 validation8개는 모델 선택과 실패 진단에 반복 사용했으므로 **개발용**이다.
현재 장면·조건의 기존 결과는 유지하고, 확대 평가는 별도 manifest/run으로 만든다.

1. 기존 hash split과 중복 scene ID 격리 규칙을 유지한다. 추가 원본 shard를 조사하되
   기존 train·개발 validation과 겹치지 않는 후보 ID·출처·제외 사유를 먼저 기록한다.
2. policy 결과를 보지 않고 기존 ODD/reset 기준으로 후보를 선정한다. 성공한 Rule-based
   장면이나 SAC 경로가 잘 통과하는 장면만 고르는 방식은 사용하지 않는다.
3. 목표는 개발용8개와 별도로 최소32개 고유 평가 장면을 확보하는 것이다. **현재 확보한
   수가 아니다.** 부족하면 실제 분모와 부족 이유를 공개하고 목표 달성을 가장하지 않는다.
4. 모델/설정을 고정한 뒤 별도 평가를 실행한다. 그 결과로 다시 모델을 수정하면 해당
   장면은 개발 자료가 되므로 다음 독립 평가 자료와 구분한다.
5. 현재 source의 drive 단위 독립성은 확인되지 않았다. scene 분리만으로 drive 독립이나
   실제 도로 일반화가 검증됐다고 주장하지 않는다. 가능하면 drive/중첩 메타데이터로 보완한다.

추가 데이터 스캔과 성능 측정은 같은 장비에서 겹치지 않게 수행한다. 기존 scanner는
test split에 정책 검사를 수행하지 않으므로, 최종 평가 자료 준비 시에도 사전 봉인과
정책 비의존 선정을 명시적으로 유지해야 한다. 현재 실험을 확대 manifest로 덮어쓰지 않는다.

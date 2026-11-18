# 같은 상태의 Q 예측과 실제 실행 결과 대조

2026-11-18. [직전 온라인1,000회 후보](sac-action-coordinates-20261118.md)를 고정해
학습 없이 진단했다. **저장된 실패 행동의 Q는 학습했지만, 같은 관측에서 현재 actor의
행동에 부여하는 Q가 크게 과대평가되는 사례를 실제 실행으로 확인했다.** 새로운
validation 장면의 문제만으로 설명할 수 없다. 이를 모든 실패의 유일한 원인으로
확정하거나, 보완·학습 성능 검증까지 완료했다고 주장하지 않는다.

## 수행 범위와 비교 방법

- 두 validation 장면의 KEEP/FOLLOW4episode·154tick을 저장된 주행과 대조했다.
  위치·속도·검사 결과·실행 출처·종료·보상이 일치해야 진단을 완료한다.
- 고정 간격, 최초 거절·제동·정지 및 충돌 직전에서31개 상태를 선택했다. 각 상태에서
  현재 평균, 확률 표본8개, 평균0, Rule-based의11개 후보, 총341개를 실제 한 단계 실행했다.
- 각 분기는 동일한 simulator snapshot·PD memory·old plan·행동 명령에서 시작한다.
  부모 주행과 분리된 ledger를 사용하며, 실제 replay 대조로 분기의 간섭도 검사한다.
- Q는 `min(Q1,Q2)`이며 거절 후 fallback까지 포함하는 미래 보상 예측이다.
  검사 통과 확률이 아니므로 “유효 경로의 Q가 항상 높아야 한다”를 정답으로 삼지 않는다.
- 한 단계 target은 실제 보상과 target Q·현재 정책 표본8개의 soft value로 계산했다.
  종료 시에는 bootstrap이0이므로 **즉시 보상이 정확한 Q target**이다. 비종료의
  target은 학습된 Q에 의존하므로 독립적인 실제 수익의 정답으로 취급하지 않는다.
- KEEP의 reset/최초 거절4개 상태에서는 평균·현재 정책 후보 중 최고 Q·유효 후보 중
  최고 Q를 선택했다. 각 후보 이후에는 고정된 확률 정책으로 실제 종료까지2개 공통
  RNG seed를 실행했다. 첫 고정 행동의 entropy는 제외하고 이후 `gamma^k*(r-alpha*ell)`을
  합산했다. 강제 종료/truncation의 bootstrap 처리는 현재 규약을 따른다.
- BC/teacher 경험 삽입, replay 변경, learner update, 검사·PD·보상 변경은 없다.
  Rule-based와 평균0 후보는 현재 정책 분포 밖일 수 있어 Q의 비교 해석에 주의한다.

## 결과1: 충돌 직전에도 Q가 실패를 예측하지 못했다

| 장면·행동 | pre-step tick | 현재 평균 경로의 Q | 실제 종료 보상 |
|---|---:|---:|---:|
| 빠름 KEEP | 53 | +0.1213 | -20.014 |
| 빠름 FOLLOW | 58 | +0.1163 | -20.017 |
| 저속 KEEP | 21 | +0.1450 | -20.014 |
| 저속 FOLLOW | 18 | +0.0755 | -20.017 |

네 경우 모두 다음 한 단계에서 충돌했다. 평균 절대 오차는20.1300이었다.
종료에서 미래 보상·entropy를 더하지 않으므로, 이 차이는 향후 주행 표본의 변동성이나
entropy 보너스로 설명되지 않는다.

충돌 상대 차량도 입력에서 빠져 있지 않았다. 주행을 다시 재현해 실제 agent tensor의
상대 위치·상대 속도·field/item mask를 확인했다. 빠른 장면의 상대 차량4877은
KEEP slot0/FOLLOW slot1, 저속 장면의504도 slot0/slot1에 포함되어 있었다.
빠른 KEEP에서는 정지한 자차 뒤5.52m의 차량이 약9.49m/s로 접근 중이었다.
이는 해당 차량의 입력 누락 가설을 배제하는 근거이며 encoder가 그 정보를 제대로
표현하거나 이용했음을 입증하는 것은 아니다.

## 결과2: 행동 간 점수 차이에도 실제 근거가 부족했다

- 현재 평균+확률 표본 안에 유효 후보가 있는 상태는14/31개였다. 유효·거절 후보가
  섞인12개 상태 중5개에서는 최고 Q 후보가 거절됐다. 평균 경로가 거절되지만 유효
  확률 표본이 있는 상태도6개였다. 이는 학습 정책 자체의 후보 분포에 대한 관찰이다.
- 거절 후보들이 **보상·실행 제어·다음 관측 전체·bootstrap·실행 tick 수**까지 같아지는
  그룹이26개였다. 현재 평균/표본끼리의 Q 차이는 최대1.1571이었다. 평균0/Rule-based까지
  포함한 차이는 최대10.0162지만, 분포 밖 후보의 영향을 혼합해 대표 수치로 쓰지 않는다.
- 이후 주행24개는 충돌21·목표 도달3이었다. 이는 선택한4개 상태·후보·공통 RNG의
  반복 측정이며 새로운24개 독립 평가 장면이나 공식 성공률이 아니다. 평균 경로를 쓰는
  기존4episode 평가와도 구분한다. 실현 soft return 대비 Q의 평균 차이는+12.4507이었다.
  후보마다2개 표본뿐이므로 정책 기대값의 정확한 오차나 신뢰구간으로 해석하지 않는다.

예를 들어 저속 KEEP 시작 상태에서는 최고 Q 표본의 Q가+0.345였지만 경로가 거절됐다.
이 후보 이후2회 주행의 soft return은 약-16.63/-16.64였고 모두 충돌했다. 평균 경로의
Q는+0.307, 이후 수익은 약-15.73/-12.97이었다. 이 작은 표본에서 높은 Q의 후보가
더 좋은 실제 결과를 내지는 않았다.

## 결과3: 기존 관측에서도 현재 actor 행동을 과대평가했다

원래 자체 warmup5,000개 중 bootstrap=false인62개 전체를 검사했다. 기록된 행동의
target은 저장된 즉시 보상과 같으므로 이 검사는 미래 가치 추정에 의존하지 않는다.
큰 벌점 그룹은 reward≤-10인44개이며 원시 event를 복원하지 않고 이를 모두 특정
충돌 종류라고 재명명하지 않는다.

| 모델 | 큰 벌점44개: 기록 행동 Q의 평균 절대 오차 | 같은 관측의 현재 평균 행동 Q 평균 |
|---|---:|---:|
| 100회 | 18.5673 | -0.4262 |
| 1,000회 | 0.1995 | +0.000094 |

1,000회 모델의 기록 행동 Q 평균은-19.8483, 실제 보상 평균은-20.0154였다.
행동을 바꾸면 다음 결과가 달라질 수 있으므로 표의 마지막 열 자체를 정답 오차로
계산하지 않았다. 대신 Q 차이가 큰 서로 다른 train 장면을 선택해 원래 행동 이력으로
종료 직전 상태를 복원하고 기록 행동/현재 평균 행동을 각각 실제 실행했다.

확인된 예: train `6a185a176a1e70e0`, KEEP tick65.

| 같은 상태의 행동 | Q | 실제 보상 | 실행·결과 |
|---|---:|---:|---|
| 저장된 자체 warmup 행동 | -20.1309 | -20.014 | EMERGENCY → COLLISION |
| 현재 actor 평균 행동 | +0.1447 | -20.014 | EMERGENCY → COLLISION |

두 행동 모두 거절됐고 다음 관측 전체의 hash도 같았다. 따라서 이 사례의 Q 상승은
실제 위험 감소가 아니다. 기존 상태에서 행동만 달라져도 발생하므로, 관측의 새로운
장면 일반화 문제만으로 설명할 수 없다. Q 차이 상위 사례를 의도적으로 선택한
반례 재현이며 발생 빈도의 불편 추정은 아니다.

두 번째 반례는 train `9f7a210d74379953`, FOLLOW tick72다. 저장 행동 Q=-20.1187,
현재 평균 행동 Q=+0.0932였지만 실제 보상은 둘 다-20.017, EMERGENCY→COLLISION이고
다음 관측 hash도 같았다. 두 상태 복원 과정의137개 선행 tick은 저장된 관측 feature·
mask 및 보상과 대조했다. 현재 평균으로 바꾼 행동까지 실제 실행한 이 두 반례에서는
행동 변화에 따른 가치 상승이 잘못된 예측임을 직접 확인했다.

## 결론과 다음 작업

현재 확인된 문제는 **현재 정책이 생성하는 행동 영역에서의 Q 과대평가와 잘못된
행동 간 가치 차이**다. actor가 이런 Q를 최적화하면 실제로 개선되지 않는 경로를
선호할 수 있다. 이 메커니즘과 학습 실패 전체의 인과 비중을 동일시하지 않는다.
평균 학습 loss 감소와 기존 replay 행동의 적합도만으로 정책을 승인할 수 없다.

다음 보완은 train 경험에서 현재 actor 행동의 분포와 실패 사례가 얼마나 포함되는지,
Q 오차가 기록 행동/현재 정책 행동에서 어떻게 달라지는지부터 기준으로 삼는다.
수집·replay 연결과 분포를 먼저 확인한 뒤 Q 보완 후보를 동일 조건으로 비교한다.
그 후보는 종료 상태 오차·동일 실행 결과의 가치 차이·새 주행 회귀를 함께 검사한다.
수집 데이터와 평가 장면을 섞거나, 안전 검사 완화·BC/teacher replay로 문제를 우회하지 않는다.
규약의 손실식·업데이트 비율을 바꾸는 제안은 별도 개정으로 검토해야 하며 이번 작업에서
그런 변경이나 장시간 재학습을 시작하지 않았다. B 평균의 주 실험 채택도 미결정이다.

## 검증·산출물·재현

- 관련 테스트8개 통과: 분기 상태 격리, 기존 주행 재현, 검사 계측 보존, 표본/상태 분모,
  종료 target 오차와 확률 continuation의 구분을 포함한다.
- core hash `96fa1d625b0d8525b1de6de05a12cc132721a03ee47eb51dff3d6e25f02e30ef`,
  모델 hash `0c59d1ac5985616246b9f557cb5caed0ca070d6eeab7e9044f6230f541cb69ee`.
  각 report는 script·manifest·contract·source 해시를 보존한다.
- [원시 결과](../../outputs/sac_q_audit_20261118/): `audit-complete-observation/report.json`,
  `warmup-terminal-actions.json`, `collision-observation-slots.json`과 대응 실행 로그.
- [집계](../../outputs/sac_q_audit_20261118/summary/): `summary.json`, 후보/순위/동일 결과/
  이후 주행 CSV, `terminal-q.png/svg`, `continuation-q.png/svg`.
  collider tensor 확인 코드는 원시 결과 폴더의 `inspect_collision_observation.py`에 보존한다.

```bash
python scripts/diagnose_sac_q_values.py \
  --diagnostic outputs/sac_action_coordinates_20261118/aligned-online1000 \
  --evaluation outputs/sac_action_coordinates_20261118/aligned-online1000-driving \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest_reset_v2.json \
  --output outputs/reproduce-q-audit --samples 8 --continuations 2

python scripts/summarize_sac_q_audit.py \
  --report outputs/reproduce-q-audit/report.json --output outputs/reproduce-q-summary

python scripts/diagnose_sac_q_warmup.py \
  --checkpoint outputs/pure_sac_stable_20261117/online-seed0/checkpoints/step-00015191 \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest_reset_v2.json \
  --diagnostics outputs/sac_action_coordinates_20261118/aligned-online100 \
                outputs/sac_action_coordinates_20261118/aligned-online1000 \
  --output outputs/reproduce-warmup-q.json --reproduce-count 2
```

`its-merge` 환경에서 실행하며, 결과 폴더는 덮어쓰지 않는다. 원래 학습 checkpoint와
가중치는 읽기 전용으로 사용한다.

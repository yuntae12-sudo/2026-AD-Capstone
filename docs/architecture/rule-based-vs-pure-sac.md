# Rule-based 경로와 순수 SAC 경로 비교

2026-11-17 사용자 결정으로 주 실험을 **Rule-based vs 순수 SAC**로 변경했다.
이는 기준서 v1.2의 BC 사전학습·teacher replay 초기화·BC 전용 진입 gate를 대체하는
명시적 실험 개정이다. 원본 기준서와 기존 BC 결과는 보존한다.
기계 판독 설정은 [pure_sac_v1.json](../../configs/experiments/pure_sac_v1.json)이다.

같은 날 Q 발산으로 [critic 안정화 개정](../records/sac-stability-20261117.md)을 추가했다.
critic head LayerNorm, float64 gradient norm 계산, 전체 학습 상태 유한성 검사를 사용한다.
기존 ReLU-only 실험과 구분하며 발산한 가중치를 새 실험에 가져오지 않는다.

## 비교 대상과 공통 조건

| 항목 | Rule-based | 순수 SAC |
|---|---|---|
| 경로 생성 | 현재 WOMD/PD 환경에 연결된 FrenetTeacher v4 | 새로 초기화한 actor가 생성한 미래 3초 XY 30점 |
| 학습 자료 | 없음 | 자신의 train 장면 주행 경험만 사용 |
| 모방학습 / teacher replay | 해당 없음 | 모두 0 |
| 행동 지시 | 고정 KEEP / FOLLOW | 동일 |
| 실제 실행 | 공통 검사 → 기존 경로 재사용/제동 fallback → PD → Waymax | 동일 |
| 최종 평가 | validation 전체의 결정적 Frenet 경로 | 같은 validation 전체의 actor 평균 경로 |

Rule-based는 원본 MORAI/Frenet/MPC 파이프라인 전체가 아니다. 실제 앵커 vx/vy·가속도와
지도 C2 좌표계를 맞춘 현재 Frenet 어댑터를 사용한다. 그 차이는 teacher 버전으로 기록한다.
두 방식의 관측·route·대상·도로/객체 정보·초기 상태·종료·충돌 판정·PD를 고정한다.
SAC는 공통 물리 기반 경로 디코딩과 안전 검사/제동을 사용한다. 따라서 여기서 ‘순수’는
**시범 데이터/사전학습 없이 강화학습한다**는 뜻이며 모든 사전 지식이 없다는 뜻은 아니다.

## 학습과 검사

- actor와 두 critic을 새로 초기화한다. actor mean head 0, 초기 sigma 0.02 등
  기존 네트워크·분포·raw action·정밀도·보상·optimizer 규약을 유지한다.
- replay는 빈 상태로 시작한다. BC checkpoint/gate 인자는 순수 SAC 모드에서 거절한다.
- 최초 유효 전이 5,000개는 초기 actor의 확률 표본으로 수집하며 업데이트하지 않는다.
  5,001번째부터 실제 유효 전이마다 batch256 SAC 업데이트를 1회 수행한다.
  초기 5,000개에 대한 업데이트를 나중에 몰아서 수행하지 않는다.
- 경로 거절·제동도 실제 정책 행동의 결과로 replay에 저장한다. 손상 전이는 제외한다.
  teacher가 대체 경로를 제공하지 않는다. validation 경험은 학습 replay에 들어가지 않는다.
- 학습 전 BC 평균/표본 통과율90% 요건은 제거한다. 합성18개 case/sigma 쌍×4096표본의
  95% 검사는 유지한다. 합성 검사는 정해진 기준 경로 주변의 탐색 분포 검사이며,
  초기 actor가 실제 도로에서 좋은 경로를 만든다는 보장은 아니다.
- 실제 매 tick 충돌·추월·도로·운동 한계·PD 추종 검사는 그대로 적용한다.
  실제 정책의 거절률과 주행 성능은 학습 전 통과 조건이 아닌 평가 결과로 기록한다.
- seed0/1/2 각각 초기 총 15,000개 유효 전이를 수집한다(초기 수집5,000 + 업데이트10,000).
  진행 중 episode는 끝내므로 실제 전이 수는 예산보다 조금 많을 수 있다.
- 5,000전이마다 validation 평가, 약1,000전이마다 episode 경계 복구본,
  10,000전이 및 예산 도달 경계에 실험 checkpoint를 저장한다.
  첫 5,000전이 평가는 업데이트 전 actor 평가다.
- 체크포인트에는 실험 개정 ID/해시, 초기 replay 크기, BC 업데이트0, 자체 전이·SAC
  업데이트 수, RNG·장면 순서·trace 경계를 저장한다. 다른 모드의 재개와 비교 혼입은 거절한다.

## 실행과 결과 해석

```bash
python scripts/run_pure_sac_experiment.py \
  --manifest <inventory/eligible_manifest_reset_v2.json> \
  --synthetic <qualified-synthetic.json> \
  --output <new-output-directory>
```

중단 복구에는 동일 인자와 `--resume`을 사용한다. `--baseline-report`는 같은 런타임·
장면 전체·seed0의 완료된 Rule-based 평가를 검증해 재사용한다. 학습 데이터는 재사용하지 않는다.
실행기는 고정된 clean revision에서 합성 검사 확인 → Rule-based 평가 → 순수 SAC
seed0/1/2 학습·평가 → 비교 보고서를 순서대로 수행한다. 오류가 나면 증거를 남기고
중단하며 실행기 자체가 소스를 고치지는 않는다.

진행 상황은 Python 표준 라이브러리만 사용하는 별도 모니터로 확인한다.
현재 실험의 기본 경로는 `outputs/pure_sac_stable_20261117`이다.

```bash
python scripts/monitor_sac_training.py
# 다른 실험 또는 한 번만 출력:
python scripts/monitor_sac_training.py --output <experiment-directory> --once
```

5초마다 전체 수집 목표와 seed별 막대·퍼센트, SAC 업데이트 수, 중간/최종 평가 상태를
갱신한다. seed가 바뀌어도 명령을 다시 실행할 필요가 없다. 수집량은 episode 경계 보고서와
완전히 기록된 최신 업데이트 로그를 사용한다. warmup 중에는 episode가 끝날 때 수치가 바뀐다.
초과 수집은 전체 목표에서 seed별100%까지만 반영한다. 이 퍼센트는 평가 시간을 포함한
시간 진행률이 아니다. 최종 비교 완료/오류 중단까지 관찰하며 `Ctrl+C`는 모니터만 종료한다.

성공률·충돌률·도로 이탈률·경로 거절률·제동 개입률·정지 비중·보상·episode 길이를
함께 비교하고, 개별 평가의 추종 RMS도 확인한다. 결정적 Rule-based 한 결과에 대해
SAC seed별 차이와 그 평균·표본 표준편차를 보고한다. BC-only/BC+SAC는 보조 실험으로만 남긴다.

15,000전이는 초기 실행 가능성 및 성능 점검 예산이며 수렴을 보장하지 않는다.
현재 validation은 8개 고유 장면으로 작고 drive 단위 독립성은 검증되지 않았다.
최종 일반화 주장은 장면 확장과 별도 test 평가 전까지 하지 않는다.
낮은 속도의 탐색 약화, 거절 후 제동에 의한 정지 편중, 주변 로그 차량의 비반응성을
별도 실패 원인으로 추적하며 결과를 좋게 보이게 하려고 검사 한계를 변경하지 않는다.

## 최근 자기 경험의 배치 반영 비교(2026-11-20)

안정 정책은 아직 확보되지 않았다. 기본 `Replay.sample`의 균등 추출은 유지하고,
진단 실행기에만 `--replay-sampling half_recent256`을 추가한다. 이는 비교 실험이며
기본 설정 채택이나 사용자가 확정한 최종 학습 규약 변경을 뜻하지 않는다.

- 전체 유효 replay에서 복원 추출 128개, 가장 최근 256개에서 복원 추출 128개를 뽑는다.
  총 batch256이며 최근 구간에 warmup이 남아 있으면 그 행도 포함한다.
- 실제 새 전이 하나당 업데이트 한 번, 기존 SAC 손실식·학습률·tau·Gaussian 행동과
  actor/target 표본·보상·검사·PD·BC0/teacher0는 유지한다.
- 경험의 추출 확률이 달라지므로 **균등 replay 목적함수와 같다고 주장하지 않는다.**
  importance weight를 추가하지 않으며 critic뿐 아니라 actor의 상태 분포도 달라진다.
- v1.2 10절은 batch256/replay50k/전이당 update1을 명시하고 균등 추출을 명시하지 않는다.
  그래도 기존 구현과 비교 조건이 달라지므로 별도 실험 개정으로 기록한다.
- `scripts/sac_diagnostic_sampling.py`가 원본 배열을 그대로 모으고 각 배치 index hash를
  `updates.jsonl`에 남긴다. coverage 감사는 모든 index hash와 최종 RNG를 재현한다.
  방식·설정·구현 hash가 다르면 continuation을 거절하며 ring wrap은 지원하지 않는다.
- 첫 후보는 seed0, 같은 자기 warmup5,000개와 fresh learner, 신규1,000전이다.
  기존 f64 균등1,000과 개발4주행·정지4주행·종료 Q·경험 노출·실제 미래 보상을 비교한다.
  정지 경험 편중 증가, 기존 성공 상실, 과거 경험 망각을 함께 확인한다.

최근 실패 경험의 노출 지연은 후보 선택의 근거이며 실패의 유일 원인 증명은 아니다.
단일 seed·작은 반복 개발 평가로 수렴·일반화를 주장하지 않는다. 구체적 실행 결과는
[수집 실패 분석 기록](../records/sac-failure-collection-20261119.md)에 누적한다.

## 자기 성공·실패 종료 경험 유지 비교(2026-11-20)

성공 종료8개를 유지한 fresh1,000 후보도 개발4장면×KEEP/FOLLOW에서0/8성공으로
미채택이다. 성공 종료 Q 오차는 줄었으나 실패 종료 오차와 실제 충돌이 남았다.
후속 비교는 `scripts/diagnose_sac_terminal_retention.py`의 별도 옵션
`own_goal8_failure8`로 실행하며 production replay와 기존 실험 코드는 유지한다.

- batch256은 전체 균등128 + 최근256행에서112 + 자기 성공 종료8 + 자기 실패 종료8이다.
  모두 복원 추출이며 빈 종료 pool만 전체 균등 추출로 대체한다.
- 종료 분류에는 bootstrap=false와 기존 실제 training reward 구간을 함께 사용한다.
  목표는[0.05,0.12], 큰 실패는[-20.05,-19.98]이다. 저장 감사는 이 분류를 실제
  ROUTE_GOAL 및 COLLISION/OFFROAD/CORRIDOR_VIOLATION/MAP_COVERAGE_EXIT 기록과 대조한다.
  새로운 보상·가상 전이·가짜 종료·teacher 경험을 만들지 않는다.
- 비교 요인은 성공 종료8을 유지하면서 최근 추출8개를 실패 종료8개로 바꾸는 것이다.
  상태/행동의 추출 분포와 actor가 학습하는 상태 분포가 달라진다. 손실 가중치나
  importance weight는 추가하지 않으며 균등 replay와 동일 목적함수라고 주장하지 않는다.
- 전이당 전체 SAC 업데이트1회, Gaussian, 보상, 안전검사, PD, BC/teacher0을 유지한다.
  초기 자기 warmup5,000행만 재사용하는 fresh 진단이며 continuation은 지원하지 않는다.
  이전 실패 모델의 학습 가중치를 초기값으로 사용하지 않는다.
- 첫 검증은 seed0·fresh1,000이다. 직전 성공 종료8 후보의 동일 예산 결과를 고정한
  대조 자료로 사용한다. 같은 버전·설정·초기 frame/map을 검증하되 독립 실행의 수치 차이를
  인정하며 단일 seed에서 배치 추출만이 성능 차이의 유일 원인이라고 단정하지 않는다.
- 성공·실패 Q, 실제 경험/노출, 동일8주행의 다중 지표를 검토한다. 이는 별도 비교안의
  구현이며 기본 채택이나 안정 정책 확보를 의미하지 않는다. 실제 결과는
  [종료 경험 유지 기록](../records/sac-own-goal-retention-20261120.md)에 남긴다.

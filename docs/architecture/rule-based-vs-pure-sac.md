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

### 같은 종료 경험 추출 방식의 이어학습 준비

`scripts/continue_sac_terminal_retention.py`는 완료된 목표8+실패8 실행을 복원하는
별도 진입점이다. `--continue-from`과 누적 목표 `--updates`를 명시해야 한다.
현재 모델·optimizer·target·learner RNG·실제 replay/RNG를 복원하고 과거 tick/update
기록을 그대로 보존한다. 장면·정책 수집 RNG와 환경은 명시적으로 재시작하므로
시뮬레이터의 정확한 중간 시점 재개와는 다르다. 기존 예산 경계에 가짜 종료를 넣지 않는다.

총10,000 update 상한과 warmup5,000을 포함한 replay의 wrap 금지를 유지한다.
모델·수집·추출 방식/구현 해시 변경이나 손상된 이력은 거절한다. 별도
`audit_sac_terminal_continuation.py`가 실제 종료 분류·모든 배치 index/RNG·누적 update와
복원 이력을 검사한다. `diagnose_sac_terminal_soft_returns.py`는 같은 이어학습 감사와
기존 현재행동/실제 미래보상 검사를 연결한다. 기존 진행 중 소스와 기본 설정은 유지한다.

관련69pytest, 실제 fresh1,000의 새 감사 및 CPU 전체 상태/replay 복원을 검증했다.
2026-11-20 19:36에는 실제 GPU 복원 후 자기 전이1개와 전체 SAC update1회,
누적 actor/Q1,001·BC0·replay6,001행·원본 이력/배치/RNG 감사도 통과했다.
이는 이어학습 실행 검증이며 주행 성능 검증과 구분한다. 같은 날 완료된1,000회 모델을
총3,000회까지 이어가는 제한 실험을 시작했다. 검사에 사용한1,001회 모델은 가져오지 않는다.
동일 개발8주행·종료 Q·경험 노출·현재행동 미래 보상·GIF 검토를 연결했으며,
후속 예산/기본 채택은 자동 결정하지 않는다. 실제 결과는 위 종료 경험 유지 기록에 남긴다.

### 이어학습 후 현재 행동과 저장 경험 대조

별도 읽기 전용 진입점 `scripts/audit_sac_terminal_coverage.py`는 실제 종료 유지
배치/RNG를 검산하고, 같은 과거 관측에서 저장 행동과 현재 Gaussian의 거리를 계산한다.
95% 타원체 포함 비율은 조건부 거리 진단이며 문자 그대로의 support, 신뢰구간,
새 정책 행동의 충분한 경험 확보나 주행 안전성을 의미하지 않는다.
기존1,000회와 이어학습3,000회는 공통6,000행과 신규2,000행을 구분해 비교한다.

`scripts/diagnose_sac_terminal_branch_q.py`는 Q를 보기 전에 선택한 각 명령의 첫 실제
실패 분기에서 저장 행동·현재 평균·원래 Gaussian4표본을 실제로 다시 실행한다.
보상/종료/다음 관측이 같은지 확인하고 실제 종료한 행동에만 Q−보상 오차를 계산한다.
이 진단의 평균 행동을 학습 replay에 넣거나 확률 수집 정책을 변경하지 않는다.
관련28pytest와 기존 실제1,000회의 배치/RNG 재구성 검증을 마쳤다. 새 GPU 거리/분기
실행은 현재3,000 학습과 주행/미래보상 평가의 성공 종료 뒤 수행하도록 연결했다.

### 실제 실패 직전 상태의 현재 정책 재수집 비교

`scripts/diagnose_sac_terminal_refresh.py`는 별도 fresh 실험 진입점이다.
기존 complete-path 수집과 `own_goal8_failure8` 배치를 보존하면서 실제 main 또는
failure_branch에서 충돌/이탈한 직전 상태를 최근8개까지 보존한다. 기존 수집 전이
4개마다 최대1개를 FIFO 퇴출·순환 선택해 최신 actor의 원래 Gaussian으로 다시 실행한다.
재수집이 만든 실패는 bank에 다시 넣지 않는다. 따라서 새 전이 중 재수집 비중은
최대20%이며 초반 bank가 비었거나 예산이 끝나면 더 작다.

이 옵션은 행동 선택·평균 출력·잡음·보상·안전검사·PD를 바꾸지 않는다. 실제 결과가
과거와 같다고 가정하지 않고 원래 collector가 계산한 보상·종료·next_obs를 저장한다.
비종료 대안의 bootstrap은 유지한다. teacher/BC0, 자기 전이당 전체SAC update1,
batch256·손실·lr·tau도 유지한다. 규약5절의 확률 행동과10절의 전이당 update 조건을
유지하는 상태 수집 분포 비교이며, 균등 상태 방문이나 기본 설정 채택을 뜻하지 않는다.

`scripts/sac_terminal_refresh.py`가 bank와 비중을 관리한다. 원래 경로의 전이 ID를
재수집 ID와 분리해 실제 successor 연결을 유지한다. 별도
`scripts/audit_sac_terminal_refresh.py` 및 `scripts/audit_sac_refresh_replay.py`는
원본warmup17배열·실제 종료·main/branch/refresh 이력·FIFO/순환/주기·동일 source
관측·배치hash/RNG·actor/Q/BC 카운터를 검산한다.
`scripts/verify_sac_terminal_refresh_execution.py`는 명령/source종류별 첫 재수집을
선택해 실제 원본 경로와 저장 행동·재수집 행동을 Waymax에서 재실행한다.

관련33pytest를 통과했고 seed0·동일 자기warmup5,000·동일 모델 설정의 fresh100회
후보/기존 수집 비교를 시작했다. 학습·저장·실제재현·같은 두 개발장면×KEEP/FOLLOW
평가를 순서대로 연결했다. 실제 실행 결과와 성능은 별도 기록하며 테스트 통과를
주행 개선으로 해석하지 않는다. continuation과 다음 예산은 아직 지원/선택하지 않는다.

### 비종료 거절 상태를 포함하는 재수집 옵션

`scripts/diagnose_sac_rejection_refresh.py`는 별도 실험 진입점이다. 앞선 terminal
refresh의 전체 개발 평가에서는 두 모델 모두 2/16 목표 도달·14/16 충돌이었고,
후보의 KEEP/FOLLOW 경로 분석에서는 종료 오차가 줄어도 비종료 구간의 오차가
남았다. 이 관찰을 근거로 충돌/이탈 직전 상태에 더해 **실제로 경로가 거절된
비종료 상태**를 재수집 대상으로 보존하는 옵션을 준비했다.

`scripts/sac_rejection_refresh.py`는 기존 bank의 실제 전이·관측·bootstrap 검사,
FIFO8·순환 선택·기존 전이4개당 최대1회·재귀 재수집 금지를 재사용한다. 바뀌는 것은
source 자격 조건이다. 실제 COLLISION/OFFROAD 직전 상태는 유지하고,
`bootstrap=true && check_valid=false`인 main/failure_branch 상태를 추가한다.
수락된 비종료 상태와 목표/기타 종료 상태는 새 자격 조건으로 넣지 않는다.
source가 늘어나 FIFO 구성, 첫 재수집 시점과 종료 상태의 노출량은 달라질 수 있다.
최대20% 비중이 같다고 방문/추출 분포까지 같다고 주장하지 않는다.

원래 Gaussian으로 실제 재실행하고 그 결과의 reward/next_obs/종료를 저장한다.
원본이 거절됐다는 이유로 새 행동도 거절하거나 충돌로 표시하지 않는다. 보상·검사·PD·
loss/lr/tau·배치256·전이당 전체 SAC update1회·BC/teacher0을 유지한다. 규약5절의
확률 행동과10절의 전이당 업데이트를 바꾸지 않는 상태 수집 분포 비교이며 기본 채택은 아니다.
새 driver는 fresh1..1000만 지원하며 첫 실제 update에 전달되는 learner를 기록하고,
지정한 초기 모델과 바이트 동일성을 검사할 수 있다.

`audit_sac_rejection_replay.py`와 `audit_sac_rejection_refresh.py`는 source 자격,
실제 원본/후속 배열·FIFO/주기·배치/RNG·업데이트 수와 초기 learner를 검산한다.
`verify_sac_rejection_refresh_execution.py`는 관측된 명령/source 종류/비종료·종료
각 조합의 첫 사례를 실제 Waymax에서 재현한다. 없는 조합을 검증한 것으로 세지 않는다.
기존 source와 실행 기록을 보존하기 위해 진입점/감사기는 별도로 두었다.

관련 CPU30pytest·AST·CLI 검사를 통과했다. 2026-11-20 23:29 KST에 실제 GPU100전이와
전체 SAC100업데이트, 저장 감사와 관측된 KEEP main/branch 비종료 source의 Waymax
재현을 완료했다. 재수집20개는 모두 실제 비종료이며 bootstrap을 유지했다.
FOLLOW와 종료 source의 새 옵션 실행은 이100회에서 관찰되지 않았다.
재수집20개 모두 경로 거절이었고 이동 상태6/정지 상태14로 주행 개선 증거는 아니다.
같은 초기 learner·자기warmup5,000의 fresh1,000회와 전체 개발16회 평가를 후속 비교로
선택했다. 완료된 terminal-only refresh1,000 모델/평가를 대조군으로 재사용하며
후속1,000회는 전체 개발16회에서 목표0·충돌16으로 대조군2목표·14충돌보다
악화돼 채택하지 않았다. 저장/실행 검증과 성능을 구분한다.

새 이력의 읽기 전용 정책 진단은 `sac_rejection_diagnostic_audit.py`를 거친다.
새 scope는 rejection 전용 lineage 감사, 과거 scope는 기존 감사기를 사용하며
서로 다른 수집 계약의 혼합/위장과 미완료 검증을 거절한다. 실행 중인 비교가 참조하는
기존 파일을 동결한 채 `diagnose_sac_rejection_branch_q.py`,
`diagnose_sac_rejection_soft_returns.py`, `audit_sac_rejection_coverage.py`를
별도 진입점으로 연결했다. 실제 행동·보상·Q·할인합 계산은 기존 진단과 같다.
이 진단은 학습 데이터/가중치를 수정하지 않으며 조건부 행동 중첩과 실제 실행 결과를
구분한다. 전용 감사기를 CPU에서 실제 pilot100/이전 두1,000 이력에 실행했고
관련30pytest를 통과했다. 새 모델의 GPU coverage·고정 실패 상태 Q·실제 종료까지의
return 검증도 완료했으나 두 train 상태×4seed 모두 충돌해 미채택이다. 최신 범위는
[실험 기록](../records/sac-own-goal-retention-20261120.md)에 남긴다.

### 종료 상태와 비종료 거절 상태의 분리 보존 비교

비종료 수집1,000회에서는 종료 source12개가 모두 비종료 source 삽입으로 퇴출됐고
8개는 한 번도 재방문되지 않았다. 같은 train 충돌 직전 상태에서 새 정책의 평균 행동을
실제로 실행했을 때 Q 과대평가도 증가했다. 이를 유일 원인으로 확정하지 않고,
상태 종류 간 퇴출 경쟁을 줄이는 별도 비교 옵션을 구현했다.

`scripts/sac_stratified_refresh.py`는 종류별 FIFO4, 총8개를 보존한다. 두 종류가 있으면
종료/비종료를 번갈아 선택하고, 한 종류만 있으면 그 종류를 선택한다. 각 종류 안에서는
기존 순환 선택을 사용한다. 기존 실제 전이4개당 최대1회, 재귀 수집 금지, 실제 snapshot
검증을 유지한다. 가중치는 새 실제 경험마다 전체 SAC로 한 번만 갱신한다.

별도 `diagnose_sac_stratified_refresh.py`와 `audit_sac_stratified_replay.py`,
`audit_sac_stratified_refresh.py`, `verify_sac_stratified_refresh_execution.py`는
원본 source·배치·실제 제어 재현을 검사한다. 기존 실험이 동결한 파일은 수정하지 않았다.
보상·검사·PD·원래 Gaussian·loss/lr/tau·BC/teacher0 조건은 같다. 규약5절/10절을
유지하는 상태 수집 분포 비교이며 기본 채택이 아니다.

관련 CPU36pytest·CLI/AST와 수집/업데이트 드라이버의 식별자 외 동등성을 확인했다.
2026-11-21 00:53 KST에 실제100전이·전체SAC100업데이트·저장 감사·관측된 KEEP
main/branch×종료/비종료4조합의 Waymax 재현을 완료했다. 재수집20개 중 비종료13/
충돌7이며 모든 경로가 거절됐다. FOLLOW 조합은 관찰되지 않아 미검증이다.
같은 초기 모델/자기warmup의 fresh1,000회는 01:19에 학습·저장 감사를 완료했고,
01:24에는 KEEP/FOLLOW×main/branch×종료/비종료8조합의 실제 Waymax 재현도 통과했다.
신규1,000전이는 main384/branch416/refresh200, 전체 actor/Q1,000·BC0이다.
refresh200 중 비종료103/충돌97·수락5이며, 종료 source 보존 증가를 주행 개선으로
해석하지 않는다. 실제 신규 주행은7장면/7쌍이고 정지 출발 신규 경험은 없었다.
전체 개발16회 평가는 01:24에 시작했다. 실행 상태와 재개 경로는 위 실험 기록에
남겼으며 새 성능·기본 채택은 아직 미검증이다.

종류별 수집 결과의 읽기 전용 Q/실제 return/coverage 진단은
`sac_stratified_diagnostic_audit.py`를 사용한다. 새 scope는 종류별 실제 lineage 검사를,
과거 scope는 기존 감사기를 반드시 통과해야 하며 계약 혼합/위장/누락을 거절한다.
`diagnose_sac_stratified_branch_q.py`, `diagnose_sac_stratified_soft_returns.py`,
`audit_sac_stratified_coverage.py`는 기존 진단의 계산을 유지한 별도 진입점이다.
관련 CPU32pytest와 과거 실제 세1,000회 이력의 감사는 통과했다. 새 종류별 모델의
실제 GPU 정책 진단과 주행 성능은 아직 미검증이다.
완료된 새100회 이력의 전용 adapter 감사도 통과했다. fresh1,000회 전체 평가 완료 후
coverage→같은 train 종료 상태의 현재 행동 Q→실제 미래보상→검산을 실행하도록
읽기 전용 후속 작업을 대기시켰다.

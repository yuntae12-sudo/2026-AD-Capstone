# SAC 종료 비용 전파와 정책 갱신 방향 진단

2026-11-19, 기준 HEAD `cee3500`, 브랜치 `feature/yuntae-sac-training`.
15:46 KST 기준 원인 분리·입력 배율 후보 검증을 완료했다.
**후보 미채택 / 안정적인 SAC 주행 정책 미확보 / 평가 확대 미실행**이다.
실행 당시 변경은 기준 HEAD에 미커밋 상태였으며 아래 파일들의 실행 hash를 보존했다.

## 범위와 사전 고정 비교

이전 행동 입력0 초기화1,000회 모델의 고정 train 반례 index1405/1277에서
각 마지막11전이의 실제 저장 보상, 기록 행동 return, 현재 평균·32개 확률 행동의 Q,
32개 다음 행동 표본의 TD target을 대조한다. 기록 return은 당시 행동의 return이며
현재 정책의 unbiased Q 정답으로 사용하지 않는다. 종료 target은 보상과 정확히 같아야 한다.

저장된 learner/replay RNG에서 uniform batch256 한 번의 원래 SAC update를 계산한다.
Critic을 갱신한 뒤 actor에 주는 Q/entropy gradient를 분리하고, **동일한 갱신 후
Critic**에서 actor 전후의 Q를 비교해 critic 변화와 혼동하지 않는다. 이 가상 update의
가중치는 버리며 학습 checkpoint나 replay를 덮어쓰지 않는다.
두 episode를 실제 Waymax로 재현해 종료0/1/5tick 전의6상태에서 저장 행동·현재 평균·
가상 update 후 평균18분기를 실행한다. 저장 관측·mask·보상과 다음 관측 hash를 대조한다.

## 원인에 대응하는 제한 후보

종료 target이 올바른데도 동일한 실패 결과를 내는 새로운 행동의 Q가 높게 남는 경우,
Critic 입력의 행동 좌표 배율만1.0→0.1로 바꿔 과도한 행동별 적합을 줄이는 가설을
검증한다. 이는 확정된 해결책이 아니라 별도 학습 좌표 비교다. 초기화만0으로 만드는
이전 실험과 달리 학습 내내 Critic 입력 복사본에 양의 상수 배율을 적용한다.

`scripts/sac_diagnostic_critic.py`의 선택적 Critic으로 구현한다. 기본 Critic은 그대로다.
배율은 가역이며60차원 자유도·무제한 입력·actor로의 gradient를 보존한다. raw u의
샘플링·entropy·decoder·replay·보상·검사·PD는 바꾸지 않는다. 두 Q와 target 모두 같은
좌표를 사용하며, 별도 factory hash와 gain을 저장하고 복원 때 확인한다.
순수 SAC의 BC/teacher0, 목적함수·학습률·batch256·uniform replay·전이당 update1회도 유지한다.
이 배율이 원래 학습과 같은 결과를 낸다고 주장하지 않는다.

같은 seed0·자체 warmup5,000개·mean_gain0.0012098536225957168·integrated/entropy
평균·critic whitened 입력·행동 입력0 초기화를 유지한 fresh learner로 비교한다.
가중치 사전학습은 하지 않는다.100회 후 고정2장면×KEEP/FOLLOW4주행에서 초기 성공
2회를 잃거나 충돌이 증가하면 확대하지 않는다. 유지한 경우에만 새1,000회 비교를 한다.
그 뒤 고정 train Q 반례와 실제 주행을 함께 본다. 초기2/4 성공을 보존하는 것만으로
학습 개선을 선언하지 않으며, 개선이 입증되지 않으면 전체 validation/다른 seed/별도
평가로 확대하지 않는다. 어떤 실행 완료도 후보 채택을 의미하지 않는다.

## 재현 입구

환경 `its-merge`, 저장소 루트에서:

```bash
python scripts/diagnose_sac_td_direction.py \
  --diagnostic outputs/sac_policy_search_20261119/zero-action-trial/online1000 \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest_reset_v2.json \
  --reference-cases outputs/sac_q_audit_20261118/warmup-terminal-actions.json \
  --output outputs/sac_td_direction_20261119/audit.json
```

산출물은 로컬 `outputs/sac_td_direction_20261119/`에 보존한다. 기존 source model은
SHA256 `11a225ca8d8e09ee9589ba381d2cf4cb41c02fce7a6e665c34caaf27505bfd07`이다.
구체적인 실행 명령/PID·모델·replay·manifest·소스 hash는 각 launch/report에 기록한다.
보상·안전 기준의 변경이나 보수적 Q 손실·teacher 자료 추가는 하지 않는다.

## 완료된 원인 분리

기존1,000회 모델의22전이·32개 표본별 평가와6상태×3행동 실제18분기를 완료했다.
6상태 모두 저장/현재 평균/가상 update 후 평균의 즉시 보상·실행 source·다음 관측
SHA256이 같았다. 마지막 상태에서는 모두 EMERGENCY→COLLISION이었다.

| 항목 | KEEP index1405/직전1404 | FOLLOW index1277/직전1276 |
|---|---:|---:|
| 실제 종료 보상·terminal target | -20.014 | -20.017 |
| 종료 상태 저장 행동 Q | -18.6601 | -19.2661 |
| 종료 상태 현재 평균 Q | -2.7583 | -2.5883 |
| 종료 상태 확률32개 Q 평균 | -2.7759 | -2.6763 |
| 직전 상태의 현재 정책 TD target 평균 | -2.8965 | -2.6648 |
| 직전 상태의 당시 기록 행동 return | -19.8278 | -19.8338 |

종료 target 표본32개는 실제 보상과 정확히 같았다. 종료 masking 오류로 설명할 수
없으며, 저장 행동 적합과 다른 행동 일반화 사이의 차이가 bootstrap에도 나타났다.
직전 상태에서 현재 행동으로 실제 한 step을 실행해도 같은 다음 관측이었다.
기록 행동 return과 현재 정책 value의 차이 전체를 unbiased Q 오차로 일반화하지 않는다.

다음 uniform256 batch에는 terminal3개가 포함됐다. actor의 Q loss gradient norm은
0.155914, entropy는0.050190, cosine0.061819였다. mean head에 전달되는 norm은
Q0.0046475 / entropy0이다. 다만 공유 hidden layer 때문에 entropy가 다른 상태의
평균을 간접 변경할 가능성을 배제하지 않는다. 갱신된 동일 Critic에서 actor를 한 번
움직였을 때 종료 상태 평균 Q는 KEEP -2.92305→-2.92196,
FOLLOW -2.66070→-2.66002로 상승했지만 실제 결과는 그대로 충돌이었다.
이를 과대평가를 이용하는 국소 갱신의 반례로 해석하며 전체 실패 원인이라고 단정하지 않는다.

기존 가중치와 replay hash가 실행 전후 일치했다. 실행 당시 진단 소스는
`outputs/sac_td_direction_20261119/audit-script-executed.py`에 보존했다.
이후 동일 진단기가 배율 후보도 정확히 복원하도록 factory 연결을 추가했으므로
최종 Git 소스 hash와 최초 실행 소스 hash는 구분한다.

### 목표값 분해 검사 실패와 수정

다음 actor와 target Q를 별도로 호출해 원래 fused target과 대조하는 첫 추가 검사는
최대0.00109251 차이로 엄격한 일치 검사에 실패했다. 실패 JSON/log/소스를 보존했다.
차이를 그대로 기록한v2에서는32표본 중 최대1.51697 차이가 관찰되어, 그 분해값을
원래 학습 목표의 정확한 구성 항으로 사용하지 않았다.

동일 모델·입력·key에서 fused 반복은 동일했고, fused 함수가 같은 중간 raw u를 반환하도록
구성해 다시 Q를 계산한 값도 일치했다. 별도 actor 호출과의 raw u 차이는 최대4.27e-10이지만
whitened 좌표에서는0.18124까지 증폭됐다. 행렬 precision을 `highest`로 둔 두 key의
읽기 전용 대조에서는 soft value 차이가 최대1.53362→0.00030613으로 줄었다.
이는 실행 그래프/정밀도와 역변환 민감성의 진단이며 GPU 원인 전체를 확정하거나
학습에서 precision 변경을 채택한 것은 아니다. `target-numerics.json`에 각 실행을 보존했다.

`scripts/sac_target_components.py`에서 원래 학습과 같은 fused actor/target-Q 경로로
항을 함께 반환하도록 진단기를 수정했다. `audit-fused.json`의22전이×32표본에서
원래 target과 최대 차이0, 허용치atol/rtol1e-10 검사 통과. 마지막 두 nonterminal의
next target Q 평균은 -2.896506/-2.659652, entropy correction은 -0.015113/-0.014895였다.
즉 종료 비용 약-20과의 큰 차이를 entropy 항으로 설명할 수 없다.
이 추가 진단은 실제 Waymax 분기를 재실행하지 않았고, 최초6상태18분기 재현과 구분한다.

진단용 가상 actor step 역시 JIT 구성에 따른 작은 수치 차이가 있으므로 최초 실제
분기에서 사용한 가중치/수치는 `audit.json`을 기준으로 한다. `audit-fused.json`의
가상 update는 저장·주행하지 않았다. 두 검증을 혼합해 bit 재현을 주장하지 않는다.

## 입력 배율0.1 비교 완료와 미채택

같은 seed0·warmup5,000개에서 별도로 초기화한100회와1,000회 실행을 완료했다.
1,000회 모두 finite이고 신규 자기 전이1,000개/replay6,000개를 저장했다.
신규 제안 수락488/1,000은 주행 성공의 증거로 쓰지 않는다.
모델 SHA256은 `96518b7ffc33390a37914ba4d97f3f2ef0325078114875bf4805145d3d2f15a1`,
replay는 `d194ae5b50cf5ae3e5090928c53dc2b5bc90ca74f7cf0d7ebb1bd837fc68ee22`,
factory는 `9986af114dd291e6664e007782287c07d0e75bcfee9c2d05526b08b384a8ed12`다.
runtime core SHA256은 이전과 같은 `96fa1d625b0d8525b1de6de05a12cc132721a03ee47eb51dff3d6e25f02e30ef`.
manifest SHA256은 `1f254359864bff2bc610497f5c78430b91433bb88302c1931daec9056c48257f`다.

고정 개발 validation 고유2장면×KEEP/FOLLOW4episode의 초기 frame/map/seed 일치를 확인했다.
100회는 기존 초기화100회와 종료·tick·거절·제동·정지 집계가 같았다.

| 모델 | 목표 도달 / 충돌 | 전체 tick | 거절률 | 제동률 | 정지율 | 평균 return |
|---|---:|---:|---:|---:|---:|---:|
| 미학습 평균0 | 2/4 · 2/4 | 216 | 26.85% | 21.30% | 10.19% | -10.01539 |
| 기존 배율1.0,1,000회 | 0/4 · 4/4 | 150 | 66.67% | 58.67% | 22.67% | -20.28212 |
| 배율0.1,100회 | 2/4 · 2/4 | 218 | 26.61% | 21.10% | 10.09% | -10.01610 |
| 배율0.1,1,000회 | 0/4 · 4/4 | 148 | 67.57% | 59.46% | 22.97% | -20.28179 |

빠른 두 주행은 각각45tick 충돌로 기존 실패와 같았고, 저속은 각각29tick 충돌로
기존30tick보다 일찍 끝났다. 총 거절/제동/정지 수는100/88/34로 같고 분모가 줄었다.
미세한 return 상승을 성능 개선으로 해석하지 않는다. `trial.json`은
`COMPLETE_WITH_REGRESSION / STOPPED_AFTER_1000`으로 끝났다.

고정 train 반례를 다시 실제 실행한 결과 현재 평균 Q는 KEEP -3.65552 / FOLLOW -2.85450,
실제 종료 보상은 -20.014/-20.017로 과대평가16.35848/17.16250이 남았다.
저장 terminal62개 MAE는 기존 배율1.0의1.02532에서0.1의7.64724로 악화했다.
입력 영향만 줄이면 일반화가 해결된다는 가설은 이 예산에서 지지되지 않았다.
초기화·단순 배율 축소를 더 반복하는 것을 다음 작업으로 삼지 않는다.

### 검증과 재현 범위

- 관련 pytest 고유18개 통과: 새 TD 구간2개·입력 gain/복원3개·fused 분해1개,
  기존 확대 검사2개·Q 진단/일반화/단기 변형10개. 전체 회귀는 미실행이다.
- 새 CLI의 원인 분리 GPU 실행, 실제6상태18분기,100/1,000회 GPU 학습,
  각각4주행과 최종 고정 Q2반례의 실제 Waymax 실행을 완료했다.
- terminal 지도 회귀 가중치, BC/teacher 경험, 보상/안전/PD 변경은 사용하지 않았다.
- 오래된 terminal-only fit 도구는 gain 변형을 조용히 잘못 복원하지 않고 명시적으로
  거절한다. 평가/Q 도구는 같은 gain과 factory hash로 복원한다.
- 최종 진단기는 실행 후 예외 발생 시 RUNNING으로 남기지 않도록 FAILED 기록을
  추가했다. 실제 실행 소스와 최종 Git 소스의 차이는 출력 폴더의 소스 사본으로 보존한다.
- 전체8장면 validation, 다른 seed, 독립 test는 주행 회귀로 미실행이다.

각 실행 명령은 로컬 `outputs/sac_td_direction_20261119/*-launch.json`과
`trial.json`의 executions에 있다. `finish_gain_trial.py`는 고정된 조건부 실행이며
소스를 고치거나 새 가설을 선택하는 무인 agent가 아니다.
`review/summary.json`, `review/evaluation.csv`, `review/fast.gif`, `review/slow.gif`,
`td-targets.png`에 비교와 원시 근거 위치를 정리했다. 이 경로는 로컬 자료다.

### 다음에 확인할 사항

현재 원인 분리로 확인한 문제는 **실제로 같은 실패를 낸 현재 정책 행동의 비용이
충분히 학습되지 않는 것**이다. 다음은 배율 변경을 반복하기보다 현재 정책의 실패
행동을 실제 경험으로 더 확보하는 수집 대안을 검토하는 것이다. 예를 들어 train의
같은 실패 직전 상태에서 현재 정책의 여러 행동을 실제로 실행하는 방법은 후보지만,
기존 수집 분포·episode 정의·전이당 update 규칙과 대조해 별도 실험으로 명시해야 한다.
아직 구현·실행·채택하지 않았으며 validation 경험이나 teacher 경로를 넣는 제안이 아니다.
정밀도 고정도 별도 대조 대상이며 이번 입력 배율 학습에 섞어 적용하지 않았다.
실제 주행 개선이 확인된 경우에만 전체 validation→다른 seed→독립 장면으로 확대한다.

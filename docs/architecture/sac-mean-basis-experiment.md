# SAC 평균 경로 좌표 변경 비교 실험

2026-11-18 사용자는 **“별도 비교 실험 허용 — 결과를 보고 채택 결정”**으로 승인했다.
이 문서는 실험 개정이며 주 실험의 기본 설정을 변경하거나 새 모델의 채택을 승인한 기록이 아니다.

기준서 v1.2 §5의 “정책이 직접 정하는 평균 60개 점을 B로 smoothing하지 않는다”를
이 비교 실험에서만 대체한다. 기존 `pure_sac_v1.json`과 Actor의 기본 설정은 유지한다.

## 평균 좌표와 유지하는 계약

기존 mean head의60개 선형 출력을 `z[30,2]`라고 부르고 다음을 비교한다.

```
direct:      mu = gain * z
integrated:  mu[:, d] = B @ (gain * z[:, d])
u[:, d] = mu[:, d] + g(state) * B @ (sigma[d] * epsilon[:, d])
```

B는 기존 sampler의 양의 대각을 가진30×30 full-rank 행렬이다. 따라서60개 자유도와
unbounded 출력을 유지하며 저차원 후보 경로 정책으로 바꾸지 않는다. 하지만 Adam이
갱신하는 좌표가 달라지므로 **학습 과정이 같은 알고리즘이라고 주장하지 않는다.**

변환은 Gaussian mean 생성 안에서 일어난다. 이미 생성한 raw u에 clip, tanh, smoothing을
적용하지 않는다. 실제 raw u를 그대로 decode·검사·replay에 사용한다. 외부 출력은 여전히
미래3초의 XY30점이며 PD에 z나 jerk 명령을 전달하지 않는다. B가 full rank라는 사실이
물리 한계 준수를 보장하지는 않으므로 기존 검사를 그대로 수행한다.

Gaussian covariance, sigma 범위, 저속 g, log_pi와 ell 식은 유지한다. mean을 생성하는
결정적 좌표를 바꾼 것이므로 log_pi에 logdet(B)를 추가로 더하지 않는다.
network parameter와 mean은float32, sampler·밀도 계산은float64다.

비교 옵션은 다음과 같다.

- `Actor.mean_basis`: `direct` 기본값 또는 `integrated`.
- `Actor.mean_gain`: 양의 유한 선형 gain, 기본1.0. 출력의 hard bound가 아니다.
- `Critic.zero_output`: 최종 Q head kernel을0으로 초기화하는 선택 사항, 기본false.
- `Actor.head_layer_norm`: mean/rho 앞의 두256차원 hidden layer에 LayerNorm을 넣는
  별도 선택 사항, 기본false. 기준서의 원래 ReLU-only head와 구분해 기록한다.
- `Critic.action_coordinates`: `raw` 기본값 또는 `whitened`.
  기준서 §5에 이미 정의된 `c=0.02*sqrt(2*pi*e)`,
  `w=(g*B)^(-1)u/c` 좌표를 Q 입력에 사용하는 비교다.
  §5의 `Q_w(s,w)=Q_u(s,u(s,w))`에 해당하며 새로운 entropy 목적함수를 추가하지 않는다.
  학습 가능한 함수의 좌표가 달라지므로 초기화 이후 학습 결과가 같다고 주장하지 않는다.
- `Actor.mean_coordinate_scale`: `raw` 기본값 또는 `entropy`.
  `integrated` basis의 별도 비교 안에서만 허용한다. 후자는
  `mu = g(state) * c * B @ (gain * h(state))`로 평균을 생성한다.
  즉 승인된 `mu=Bz`에서 `z=g*c*gain*h`로 좌표를 정하는 추가 비교다.
  g의 최솟값0.002와 c는 양수이므로 매 상태에서60개 자유도·무제한 출력을 유지한다.
  기존 주 실험의 직접 평균 출력 규칙을 만족한다고 주장하거나 자동 채택하지 않는다.

Q 초기화나 actor LayerNorm도 표본과 실행 규칙을 변경하지 않는다. BC, teacher replay,
보상 변경, 미래 로그 기반 검사, 물리 한계 완화는 추가하지 않는다.

`whitened` 입력은 raw float64 u에서 B의 삼각 연립방정식을 float64로 풀고,
g와 고정 c로 나눈 뒤 critic 입력 사본만 float32로 바꾼다. 학습된 sigma로 나누지 않고,
action을 clip하거나 gradient를 끊지 않는다. replay/decoder/검사/PD에는 원래 raw u를
사용하며 actor·target·temperature의 기존 ell 식도 유지한다. Q1/Q2와 target들은
동일 좌표 설정을 공유한다. 복원 시에도 저장된 설정으로 모듈을 구성해야 한다.

2026-11-18 선행 점검에서 온라인100회 후보의512개 자체 warmup 관측을 조사했다.
raw u RMS는0.000374, 위 w 좌표 RMS는0.242였다. 해당 Q의 raw action을0으로 바꿀 때
Q 변화 RMS는0.000413이었고, actor 평균의 상태 간 표준편차 RMS는2.28e-7이었다.
이는 짧은 학습에서 행동 구분과 상태 의존성이 약했던 관측이며, 입력 좌표 변경이
이를 해결한다는 증명은 아니다. 같은 replay·seed·학습 예산의 대조와 실제 주행으로 검증한다.

Q만 w 입력으로 바꾼100회 후보에서도 실제 주행 회귀가 발생했다.512개 자체 warmup
관측의 한 고정 난수 sample에서 mean head에 전달되는 Q gradient를 분리했다.
정지337개/이동154개의 평균 norm은0.07696/0.01048이었다(중간 저속21개는 별도).
이는 parameter 전체 gradient의 합이나 원인의 확정이 아니라 출력 좌표별 국소 측정이다.
Q 입력에는1/g가 있는데 평균 출력에는 g가 없는 비대칭을 다음 비교에서 점검한다.
actor도 w 평균을 정하면 이 좌표 변환의 g 요인이 상쇄된다. 실제 Gaussian covariance,
rho/sigma, ell, raw 저장, 실행 검사는 그대로이며 float32 평균의 반올림 오차는 남는다.

## 비교와 결과 보존

`diagnose_sac_short_updates.py`는 동일 seed의 최초5,000개 자체 수집 경험만 사용하고,
각 후보에서 같은 replay index·learner RNG로100회 업데이트한다.
0/1/10/100회에 train reset8개와 validation reset16개에서 mean과 고정 확률 표본을 검사한다.
이는 고정 데이터 진단이며 새 온라인 경험이 추가되지 않는다. 성공적인 주행 학습이나
SAC 수렴을 인증하지 않는다. validation을 후보 선택에 쓰므로 독립 test라고 부르지 않는다.

보고서에는 gain·basis·Q 초기화·actor LayerNorm·source/runtime/script/weight 해시를 기록한다.
저장 가중치는 `qualified=false`인 진단 산출물이며 production checkpoint manifest를 만들지
않는다. `evaluate_sac_short_updates.py`는 기록된 동일 구조로 가중치를 복원해 별도 실제
Waymax rollout으로 검증한다. 이 결과도 진단 범위로 표시한다.

원래 런타임의 실패 결과와 새 실험을 섞어서 재라벨링하지 않는다. 채택 시에는 별도
production 설정·초기화·checkpoint provenance 개정이 필요하다.

기본 fresh 모드의 `diagnose_sac_online_updates.py`는 고정 replay 비교에서 선택한 **구조 설정만** 읽고
learner를 새로 초기화한다. 같은 seed의 자체 warmup5,000개에서 출발해 새 train 전이마다
업데이트1회를 수행한다. 고정 replay로 학습한 가중치나 validation 경험을 넣지 않는다.
장면/정책 RNG를 재시작하는 별도 진단이며 기존 실행의 정확한 재개가 아니다.
진단 예산에서 끊긴 episode는 `DIAGNOSTIC_BUDGET`으로 기록하고 성공으로 세지 않는다.

2026-11-19 추가된 opt-in `--continue-from`은 완료된 동일 설정 진단의 learner와
optimizer/target/learner RNG, 실제 replay와 그 RNG를 그대로 이어받는다. 동일 모델·수집
설정·runtime·원본 warmup·완전한 trace/hash를 검사하고 BC/teacher0을 유지한다.
`--updates`는 누적 목표이며 상한5,000, fresh 모드는 기존 상한1,000이다. 이전 예산
종료에 가짜 terminal을 넣지 않는다. 환경과 장면/정책 RNG는 명시적 `--collection-seed`
(기본 seed+기존 update 수)로 새로 시작하므로 production의 exact rollout resume와
다르다. source별 hash·누적 이력·승계/신규 update 수를 분리한다. 이는 수집량의 영향을
분리하는 제한 진단 기능이며 회귀한 후보의 자동 채택이나 무제한 예산 확대가 아니다.
실제1,000→1,001 업데이트·직렬화 동일성·replay RNG 검증과 후속2,000회 비교는
[실패 수집 기록](../records/sac-failure-collection-20261119.md)에 보존한다.

2026-11-18의 비교 결과와 실제 주행 회귀는
[초기 업데이트 진단 기록](../records/sac-short-update-20261118.md)에 보존한다.

## 2026-11-19 별도 Q 내부 좌표 비교

`physical_whitened`는 critic의 내부 특징만 `v=B^-1*u/c`로 만드는 opt-in 진단이다.
실제 actor Gaussian, entropy 좌표 w/ell, decoder, checker, replay의 raw u는 바꾸지 않는다.
입력 사본에 같은 현재 관측의 g를 float64로 곱한 뒤 기존 whitened critic의 float64 solve를
통과시키므로 내부에서 g가 상쇄된다. g≥.002이고 B가 full rank이므로 60자유도를 유지한다.
같은 고정 c와 MLP/초기화를 쓰며 sigma로 정규화하거나 출력을 clip하지 않는다.

기존 w 특징은 정지 부근 raw 잡음도 1/g로 확대한다. 실제 같은 결과인 0.12mm 미만 경로
변화에 큰 Q 차이가 남았고, 같은 모델의 추가 학습도 4/4충돌이어서 검토하는 후보이다.
정지 행동의 Q 민감도를 줄이는지 보는 가설이며 원인 확정·성능 개선·기본 채택이 아니다.
평균의 기존 integrated/entropy 설정은 그대로이며 승인된 B 비교 범위로 계속 표시한다.
critic의 함수 표현과 최적화가 달라지므로 동일한 학습 과정이라고 주장하지 않는다.

`--critic-action-coordinates physical_whitened`로 선택하고 다른 input gain과 동시에
사용하지 못하게 한다. `sac_physical_critic.py`와 factory 두 소스의 합성 hash로 복원
설정을 검증한다. 기존 raw/whitened와 과거 gain0.1의 실행 의미는 보존하며, 알려진
과거 factory hash만 호환한다. 알 수 없는 hash나 과거 hash를 새 physical 모델에
붙이는 복원은 거절한다. 같은 자체 warmup·seed·mean·수집·학습 예산 대조와 실제
주행 검증이 필요하다.

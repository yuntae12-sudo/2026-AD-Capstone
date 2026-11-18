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

Q 초기화나 actor LayerNorm도 표본과 실행 규칙을 변경하지 않는다. BC, teacher replay,
보상 변경, 미래 로그 기반 검사, 물리 한계 완화는 추가하지 않는다.

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

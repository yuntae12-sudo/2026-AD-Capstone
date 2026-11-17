# 순수 SAC 수치 발산 진단과 재시작

초기 실험 `outputs/pure_sac_20261117/online-seed0`은 유효 전이5,810개,
성공한 SAC 업데이트809회 이후 비유한 업데이트 검사에서 중단됐다.
기존 report.json의793회는 마지막 episode 경계 기록으로 로그와 시점이 달랐다.
이제 오류 발생 시에도 실제 learner 업데이트 수와 미완료 업데이트 여부를 기록한다.

원래 고정 소스와 5,296개 복구 지점의 복사본으로 재실행해 5,810개/810번째 업데이트에서
같은 중단을 재현했다. 이때 Q1/Q2 gradient norm은 `inf`이고 loss는 약2.5e20이었다.
parameter 자체는 여전히 유한했다. float32 제곱합의 overflow가 직접적인 중단 원인이며,
norm 계산만 바꿔도 이미 비정상적으로 커진 Q를 정상화할 수 있다는 뜻은 아니다.
GPU 재실행의 수치는 원본과 bit 단위로 같지 않았으므로 정확한 수치 재현으로 주장하지 않는다.
증거는 `outputs/sac_stability_20261117/original-reproduction/failure_metrics.json`에 있다.

## 관측된 발산

296회 업데이트 복구 지점을 GPU로 조사했다. 관측은[-5,5] 안이고 sampled 보상 크기는
약20 이내였지만, actor raw_u는 최대약102만, actor 경로의 Q는 최대약187만,
bootstrap target은 최대약73만이었다. 같은 관측에서 actor 경로를10배/100배 확대하자
Q도 최대약1,612만/1억5,873만으로 커졌다. 경험 범위 밖의 큰 행동을 Q가 높게 평가하고
그 값이 다시 target으로 들어가는 발산 경로가 관찰됐다.

## 명시적 안정화 개정

사용자의 원인 수정·재개 요청에 따라 다음을 적용한다. **원래 Q head에 없던 LayerNorm을
추가하는 구조 변경**이며 동일한 함수의 단순 정밀도 수정이라고 주장하지 않는다.

- Q1/Q2와 target Q의 head: `Dense256 → LayerNorm → ReLU`를2회 적용.
- gradient norm의 제곱합을 float64로 누적하고 기존 전역 norm10으로 제한.
  parameter/optimizer moment는 float32, lr3e-4/gamma.99/tau.005는 유지.
- metric과 함께 새 parameter·optimizer·target 상태의 유한성을 검사하고 실패 시 이전 상태 유지.
- actor raw_u 크기, actor 경로의 Q 크기, target 크기를 매 업데이트 기록.
- actor/관측 encoder/Gaussian/경로 디코딩/보상/검사/PD는 불변.
  경로 clamp/tanh/smoothing, teacher/BC는 추가하지 않는다.

Q 외삽과 LayerNorm의 참고는 [Kim et al., 2025, §2–3](https://arxiv.org/html/2507.08761v1)이다.
이 논문의 전체 알고리즘을 도입하지 않았으며 여기서는 critic 정규화만 시험했다.

## 같은 데이터의 대조 실험

자체 수집한 최초5,000개 warmup 전이, 동일 seed·batch index를 사용했다.
양쪽 모두 float64 gradient norm을 사용해 norm overflow 수정과 구조 효과를 분리했다.
환경을 재실행하지 않은 진단이며 본 온라인 업데이트 수에 포함하지 않는다.

| Q 구조 | 결과 | 마지막 actor 경로 Q 최대 | 마지막 target 최대 |
|---|---|---:|---:|
| 기존 ReLU MLP | 413회에 사전 지정 크기1e9 초과, 진단 중단 | 약1.001e9 | 약3.758e8 |
| LayerNorm 추가 | 2,000회 모두 유한 | 약8.52 | 약20.01 |

정규화 모델도 raw_u 최대약24.57이었다. **유한함은 경로 검사 통과나 주행 성공을 뜻하지 않는다.**
고정 replay의 안정성을 장시간 온라인 수렴·성능 향상으로 해석하지 않는다.
증거는 `outputs/sac_stability_20261117/{checkpoint_inspection.json,critic_ablation/report.json}`에 있다.

```bash
python scripts/diagnose_sac_stability.py \
  --checkpoint <old-run/recovery/step-00005296> \
  --output <new-diagnostic-directory> --updates 2000
```

## 새 실험과 자체 warmup 재사용

발산한 모델을 재사용하지 않는다. `step-00004292`는 BC0·SAC0인 episode 경계이며,
같은 seed0 초기 actor가 자체 생성한 경험4,292개만 포함한다.
`fork_sac_warmup.py`는 파일 해시, source 업데이트0, teacher replay 없음,
새 actor와 원래 초기 actor parameter의 정확한 일치, trace 확정 byte와 종료 경계를 확인한다.
critic/optimizer/target/learner RNG는 새로 초기화하고 policy RNG·장면 순서는 warmup 경계에서 잇는다.

이는 **다른 critic으로 시작하는 새 실험 fork**이며 이전 learner의 정확한 재개가 아니다.
`warmup_reuse`에 상속 전이 수·source 버전/해시·새 learner 여부를 기록한다.
`initial_replay_size=0`은 lineage의 최초 시작이고 새 실험은 명시된 상속 warmup4,292개로 시작한다.
seed1/2는 각각 빈 replay에서 자체 수집한다. 원본 실패 로그·checkpoint는 보존한다.

`revalidate_sac_evidence.py`는 원본 code hash와 현재 소스를 대조한다. 변경은 learner,
network의 Critic, runtime의 versions 정의에만 허용한다. sampler/actor/관측/보상/collector/
물리 검사/PD/manifest/contract/의존 라이브러리와 합성 runner·baseline 실행 함수가 같아야 한다.
그 경우에만 합성 검사와 Rule-based 평가를 재사용하고 `evidence_reuse`에 원본 버전·해시와
대조한 compatibility revision을 기록한다. 새 측정이라고 표시하지 않는다.

검증: learner/protocol/실험과 evidence/gate/recovery/collector 테스트37개를 통과했다.
큰 유한 gradient의 norm10 제한, Q 외삽 대조, 비유한 target 상태 거절,
sampler/actor/검사 변경 시 증거 재사용 거절을 포함한다.
GPU에서도 최대1e38 float32 gradient의 방향과 norm10 제한을 별도로 확인했다.

재시작용 증거는 `outputs/sac_stability_20261117/revalidated-final/`,
새 초기 상태는 `outputs/sac_stability_20261117/warmup-fork-final/`에 보관한다.
온라인 결과는 기존 실패 디렉터리와 분리한 `outputs/pure_sac_stable_20261117/`에 기록한다.
seed0의 남은 warmup 수집과 5,000개 시점 평가가 끝난 뒤 실제 SAC 업데이트가 시작된다.
앞의 진단 2,000회는 이 온라인 실험의 업데이트 수에 포함하지 않는다.

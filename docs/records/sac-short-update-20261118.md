# SAC 초기 업데이트 붕괴: 비교 실험과 실제 주행 재검증

2026-11-18. 초기 경로 검사 안정성은 개선했지만, 실제 주행에서는 회귀가 확인됐다.
따라서 아래 후보는 **채택하지 않는다**. 기존 주 실험 설정을 유지하며, 성공한 학습
모델이나 Rule-based보다 우수한 정책을 얻었다고 해석하지 않는다.

## 실험 범위와 규약

사용자가 “별도 비교 실험 허용 — 결과를 보고 채택 결정”으로 승인한
[평균 좌표 변경 실험](../architecture/sac-mean-basis-experiment.md)이다.
기준서 §5의 평균에 B를 적용하지 않는 규칙은 이 비교 실험에서만 명시적으로 대체한다.
`mu = gain * z`와 `mu = B @ (gain * z)`를 비교하며, B는 full rank이므로
60개 자유도와 무제한 출력을 유지한다. full rank는 경로 안전성의 보장이 아니다.

표본 분포의 공분산·밀도, raw action 저장, 보상, 물리 검사 한계, PD, 학습률,
batch256, 업데이트 순서는 유지했다. BC나 teacher 경험은 사용하지 않았다.
Q 출력의 0 초기화, actor head LayerNorm, mean gain은 별도 실험 인자로 기록한다.
기본 Actor/Critic 설정과 `configs/experiments/pure_sac_v1.json`은 변경하지 않았다.

원래 완료된 `pure_sac_stable_20261117`의 seed별 replay에서 **최초 5,000개의 자체
warmup 경험만** 읽었다. 이후 학습 중 생성된 경험은 고정 replay 비교에서 제외한다.
각 후보는 새로 초기화하며, 동일 seed에서 같은 minibatch 순서와 learner RNG를 사용한다.
이것은 기존 최종 모델의 이어 학습이 아니다.

검사는 각 seed에서 train reset8개와 validation reset16개, 총24개 상태를 사용했다.
각 상태마다 평균 경로와 고정 난수 표본1개를 검사했다. 표본1개로 확률적 통과율 전체를
추정할 수 없으며, 초기 상태 검사 결과를 실제 주행 성공률로 취급하지 않는다.

## seed0: 실패한 후보도 포함한 전체 비교

아래 수치는 **24개 reset 상태 중 평균 경로 통과 개수**다. 모든 후보의 초기 통과는
14/24였다. 이미 실패하던10개를 분모에서 제거하지 않았다.

| 평균 좌표 | gain | Q 0 초기화 | actor LayerNorm | 1회 | 10회 | 100회 |
|---|---:|:---:|:---:|---:|---:|---:|
| 직접 출력, 기존 설정 | 1 | 아니오 | 아니오 | 0 | 0 | 0 |
| 직접 출력 | 1 | 예 | 아니오 | 0 | 0 | 0 |
| 직접 출력 | 0.0001 | 아니오 | 아니오 | 14 | 0 | 0 |
| 직접 출력 | 0.0001 | 예 | 아니오 | 14 | 6 | 0 |
| 직접 출력 | 0.00001 | 예 | 아니오 | 14 | 14 | 0 |
| 직접 출력 | 0.000001 | 예 | 아니오 | 14 | 14 | 6 |
| B 좌표 | 1 | 아니오 | 아니오 | 14 | 14 | 0 |
| B 좌표 | 1 | 예 | 아니오 | 14 | 0 | 0 |
| B 좌표 | 0.1 | 예 | 아니오 | 14 | 14 | 0 |
| B 좌표 | 1 | 예 | 예 | 14 | 0 | 0 |
| B 좌표 | 0.01 | 예 | 예 | 14 | 14 | 8 |
| B 좌표 | 0.001 | 예 | 예 | 14 | 14 | 14 |

고정 표본 결과도 대부분 같은 개수였다. 예외는 B/gain0.1/Q0의10회 지점으로,
평균14개·표본13개가 통과했다. 기존 설정은 첫 업데이트 직후14→0으로 붕괴했고,
100회에는 평균 raw action 최대 절댓값21.02, reset의 min(Q1,Q2)가+4.44~+4.48이었다.
Q0 초기화만으로는 첫 업데이트의 경로 붕괴를 막지 못했다.

한 고정 replay 관측에서 조사한 B/gain1 후보의 두 번째 actor hidden 출력 최대 절댓값은
초기1.08에서100회138.29로 증가했다. LayerNorm 후보에서는 해당 정규화 출력이1.19였지만
경로는 여전히 실패했다. 이는 feature 크기 증가를 관찰한 근거이며, 단일 관측으로
전체 실패의 인과관계를 확정한 것은 아니다.

![초기 상태 경로 검사 결과](../../outputs/sac_short_updates_20261118/reset-acceptance.png)

## 가장 안정적인 reset 후보의 seed 재검증

B/gain0.001/Q0/actor LayerNorm을 동일하게 적용했다.

| seed | 처음 통과 / 전체 | 100회 평균 통과 / 전체 | 100회 고정 표본 통과 / 전체 | validation 평균 통과 |
|---|---:|---:|---:|---:|
| 0 | 14/24 | 14/24 | 14/24 | 8/16 |
| 1 | 16/24 | 16/24 | 16/24 | 8/16 |
| 2 | 15/24 | 15/24 | 15/24 | 8/16 |

처음 통과하던 상태를 모두 유지했다는 뜻이다. 모든 상태에서100% 통과하거나,
새롭게 안전한 경로를 학습했다는 뜻은 아니다. 작은 gain으로 학습 변화를 늦춘 효과와
유용한 정책 학습 효과도 아직 분리되지 않았다.

## 실제 Waymax 주행: reset 결과로 놓친 회귀

seed0의 같은 후보를 업데이트 전후 각각 실행했다. 빠른 장면
`8b4c1e7e09f5f4c3`와 저속 장면 `8a4fba3463c6c62d`에서 KEEP/FOLLOW를 모두 사용했다.
두 장면은 실패 분석용으로 선택했으며 독립 test가 아니다. 아래 미학습 모델은 평균0인
SAC 초기화 대조군이며, **Rule-based 정책이 아니다**.

| 장면 | 정책 | KEEP / FOLLOW 결과 | 각 episode 거절 | 각 episode 제동 | 각 episode 정지 tick |
|---|---|---|---:|---:|---:|
| 빠른 장면 | 미학습 | 둘 다 목표 도달,72tick | 11/72 | 8/72 | 0/72 |
| 빠른 장면 | 고정 replay100회 | 둘 다 충돌,68tick | 33/68 | 30/68 | 8/68 |
| 저속 장면 | 미학습 | 둘 다 충돌,36tick | 18/36 | 15/36 | 11/36 |
| 저속 장면 | 고정 replay100회 | 둘 다 충돌,44tick | 14/44 | 11/44 | 7/44 |

빠른 KEEP의 학습 후보는3.7초부터 `PREDICTED_OFFROAD` 거절이 이어졌다.
4.4초에는 가속도·곡률·jerk·곡률 변화율 위반으로 `KINEMATIC_LIMIT`이 발생했고,
긴급제동 후6.8초에 정지한 ego와 뒤차4877이 충돌했다. FOLLOW도 같은 사건 순서였다.
미학습 대조군은 두 조건 모두7.2초에 목표에 도달했다.

따라서 경로 변화가 작고 reset에서 유효하다는 사실만으로, 반복 재계획과 PD 실행을
거친 후에도 유효하다고 판단할 수 없다. 이 결과로 후보의 채택을 보류한다.
저속 장면에서 거절·정지 tick이 줄었어도 충돌했으므로 성공으로 재분류하지 않는다.

## 새 경험을 넣는 온라인 진단

`diagnose_sac_online_updates.py`는 같은 자체 warmup5,000개와 새 learner로 시작하고,
실제 train 전이를 replay에 넣을 때마다 업데이트1회를 수행한다. 장면/정책 RNG는 새로
시작하므로 원래 실행의 정확한 재개가 아니다. source/current runtime을 모두 기록하고,
진단 가중치에는 `qualified=false`를 유지한다. validation 경험은 replay에 넣지 않는다.

seed0의 실제 신규 전이100개·업데이트100회를 완료했다. 첫 두 train episode는29tick과
54tick에서 충돌했고, 세 번째는17tick에서 진단 예산 종료로 기록했다. 총 경로 통과는
28/100이다. 예산으로 끊긴 episode를 성공으로 집계하지 않는다.

온라인 모델의 동일 장면 재평가8회(미학습4회·학습 후4회)를 완료했다.
미학습 결과는 앞선 고정 replay 비교의 대조군과 일치했다.

| 온라인100회 모델 | KEEP / FOLLOW 결과 | 각 episode 거절 | 각 episode 제동 | 각 episode 정지 tick |
|---|---|---:|---:|---:|
| 빠른 장면 | 둘 다 충돌,70tick | 36/70 | 31/70 | 8/70 |
| 저속 장면 | 둘 다 충돌,43tick | 15/43 | 12/43 | 8/43 |

빠른 장면은3.5초부터 도로 이탈 예상 거절이 다시 발생했고,4.6초부터 운동 한계
위반과 긴급제동이 이어졌다.7.0초에 정지한 ego와 뒤차4877이 충돌했다.
신규 경험을100개 추가하는 짧은 온라인 학습에서도 회귀가 해결되지 않았다.
따라서 고정 replay만 사용한 탓이라고 단정하거나, 이 후보로 장시간 학습을 재개하지 않는다.
반대로100회 진단만으로 장기 온라인 학습의 모든 가능성을 부정하는 것도 아니다.

![같은 장면의 실제 ego 속도](../../outputs/sac_short_updates_20261118/closed-loop-speed.png)

## 검증·재현·산출물

- SAC learning/stability/pure protocol/새 실험 옵션 관련 테스트21개 통과.
- 동일 GPU에서 기본 초기화의513개 learner leaf가 원래 zero-update checkpoint와 일치했다.
  CPU/GPU 간 bitwise 일치를 주장하지 않는다.
- full-rank60차원, 무제한 mean, Gaussian noise/밀도 유지, Q0 이후 실제 학습을 테스트했다.
- 핵심 소스: 직접 출력 비교 `c3a8f1b`, B/LayerNorm 비교 `f4547b1`.
  B 비교 runtime SHA256은 `4ec35cfaccb5a2c36a6ce3675b6bc327526c5244192cb2dd95ce67a9a8fd3a64`.
- [원시 결과 폴더](../../outputs/sac_short_updates_20261118/): 각 실험의 report,
  업데이트 로그, 진단 weights, 전체 주행 frame JSON을 보존한다.
- [수치·실패 시점 요약](../../outputs/sac_short_updates_20261118/summary.json),
  [초기 상태 비교 그림](../../outputs/sac_short_updates_20261118/reset-acceptance.svg),
  [파일 해시 목록](../../outputs/sac_short_updates_20261118/artifact-index.json).
  outputs는 로컬 산출물이며 Git으로 업로드하지 않는다.
- 온라인100개 전이/업데이트 로그의 연속 번호와 유한값, 저장 가중치 해시, 세 CLI의
  도움말 실행, 변경 Python 파일 포맷과 `git diff --check`를 확인했다. 진단 실행 중
  핵심 runtime 변경 여부도 각 실행의 종료 검사에서 확인했다.

재현 예시(프로젝트 루트, `its-merge` 환경):

```bash
python scripts/diagnose_sac_short_updates.py \
  --checkpoint outputs/pure_sac_stable_20261117/online-seed0/checkpoints/step-00015191 \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest_reset_v2.json \
  --variants basis_norm_zero --mean-gain 0.001 --seed 0 \
  --output outputs/reproduce-short-seed0

python scripts/diagnose_sac_online_updates.py \
  --checkpoint outputs/pure_sac_stable_20261117/online-seed0/checkpoints/step-00015191 \
  --configuration outputs/reproduce-short-seed0 --variant basis_norm_zero \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest_reset_v2.json \
  --output outputs/reproduce-online100-seed0

python scripts/evaluate_sac_short_updates.py \
  --diagnostic outputs/reproduce-online100-seed0 --variant online \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest_reset_v2.json \
  --scenes 8b4c1e7e09f5f4c3 8a4fba3463c6c62d \
  --output outputs/reproduce-online100-evaluation
```

## 다음 판단 기준

장시간 학습을 재시작하기 전에, 반복 재계획에서 도로 이탈 예상이 누적되는 원인과
제동 후 현재 속도·가속도를 기준으로 생성하는 경로가 다시 거절되는 과정을 분리해야 한다.
같은 중간 상태에서 미학습/학습 경로와 PD 추종을 비교해야 하며, 검사 한계를 낮추거나
실패 장면을 제외해서 통과시키지 않는다. Q의 행동 민감도와 실제 수익의 관계도 별도로
점검해야 한다. 현재의 짧은 실험만으로 B 좌표, gain, Q 초기화 중 하나를 정답으로
확정하거나, 순수 SAC가 불가능하다고 결론 내릴 수 없다.

validation8개를 반복 진단·선택에 사용했으므로 최종 일반화 성능은 별도 장면에서
평가해야 한다. 우선 짧은 실제 주행에서 회귀를 없애고 나서 장기 학습 여부를 결정한다.

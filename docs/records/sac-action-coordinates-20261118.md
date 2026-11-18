# SAC 행동 좌표 비교와 학습량 진단

2026-11-18. Q 입력의 행동 차이 감도와 초기 경로 검사에는 개선이 있었지만,
100회 및1,000회 학습 후보의 실제 주행 회귀는 해결되지 않았다. 제한 진단과 동일
장면 평가를 완료했으며, 현재 후보를 주 실험에 채택하지 않는다.

## 원인 점검과 변경 범위

[직전 재계획 진단](sac-replanning-20261118.md)에서 제안 경로 자체의 실패를 확인했다.
기존 온라인100회 모델의512개 자체 warmup 관측에서 raw action RMS는0.000374,
actor 평균의 상태 간 표준편차 RMS는2.28e-7이었다. 평균 경로가 상태에 따라 달라지는
정도가 매우 작았다. 이 관측만으로 전체 학습 실패의 원인을 확정하지는 않았다.

기준서 §5의 `c=0.02*sqrt(2*pi*e)`, `w=(g*B)^(-1)u/c`를 Q 입력으로 선택할 수 있게
구현했다. raw u를 float64 삼각 solve로 변환한 뒤 Q 입력 사본만 float32로 바꾼다.
replay/decoder/검사/PD에는 실제 raw u를 그대로 전달한다. ell·보상·학습률·업데이트
순서·sampler·검사 한계를 바꾸지 않았다. BC와 teacher 경험도 사용하지 않았다.

Q만 w를 사용한 후보에서 mean head로 가는 Q gradient를 고정 난수 sample로 측정했다.
정지337개·저속21개·이동154개 관측의 평균 norm은 각각0.07696·0.05086·0.01048이었다.
이는 출력 좌표의 국소 gradient norm이며, parameter gradient의 최종 합이나 원인의
확정은 아니다. Q 입력의1/g가 정지 부근의 mean 업데이트에 미치는 영향을 분리했다.

승인된 B 평균 **비교 실험 범위**에서 actor도
`mu=g*c*B@(gain*h)`로 평균을 표현하는 옵션을 추가했다. 기존 `mu=Bz`의
`z=g*c*gain*h`에 해당하며, g≥0.002이므로 상태마다60개 자유도·무제한 출력이 유지된다.
이 변경은 기본 직접 출력 규약을 만족한다고 주장하지 않고 별도 설정으로 기록한다.
자세한 정의는 [평균·Q 좌표 실험 규약](../architecture/sac-mean-basis-experiment.md)을 따른다.

기존 기본값과 `pure_sac_v1.json`은 유지했다. 과거 실행 소스는 별도 worktree로 보존했다.

## 고정 자체 warmup100회 비교

seed0의 처음5,000개 자체 warmup 경험만 사용했다. 각 후보는 새로 초기화하고 동일
minibatch 순서와 learner RNG를 쓴다. train reset8개+validation reset16개의 평균 경로와
고정 확률 표본1개를0/1/10/100회에 검사했다. 이는 실제 주행 성공률이 아니다.

| 후보 | 평균 구성 | Q 입력 | 100회 평균 통과 / 24 | 고정 표본 통과 / 24 |
|---|---|---|---:|---:|
| 기존 좌표 대조 | B×0.01×h | raw u | 8 | 8 |
| Q 정규화만 | B×0.01×h | w | 18 | 18 |
| actor/Q 좌표 정렬, 이동 배율 유지 | g×c×B×0.12098536×h | w | 14 | 14 |
| actor/Q 좌표 정렬, 기본 배율 | g×c×B×h | w | 12 | 12 |

모두 초기14개가 통과했으며1/10회에도14개였다. Q 정규화만 적용한 후보는 기존14개를
유지하고4개가 추가 통과했다. validation은8/16→12/16이었다.
배율 유지 후보는 c×gain=0.01이어서 g=1인 이동 상태의 직접 출력 배율을 대조군과 맞췄다.
기본 배율 후보는 처음 통과하던2개를 잃어 다음 온라인 진단에는 배율 유지 후보를 사용했다.

같은512개 관측에서 raw action을0으로 바꿨을 때 Q 변화 RMS는 기존 좌표0.000174,
Q 정규화0.046971이었다. 반응 크기가 커졌다는 뜻이며 정확한 Q값을 학습했다는 뜻은 아니다.
두 모델의 평균 경로 상태 간 표준편차 RMS는 각각3.22e-6과3.51e-7로 여전히 작았다.

## 실제 주행100회 모델 비교

빠른 장면 `8b4c1e7e09f5f4c3`와 저속 장면 `8a4fba3463c6c62d`에서 KEEP/FOLLOW를
각각 실행했다. 아래 각 행의 두 행동은 종료 tick과 거절/제동/정지 개수가 같았다.
미학습 평균0 대조군은 빠른 장면에서72tick에 목표 도달, 저속 장면에서36tick에 충돌했다.

| 후보 | 데이터/업데이트 | 장면 | 결과 | 거절 tick | 제동 tick | 정지 tick |
|---|---|---|---|---:|---:|---:|
| Q 정규화 | 고정 replay100회 | 빠름 | 충돌52tick | 30 | 27 | 4 |
| Q 정규화 | 고정 replay100회 | 저속 | 충돌29tick | 21 | 18 | 14 |
| Q 정규화 | 신규 전이+온라인100회 | 빠름 | 충돌50tick | 29 | 26 | 4 |
| Q 정규화 | 신규 전이+온라인100회 | 저속 | 충돌30tick | 21 | 18 | 14 |
| actor/Q 정렬, 배율 유지 | 고정 replay100회 | 빠름 | 충돌58tick | 39 | 30 | 5 |
| actor/Q 정렬, 배율 유지 | 고정 replay100회 | 저속 | 충돌46tick | 11 | 10 | 6 |
| actor/Q 정렬, 배율 유지 | 신규 전이+온라인100회 | 빠름 | 충돌37tick | 29 | 26 | 2 |
| actor/Q 정렬, 배율 유지 | 신규 전이+온라인100회 | 저속 | 충돌26tick | 21 | 18 | 14 |

초기 경로 검사 개선을 실제 주행 개선으로 해석할 수 없다. 고정 replay에서는 저속
거절이 줄어도 충돌했고, 온라인100회 모델은 빠른 장면의 회귀를 해결하지 못했다.
소수의 선택 장면과 seed0 결과를 전체 validation이나 다른 seed로 일반화하지 않는다.

온라인 진단은 동일 자체 warmup5,000개와 새 learner에서 시작한다. 고정 replay100회
가중치를 이어 학습한 것이 아니다. 신규 전이를 넣을 때마다 업데이트1회를 수행하며,
예산에서 끊긴 episode는 `DIAGNOSTIC_BUDGET`으로 기록한다.

## 온라인1,000회 제한 진단

100회 후보들의 Q1 loss는 첫10회 평균 약3.41에서 마지막20회 평균 약2.88~2.93이었다.
Q 오차가 남아 있지만 이것만으로 원인을 단순한 학습 부족이라고 확정하지 않는다.
actor/Q 정렬·이동 배율 유지 설정을 고정하고 seed0 신규 경험1,000개·업데이트1,000회로
제한한 진단을 완료했다. 신규 전이와 업데이트가 각각 정확히1,000개이고 모든 업데이트의
수치 유한성을 확인했다. BC·teacher 경험은 사용하지 않았다. 100회 가중치에서 이어가는
방식이 아니라 동일 초기화·자체 warmup에서 새로 실행했다.

- Q1 loss는 첫10회 평균4.663에서 마지막100회 평균0.237로 낮아졌다. 이동하는 target과
  변화한 replay 분포의 학습 오차이며, Q값의 실제 주행 예측 정확도를 입증하지 않는다.
- train13개 고유 장면에서14episode를 실행했다. 종료는 충돌11·RECORD_END2·예산 중단1,
  ROUTE_GOAL0이었다. 기록 끝까지 멈춰 있었던 것을 성공으로 세지 않았다.
- 경로 통과467/1,000개 중357개는 제안 anchor 속도<0.1m/s였다. 전체의611개가 이
  정지 조건이었다. 이는 **제안 시점 속도** 분석이며 평가의 post-step 정지 비중과 구분한다.
- 동일 자체 warmup 관측512개에서 명시적 g 배율 이전 mean head의 상태 간 표준편차
  RMS / 출력 RMS는100회0.000114에서1,000회0.10334로 증가했다. 관측별 출력 차이는
  늘었지만, 안전한 경로 생성이나 새로운 장면 일반화를 의미하지 않는다.

### 100회 vs1,000회 실제 주행

두 실행은 같은 core/manifest/contract를 사용했고, 아래4쌍의 초기 frame과 지도도
정확히 일치함을 확인했다. 두 모델 모두 결정적 평균 경로를 실행했다.

| 장면·행동 | 100회 결과·종료 tick | 1,000회 결과·종료 tick | 1,000회 거절 / 제동 / 정지 tick |
|---|---|---|---|
| 빠름 KEEP | 충돌37 | 충돌54 | 40 / 31 / 5 |
| 빠름 FOLLOW | 충돌37 | 충돌59 | 44 / 33 / 5 |
| 저속 KEEP | 충돌26 | 충돌22 | 20 / 18 / 14 |
| 저속 FOLLOW | 충돌26 | 충돌19 | 19 / 19 / 15 |

각각 성공0/4·충돌4/4였다. 저속 장면은 더 일찍 충돌했다. 전체 tick 기준 거절률은
79.37%→79.87%, 제동률69.84%→65.58%, 정지 비중25.40%→25.32%다. episode 길이가
달라졌으므로 제동 비율의 감소만으로 개선을 선언하지 않는다. 두 고유 장면을4개 독립
장면으로 세거나, 이 결과를 전체 validation/다른 seed 성능으로 일반화하지 않는다.

실제 주행 개선은 입증되지 않았다. 1,000회는 수렴 여부를 판정할 충분한 근거도 아니지만,
현재 결과만으로 큰 예산이나 다른 seed로 확대하지 않는다. 다음 진단은 동일 상태에서
정책이 높은 Q값을 주는 경로가 실제 검사·주행에서도 유효한지, 거절·제동·정지 경험을
어떤 가치로 학습했는지 분리하는 것이다. 보상·검사 완화나 teacher 경험 주입으로
실패를 감추지 않으며, B 평균 구조의 주 실험 채택은 미결정 상태로 유지한다.

산출물은 [비교 폴더](../../outputs/sac_action_coordinates_20261118/online-budget-comparison/)에
`summary.json`, episode별 CSV, 학습 지표 PNG와 두 KEEP 장면의100회/1,000회 비교 GIF로
저장했다. GIF는 실제 저장 frame만 사용하며, 먼저 종료된 차량은 종료 표시와 함께 고정한다.
이 GIF는 Rule-based와의 최종 비교가 아니라 SAC 후보의 학습 예산 비교다.

## 검증·재현·산출물

- 최신 관련 테스트29개 통과. 저속/이동 좌표 역변환, 기본 parameter layout 유지,
  raw u 비변경, actor gradient 흐름, full-rank60차원, 무제한 평균 출력을 포함한다.
- Q 좌표 비교 소스 `78764f6`/`7f49a81`, actor 좌표 정렬 소스 `ca16780`.
  core hash는 각각 `33fd158a9987a5ea6ea975134c45c61a51108a45121f01300bd7bdf19941da47`와
  `96fa1d625b0d8525b1de6de05a12cc132721a03ee47eb51dff3d6e25f02e30ef`다.
- [원시 결과 폴더](../../outputs/sac_action_coordinates_20261118/)에 모든 설정의 report,
  업데이트/실제 전이 로그, 가중치, 실제 주행 frames, 감사 코드를 보존한다.
  단일 모델 평가에는 `--labels updated`를 명시하고 그 보고서만으로 paired 비교를
  완료했다고 표시하지 않는다. 대조군의 초기 상태·지도와 결과 출처를 별도 대조한다.
- 평가 지표의 분모와 추가 장면 확보 계획은
  [평가 운영 기준](../architecture/sac-evaluation-protocol.md)에 정리했다.
  추가32개 고유 장면은 확보 목표이며 현재 확보한 수가 아니다.

재현 예시(`its-merge` 환경):

```bash
python scripts/diagnose_sac_short_updates.py \
  --checkpoint outputs/pure_sac_stable_20261117/online-seed0/checkpoints/step-00015191 \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest_reset_v2.json \
  --variants basis_norm_zero --mean-gain 0.12098536225957167 \
  --critic-action-coordinates whitened --mean-coordinate-scale entropy \
  --output outputs/reproduce-aligned-seed0

python scripts/diagnose_sac_online_updates.py \
  --checkpoint outputs/pure_sac_stable_20261117/online-seed0/checkpoints/step-00015191 \
  --configuration outputs/reproduce-aligned-seed0 --variant basis_norm_zero \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest_reset_v2.json \
  --updates 1000 --output outputs/reproduce-aligned-online1000
```

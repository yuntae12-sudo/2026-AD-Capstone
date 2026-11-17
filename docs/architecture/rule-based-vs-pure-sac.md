# Rule-based 경로와 순수 SAC 경로 비교

2026-11-17 사용자 결정으로 주 실험을 **Rule-based vs 순수 SAC**로 변경했다.
이는 기준서 v1.2의 BC 사전학습·teacher replay 초기화·BC 전용 진입 gate를 대체하는
명시적 실험 개정이다. 원본 기준서와 기존 BC 결과는 보존한다.
기계 판독 설정은 [pure_sac_v1.json](../../configs/experiments/pure_sac_v1.json)이다.

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

성공률·충돌률·도로 이탈률·경로 거절률·제동 개입률·정지 비중·보상·episode 길이를
함께 비교하고, 개별 평가의 추종 RMS도 확인한다. 결정적 Rule-based 한 결과에 대해
SAC seed별 차이와 그 평균·표본 표준편차를 보고한다. BC-only/BC+SAC는 보조 실험으로만 남긴다.

15,000전이는 초기 실행 가능성 및 성능 점검 예산이며 수렴을 보장하지 않는다.
현재 validation은 8개 고유 장면으로 작고 drive 단위 독립성은 검증되지 않았다.
최종 일반화 주장은 장면 확장과 별도 test 평가 전까지 하지 않는다.
낮은 속도의 탐색 약화, 거절 후 제동에 의한 정지 편중, 주변 로그 차량의 비반응성을
별도 실패 원인으로 추적하며 결과를 좋게 보이게 하려고 검사 한계를 변경하지 않는다.

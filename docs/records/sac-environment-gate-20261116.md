# SAC 연결 전 환경 보완 결과

기준: v1.2/schema3. 기존 PR #5의 제어 환경 보완이며, SAC 성능 결과가 아니다.

## 변경

- 손상된 post-step 상태는 `truncated=true`, `sample_valid=false`. 유한한 빈 final observation을 반환하며 reset observation과 섞지 않는다.
- 실제 corridor 측방 이탈과 route 종방향 지도 coverage 이탈을 구분한다. 두 경우 모두 정책 실패이며 replay 제외 대상이 아니다.
- collision/offroad/corridor/coverage의 동시 사건 flags를 보존하고 기존 종료 우선순위를 유지한다.
- Waymax에 적용하는 최종 float32 입력을 결과와 controller memory에 기록한다.
- 정적 route 계산 캐시, 불변 JAX 상태별 host snapshot, polygon/OBB 검사 배열화와 보수적인 broad phase로 반복 계산을 줄인다. 시간 표본·물리 한계·충돌/도로 margin은 유지한다.

## 실행 증거

로컬 결과는 `outputs/sac_environment_gate/`에 있다. 원시 데이터와 실행 산출물은 Git에 추가하지 않는다.

| 검증 | 결과 |
|---|---|
| 기존 `568d6d9` 검사기 대조 | 실제 장면 3개, planner 2개, 3tick의 18개 입력 일치. 판정·이유·표본 수 일치, 오차 수치는 절대차 1e-8 이내 |
| 경계/오목 polygon/큰 좌표/접촉 | 기존 scalar oracle과 대조하는 회귀 통과 |
| 고정 경로 PD | production Hermite·PD와 설치된 Waymax dynamics로 합성 13개 통과 |
| 실제 도로 고정 3초 경로 | `d675f8a070ebe2e4`, `8b4c1e7e09f5f4c3` 모두 30tick, 실제 도로·충돌 실패 0 |
| 실제 고정 경로 위치 RMS/최대 | 각각 0.0552/0.0950m, 0.0433/0.0732m |
| 입력 교체 | 새 의미 명령은 오류 기억 초기화·최종 적용 입력 유지. 같은 명령 재계획은 기억 유지. 가상 검사로 실제 기억이 바뀌지 않음 |
| CPU x64, 동일 KEEP 장면 10tick×2 | 전체 tick 평균 0.762/0.732초. 각 첫 tick 제외 0.747/0.730초. 기존 warm 약 2.15초 |

위 시간은 경로 제안·전체 검사·실제 step·next observation이며 teacher/학습기 비용은 포함하지 않는다.
다른 장면과 fallback 비중에 일반화하지 않는다. 전체 WOMD에서의 안전 보증도 아니다.

## 실패에서 수정한 시험 입력

처음 만든 두 정지 경로는 3초 끝점에만 정지 조건이 있었다. 규약의 끝점 유한 차분과
Hermite를 적용하니 마지막 속도가 역방향이 되어 heading 시험이 실패했다. 원본 실패는
`pd_fixed_paths.json`에 남아 있다. 정상 정지 시험에는 2.7초 이후 실제 정지 suffix를
넣었고 기존 편집 규칙으로 처리했다. endpoint-only 역방향 경로를 검사기가 거절하는
회귀도 추가했다. PD 이득·heading 기준을 바꿔 통과시킨 것이 아니다.

## 재현

```bash
JAX_ENABLE_X64=true JAX_PLATFORMS=cpu python scripts/compare_womd_checker.py
JAX_PLATFORMS=cpu python scripts/validate_pd_fixed_paths.py \
  --output outputs/sac_environment_gate/pd_production.json \
  --real-proto outputs/sac_readiness_20261116/inventory/candidates/00005-00044.pb \
  --real-proto outputs/sac_readiness_20261116/inventory/candidates/00060-00013.pb
```

위 입력은 준비 단계에서 만든 로컬 cache다. 서로 다른 데이터에서는 해당 proto를 인자로 전달한다.
환경·계약·데이터·준비 도구 관련 CPU 회귀 106개가 최종 통과했다(27.55초, skip 없음).
학습 연결 회귀의 집계는 해당 학습 기능 브랜치에 별도로 남긴다.

FOLLOW의 목표 간격을 hard rejection으로 사용할지에 관한 해석은 이번 보완에서 바꾸지 않았다.
이후 학습 기능 브랜치에서는 사용자가 목표 간격은 보상, 충돌·추월은 hard 검사로 승인했다.
후속 구현과 검증은 [SAC 진행 기록](sac-development-progress.md)에 누적한다.

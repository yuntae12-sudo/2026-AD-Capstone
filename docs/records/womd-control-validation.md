# WOMD 제어 루프 장기 실행 검증

2026-11-12, PR #5의 기존 5 tick smoke를 종료 시점까지 확장했다.
공통 계약과 PD gain·운동/충돌/도로 검사 기준은 유지했다.
실제 장면에서 목표 종료와 제동 후보 실행을 확인했지만, 아래 실패 사례도 남아 있다.
전체 장면 수락이나 정책 성능 통과를 의미하지 않는다.

## 조건과 재현

- WOMD v1.3.1 `scenario/training_20s`, CPU Waymax, seed 0, `fixed_keep`.
- 각 실행은 `--steps 200` 상한이며 실제 사건 또는 목표가 먼저 발생하면 종료한다.
- 비교 기준은 커밋 `ea131987`이다. 보완 실행은 이 커밋 위 PR #5 변경으로 수행했으며
  원본 보고서의 `working_tree_dirty=true`는 커밋 전 검증을 뜻한다.
- 결과는 로컬 `outputs/womd_control_extended/{before,final,route_reference}/`에 있다.
  원시 장면·상세 JSON은 Git에 추가하지 않는다. 이 문서는 해당 실행의 요약이다.

```bash
conda activate its-merge
export WOMD20_ROOT="<WOMD training_20s 디렉터리>"
JAX_PLATFORMS=cpu PYTHONPATH=. python scripts/smoke_womd_control.py \
  --shard "$WOMD20_ROOT/training_20s.tfrecord-00005-of-01000" \
  --record-index 44 --steps 200 --planner constant_acceleration
JAX_PLATFORMS=cpu PYTHONPATH=. python scripts/smoke_womd_control.py \
  --shard "$WOMD20_ROOT/training_20s.tfrecord-00005-of-01000" \
  --record-index 44 --steps 200 --planner route_reference
```

나머지는 shard/record를 아래 표대로 바꾼다. shard `00006`도 전체 파일명은
`training_20s.tfrecord-00006-of-01000`이다. 모든 장면은 frame 2에서 시작했다.

| shard / record | scene ID |
|---|---|
| 00005 / 44 | `d675f8a070ebe2e4` |
| 00006 / 29 | `baac7893b025167a` |
| 00005 / 10 | `d96e67ea2a0669d3` |

## 실제 결과

N/R/B/E는 각각 NEW/REUSED/BRAKE/EMERGENCY의 실행 tick 수다.

| 코드 / 입력 | shard / record | tick / 시뮬레이션 시간 | N/R/B/E | 실제 종료 |
|---|---|---:|---|---|
| 보완 전 / constant_acceleration | 00005 / 44 | 29 / 2.9초 | 3/2/0/24 | COLLISION |
| 보완 전 / constant_acceleration | 00006 / 29 | 44 / 4.4초 | 5/3/0/36 | COLLISION |
| 보완 후 / constant_acceleration | 00005 / 44 | 34 / 3.4초 | 3/2/5/24 | COLLISION |
| 보완 후 / constant_acceleration | 00006 / 29 | 59 / 5.9초 | 7/7/10/35 | COLLISION |
| 보완 후 / constant_acceleration | 00005 / 10 | 51 / 5.1초 | 35/0/0/16 | COLLISION |
| 보완 후 / route_reference | 00005 / 44 | 72 / 7.2초 | 72/0/0/0 | ROUTE_GOAL |
| 보완 후 / route_reference | 00006 / 29 | 35 / 3.5초 | 0/0/0/35 | COLLISION |

`route_reference`의 00005/44는 진행 거리 80.868m에서 목표 종료했다.
72개 제안이 모두 공통 검사를 통과했고 실제 Waymax overlap/offroad는 없었다.
00006/29에서는 첫 제안부터 보간 후 tangential jerk 제한을 넘었다.
수락된 과거 경로가 없어 바로 보호 입력을 적용했고, 이후 충돌로 종료했다.
검증용 도로 경로도 장면마다 유효하다는 보장은 없다.

보완 후 constant 입력의 새 제안 거절 횟수는 다음과 같다.

| shard / record | OFFROAD | KINEMATIC_LIMIT | COLLISION | UNRESOLVED_SWEEP |
|---|---:|---:|---:|---:|
| 00005 / 44 | 8 | 19 | 4 | 0 |
| 00006 / 29 | 18 | 11 | 23 | 0 |
| 00005 / 10 | 1 | 1 | 12 | 2 |

OFFROAD/COLLISION은 제안 경로의 예측 검사 결과다. 모든 실제 충돌 사례에서
마지막 tick의 Waymax overlap도 확인됐다. 실제 Waymax offroad는 이 실행들에서 없었다.
충돌 때 자차는 정지 상태였고, 주변 객체는 제어 자차에 반응하지 않는 원본 로그를 따른다.
이 사실을 충돌 성공이나 안전 보증으로 해석하지 않는다.
충돌까지의 시간이 길어진 것도 정책 개선의 증거로 사용하지 않는다.

## 발견한 결함과 보완

1. 제동 후보 생성 시 입력 jerk를 20m/s³ 이내로 맞춰도, 30점 유한 차분과
   5차 보간 후 경로 jerk가 제한을 초과했다. 후보의 가속도 변화율을 1.5m/s³로
   낮춰 보간 여유를 확보했다. 최종 경로 검사의 20m/s³ 상한은 변경하지 않았다.
   합성 회귀에서 보간 경로 3,001개 시각을 검사하고 같은 수락 검사를 통과했다.
   실제 00005/44의 5 tick, 00006/29의 10 tick에서 BRAKE 선택을 확인했다.
   큰 초기 가속도·정지 전환·도로/충돌 제약에서는 여전히 제동 후보가 거절될 수 있다.
2. 이전 보고서의 `CONTROL_STEPS_EXECUTED`만으로는 충돌 종료와 실행 상한을
   구분하기 어려웠다. `status`, `outcome`, `terminal_reason`, 실행 시간과 후보별
   거절 원인을 추가했다. 각 tick 저장과 예외 발생 시 부분 결과 보존을 적용했다.
3. 실제 PD 추종 오차를 선택된 경로와 post-step 시각으로 측정한다.
   거절된 새 제안의 오차를 실행 오차로 사용하지 않는다. REUSED의 원래 생성 시각을
   보존하고, 경로 없는 EMERGENCY는 오차 `null`과 제외 표본 수를 기록한다.
4. 현재 지도와 명령만 사용하는 선택형 `route_reference`를 추가해
   단순 속도/가속도 연장 입력의 한계와 제어 연결을 구분했다.
   미래 로그 접근과 검사 완화는 없다. 기본 입력은 기존 constant 방식이다.

## 실제 선택 경로의 추종 오차

매 tick 실제 실행한 경로의 0.1초 후 위치·속도와 비교한 값이다.
새 경로는 매번 현재 상태에 다시 앵커를 두므로 **고정 경로를 3초간 추종한 G3 평가가 아니다**.

| 입력 / shard / record | 참조 있음 / 없음 | 위치 RMS / 최대 (m) | 속도 RMS / 최대 (m/s) |
|---|---:|---:|---:|
| constant / 00005 / 44 | 10 / 24 | 0.000446 / 0.000673 | 0.009644 / 0.017401 |
| constant / 00006 / 29 | 24 / 35 | 0.001142 / 0.002709 | 0.013739 / 0.023742 |
| constant / 00005 / 10 | 35 / 16 | 0 / 0 | 0 / 0 |
| route / 00005 / 44 | 72 / 0 | 0.000269 / 0.000513 | 0.004068 / 0.004784 |
| route / 00006 / 29 | 0 / 35 | 측정 불가 | 측정 불가 |

constant 00005/10은 계속 정지한 사례이므로 0 오차를 주행 성능으로 해석하지 않는다.
route 00005/44의 방향 오차 RMS/최대는 0.006463°/0.052606°다.
비상 입력을 0 오차로 섞지 않으며 정지 상태의 방향도 측정 불가로 둔다.

## 종료와 회귀 검증 범위

[실행 가이드](../architecture/womd-control.md)의 pytest 명령으로 계약·제어·기존 행동·
WOMD 어댑터·물리 합류 검증을 함께 실행했다. **87 passed (96.37초), skip 없음**.
추가 회귀는 다음을 다룬다.

- 제동 경로 보간 후 jerk, REUSED 만료 후 BRAKE 선택과 실제 참조 시각.
- KEEP/FOLLOW 검증 입력의 현재 앵커와 미래 로그 차단.
- 실제 pose/속도/방향 오차 산출, 비상 입력의 RMS 제외, 후보별 거절 집계.
- 부분 실행·충돌·목표·기록 끝·실행 오류의 보고서 구분.
- 200프레임 합성 장면에서 197 tick(19.7초) 실행 후 마지막 frame 199에서
  `RECORD_END` truncation, 종료 후 추가 step 거절. 초기 3프레임은 관측 이력이다.
- 실제 사고를 목표보다 우선 처리하고 손상 데이터는 학습 표본에서 제외하는 기존 회귀.

실제 장면은 모두 기록 끝 이전에 목표 또는 충돌로 종료했다.
따라서 실제 WOMD에서 20초 연속 무사고 실행이나 실제 기록 끝 종료를 달성했다고
주장하지 않는다. 전체 데이터/GPU/실시간 성능/학습 성능 및 G3/G4 전체 수락도 미검증이다.
다음 경로 생성 개발에서는 동일 기준 아래 초기 가속·정지 전환의 보간 jerk와
주변 차량을 고려한 경로를 개선하고 이 실패 장면들을 다시 검증해야 한다.

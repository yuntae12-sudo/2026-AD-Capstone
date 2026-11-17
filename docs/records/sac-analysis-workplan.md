# 학습 중 진행할 시각화·분석 작업

2026-11-17 사용자 요청을 보존한 작업 목록이다. 순수 SAC 학습은 고정 worktree에서
계속 실행하며 이 작업 때문에 실행 중인 코드·manifest·평가 조건을 바꾸지 않는다.

1. **주행 결과 시각화 — 초기 버전 완료, KEEP/FOLLOW GIF 검증 완료**
   - 실제 Waymax 제어 rollout의 도로·주변 객체·제안 경로·실제 궤적을 GIF로 표시한다.
   - 제안 경로의 거절 이유, 선택된 제어 경로, 제동 시점과 종료 상태를 구분한다.
   - 같은 장면·행동·초기 상태의 Rule-based / 순수 SAC를 나란히 비교한다.
   - 학습 전 checkpoint는 UNTRAINED로 표시한다. 성공 장면만 골라 성능 증거로 삼지 않는다.
2. **실패 분석 보고서 — 다음 작업**
   - 충돌 전 연속 거절·제동 이력과 충돌 객체의 상대 위치로 원인을 구분한다.
   - KEEP/FOLLOW·초기 속도 구간별 결과를 집계한다.
   - 보상 상승과 함께 정지 비중이 늘어나는지 확인한다.
3. **평가 기준 정리 — 후속 작업**
   - 성공·충돌·거절·제동·정지 비중을 함께 판단하는 기준을 문서화한다.
   - 현재 validation8장면의 한계와 추가 장면 확보 계획을 정리한다.
   - 현재 실험은 동결하고 확대 평가는 별도 실험으로 구성한다.

## 시각화 실행

```bash
python scripts/visualize_sac_waymax.py capture \
  --manifest <eligible_manifest.json> \
  --checkpoint <online-seed0/checkpoints/step-NNNNNNNN> \
  --scene <scene_id> --provider fixed_keep --output <new-directory>

# 시뮬레이션을 다시 실행하지 않고 저장된 실제 상태에서 GIF를 재생성
python scripts/visualize_sac_waymax.py render \
  --evidence <directory/evidence.json> --output <new-name.gif>
```

`comparison.gif`, 마지막 프레임 `comparison.png`, 원본 상태·경로·검사·제어·보상·충돌 객체
`evidence.json`, 영상 설정 `render.json`을 저장한다. 모든 출력은 로컬 outputs 아래에 둔다.
CPU 추론으로 생성하는 미리보기이며 학습/공식 GPU 평가 trace와 동일하다고 주장하지 않는다.
표시된 자차 움직임은 PD 제어를 적용한 실제 Waymax 상태이고 주변 객체는 WOMD 로그다.
학습 replay나 optimizer를 사용하지 않는다.

두 패널은 같은 시각·축척·카메라로 표시한다. 한쪽이 먼저 종료되면 마지막 상태를 `[held]`로
고정하며 계속 주행한 것처럼 외삽하지 않는다. 제안은 직전 제어 tick에서 생성된 3초 경로이고,
차량 위치와 실제 궤적은 적용 후 상태다. 청색 점선은 통과 제안, 적색 점선은 거절 제안,
황색 실선은 실제 선택된 추종 경로, 초록 실선은 실제 자차 궤적이다.
EMERGENCY 제동에는 추종 경로가 없으므로 황색 경로를 만들어 표시하지 않는다.
하단 속도 그래프의 붉은 배경은 BRAKE/EMERGENCY 개입, 상태 띠는 제안 통과/거절이다.
기본 재생은 실제 시뮬레이션 시간과 같은1배속이고 마지막 프레임만1.5초 유지한다.

## 검증 기록

- `tests/visualization/test_sac_gif.py`: 5개 통과. 원래 환경과 실제 위치·제어 source·검사·보상
  일치, EMERGENCY에서 가짜 실행 경로를 그리지 않음, 초기 조건 대조, 종료 상태 유지,
  좌표 변환·마지막 프레임 보존, GIF의 실제 프레임 수·재생 시간을 확인했다.
- 실제 validation `8b4c1e7e09f5f4c3` KEEP 미리보기: Rule-based74tick / 초기 SAC72tick,
  1440×900, 75프레임, 시뮬레이션7.4초 + 마지막 화면1.5초 유지.
  두 정책 모두 ROUTE_GOAL이었지만 초기 SAC는 업데이트0이며 이 장면의 성공을 학습 성능으로
  해석하지 않는다. 원본 상태와 최종 그림을 함께 보존한다.
- 초기 모델은 `step-00000000`이다. 학습된 모델 비교에는 이후 완성된 불변 checkpoint를
  명시적으로 선택한다. 진행 중인 recovery 경로는 자동 삭제될 수 있으므로 캡처 중 교체하지 않는다.
- 동일 validation 장면 FOLLOW도 Rule-based74tick / 초기 SAC72tick의 실제 제어 결과를
  75프레임 GIF로 저장했다. FOLLOW 대상4874가 각 프레임에서 유지되며 금색 차량으로 표시된다.
  KEEP/FOLLOW 모두 최종 그림을 직접 확인했다. 두 미리보기는
  `outputs/waymax_gif_20261117/{keep-initial,follow-initial}/`에 보존한다.
  최초 FOLLOW 캡처 프로세스는 SIGTERM(exit143)으로 종료되어 중단 자료를 별도 보존했고,
  재실행으로 두 정책의 전체 기록과 GIF 생성을 완료했다. 학습 프로세스는 계속 실행됐다.

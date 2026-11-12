# PPO–SAC–PD 공통 계약

기준은 소유자가 확정한 v1.2/schema3이다. 원본 기계 계약은
[planning_v3.json](../../configs/contracts/planning_v3.json), 원본 직렬화 예제는
[테스트 fixture](../../tests/contracts/fixtures/planning_v3.json)에 보존한다.
JSON의 `production_implementation_complete: false`는 원본 채택 당시 상태다.
현재 구현 범위와 미완료 항목은 이 문서에서 구분한다.

## 구현된 경계

- [planning.py](../../src/contracts/planning.py): 이름으로 직렬화하는 5행동
  `KEEP/FOLLOW/MERGE/AVOID/STOP`, 명령 초안·요청·수락 명령, episode/tick/revision
  token, task, waypoint proposal, 제어·피드백 자료형과 wire codec.
- [observation.py](../../src/contracts/observation.py): float32 feature 5,975개,
  동일 shape의 bool field mask, 별도 item mask, PPO 후보 5개·슬롯·action mask.
- 정수 ID는 metadata이며 정책 feature에 정규화하여 넣지 않는다. 슬롯은 int32이고
  `-1`은 미지정이다. wire optional은 `{value: null, valid: false}` 또는 유효 값이다.
- 물리 시간은 episode tick 0부터 0.1초, 미래점은 현재점을 제외한 30개다.
  source frame은 `episode_start_frame + sim_tick`으로 별도 관리한다.
- World 위치는 박스 중심 float64다. 속도 방향을 yaw로 대체하지 않는다.
  decode는 현재 속도와 가속도를 포함하며, inverse codec은 teacher 위치를 같은 raw_u로 바꾼다.
- proposal의 시간·생성 tick·shape·finite·버전과 raw_u/World XY 일치(1e-7m)를 확인한다.
  raw_u replay는 C 순서 `[60]`으로 펼친다. STOP은 새 인덱스 4이며 기존 인덱스 3과 호환되지 않는다.

토큰은 episode UID, scene ID, tick, revision, schema 전체를 비교한다.
정책에 전달하는 배열은 독립적인 읽기 전용 snapshot이다. 물리 환경과 미래 로그를
이 관측 자료형에 넣지 않는다. 잘못된 payload는 `ContractError`로 보고한다.

## 후속 구현

이 PR은 자료형과 경계 검증이다. SceneState에서 전체 feature를 만드는 관측 생성기,
신경망 encoder/sampler, 명령 관리자의 상태 전이, 후보 생성, 학습 collector는 포함하지 않는다.
5개 행동을 표현할 수 있다는 사실이 각 행동의 실행 검증을 의미하지 않는다.
제어 환경은 우선 KEEP/FOLLOW 지원 범위를 별도로 검증하고 나머지를 mask해야 한다.

## 검증

```bash
PYTHONPATH=. python -m pytest -q tests/contracts
```

원본 JSON/예제와 enum·shape·wire 왕복을 대조한다. 회귀는 stale token, TTL 경계,
잘못된 optional/ID, 비유한·float32 World 경로, decoder 불일치, 좌표 변환,
읽기 전용 관측, masked 행동 및 슬롯 오류를 다룬다.

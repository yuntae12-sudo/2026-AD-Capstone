# 현재 정책 실패 행동의 실제 자기 경험 수집

2026-11-19 KST. 기준 HEAD `0605090a73c7c8009f5f9b6bae77456aa6923be8`,
브랜치 `feature/yuntae-sac-training`. 아래 새 구현은 미커밋 진단 실험이며 기본 학습 설정의
채택이나 안정적인 정책 확보를 뜻하지 않는다.

## 규약 대조와 실험 정의

사용자가 현재 정책 실패 행동의 실제 자기 경험 확보 방안 검토와 지속적인 실험을 요청했다.
[순수 SAC 규약](../architecture/rule-based-vs-pure-sac.md)의 자기 train 경험·BC/teacher0·
전이당 update1·uniform replay·공통 보상/검사/PD를 유지한다. 기존 규약은 reset부터 진행하는
순차 수집을 구현한 것이므로 **중간 simulator snapshot 분기는 기존 본 실험과 동일한 수집
분포가 아니다**. 별도 진단으로 명시하고 기존 production 실험의 결과를 덮어쓰지 않는다.

- train의 원래 episode에서 마지막6개 정책 경계 상태를 보존한다.
- 실제 COLLISION/OFFROAD로 끝나면 각 상태에서 현재 actor의 Gaussian 표본을 최대8개
  실제 공통 collector/Waymax로 실행한다. 정책은 각 새 전이의 SAC 갱신 후 최신 상태다.
- 최근 상태부터 round-robin으로 실행한다. 가장 좋은 경로만 고르거나 충돌 보상을
  복사하지 않는다. 거절된 raw action과 실제 fallback 결과도 그대로 저장한다.
- 추가 전이도 전체100/1,000개 신규 경험 예산에 포함한다. 경험마다 즉시1회 업데이트하고
  같은 uniform replay에서 batch256을 뽑는다. validation 경험은 학습하지 않는다.
- 원본 episode UID/tick·명령·시간·정책 관측·PD 메모리를 유지한다. 분기에는 별도의
  collection_id와 parent lineage를 기록하며 새 reset/독립 장면/성공 episode로 세지 않는다.
- 실제1step 뒤 nonterminal이면 정상 bootstrap을 유지한다. 분기 길이가1이라는 이유로
  terminal/truncated를 만들지 않는다. 원본 경계와 분기 복사본의 관측 일치를 검사한다.
- 분기는 다음 분기를 재귀적으로 만들지 않는다. main/branch/branch-terminal 수를 분리한다.
- 기존 own warmup5,000개를 승계한 fresh learner 진단이다. 이전 학습 가중치·teacher 데이터·
  fitted terminal-only 가중치를 사용하지 않는다. 원래 빈 replay 본 실험의 재개도 아니다.

첫 후보는 이전 zero-action-initialization 후보와 같은 actor/critic 설정을 유지한다.
Critic input gain은1, mean gain은0.0012098536225957168, integrated/entropy mean,
actor LN·whitened critic·critic 행동 입력 초기값0이다. input gain0.1 실패 후보와 섞지 않는다.
학습 precision·SAC 목적함수·optimizer·보상·안전검사·PD는 변경하지 않았다.

## 구현과 검증

- `scripts/sac_failure_branches.py`: opt-in snapshot 보존/분기/parent lineage.
- `scripts/diagnose_sac_online_updates.py`: `--failure-branch-samples`,
  `--failure-branch-window`, 실제 수집→replay→원래 update, 별도 수집량 및 raw log.
- branch 기본0으로 순차 진단 경로를 유지한다. production 학습기/환경 코드는 변경하지 않았다.
- 관련 pytest8개 통과: branch 격리·실패/성공 trigger·현재 경계 검사3,
  기존 비용 gate2, diagnostic critic3. 실제 Waymax 검증은 아래 실행 결과와 구분한다.

## 첫100전이 실행 확인 및 평가 중

15:58 KST seed0 신규 전이100 진단을 시작했다. 로컬
`outputs/sac_failure_collection_20261119/branch100-launch.json`에 실제 명령·PID·시각·소스
hash, 같은 디렉터리에 실행 소스 사본을 보존했다. runtime core·manifest는 기존 실험과
같고 보고서에서 실행 후 다시 검증한다. 100회 학습 완료 후 main52·분기48을 확인했다. 분기는 같은6상태×8회이며 실제 충돌8,
nonterminal40이고 모두 거절됐다. 고유 train2장면의 KEEP 경험만 포함돼 FOLLOW 경험
확보와 다양성은 후속1,000전이에서 확인해야 한다. 이것은 수집 기능 확인이지 주행 개선이 아니다.

`audit_sac_branch_replay.py`로 전체 replay의 train 소속·teacher0, 신규 raw action/보상/
종료/UID/tick, source obs/mask, actual terminal trigger, 신규 전이당 update1을 대조했다.
`audit100-final.json` COMPLETE. parent_tick을 변조한 별도 임시 trace는 거절됨을 확인했다
(`audit-negative-test.json`). 기존 파일은 변경하지 않았다. 고정4주행 평가는 진행 중이다.

다음은 분기 실행/원본 보존·replay 원시 행동 일치 확인 → 고정2장면×KEEP/FOLLOW의4회
주행 → 회귀가 없으면1,000회 및 Q 진단이다. 후보가 실패하면 원인을 분석해 다음 비교로
이어가며 전체 validation·다른 seed·별도 평가 없이 정책 확보 완료로 표시하지 않는다.

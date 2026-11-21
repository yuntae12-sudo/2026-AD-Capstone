# 2026-11-21 로컬 웹 결과 확인과 터미널 실행 도구

## 요청과 구현

사용자가 현재 학습 상황을 웹에서 보고 추가 학습은 터미널에서 직접 실행하기를 요청했다.
기존 브랜치 `feature/yuntae-sac-training`, 기준 HEAD
`c164e5585ccb20d1b5c9ad1df72b4d7f00506f94`에서 별도 도구만 추가했다.
진행 중 학습/후속 worker의 소스·보상·검사·PD·수집·SAC 설정은 수정하지 않았다.

- `scripts/serve_sac_dashboard.py`: localhost:8765 읽기 전용 HTTP/API, Python 표준 라이브러리.
- `scripts/sac_dashboard_data.py`: matched/manual continuation 진행률·평가·곡선·미디어·로그 집계.
- `web/sac_dashboard/`: 한국어 화면, 5초 갱신, KEEP/FOLLOW 필터, 과거 결과/미평가 구분.
- `scripts/sac_experiment.py`: `plan`/`run`, 기존1,000→총10,000 이내, 학습→감사→평가 순차 실행.
- [사용 설명서](../sac-experiments.md): 웹 접속, 정확한 터미널 명령, 로그·중단·지원 범위.

웹과 수동 실행기는 새/이전 실험을 구분한다. 평가 전 수치는0%로 만들지 않고 미평가로
표시한다. 표의 collision은 종료 사유 기준이며 별도 이벤트율/추종 RMS 분석과 다르다.
가중치와 데이터는 웹으로 전송하지 않고, API는 GET만 제공한다. 결과 파일 경로는
outputs 내부로 제한하고 symlink 탈출을 거절한다. 학습 실행 기능은 웹에 넣지 않았다.

## 검증

2026-11-21 17:10 KST까지 다음 검증을 완료했다.

- `PYTHONPATH=. python3 -m pytest -q tests/training/test_sac_dashboard.py`: **16 passed / 0.89초**.
  미평가·tick/episode 분모·부분 JSONL·NaN 그래프 제외·상속 진행률·경로 탈출·HTTP GET 제한,
  plan 무실행·예산 상한·덮어쓰기·동시 실행·죽은 부모의 살아있는 자식 탐지,
  자식 실패 후 후속 단계 금지·SIGTERM 중단 시 자식 종료와 STOPPED 상태를 포함한다.
- 새 Python3개 `py_compile`, `git diff --check` 통과.
- 실제 preset으로 `sac_experiment.py plan --arm both --updates 10000` 실행: 올바른 원본1,000,
  checkpoint/manifest 및6단계 명령 확인, 결과 폴더 생성/새 학습0.
- 실제 HTTP 200, API는 후보10,000·대조군 진행 수와 평가16/0개를 정확히 표시했다.
- Chromium 데스크톱1440×1000/모바일390×844 검증: 실제 API·5초 갱신·미평가 표시·KEEP 필터,
  alpha 전환·명령 clipboard·기존 GIF10개 decode·연결 끊김 시 마지막 자료 보존/복구,
  모바일 가로 넘침 없음. page error0. desktop/mobile/learning screenshot 직접 확인.
- 본 실험과 네 후속 worker의 동결 파일을 다시 해시해 변경0 확인.

브라우저 QA 첫 실행은 호스트의 libnspr4/libnss3 누락으로 시작하지 못했다.
시스템/학습 환경을 변경하지 않고 `/tmp`의 QA 전용 라이브러리를 지정해 해결했다.
다음 실행에서는 테스트의 문자열 wait predicate가 앱 CSP의 unsafe-eval 금지에 막혀,
검사 코드를 함수 predicate로 바꾸었다. 앱의 CSP는 완화하지 않았다. 이후 위 검증 통과.
Playwright/브라우저/한글 폰트는 `/tmp` QA 전용이며 제품 실행 의존성이 아니다.

로컬 근거: `outputs/sac_dashboard_20261121/browser-review.json`, `browser_qa.py`,
`desktop.png`, `mobile.png`, `learning.png`, `frozen-review.json`.
새 GPU 학습·새 정책 평가·전체 저장소 회귀는 실행하지 않았다. 기존 실험은 계속 진행됐다.

## 당시 학습 상태와 남은 범위

17:10:06 KST: 후보 누적10,000 학습/16평가 완료, 목표0/16·충돌16/16으로 안정 정책 미확보.
대조군6,682/10,000, 전체 추가학습14,682/18,000=81.6%. 진행률은 복원/평가 시간을 제외한다.
기존 Q/MC·실패분석·지표·GIF worker는 부모 실험 완료 후 진행하도록 유지한다.

수동 실행은 **원본1,000 모델의 동일조건 재실험**이며 최종10,000 모델의 무제한 연장이 아니다.
seed 새 생성/임의 학습 설정 변경/중간checkpoint 재개는 지원하지 않는다. 기존 학습과
후속 작업이 실행 중이면 수동 시작을 거절한다. 별도로 원래 스크립트를 직접 실행하는
경쟁까지 차단하지는 않는다. 수동 실행에는 기존 네 분석/GIF worker를 복제하지 않았다.
일반적인 모든 과거 PPO/SAC 결과 형식의 조회도 이번 범위 밖이다.

17:13 후속 확인에서 브라우저 연결 취소 검사 중 서버가 BrokenPipeError 뒤 오류 응답을
다시 보내려는 로그를 발견했다. 이미 종료된 연결은 응답을 재시도하지 않도록 수정했다.
해당 회귀 검사를 추가한 최종 관련 테스트는17passed이며, 이전16pass는 당시 결과로 보존한다.
검증용 웹 서버는 종료했고 사용자가 설명서 명령으로 직접 실행한다. 학습은 계속 진행 중이다.

향후 실험 결과 판단은 대조군 완료와 같은예산 비교·실제 Q/return·실패/GIF 검토 후 수행한다.
학습 예산 완료/웹 구현을 안정 정책 확보로 기록하지 않는다. 추가 PR/병합은 하지 않는다.

## 후속 개선: 진행률보다 주행 결과와 실험 의미를 먼저 설명

사용자가 기존 화면만으로 실험의 의미와 결과를 이해하기 어렵다고 지적해,
30afd2350b47f56fe71ed1a6bbd4e674e5758854 기준에서 화면 구조를 개선했다.

- 첫 화면에 실제 평가의 목표/충돌 횟수와 지금 할 일을 배치했다. 후보/대조군 모두 SAC이며
  자기 경험 수집 방식의 비교이고, Rule-based가 별도 기준임을 설명했다.
- 학습 예산 완료·주행 평가 완료·목표 도달을 구분한다. 부분 평가/중복 pair/프로세스 중단을
  처리하며 성공률이100%인 단일 개발 결과도 자동으로 안정 정책이라고 판정하지 않는다.
- 기본 평가 카드는 이번 SAC들과 Rule 참조다. 이전1,000 결과는 체크박스로 펼친다.
  같은 scene/provider 목록의 완료된 이전 후보 평가만 변화량을 계산하며 %p로 표시한다.
- KEEP/FOLLOW 필터를 카드·표·변화량에 적용한다. 그래프는 설명과 함께 펼쳐 보는 보조 정보다.
- 이번/이전 GIF를 선택하고 한 영상만 재생한다. 갱신 중 같은 영상의 DOM을 보존해 재생을
  초기화하지 않는다. 현재 영상이 없을 때 이전 영상을 기본 표시하지 않는다.
- 터미널 메뉴는 현재 학습 모니터 명령을 먼저 보여주고 새 실행은 펼침 메뉴로 분리한다.
  로그 펼침 상태도 갱신 중 보존한다. API에 scope를 추가해 수동 실행의 확인 명령을 구분한다.

실행 검증: 기존17pytest pass/0.89초, 신규 Node 내장 테스트10pass, JS 구문/Python compile/
diff 검사 통과. Node 테스트는 미평가·부분/중복 평가·성공 후 과잉판정 방지·중단/자식 프로세스 누락 감지·
cohort 불일치·이전 결과 선택·single-arm 완료를 포함한다.

Chromium 실제 결과 연결에서 현재 후보0/16목표·16충돌/대조군 학습 중을 확인했다.
기본3행/이전결과5행, KEEP 필터, 후보 거절 변화+40.6%p, 그래프 펼침/alpha 설명,
영상 시점 선택·하나씩 재생·자동갱신 중 재생 DOM 보존, 모니터/plan clipboard,
로그 펼침 보존, 연결 끊김/복구, 모바일 가로 넘침 없음 검증 통과·page error0.
데스크톱 첫 화면/결과 카드/모바일 screenshot을 직접 확인했다.

로컬 `outputs/sac_dashboard_ux_20261121/`: `review.json`, `browser_qa.py`,
`overview.png`, `results.png`, `learning.png`, `history-video.png`, `mobile.png`,
`frozen-review.json`. 본 실험/네 worker의 고유634동결파일 재해시 변경0이다.
검증용 서버는8766포트를 사용하며 사용자의8765서버나 학습을 종료하지 않는다.
새 학습/정책평가/규약변경0, 안정 정책 확보는 여전히 미완료다.

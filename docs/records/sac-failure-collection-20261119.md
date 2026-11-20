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

## 16:07 KST 중간 결과: 100회 주행 평가 완료, 1,000회 진행

구현 보존 commit `c99036d05895153555ed46efa8534d0c1431395b`를 같은 브랜치에 push했다.
새 PR이나 병합은 하지 않았다. 위 실행 당시 HEAD와 보존 commit을 구분한다.

고정 개발 validation 고유2장면×KEEP/FOLLOW=4회, 총216tick에서 목표2/4·충돌2/4다.
거절58/216(26.85%)·제동46/216(21.30%)·정지22/216(10.19%), 평균 보상-10.015397.
빠른 장면72tick 목표 도달2회, 느린 장면36tick 충돌2회다. 기존 성공을 유지했지만
주행 개선은 확인하지 못했다. 같은 설정의 fresh1,000전이 실행을 시작했다.

`exposure100.json`은 저장된 replay RNG와 동일한 batch 추출을 재현한다.
25,600개 추출 중 warmup25,314·main162·branch124회였다. 분기48개 중4개가 미추출,
분기 종료8개는14회 추출되고2개는 미추출이었다. 단순 저장량을 충분한 학습 노출로
해석하지 않는다. 다음1,000회에서도 실제 노출과 KEEP/FOLLOW·장면 편중을 확인한다.

100회 모델 SHA `9a380bf07b0a8106c7a2b49339408abed80099e1d1ae108985b99cc0bb268968`,
replay SHA `efd3f1291fef70be5bc84abba8716a3f93a25c1d507f39689ba350af248a6c88`.
core `96fa1d625b0d8525b1de6de05a12cc132721a03ee47eb51dff3d6e25f02e30ef`,
manifest `1f254359864bff2bc610497f5c78430b91433bb88302c1931daec9056c48257f`.
보고서·모델·평가 hash와 시각은 로컬 `milestone100.json` 및 원본 report에 보존했다.

Notion 첫 화면·02·03·04를 갱신하고 모두 다시 읽어 실제 반영, 기존 실패 이력과 링크
보존을 확인했다. 03의9절이 이번 결과이고04 첫 절이 현재 실행 입구다. 정책 확보는
미완료이며 지속 작업 목표는 active다. 지정 runner의 완료/실패는 전체 연구 목표 종료가
아니다. 1,000회 평가 후 원인 분석과 다음 후보 실험을 계속한다.

## 16:13 KST 후속 검증 준비

`audit_sac_replay_coverage.py`는 분기 trace와 저장 action을 대조하고 main/branch별
실제 batch 추출·95% Gaussian 영역·저장 terminal Q 적합을 따로 집계하도록 확장했다.
Critic 복원도 공통 diagnostic factory를 사용해 input gain 옵션을 누락하지 않는다.
`diagnose_sac_branch_q.py`는 Q를 보기 전에 provider별 최초 실제 종료 분기를 고르고,
원래 main 행동으로 상태를 재현한 뒤 현재 평균·확률4표본·저장 분기 행동을 다시
Waymax로 실행한다. 실제 대체 행동이 종료된 경우에만 Q-보상을 terminal 오차로 표시한다.
새 검증기의 실제 GPU/Waymax 실행은 아직 미실행이며 학습 완료 후 실행한다.

관련 pytest: branch 선택2개와 sampling exposure1개 통과, 공통 Critic factory3개도 통과.
기존8개와 중복되는 Critic3개를 제외하면 이번 작업의 고유 테스트는11개다.
`git diff --check` 통과. 전체 일반 회귀는 미실행이다.

로컬 `review_after_trial.py`를 실행해 현재 학습→4주행→기존 고정 Q 검증 뒤
저장 대조→분기별 coverage/Q→새 분기 실패 상태 실제 재현→동일 모델의 계산 정밀도
읽기 전용 진단을 순서대로 연결했다. 상태는 `review-status.json`, 실행 식별자는
`review-launch.json`이다. 학습과 후속 GPU 진단은 겹치지 않는다. highest 정밀도는
읽기 전용 비교이며 학습 설정으로 채택한 것이 아니다. 현재1,000전이 실행 설정은 고정했다.

## 16:25 KST: 1,000전이 수집 완료, 주행 평가·후속 진단 진행

학습 자체는 COMPLETE다. main616·branch384, 고유 train10장면, main10episode
(충돌8·목표1·예산중단1). main KEEP322/FOLLOW294, branch KEEP240/FOLLOW144다.
분기는48개 원본 상태×8회이며384개 모두 거절됐다. 각 상태의8개 표본은 실제 제어와
보상이 동일했고 실제 종료64·nonterminal320을 저장했다. branch352/384(91.7%)는
pre-step 속도<0.1m/s의 정지 상태다. main 정지269/616을 합친 신규 전체 정지는621/1000이다.
대조군의 신규 정지는624/1000이므로 **전체 정지 비중이 대조군보다 늘었다고 주장하지
않는다**. 다만 새 분기 자체가 늦은 정지 상태에 집중돼 있다는 한계가 있다.

정확한 replay RNG 대조에서 총256,000회 batch 추출은 warmup233,361·main12,930·
branch9,709회다. branch terminal64개는1,603회 추출됐고1개는 미추출이다.
새 terminal은73개(main9+branch64)로 순차 대조군12개보다 늘었지만 main 주행은
1,000→616tick, episode13→10으로 줄었다. main 수락215/616, branch 수락0/384다.
서로 다른 장면 방문 이력의 수치를 같은 상태에서의 성능 차이로 해석하지 않는다.
로컬 `collection-summary.json`에 실제 분모와 모델/replay 해시를 보존했다.

모델 SHA `673ad1c31481384cda7524f2b7ec308774d1259df17bae1032dcefe32ea2a085`,
replay SHA `3cfd7cfdc0f34282800db9223763218a7ebfd01be93c88eee6fcee64bbc4b35d`.
중간 주행 평가에서 빠른 장면 KEEP/FOLLOW 모두45tick COLLISION이 확인됐다.
전체4주행·Q/coverage·새 분기 재현은 아직 완료되지 않았으며 후보를 채택하지 않는다.

후속 수치 진단의 해석 근거: JAX0.6.2 설치 소스의 `Precision`은 GPU highest가float32
내적을 사용하고 입력/출력 dtype을 바꾸지 않는다고 명시한다. [JAX 공식 정밀도 문서](https://docs.jax.dev/en/latest/201/precision.html)도 계산 정밀도와 저장 dtype을 구분한다.
이는 향후 진단 선택지의 규약 검토이며 학습 설정 변경이나 주행 문제 원인 확정이 아니다.
현재 실험의 학습 precision은 그대로이고 후속 읽기 전용 진단에서만 두 정밀도를 비교한다.

## 첫 후보 후속 진단 완료와 다음 비교 (16:37 KST)

첫 runner는 `COMPLETE_WITH_REGRESSION / STOPPED_AFTER_1000`, 후속 읽기 전용 runner는
`COMPLETE_PENDING_REVIEW / COMPLETE`다. 전체4주행은 목표0/4·충돌4/4,
총152tick, 거절98(64.47%)·제동86(56.58%)·정지34(22.37%), 평균 보상-20.277144다.
안정적인 정책 확보는 미완료이며 이번 후보를 채택하지 않는다.

### Q 적합과 새 행동 일반화

기존 고정 train 반례 KEEP/FOLLOW에서 실제 현재 평균 행동은 모두 충돌했다.
Q는-5.210397/-4.012495, 실제 보상-20.014/-20.017이므로 과대평가는
14.803603/16.004505다. 순차 대조군의17.2557/17.4287보다 줄었지만 해결되지 않았다.
원래 warmup terminal62개의 저장 행동 MAE2.075814는 대조군1.025323보다 악화했다.

이번에 추가 수집한 실제 terminal 분기 중 provider별 최초 사례를 **Q를 보기 전에**
선택했다. 원본 main 행동으로 상태를 재현하고 저장 행동·현재 평균·새 확률4표본을
실제로 각각 실행했다(고유 train2상태·12분기). 모두 EMERGENCY→COLLISION이며
각 상태 안에서 보상과 다음 관측 해시가 같았다.

| 이번 수집 상태 | 저장 행동 Q 오차 | 현재 평균 Q 오차 | 새 확률4표본 Q 오차 범위 |
|---|---:|---:|---:|
| KEEP, 77ae312df21e80b9 tick28 | 0.6384 | 3.3637 | 3.4322~13.4314 |
| FOLLOW, baac7893b025167a tick68 | 7.3513 | 12.7384 | 10.4843~14.8041 |

원본 상태·시드·raw action·모델/소스 hash는 `branch-q1000.json`에
보존했다. 모두 이번에 실제 실행한 결과이며, 저장 행동의 보상을 다른 행동에 복사하지 않았다.
분기 terminal64개(고유8원본 상태)의 저장 행동 MAE는1.668666이고, 분기 기록 행동의
최종 정책95% Gaussian 영역 포함 비율은91.93%다. 이는 새 행동의 오차가 작다는 뜻이
아니다. 같은 상태의 새 행동 일반화 문제가 여전히 관찰됐다.

### 수치 가설을 현재 결과로 재검토

`numerics1000.json`은 같은 고정 train suffix22개 중 nonterminal20개·PRNG key2개를
진단했다. 이번 모델의 default에서 fused target과 별도 호출의 최대 차이는4.36e-9,
동일 fused 반복과 joint 분해 차이는0이었다. highest에서는 별도 호출 차이가
0.000165/0.000129였다. 이전 모델의1.5 수준 차이를 이번 모델의 실패 원인으로
그대로 적용할 수 없다. 이 검사 범위에서 현재10점 이상의 Q 과대평가를 수치 불일치로
설명할 근거가 없으므로 **정밀도 변경 실험은 현재 우선순위에서 제외**했다.
준비했던 미사용 정밀도 helper/test는 삭제했고 학습 precision은 변경하지 않았다.

### 다음 후보: 수집 상태 선택만 변경

16:37 KST `outputs/sac_moving_failure_collection_20261119/`에서 fresh100전이를 시작했다.
같은 seed0·자체 warmup5,000·actor/critic·gain·학습률·uniform replay·전이당 update1·
보상/검사/PD를 유지한다. `--failure-branch-selection moving_rejections`만 추가했다.

- 기존: 마지막6상태에서 상태당8표본, 실제 종료 후 최대48개 추가 수집.
- 다음: 실제 종료 직전 상태1개는 유지하고, 나머지는 원래 main 주행에서 최초로
  경로가 거절된 **pre-step 속도≥0.1m/s** 상태를 최대5개 우선 선택한다.
- 움직이는 거절 상태가 부족하면 최근 상태로 채우고 중복 tick은 제거한다.
  원래 episode가6tick 이상이면 동일한6상태×8표본 예산을 유지한다.
- 분기는 여전히 실제 COLLISION/OFFROAD 후 현재 정책으로 실행한다. main 주행 도중
  새 branch를 끼워 넣거나 좋은 행동을 선택/복사하지 않는다. 과거 상태 선택에 실패
  결과를 쓰는 것은 이전 후보와 같고, actor 관측에 미래 로그를 넣지 않는다.
- 원본 snapshot·UID/tick·실제 reward/next obs·nonterminal bootstrap을 유지한다.
  감사기는 moving_rejection이 원래 거절 상태이며 실제 pre-speed 조건을 만족하는지도
  저장된 원본 관측과 대조한다.

관련13개 테스트 전체 통과(선택/격리5·branch Q선택2·coverage RNG1·비용 gate2·
critic3). 실제 새 선택 방식의 수집과 주행 검증은 진행 중이다. 100회→고정4주행→
회귀가 없으면fresh1,000회→같은 주행/Q 검증 순서를 유지한다. source 사본과 실행
명령/PID는 새 출력 디렉터리에 보존했다. 새 설정도 기본 채택이 아닌 별도 진단이다.

새 moving_rejections 후보의 첫100전이도 COMPLETE다(main52/branch48). 감사기가 원본
obs/mask·실제 행동/보상·종료·train 소속·중복/개수·moving 역할의 pre-speed 조건을 통과했다.
원래 recent 후보의 분기 정지48/48에서 새 후보는8/48로 줄고40개가 이동 상태 경험이었다.
하지만 같은 최초 장면의40개 이동 분기도 모두 거절됐고, 각 상태8개 표본의 제어·보상은
같았다. 상태 선택이 의도대로 바뀐 확인과 유효 경로/주행 개선을 구분한다.
현재 고정4주행 평가 중이며 결과는 새 출력 디렉터리의 report에 기록한다.

16:44 KST 새 moving 후보100회 평가도 완료했다: 고유2장면×KEEP/FOLLOW=4회,
목표2/4·충돌2/4,216tick·거절58·제동46·정지22, 평균 보상-10.01539665다.
기존 성공만 유지해 같은 조건 fresh1,000전이 비교를 시작했다. 후속 검증은 새 출력
디렉터리의 `review-status.json`/`review-launch.json`에 연결했다. 구현·진단 보존 commit은
`263097ac5af85a87e0493e9acac3b759ae71ba22`(push 확인), 이 중간 결과 추가 기록은 미커밋이다.
Notion 첫 화면·02·03·04의 현재 상태/결과/commit/다음 작업을 갱신한 뒤 모두 다시 읽어
반영과 과거 실패 이력 보존을 확인했다. 새 후보도 성능 확보·기본 채택은 미완료다.

## 16:53 KST 추가 읽기 전용 점검과 진단 준비

앞선 recent 모델의 실제2상태12분기 중 각 상태의 새4확률 표본과 평균 경로 차이를
같은 앵커에서 decoder의 `3*tau^2*(u-mu)`로 계산했다. 회전은 Euclidean 길이를
보존하므로 원본 yaw 없이도 XY 분리 거리를 계산할 수 있다. 표본별 경로 최대 분리는
0.012~0.116mm인데, 표본 Q 범위는 KEEP9.9992점/FOLLOW4.3198점이었다.
각 상태의 보상과 실제 다음 관측은 같았다. 로컬 recent 결과 디렉터리의
`terminal-action-physical-scale.json`에 입력 보고서 hash·각 표본 수치를 보존했다.
이는 두 상태의 국소 증거이며 모든 작은 경로 차이가 무의미하다는 주장은 아니다.
critic이 실제 효과가 같은 저속 행동을 과하게 구분하는지 후속 보완 근거로 검토한다.

현재 moving 후보의 학습/평가/Q 작업 뒤 최초 이동 중 거절 상태를 provider별로 선택해
현재 평균·Gaussian32표본·raw0·동일 상태 Rule-based 경로를 실제 실행하는 추가
**읽기 전용** 진단을 준비했다. Rule-based 자료는 replay나 학습에 넣지 않는다.
`outputs/sac_moving_failure_collection_20261119/diagnose_moving_actions.py`와
`probe_after_review.py`가 로컬 실행 소스이며, `moving-probe-status.json`과
`moving-probe-launch.json`으로 확인한다. 실제 실행은 아직 미완료다. 상태 선택은
Q/teacher 결과를 보기 전 첫 실패 episode의 가장 이른 이동 중 거절 tick으로 고정했다.
선택 함수의 별도 검증은 통과했으며 저장소13개 pytest와 중복 계산하지 않는다.
nonterminal 행동에서는 한 스텝 보상을 정확한 Q target으로 취급하지 않는다.

## 17:07 KST moving1,000회 평가 및 초기 경험 구성 확인

moving 후보1,000회 학습과 고정4주행 평가 완료. main664/branch336, 분기 정지72/336,
분기 수락41/336이다. recent 후보352/384정지·0/384수락에 비해 이동 상태와 수락 경험은
늘었지만, 목표0/4·충돌4/4로 기존2성공을 잃었다. 총156tick·거절98(62.82%)·
제동86(55.13%)·정지32(20.51%), 평균 보상-20.27173457이다. 서로 다른 길이의
실패 주행이므로 tick 비율 개선만으로 주행 개선을 주장하지 않는다. 기본 설정 미채택.
모델 SHA `d3402ee234c99ce63be4970167fd5034f6ca8a80710c9cb0d67579a318961094`,
replay SHA `48036622b117d3ddee03a0aa6befe27a3046bfa54ee948ce867319ae7a906430`.
기존 고정 Q·coverage·실제 분기 재현·moving-state 읽기 전용 검사는 아직 진행 중이다.

원본 seed0 초기5,000 자기 전이의 trace와 저장 replay를 raw_u/UID/tick으로 일치 확인한
후 pre-step 속도로 다시 나눴다. 로컬 recent 출력의 `warmup-speed-acceptance.json`:

| pre-speed | 전이 | 수락 | 양의 보상 | 실제 goal/collision |
|---|---:|---:|---:|---:|
| <0.1m/s | 3,347 | 2,484 | 0 | 0/42 |
| 0.1~2m/s 미만 | 213 | 63 | 0 | 0/1 |
| ≥2m/s | 1,440 | 662 | 594 | 6/1 |

추가 종료는 정지 상태 RECORD_END10/TASK_LOST2다. 전체 수락3,209개 중2,484개가
정지 상태이며 양의 보상은594/5,000개뿐이다. **경로 수락률64.18%를 유익한 이동
경험64.18%로 해석하면 안 된다.** 초기 경험 편중과 신규 경험의 실제 batch 노출을
함께 분석하되, 이 관찰만으로 curriculum이나 replay 가중치 변경을 채택하지 않는다.
원본 초기 validation 기록도2/16goal·14/16collision이다(당시 core SHA c217dc…, 이번
재실행 아님). 좁은 pilot의2/4초기 성공 유지는 회귀 방지이며 안정적 정책의 증거가 아니다.

## 17:16 KST moving 후보 후속 검증 완료와 재개 검증

저장 감사와 전체 replay coverage는 COMPLETE다. 실제256,000 batch 추출 중 기존
warmup233,361(91.16%)·main12,920·branch9,719였다. branch336개 모두 적어도1회
추출됐고 실제 종료56개의 추출은1,569회다. 그 저장 행동의 terminal MAE는0.572838로
recent1.668666보다 작지만, 서로 방문한 상태가 달라 동일 상태의 인과 비교는 아니다.

기존 고정2반례에서 현재 평균의 실제 terminal 과대평가는13.601100/15.133336으로
남았다. 새로 수집한 최초 provider별 terminal2상태에서도 저장 행동 오차는
KEEP0.010012/FOLLOW0.889326인데, 현재 평균은7.656383/6.047168,
새4표본은1.440378~14.878970/8.064475~13.225845다.12실행 모두 실제 충돌이었다.
현재 모델의 default fused/분리 target 최대 차이는1.83e-9, highest는8.82e-5였다.
수치 정밀도 변경을 실패 해결책으로 채택할 근거는 이번에도 없었다.

최초 이동 중 거절 상태2개를 Q/Rule-based 결과 전에 고정한 읽기 전용 검사가 완료됐다.
각 상태의 stored/current mean/raw0/Rule-based 및Gaussian32표본, 총72actual step이다.

| 상태 | 현재 평균 | 자기32표본 수락 | Rule-based 참조 수락 | 참조 Gaussian 거리(RMS sigma) |
|---|---|---:|---|---:|
| KEEP77ae… tick1,3.739m/s | 도로 이탈 예측 거절 | 0/32 | 통과 | 20.8669 |
| FOLLOWbaac… tick5,6.857m/s | 도로 이탈 예측 거절 | 10/32 | 통과 | 15.1087 |

KEEP32표본은 모두 같은REUSED·보상·다음 관측이었다. FOLLOW는 수락10/거절22로 실제
다른 다음 상태가11개였다. Rule-based가 유효 경로를 제시한다는 것과 그것이 유일한
해결 경로라는 주장은 다르다. Gaussian 거리는 성공 확률이 아니며 전체 주행 성공도
검사하지 않았다. nonterminal이므로 여기서 Q와1step 보상의 차이를 정확한 Q 오차로
계산하지 않는다. Rule-based 참조는 replay/학습에 넣지 않았고 모델/replay SHA 불변을
재확인했다. 원본 `moving-actions1000.json`, 요약 `moving-actions1000-summary.json`.

재개 진단을 위해 `--continue-from`, 총 예산 상한5,000, 별도 collection RNG seed를
구현했다(기준HEAD263097a의 미커밋). 기존100/1,000 fresh 실행은 유지한다. 완료된 같은
구조·수집/환경/규약·원본warmup lineage·hash만 받아 학습기/optimizer/targets/learner RNG와
uniform replay RNG를 복원한다. learner의 전체 직렬화 바이트 일치도 확인한다. 이전
trace/updates/episode 이력을 복사하고 누적1전이=1update를 유지한다. **환경과 scene/policy
RNG는 명시적으로 다시 시작하는 collection fork**이며 exact rollout resume가 아니다.
이전 예산 종료의 nonterminal 전이에 가짜 terminal을 넣지 않는다.

관련18pytest 통과(기존13+재개 검증5), py_compile/diff check 통과. 실제 복원 검증은
`outputs/sac_continuation_20261119/verification-status.json`으로 이어지는1,000→1,001회
한 스텝 학습·저장 감사·replay RNG coverage 순서다. 아직 실제 검증은 진행 중이다.

다음 제한 실험 계획: 실제 재개 검증 통과 시 moving1,000 모델을 총2,000회까지만 이어
현재 정책 경험의 학습 노출을 늘려 본다. 모델/분포/보상/검사/PD/순수 SAC는 고정한다.
새 architecture나 배율 변경 효과와 혼합하지 않는다. 이전 회귀를 공개한 채 시행하는
학습량 진단이며 성능 gate를 통과한 본 실험 확대가 아니다. 같은 고정4주행·고정 Q2반례·
coverage로 다시 판단하고, 추가 연장은 그 결과를 검토한 뒤 결정한다. 최종 주행 성능
및 수집 방식의 인과 효과를 주장하려면 이후 같은 예산의 순차 수집 대조가 필요하다.

17:17 KST 실제 재개 검증 완료: `verification-status.json` COMPLETE,
`restored_learner_bytes_identical=true`, 승계1,000+실제 신규1=총1,001update.
저장 감사와256,256회 replay 추출 RNG 재현 모두 통과했다. 원본 moving1,000모델은
변경하지 않았다. 한 스텝 smoke 모델은 이어 학습할 모델로 선택하지 않는다.

동일 원본 moving1,000모델에서 총2,000회로 이어지는 제한 실험을 시작했다.
로컬 `outputs/sac_continuation_20261119/extended-status.json`, `extended-launch.json`,
`extended-sources/`에 명령/PID/동결 소스가 있다. 실제 추가1,000전이→저장 감사→
고정4주행→기존Q2반례→coverage→새 수집2상태Q 순서이며 그 후 자동 연장은 없다.
정책/장면 RNG는1000으로 재시작하고 learner/replay RNG는 그대로 잇는다.

## 17:25 KST 같은 모델의 물리 경로 변화와 Q 민감도 재분석

실행 중인2,000회 모델/수집 설정은 바꾸지 않았다. 이미 실제 실행한 moving1,000모델의
terminal2상태8확률 행동과 이동2상태64확률 행동을 같은 decoder 식으로 재분석했다.
모델 SHA가 같은지, 원본 두 보고서가 COMPLETE인지 확인한 뒤 결과와 입력 hash를
로컬 moving 출력의 `physical-q-sensitivity.json`에 보존했다. 새 rollout은 아니다.
`compare_action_scales.py`가 재현 코드, `physical-q-sensitivity.png`가 확인한 산점도다.

| 상태 | 표본수 | 평균과 경로 최대 분리의 범위 | 표본 Q 최대−최소 | 실제 다음 상태 수 |
|---|---:|---:|---:|---:|
| KEEP 정지/충돌 직전 | 4 | 0.0120~0.0975mm | 13.4386 | 1 |
| FOLLOW 정지/충돌 직전 | 4 | 0.0505~0.1182mm | 5.1614 | 1 |
| KEEP 이동 tick1 | 32 | 6.017~107.988mm | 0.0747 | 1 |
| FOLLOW 이동 tick5 | 32 | 11.254~116.490mm | 0.1145 | 11 |

동일한2장면의 서로 다른 시점이다. 정지 사례의 실제 reward/next state는 표본 사이에
같았고, 0.12mm 미만 변화에도 Q 차이는 컸다. 반면 이동 사례의 Q 변화는 작았다.
상태·분모가 다르므로 causal 효과나 전체 장면의 통계로 해석하지 않는다. nonterminal의
Q 변화는 Q 오차가 아니다. 이전 recent 모델의 비슷한 관찰과 새 모델 수치를 섞지 않는다.

원본 v1.2 텍스트(`PPO_SAC_PD_구현기준선_v1_2.txt`)의SHA가runtime의
`10b2a882e5dfb3f31b4b300793a67a3d10aa19886421a3782e7b9a58394b4a64`와 일치함을
재확인하고§5/§10을 읽었다. 개인 reference의 `Implementation_Basis_v1.pdf`는 과거
v1(tanh) 문서로 현재 규약 근거와 구분했다. 임시PDF 읽기 도구는 `/tmp`에만 설치했고
실행 중인 conda환경의 dependency는 변경하지 않았다.

**후속 가설, 미구현/미채택:** 현재 critic의whitened 입력 `w=B^-1*u/(g*c)`는 정지 부근
작은 raw 잡음을1/g로 확대한다. g=.002이면 같은 물리적 변화에 이동g=1 대비500배의
좌표 배율이다. 현재 entropy mean의g와 이1/g가 actor의 mean gradient에서 상쇄되므로
정지 경험이 많다고 그 평균 갱신 영향도 작다고 가정할 수 없다. 필요하면 critic 내부
입력만 `v=g*w=B^-1*u/c`로 정하는 별도 좌표 비교를 검토한다. actor/표본/entropy/raw
저장/보상/검사/PD는 그대로 두는 안이다. g가양수여서 자유도는 유지되지만 학습 과정이나
Q 함수의 적합이 같다는 뜻은 아니다. 현재§5는Q를raw u로 계산하는 것도 허용하지만
이 새로운 내부 조건화를 기본 설정으로 채택한 것은 아니다. 기존1000→2000 결과를
먼저 검토하며, 이 가설을 확인된 원인이나 성능 개선으로 기록하지 않는다.

## 17:43 KST 총2,000회 제한 진단 완료: 추가 연장하지 않음

`outputs/sac_continuation_20261119/extended-status.json`의 모든 지정 단계가
COMPLETE_PENDING_REVIEW/COMPLETE로 종료됐고 실행 PID2264605/마지막 검증 PID2276348의
종료를 확인했다. 원본1,000회+신규1,000회, replay7,000개(자체 warmup5,000 포함), BC/teacher0.
모델 SHA `0a34e50663e4d7d79bc2f1bfe784382b1133506642f040f6c33740c46ae68ea6`,
replay SHA `f5a6934e0811f67093ec60ecf28726aae7cc1b05f37531cfed505fa4035b57cf`.
실행 소스는 commit2fb3e4c와 동일하며 `extended-sources/`에도 보존했다.

추가1,000전이: main684/branch316, 정지521+93=614, 양의 보상31+17=48.
전체 누적 main1,348/branch652, main 종료14충돌·3RECORD_END·이전 예산 경계1이고 목표 도달0.
원본의 이전 예산 경계는 그대로 이력에 남았으며 새 episode나 가짜 terminal로 재분류하지 않았다.
고정4주행: **목표0/4·충돌4/4**,157tick·거절96(61.15%)·제동84(53.50%)·정지32(20.38%),
평균 보상-20.26705714. 빠른 장면47tick씩, 저속31/32tick 충돌이다. 기존1,000회와 같이 실패다.

기존 고정 Q2반례의 현재 평균 과대평가는16.679419/15.490440으로 1,000회의
13.601100/15.133336보다 줄지 않았다. 새 수집 고정2상태12행동도 모두 충돌했다.
저장 행동 오차1.581696/1.684767, 평균11.980103/7.744276,
새4표본9.960168~13.965941/8.241598~10.120334였다. 동일2상태에서 1,000회 모델보다
현재 평균 오차가 커졌다.

replay RNG를 검증한512,000 batch 추출 중 warmup430,763/online81,237이다.
추가 학습1,000회의 online 추출 비중은22.8898%(앞선1,000회8.8434%)로 늘었다.
동일한 기존 branch336개를 CSV index로 대조하면 최종 정책95% Gaussian 영역 포함은
318/336→156/336으로 감소했다. 경험 노출을 늘리는 것만으로 Q 일반화/주행이 개선되지 않았다.
저장 감사·주행·고정 Q·coverage·새 분기 Q는 모두 실행했고 새 seed/전체 validation/독립 test는
미실행이다. 현재 설정의 추가 연장을 중단하고 별도 Q 좌표 후보를 검증한다.
전체 정책 확보 목표는 계속 진행한다.

## 17:44 KST 다음 별도 후보: 물리 크기를 반영한 Q 내부 특징

기준 HEAD2fb3e4c 이후 미커밋 구현으로 `PhysicalActionCritic`을 추가했다.
critic 입력 사본만 g를 곱해 내부 특징을 `B^-1*u/c`로 만든다. actor/표본/entropy/보상/검사/PD와
기존 moving_rejections 수집은 동일하다. `physical_whitened`는 별도 설정이며 production
기본값이나 기존 평균 B 비교의 기본 채택이 아니다. 원본 raw u는 그대로 저장/실행한다.

관련22pytest 통과. 새2개는 독립 선형 solve/재구성·고속 동일 동작·동일 초기 파라미터·
저속 gradient 유한성/비영·raw 불변을 검증했다. 최초 원소별 비교는240성분 중1개에서
상쇄 오차 relative1.31e-11로 실패했다. 조건수 약2.89e4인 B의 float64 normwise 오차와
역재구성 검사로 바꿔 통과했으며 물리 검사 기준은 바꾸지 않았다. factory 관련2추가 검사는
과거 gain 모델 호환과 새 모델의 구현 hash 검증을 확인했다. 새 factory hash는 두 소스를 포함한다.

17:44 KST `outputs/sac_physical_q_20261119/branch100` fresh100학습을 시작했다.
동일 seed0·원본 자체 warmup5,000·mean gain0.0012098536225957168·integrated/entropy·
actor LN·zero-Q/action-init·moving 상태당8/최대6상태를 유지하고 Q 내부 좌표만 변경했다.
`branch100-launch.json`/`runner-launch.json`에 명령·PID, `sources/`에 실행 소스가 있다.
100→저장 감사→고정4주행→회귀가 없으면 fresh1,000→저장/4주행/Q/coverage/분기 Q/수치
진단을 순차 수행한다. 초기 학습 프로세스의 실제 종료를 기다린 뒤 평가하므로 GPU 작업은
겹치지 않는다. 실행 상태는 `trial.json`, 현재 새 후보의 주행 개선은 아직 미확정이다.

17:48 KST 새 physical 후보100회 학습과 저장 감사 완료. main52/branch48, 실제 종료8/
nonterminal40, 고유 train2장면·6분기 상태이며 teacher0·원본 관측·raw/보상/종료 일치와
전이당 update1을 확인했다. branch48개는 모두 거절됐다. 모델 SHA
`61343fcf09b52dbe7c0d021095fcc60646ab1678a0cb23a6a9c988cf8240572e`, replay SHA
`f87351f5453ed9b1f993da5e1bff8e0d5b786d71dd9df0ff356c4893754d1ea6`.
factory/helper 합성 SHA `6e9bb1e31c01f1e8b9ea7b118d07329ef6bf798628a70a3f2ddc539df3882fc1`.
core SHA는96fa1d…로 그대로다. 현재 고정4주행 평가 중이며 성능 개선은 아직 미확정이다.

## 17:58 KST physical100회 평가 실패와 동일 예산 진단 결정

첫100회 평가도 **목표0/4·충돌4/4**였다. 총228tick·거절110·제동98·정지46,
평균 보상-20.26424405. 빠른 장면78tick씩, 느린 장면36tick씩 충돌했다.
`trial.json`은 COMPLETE_WITH_REGRESSION/STOPPED_AFTER_100으로 끝났고 프로세스 종료를
확인했다. 코드 보존 commit `dc0fd000e68e32b27507d91ffa7723f92aa6df70`(push 확인).
위 실행은 기준2fb3e4c의 미커밋 변경으로 시작했으며 보존된 코드는 같은 내용이다.

고정 충돌2상태의 저장/평균 과대평가는18.281221/18.281210,18.332859/18.332913이었다.
행동 간 차이는 작지만 두 값 모두 실제 terminal 보상과 멀다. 이후 같은100회 시점의
기존 moving과 새 physical 모델을 같은 원본 terminal62관측에서 읽기 전용 비교했다.
`matched100-fit.json` COMPLETE; `--reproduce-count 0`으로 새 rollout은 하지 않았다.
severe44개 저장 행동 MAE18.463951/18.372970으로 양쪽 모두 아직 크다. 고정 두 관측의
기존 stored/mean Q 간격은0.105260/0.156643, 새 후보는-0.000011/0.000054다.
이 간격 감소만으로 Q 정확도나 주행 개선을 주장하지 않는다.

100회 비용 gate 실패를 보존한 채, 기존 대조군과 같은 fresh1,000회 예산으로 Q 적합·
새 행동 일반화·주행을 비교하는 제한 진단을 별도로 시작했다. 100회 결과를 통과로
변경하거나 본 실험으로 확대하는 것이 아니다. 동일 설정의 반복 실험에서 양쪽 Q가
아직 적합하지 못한 시점만으로 좌표 가설을 판정하지 않기 위한 결정이다.
`reviewed1000-status.json`에 이유·원본 실패/비교 보고서 hash·실행 명령·소스 hash를
기록했다. 실행기 PID2286883, 학습 PID2286884이며 코드 dc0fd00을 사용한다.
추가 자동 연장은 없다. 학습→저장→고정4주행→고정Q→coverage→새 분기Q→수치 진단 후
검토하며 평가 최초 프레임/지도/seed도 이전 대조와 대조한다. 현재 성능 결과는 미확정이다.

## 18:17 KST 기존 고속 KEEP 주행의 실패 순서 재분석

새 rollout 없이 같은 개발 장면 `8b4c1e7e09f5f4c3`의 고정 KEEP 기록을 비교했다.
초기 평균0·physical100·moving1000·moving2000의 최초 frame/지도/seed/versions가 같음을
확인했다. 서로 다른 정책의 같은 한 장면이므로 독립 장면4개로 집계하지 않는다.
3회 연속 거절이 시작되는 tick → 첫 BRAKE → 실제 정지 → 종료 순서는 다음과 같다.

| 정책 | 연속 거절 시작 | 첫 BRAKE | 속도<0.1m/s | 종료 |
| --- | ---: | ---: | ---: | --- |
| 초기 평균0 | 63 | 65 | 없음 | 72 ROUTE_GOAL |
| physical100 | 44 | 46 | 68 | 78 COLLISION |
| moving1000 | 20 | 22 | 45 | 47 COLLISION |
| moving2000 | 21 | 23 | 45 | 47 COLLISION |

세 학습 정책은 모두 정지 후 차량4877과 충돌했다. 충돌 frame에서 그 차량의 중심은
자차 진행 방향으로 각각 -4.678/-5.257/-5.297m 뒤쪽, 횡방향 -0.186/-0.330/-0.329m였고,
자차 속도0, 상대 속도7.342/9.912/9.912m/s였다. 로그대로 움직이는 주변 차량과의
후방 접촉이라는 근거이며 환경 전체의 사고 원인을 일반화한 결론은 아니다.
physical100의 첫 연속 거절 tick44에서 초기 평균0 주행 대비 실제 XY 차이는1.24cm,
제안한 미래 경로의 최대 XY 차이는5.53cm였다. moving1000의 tick20은 각각1.20cm/16.23cm다.
정책별 이후 방문 상태가 다르므로 이 차이를 동일 상태의 행동 인과 효과로 취급하지 않는다.
검사·PD를 낮추거나 바꾸지 않고, 거절 이전 상태의 자기 경험 확보 가능성을 조사한다.

로컬 `outputs/sac_physical_q_20261119/analyze_failure_timeline.py`로 재분석했고
`failure-timeline.json`에 원본 보고서/rollout hash·차량 위치·시계열을 저장했다.
`failure-timeline.png`를 실제로 열어 확인했다. 아래 두 패널은 작은 초기 차이를 보려는
축 확대이며 전체 거리 범위를 뜻하지 않는다. 기준 코드dc0fd00 이후 분석 스크립트는
로컬 산출물, 이 문서 변경은 미커밋이다.

현재 physical1000 학습/저장 감사는 완료했다. 후속4주행/Q/coverage/분기Q/수치 진단은
아직 실행 중이다. 종료 뒤 같은4주행의 초기 frame/지도/seed를 다시 확인하고,
최초 이동 거절 상태와 바로 이전에 통과했던 상태에서 현재 평균·32표본을 실제 실행하는
읽기 전용 검사를 예약했다. 상태는 결과/Q/Rule-based를 보기 전에 고정하며,
Rule-based 참조는 학습/replay에 넣지 않는다. 학습 수집 방식은 아직 바꾸지 않았다.
실행기 `run_predecessor_probe.py` PID2294028, 상태 `predecessor-probe-status.json`,
분석 `diagnose_predecessor_actions.py`, 결과 예정 `predecessor-actions1000.json`이다.

## 18:35 KST physical1,000회 후속 검증 완료와 수집 위치 비교

코드 dc0fd00의 physical1,000회 파이프라인과 후속 predecessor probe는 모두
`COMPLETE_PENDING_REVIEW`로 종료됐다. 지정 프로세스의 실제 종료를 확인했고, 현재 모델의
성능을 달성/채택으로 바꾸지 않는다. 모델 SHA
`1c1dcc8443a6fa245a17170f53c727ebdc46145af0d2795eae218e39ebbbd7c9`, replay SHA
`e438109b34666174c6201b471a7d98651d9c20f70bf2321f9b2b899faf33d425`.

실제 main664/branch336, 고유 train10장면·분기 원본42상태, 분기 통과38/336·실제 종료56·
nonterminal280이다. 저장 감사에서 원본 raw/보상/관측/mask/종료·teacher0·전이당 update1을
확인했다. 고정4주행은 **목표0/4·충돌4/4**,276tick·거절104(37.68%)·제동96(34.78%)·
정지44(15.94%), 평균 보상-20.23615458이다. 빠른 장면92tick씩, 저속46tick씩 충돌했다.
초기 frame/지도/seed/versions도 기존 고정 대조와 네 쌍 모두 같았다. 더 늦게 충돌하거나
tick 비율이 낮아진 것을 목표 성공으로 취급하지 않는다.

기존 고정 두 충돌 상태에서 현재 평균 행동의 실제 과대평가는6.143699/4.173077이다.
같은1,000회 moving 대조의13.601100/15.133336보다 작지만 아직 크다. 저장 행동의 오차는
6.129748/4.166669로 현재 평균과 비슷하다. 원본 severe44개 저장 행동 MAE는5.082724로
대조의1.184733보다 나빠져, 저장 적합과 새 행동 일반화를 함께 판단해야 한다.
새 수집 첫 terminal2상태(KEEP77ae… tick28/FOLLOWbaac… tick63)의12실제 분기는 모두
충돌했다. 저장/평균 오차는3.011593/3.018532와4.677645/4.656944,
새4표본 범위는3.013680~3.025696와4.639962~4.671277이었다.
FOLLOW의 원본 상태 tick은 기존 moving의68과 다르므로 동일 상태 인과 비교가 아니다.

coverage는256,000 batch 추출 중 warmup233,361·main13,409·branch9,230을 실제 replay RNG로
검증했다. 마지막 수집 분기4개는 아직 batch에서 추출되지 않았다. 분기 정지112/336,
최종 정책95% Gaussian 영역 포함313/336이다. 원본 warmup의 높은 비중은 계속 남는다.
고정 nonterminal20상태×2key 수치 대조에서 default fused/분리 target 최대차이는
4.95e-10이었다. precision 변경을 채택하지 않았다. 실제 수치는 `numerics1000.json`에 있다.
초기 평균0과 physical1,000의 빠른 KEEP 실제 기록을 기존 renderer로 다시 그려
`review/fast.gif`(32frame,1440×900,실시간9.2초+종료1.5초 hold)를 저장하고 PNG를 열어
확인했다. 이는 Rule-based 비교가 아니며 새 주행을 추가 집계하지 않는다.

### 읽기 전용: 거절 상태와 직전 통과 상태

현재 physical1,000 모델을 고정하고 각 provider의 첫 실패 episode에서 첫 이동 거절 상태와
바로 전 통과 상태를 결과/Q/참조 생성 전에 골랐다. 동일 Gaussian seed77119의32표본을
각 상태에서 실제 실행했다. 평균·raw0·저장 행동·Rule-based 참조까지4상태×36=144실제
1step을 실행했고 모델/replay hash 불변을 확인했다. 모두 nonterminal이며1step 보상을
정확한 Q target으로 간주하지 않는다. 참조는 학습/replay에 넣지 않았다.

| train 장면·행동 | 원본 tick | 현재32표본 통과 | 현재 평균 통과 | 표본 다음 관측 고유 수 |
| --- | ---: | ---: | --- | ---: |
| 77ae312df21e80b9 KEEP 거절 상태 | 1 | 0/32 | 아니오 | 1 |
| 같은 KEEP 직전 통과 상태 | 0 | 32/32 | 예 | 32 |
| baac7893b025167a FOLLOW 거절 상태 | 5 | 9/32 | 아니오 | 10 |
| 같은 FOLLOW 직전 통과 상태 | 4 | 16/32 | 예 | 17 |

고유 train2장면의 paired4상태이며, 전체 주행 성공이나 다른 장면 일반화의 증거는 아니다.
다만 같은 현재 정책으로 거절 이전 상태에서 다양한 실제 유효 경험을 얻을 수 있음을
보였다. `predecessor-actions1000.json`/`predecessor-actions1000-summary.json`,
`predecessor-probe-status.json`에 선택 상태·행동·실제 결과·명령·해시가 있다.

## 18:35 KST 다음 제한 비교: 통과했던 이동 상태를 우선 포함

기준 HEAD dc0fd00 이후 미커밋 변경으로 `moving_predecessors` 선택 옵션을 추가했다.
`FailureBranches.after_step`에서 실제 이동 거절이 발생하면 그 직전5tick 안에서
통과했던 이동 상태를 가까운 순서로 최대5개 보존한다. 최종 실제 실패 직전1상태는
유지한다. 부족하면 기존 최초 이동 거절 상태, 최근 상태로 채워 고유 최대6상태×8표본을
유지한다. 동일 상태의 중복 슬롯을 만들지 않으며 전체1,000전이 예산을 공유한다.

**수집 분포만 바꾸는 별도 진단**이다. 실제 COLLISION/OFFROAD 뒤에만 분기를 실행하고,
분기에서는 당시 최신 정책의 원래 Gaussian 행동을 실제 collector로 실행해 결과 전부를
저장한다. 새 분기의 통과 결과를 보고 좋은 행동만 골라 replay에 넣지 않는다.
branch nonterminal의 bootstrap은 유지하며 부모 UID/tick·PD memory·관측을 보존한다.
전이당 update1·uniform replay·BC/teacher0·보상·검사·PD·sigma·physical Q는 동일하다.
`predecessor_rejection_tick`으로 어떤 실제 후속 거절 때문에 선택됐는지 기록하고 감사한다.

관련25pytest 통과. 새3검사는 유효 이동 이력·실제 최종 실패 상태·동일 최대 예산,
중복 없는 fallback, 정지/오래된 상태 제외와 window1 경계를 확인했다. 첫 실행은
fixture가 두 번째 최초 거절 tick3을 tick5로 잘못 기대해24통과/1실패였고, 실제 이력을
대조해 기대값을 수정한 뒤25통과했다. 기존 두 선택 모드도 기록된7실패 episode의
main 판정/속도로 재생해 이전 보존 소스와 선택 tick이 모두 같음을 확인했다.
이 선택 전용 점검은 시뮬레이션 재실행이 아니다(`legacy-selection-check.json`).
구문 검사와 `git diff --check`도 통과했다.

새 실행은 `outputs/sac_predecessor_collection_20261119/`, 실행기 PID2305535,
첫100학습 PID2305536이다. `launch.json`과 `trial.json`에 명령/소스 hash/PID/판단 근거,
`sources/`에 동결 실행 소스를 보존했다. **첫100회는 저장·통합 검증만** 수행한다.
기존 physical100의 주행 gate 실패를 통과로 재분류하지 않는다. 실제 predecessor 전이가
저장·검증되면 기존 physical1,000 대조와 같은 fresh1,000회 예산의 수집 비교를 실행한다.
학습→저장→같은4주행/초기조건 대조→고정Q→coverage→새분기Q→수치 진단 뒤 검토하며
1,000회 이상 자동 연장은 없다. 전체 validation/새seed/독립평가/정책 확보는 미완료다.

18:39 KST 첫100회 실제 저장 검증 완료: main52/branch48, 분기40거절·8통과,
실제 종료8/nonterminal40. moving_predecessor8개는 모두 실제로 통과했고 원본 관측·raw·
보상·종료·후속 거절의 lineage·teacher0을 대조했다. 고유 train2장면·분기 원본6상태다.
모델 SHA `88687235d206c4b5e6bd132e9101c22c8ad68bd3e0bc03253f18bfbd6389cf85`,
replay SHA `a09d39f0fd1e68bfe8fa7962ed85351f0b77c60888b575addb6d9ff0b59ccf33`.
100회 주행 평가는 이번 수집 비교에서 미실행이다. 학습/저장 통과를 성능 통과로
기록하지 않는다. 현재 같은 설정 fresh1,000회 학습 PID2309042로 진행 중이다.

## 18:50 KST 수집 중간 구성과 종료까지의 가치 검증 준비

현재 predecessor1,000회 실행은 계속 진행 중이다. 첫300개 실제 로그만 고정해서 기존
physical 대조의 첫300개와 비교했다. 양쪽 main156/branch144이며, 분기 통과는7→48,
분기 pre-step 정지는40→24였다. 새 predecessor48개는 전부 통과했고 양의 보상8개였다.
main의 통과32·양의 보상8·정지68은 집계상 같지만 실제 경로/상태의 동일성을 뜻하지 않는다.
완료된 replay 감사나 주행 평가가 아닌 중간 trace 분석이며, 고정 prefix 바이트 hash와
분모를 로컬 `outputs/sac_predecessor_collection_20261119/prefix300-comparison.json`에 보존했다.

추가 읽기 전용 `diagnose_frozen_soft_returns.py`를 같은 로컬 폴더에 준비했다.
현재 모델의 terminal Q 일반화가 나아져도 그 이전 상태의 미래 실패 비용이 여전히
작게 예측되는지 확인하려는 진단이다. 기존 physical1,000의 paired 검사에서 결과/Q와
무관하게 선택했던 KEEP tick0/FOLLOW tick4를 그대로 사용한다. 각 모델의 평균을 첫 행동으로
고정하고, 이후는 기존 `Policy`의 확률 행동으로 실제 계약 종료까지4seed씩 실행한다.
두 모델은 각자의 평균 첫 행동/후속 정책을 쓰므로 동일 행동 인과 비교는 아니다.

Q의 max-entropy 정의에 맞춰 첫 행동에는 entropy를 넣지 않고 이후에는 원래 ell과
고정 alpha를 사용한다. 할인율0.99, 실제 reward·delta_tick·terminal을 유지하며 종료에서
critic bootstrap을 넣지 않는다. 표본 soft return과 Q의 차이는 작은 MC 추정치이고
정확한 기대 Q 오차나 독립 test 성능이 아니다. Rule-based/teacher는 사용하지 않는다.

구문 검사와 첫 행동 entropy 제외·할인 보상/entropy·1tick guard·유한 Bellman 전개와
forward 합 일치의 산술 검사를 실행해 통과했다. **GPU/Waymax 실행은 아직 미실행**이다.
실제 실행에서는 원본 parent 관측/mask/reward를 재현하고, sample_action과 실제 Policy
raw의 일치, 부모 상태와 모델/replay hash 불변, 실제 종료 도달을 확인한다.
실행기 `run_soft_return_comparison.py` PID2313732가 현재 파이프라인 종료 뒤
physical1,000→predecessor1,000 순서로 실행한다. `soft-return-status.json` WAITING,
`soft-return-launch.json`에 명령·실행기 hash를 보존했다. 학습 코드/가중치/규약은
변경하지 않았으며, 이 로컬 진단과 기록은 HEAD3180615 이후 미커밋 작업이다.

## 19:00 KST predecessor1,000회 완료: 고정4주행의 성공 두 건 회복

학습과 실제 replay 감사가 완료됐다. 코드3180615, 모델 SHA
`d9061a908bb3c6f2f5e41d06ad55afc5b66851eb2c00e49236f8d553eaec52ee`, replay SHA
`56044f85ecbe05fb52a3c0e9a0eacaba236873f95acb45f0cc06dc3c78958f3b`.
main664/branch336, 고유 train10장면·분기 원본42상태, 실제 분기 종료56/nonterminal280이다.
원본 행동·보상·관측/mask·종료·teacher0·전이당update1·선택 근거를 대조했다.
같은 physical 대조와 비교해 분기 통과38→136/336, 양의 보상22→81/336으로 늘었다.
predecessor144개 중136개 통과·81개 양의 보상이고 나머지8개 거절도 그대로 저장됐다.
main 종료 집계는 양쪽 모두7충돌·1목표·1RECORD_END·1예산경계로 같았다.
`completed-collection-comparison.json`에 데이터 구성과 원본 hash를 보존했다.

새 고정 개발2장면×KEEP/FOLLOW의 결과는 **목표2/4·충돌2/4**다. 빠른 장면8b4c…에서
KEEP/FOLLOW 모두72tick에 목표를 도달해 기존 physical 후보의 두 충돌을 회복했다.
초기 평균0도 성공했던 장면이므로 최초 성공이나 초기 정책보다 높은 성공률로 표현하지
않는다. 저속8a4f…는 여전히KEEP/FOLLOW 모두46tick에 충돌했다. 초기 frame/지도/seed를
네 쌍 모두 대조했고 비용 gate는passed=true다. 정책 확보/최종 성능 기준 통과는 아니다.

새 후보236tick·거절44·제동36·정지12이며, 기존 physical은276tick·거절104·제동96·정지44다.
초기 평균0의216tick·거절58·제동46·정지22와도 구분한다. episode 길이와 종료 이유가
달라진 지표를 분리해서 읽는다. `pilot-review.json`에 원본 report hash와 분모를 저장했다.
현재 고정Q→coverage→새분기Q→수치 진단을 이어가고 있으며, 예약한 실제 종료까지의
soft-return 비교도 아직 대기 중이다. source/가중치/학습 조건은 변경하지 않았다.

짧은 주행 회귀가 회복됐으므로 같은 고정 모델을 **전체 개발 validation8장면×KEEP/FOLLOW**에서
초기 평균0과 비교하는 다음 검증을 예약했다. 기존8장면은 반복 사용한 개발 자료이며
독립 test가 아니다. 각 정책16회, 총32회 실행하고 최초 frame/지도/seed16쌍, 기존 pilot4회
결정적 결과 재현을 검사한다. Rule-based/다른 seed/독립 장면 최종 비교는 아직 미실행이다.
`run_full_validation.py` PID2318739, `full-validation-status.json` WAITING,
`full-validation-launch.json`에 명령·소스 hash가 있다. 실제 실행은 현재 파이프라인과
읽기 전용 soft-return 비교 종료 후 순차 진행한다. 이 후속 기록은 HEAD3180615 이후 미커밋이다.

## 19:25 KST 실제 종료까지의 가치 비교 완료: 앞선 상태의 과대평가 지속

predecessor1,000의 고정 Q·coverage·새 분기 Q·수치 진단까지 완료했다. 기존 고정 두
충돌 상태의 현재 평균 Q 오차는6.1073/4.1387로 physical의6.1437/4.1731과 비슷했다.
새 후보가 실제로 수집한 terminal 두 상태의 평균/확률 행동도12/12회 모두 충돌했고,
Q 과대평가는 KEEP 약3.28, FOLLOW 약4.33이다. batch256×1,000의 실제 추출 중
초기 자기 warmup233,361/256,000(91.16%), 새 main13,409, 분기9,230이었다.
분기336개 중4개(삽입 update930/940/971/972)는 해당1,000회 batch에서 추출0회였다. 저장된 severe terminal
warmup44개의 Q MAE도5.4065여서 충분한 적합을 주장하지 않는다.

읽기 전용 soft-return 비교를 두 모델 모두 완료했다. 이전 paired 검사에서 미리
고른 train KEEP77ae… tick0/FOLLOWbaac… tick4에 각 모델의 평균 첫 행동을 적용한 뒤,
원래 확률 정책으로 실제 종료까지 seed88019~88022 네 번씩 실행했다. 양쪽 모두
**8/8회 실제 충돌**이었다. 아래는 각 상태의4회 표본 평균이며 정확한 기대 Q가 아니다.

| 후보 | 상태 | 현재 평균 행동 Q | 실제 soft return 평균 ± 표본 표준편차 | Q − 표본 평균 |
|---|---|---:|---:|---:|
| physical1,000 | KEEP tick0 | -0.5898 | -13.2774 ± 2.0451 | 12.6875 |
| predecessor1,000 | KEEP tick0 | -0.3466 | -13.3945 ± 2.0569 | 13.0480 |
| physical1,000 | FOLLOW tick4 | -0.2683 | -10.5707 ± 0.4755 | 10.3024 |
| predecessor1,000 | FOLLOW tick4 | -0.2210 | -10.8831 ± 0.6225 | 10.6621 |

각 모델은 자신의 첫 평균/후속 정책/고정 alpha를 사용하므로 동일 행동 인과 비교는
아니다. 두 train 상태와 반복 seed를 독립 장면16개로 세지 않는다. 확률 후속 주행과
평균 정책의 개발 pilot2/4성공은 상태·평가 방식이 달라 같은 분모로 섞지 않는다.
**수집 통과와 pilot 회복은 확인했지만 장기 가치 예측 개선은 확인하지 못했다.**

원본 두 상태의 입력 fingerprint와 반복 seed가 모델 간 같음을 확인했다. 진단기의
실제 Policy/raw 대조·부모 상태/가중치/replay 불변 검사를 통과했고, 저장된16개 trace를
별도로 읽어 hash·실제 tick·최종 terminal·첫 entropy 제외·할인 reward/entropy 합·역방향
유한 Bellman 전개·거절/제동 횟수도 대조했다. 결과는 로컬
`outputs/sac_predecessor_collection_20261119/soft-return-comparison-review.json`,
`soft-returns-physical/`, `soft-returns-predecessor/`에 있다.

수치 한계도 보존한다. 같은 physical 모델의 이전1step probe와 이번 재시작 진단 사이
평균 raw 차이는 최대1.31e-7, Q 차이는 최대0.00572였다. 이번 복원 입력은 원본 replay와
바이트 단위로 같았으나 이전 probe 입력은 당시 allclose 검사만 했으므로 차이의 원인을
단정하지 않는다. 관찰된 차이는 이번10~13점 격차보다 작다. 새 후보의 별도20nonterminal×
2key 수치 진단에서 default fused/분리 계산 차이는5.95e-10였으며 정밀도는 변경하지 않았다.

19:25 KST 전체 개발 validation 실행 PID2329716을 확인했다. 초기 평균0/새 후보 각각
8장면×KEEP/FOLLOW16회, 총32회이며 아직 실행 중이다. `full-validation-status.json`과
`full-validation/report.json`에서 확인한다. source/보상/검사/PD/순수 SAC 조건은 그대로다.
새1,000회 이후 학습은 아직 미실행이고, 전체 결과·초기 조건16쌍·pilot4회 재현을 검토한
뒤 같은 모델의 추가 학습 또는 수집의 빠진 구간을 선택한다. 안정적인 정책 확보 목표는
계속 진행 중이며, 새seed·동일 조건 Rule-based·독립 평가 미완료를 유지한다.

## 19:38 KST 정지 경험의 실제 움직임 부족 추가 점검

전체 개발 평가의 초기 평균0 기준16회는 목표2·충돌14로 완료됐고 새 후보16회를 이어가고
있다. 평가 코드는 동결했다. 완료된4회 pilot의 원본 프레임으로 tick·거절·제동·정지·보상·
초기 속도·접촉을 별도로 재집계해 기존 결과와 일치를 확인했다. 전체32회 완료 뒤 사용할
읽기 전용 집계기는 로컬 `review_full_validation.py`이며, 독립 test/Rule-based 비교는 아니다.

기존1,000개 실제 수집 제안을 `from_wire`→raw decode→원래 `Trajectory`로 재구성했다.
새 simulator 실행이나 learner update는 하지 않았다. 정지 상태의 작은 경로 변화가
검사 통과 후 실제 출발을 유도하는지 확인하려는 점검이다. raw 재해석 결과는 저장 XY와
절대오차1e-9 내에서 같았다. 속도 구분은 제안 전 속도이며 평가의 post-step 정지 비중과 다르다.

| 실제 수집 집단 | 전이 수 | 검사 통과 | 3초 raw 잔차 이동 중앙값 / 최대값 |
|---|---:|---:|---:|
| main, 속도≥2m/s | 257 | 101 | 43.46 / 156.82mm |
| main, 0.1≤속도<2 | 48 | 24 | 5.56 / 51.84mm |
| main, 속도<0.1 | 359 | 208 | 0.0850 / 0.2586mm |
| branch, 속도≥2m/s | 200 | 96 | 38.77 / 115.02mm |
| branch, 0.1≤속도<2 | 40 | 40 | 5.07 / 18.49mm |
| branch, 속도<0.1 | 96 | 0 | 0.0944 / 0.2506mm |

잔차는 원래 앵커 운동 `p+v*t+0.5*a*t²`에 raw action이 더하는 `3*t²*u`의 크기다.
전체 실제 경로 이동량이나 최종 고정 정책의1,000표본으로 해석하지 않는다. main 정지345개와
branch 정지96개, **총441개는 원래 정지 suffix 규칙으로 미래30점 전체가 정렬됐다**.
이들의 다음 tick 목표 속도 최대는2.65e-23m/s이고, 통과한208개에서 실제 적용한 양의
가속도 최대도1.59e-22m/s²였다. 수치적으로 거의0이며 “검사 통과=유효 출발 경험”이 아님을
보여준다. `motion-scale-review.json`에 원본/소스 hash·모든 전이별 값, 재현용
`review_motion_scale.py`에 계산을 보존했다.

v1.2 6절의 1mm 정지 suffix/전체 정렬 조건과 현 구현을 대조했다. **정렬·검사·PD 규칙을
완화하거나 고치지 않았다.** 원 규약의 g는 탐색 잡음에 적용하며, 현재 별도 entropy 평균
좌표 옵션은 평균에도 g를 곱한다. 정지에서g=0.002이므로 같은 latent 출력의 평균 변화도
이동 상태보다500배 작아지는 조건화가 있다. 이것은 코드식과 실제 작은 수집 움직임을
연결한 추가 가설이며 학습 실패의 유일 원인이나 대체 구조의 개선을 입증하지 않는다.
현재 모델·보상·sigma·SAC update는 그대로 유지한다.

같은 모델 총2,000회까지 이어가는 로컬 실행기도 준비했지만 **미실행**이다. 로컬
`outputs/sac_predecessor_continuation_20261119/run_bounded_continuation.py`는 전체32회
평가의 실제 검토 기록·모델 hash·초기 성공 유지 확인이 있어야 시작할 수 있고 현재 PID나
학습 결과가 없다. 전체 결과와 이번 정지 경험 한계를 함께 검토해 다음 진단을 결정한다.
이 추가 분석·기록은 문서 HEADc667a2a 이후 미커밋이며 실행 코드는3180615와 동일하다.

## 19:47 KST 전체 개발 평가 완료와 19:48 KST 제한 추가 학습 시작

전체 개발8장면×KEEP/FOLLOW×초기 평균0/새 후보, 총32회를 완료했다. 최초 frame·지도·seed
16쌍과 기존 pilot4회의 결정적 결과를 대조했고, 원본32개 프레임 기록으로 모든 tick·
거절·제동·정지·보상·사건·속도 구분을 다시 계산했다. 원본 보고서 SHA
`34035c907a6526c44c3a8652299e825a90b876554c2853bf589ea99c106f76c0`, 분석 SHA
`b59dff5b15e8767db0b0797c7461cd01b6ab067a40af0c1de3929a13c6490754`.
로컬 `full-validation-review.json`에 개별 episode·원본 hash·paired 차이를 보존했다.

| 정책 | 목표/충돌 | 실행 tick | 거절 | 제동 | post-step 정지 | 평균 보상 |
|---|---|---:|---:|---:|---:|---:|
| 초기 평균0 | 2/16, 14/16 | 886 | 356(40.18%) | 334(37.70%) | 434(48.98%) | -17.76736 |
| predecessor1,000 | 2/16, 14/16 | 908 | 340(37.44%) | 322(35.46%) | 424(46.70%) | -17.75719 |

두 정책 모두 KEEP1/8·FOLLOW1/8 목표이며 각각7/8충돌이다. 새 성공도 잃은 초기 성공도
없다. 감소한 거절·제동·정지는 초기속도2~10m/s의4회에 집중됐지만 그4회도 모두 충돌했다.
초기 정지2장면×2행동은 양쪽 모두328/328tick 정지·4/4충돌이다. 양쪽14충돌 모두 자차가
정지했을 때 충돌 객체 중심이 뒤쪽에 있는 접촉을 기록했다. 이것은 접촉 기하와 이력의
근거이며 책임/단일 원인 판정은 아니다. 독립 test나 Rule-based 비교로 바꾸어 표현하지 않는다.

같은 최초 상태에서 새 평균 제안과 초기 평균0 제안의 최대점 거리는 이동 장면 약8.33~
8.35mm, 1.82m/s 장면6.30mm, 정지 장면약0.0166mm였다. 이후 다른 상태 방문의 인과 효과와
섞지 않는다. **초기 정책보다 높은 주행 성공률을 확보하지 못했으며 기본 채택하지 않는다.**

이 결과와 Q 오차를 검토한 뒤, 모델/수집 조건을 그대로 두고 총2,000회까지 추가1,000개
실제 자기 경험·update만 실행하기로 했다. 초기 성공 회귀는 없었으나 저장 terminal 적합과
앞선 상태의 미래 실패 비용 학습이 아직 부족했고, 첫1,000update batch의91.16%는 초기
warmup이었다. 새 경험 노출에 따라 Q/주행이 변하는지 분리하는 제한 진단이다. 정지
제안이 실행 가능한 출발로 이어지지 않는 한계가 추가 학습만으로 해결된다고 가정하지 않는다.

실제 시작은19:48:24 KST이며 실행기 PID2335763, 학습 PID2335764가 살아 있음을 확인했다.
로컬 `outputs/sac_predecessor_continuation_20261119/`의 `review-decision.json`에 검토 근거,
`launch.json`·`status.json`에 명령/PID/소스 hash, `sources/`에 동결 소스를 보존했다.
결정 파일의 수기 시각19:49는 분 단위로 앞서 적힌 값이라 파일 생성 관측 시각을
`launch-verification.json`에 별도로 남겼고 원본 hash는 유지했다.

원본 d9061a…의 learner·optimizer·target·learner RNG·replay RNG는 이어받고,
환경/장면/정책 수집 RNG만 명시적으로1000으로 재시작한다. 정확한 simulator resume가
아니며 이전 예산 경계에 가짜 terminal을 넣지 않는다. 실제 저장 감사→전체16주행→
고정 Q 반례→coverage→수집 상태의 새 행동 Q 순서다. 규약·보상·안전검사·PD·sigma·
uniform replay·전이당update1·BC/teacher0은 유지한다. 총2,000회 뒤 자동 연장은 없다.
기본 채택·새seed·Rule-based 최종 비교·독립 평가와 안정적인 정책 확보는 계속 미완료다.

## 20:02 KST 추가 수집 중간 구성과 후속 원인 분리 검사 준비

현재 추가 학습 PID2335764는 살아 있으며 누적1,806회(이번 재개806개)를 확인했다.
추가 첫500전이만 고정 prefix로 보면 main356개(통과187·pre-step 정지208·양의 보상37),
branch144개(통과77·정지24·양의 보상77)다. main 고유5장면·branch3장면이고 전체 replay
저장 감사나 주행 성능 판정은 아직 아니다. prefix 원본1,500줄 hash와 분석 구간
update1001~1500을 로컬 `first500-additional-review.json`에 보존했다.

정지 경험의 움직임 부족을 더 분리할 읽기 전용 `probe_mean_conditioning.py`를 같은
continuation 폴더에 준비했다. 현재 평균과 `mean/g` 가상 평균을 같은 실제 상태에서
비교하되, 원래 Gaussian 잡음·rho/sigma·검사·PD는 유지한다. **가상 평균은 원래 정책과
다르며, 학습된 새 구조나 채택된 정책이 아니다.** 결과는 replay/learner에 넣지 않는다.

결과/Q를 보기 전에 predecessor1,000의 실제 main에서 provider별 최초 통과 상태 중
정지(speed≤0.001, |a|≤0.01)와 이동(speed≥2)을 각각 선택했다. 고정된 상태는 KEEP
fcbcfcb416ca17c0 tick54와77ae312df21e80b9 tick0, FOLLOW1bedf33cd20c1e7b tick3와
baac7893b025167a tick0다. 원본 trace index는693/0/837/271이다. `mean-probe-selection.json`에
원본/소스 hash와 선택 시각을 보존했다. validation이나 teacher로 상태를 고르지 않았다.

각 정지 상태에서 두 평균과 동일 noise의4표본씩, 이동 상태에서는 두 평균과1표본씩
실제1step을 비교한다. 모델당28개, 1,000/2,000모델 총56개이며 고유4train상태의 반복이다.
평균만 바꾼 뒤 보간 정렬·제안 목표 속도·실제 수락/거절·제동/가속도·post-step 속도·
보상/종료를 분리한다. 이동g=1에서는 raw와 실제 결과가 같아야 한다. CPU 산술 검사로
동일 noise·자기 평균 기준 entropy/density 유지·g=1 raw 일치를 확인했고 구문 검사도 통과했다.
**이 새 probe의 GPU/Waymax 실행은 아직 미실행**이다.

실행기 `run_mean_conditioning.py` PID2341308은 현재 WAITING이며2,000회 학습·저장 감사·
전체16평가·Q/coverage 검사가 끝나면1,000→2,000모델을 순서대로 검사한다.
`mean-probe-status.json`·`mean-probe-launch.json`에서 확인한다. 이 검사도 전체 주행 평가나
새 평균 구조의 학습 성능으로 취급하지 않는다.

그 뒤 같은 원래 두 train 상태 KEEP77ae… tick0/FOLLOWbaac… tick4에서2,000모델의 평균
첫 행동→원래 확률 정책→실제 종료의4seed(88019~88022) soft return도 재측정한다.
1,000모델과 같은 입력 fingerprint·seed·첫 entropy 제외·할인율0.99를 대조하고 trace를
다시 계산한다. 앞선 상태의 미래 실패 비용 적합이 나아졌는지 확인하기 위한 작은 MC
검사이며 정확한 기대 Q가 아니다. `run_soft_return_2000.py` PID2342077,
`soft-return-status.json` WAITING, `soft-return-launch.json`에 식별자를 보존했다.
두 후속 검사는 GPU 작업이 겹치지 않게 순차 실행하며 학습을 자동으로 더 늘리지 않는다.
현재 실행 코드는3180615와 동일하며 추가 로컬 진단/기록은 HEADcf3dbca 이후 미커밋이다.

## 20:20 KST 총2,000회 주행 회귀 확정과 종료 Q 적합 개선의 분리

추가1,000개 실제 전이/update와 전체 저장 감사, 개발8장면×KEEP/FOLLOW16회 평가를
완료했다. 모델 SHA `1b9655222ee7c8f435e9cee9b5e23e9bc3beeb6b81d8c3af55e7e329716443be`,
replay SHA `393f520c8452590bfbfa65d712aa911cc9d9344d76c51418fdd54d7728291f2b`,
평가 원본 SHA `76f2d4eb68946d880a7e1122db5868b854b538063bbaf41c996a1fdf77b11cb0`.
실행 코드3180615·core/계약/manifest는 직전 비교와 같다. seed0 learner를 이어받고
수집 RNG만1000으로 재시작한 동일 실험이며 새로운 seed 검증이 아니다.

| 정책 | 목표/충돌 | tick | 거절 | 제동 | post-step 정지 | 평균 보상 |
|---|---|---:|---:|---:|---:|---:|
| predecessor1,000 | 2/16, 14/16 | 908 | 340(37.44%) | 322(35.46%) | 424(46.70%) | -17.75719 |
| 같은 learner2,000 | 0/16, 16/16 | 814 | 388(47.67%) | 364(44.72%) | 436(53.56%) | -20.32969 |

`review_evaluation2000.py`를 실제 실행해 초기 평균0/1,000/2,000의48개 원본 episode를
다시 계산하고 초기 frame·지도·seed의 동일성을 확인했다. 고유 장면은 여전히8개다.
`evaluation2000-review.json`에 원본 hash·개별 episode·속도 구분·paired 차이를 보존했다.
기존 빠른 장면8b4c1e7e09f5f4c3의 KEEP/FOLLOW 성공2개를 모두 잃었고 새로운 성공은 없다.
KEEP/FOLLOW는 각각tick21/20부터 연속 거절,23/22부터 제동,45부터 정지해47에 충돌했다.
16충돌 모두 정지 상태 뒤쪽 접촉 기하를 기록했으며 책임/유일 원인 판정은 아니다.

추가1,000개는 main683개(통과401·양의 보상37·pre-step 정지514), branch317개(통과86·
양의 보상85·정지93)다. 합계 정지607/1,000으로 이전455/1,000보다 늘었다.
`completed-collection-review.json`에 구간과 원본 hash를 보존했다. 실패 직전 분기만
늘려도 유용한 출발/주행 경험이 충분해지는 것은 아니었다.

반면 고정 warmup severe-penalty44개의 저장 Q MAE는1,000회5.40649에서2,000회0.92579로
감소했다. 같은 두 train terminal 반례의 현재 평균 행동 Q 오차도6.10726/4.13873에서
1.36683/0.87655로 줄었으며 실제 결과는 여전히 EMERGENCY→COLLISION이다. 이 수치는
terminal 훈련자료 적합이지 앞선 상태의 미래 비용이나 주행 성공 개선을 뜻하지 않는다.
원본 `q2000.json`과 실제 환경 재실행 결과에 근거한다. 동일 조건의 무조건 추가 연장은
하지 않고 후속 branch Q·평균 조건화·실제 종료 soft-return 검사를 마친 뒤 원인을 분리한다.

실제 프레임으로 `review/fast-1000-vs-2000.gif`와 PNG를 생성했다. PNG를 열어 도로·차량·
제안/실행 궤적·속도·거절·제동·종료 후 held 표기를 확인했다. 동일 최초 상태의1,000/2,000
SAC 비교이며 Rule-based 비교가 아니다. GIF는25frame,1440×900,7.2초 주행 시간이다.
GIF SHA `62b45ef25737a0f0fcc8fbd7c65223dd584146637b9b92fd60c88ee3e415aa89`.
원시 자료/진단/GIF는 모두 로컬 `outputs/sac_predecessor_continuation_20261119/`에 있다.

20:20 관측 상태는 main pipeline의 branch-q2000 진행, 평균 조건화 실행기2341308과
soft-return 실행기2342077은 순차 대기다. 후속 결과는 미확정이며 보상·안전검사·PD·
원래 Gaussian·BC/teacher0은 그대로다. 안정적인 정책·다른 seed·동일 조건 Rule-based
최종 비교·독립 평가는 미완료이고 목표는 계속 active다.

## 20:28 KST 평균 조건화 실제56step 완료: 변경 자체를 해결책으로 채택하지 않음

고정4train상태×1,000/2,000모델의 실제56step을 모두 완료했다. 최초 관측 fingerprint·
같은 noise·g=1일 때 raw와 실제 결과의 완전 일치·모델/replay 불변을 확인했다.
`review_mean_and_terminal_q.py`도 실제 실행해 raw 차이·27||u_last||·56개 분모·모델별
Q/coverage 식별자를 독립 대조했다. 로컬 `mean-and-terminal-q-review.json` SHA
`4cd9d8f9ce7b2908e0789ae8d277899fee3c10197467114076a77b058eb49f86`.

정지 두 상태에서 원래 평균과4표본은 두 모델 모두30점 전체 정렬 후 속도≈0이었다.
1,000모델의 mean/g는 전체 정렬이 없어지고 두 평균 모두 검사 통과했지만 post-step 속도는
KEEP0.00006799m/s/FOLLOW0.00002965m/s에 불과했다. 같은 noise4표본씩도 통과했으나
실용적인 출발이나 주행 성능으로 볼 수 없다. 2,000모델의 mean/g는 두 평균과8표본 모두
curvature/curvature_rate 위반으로 거절돼 실제 속도는 그대로≈0이었다.

가상 평균은 원래 분포에서1,000모델 약92~131 sigma RMS,2,000모델 약202 sigma RMS
떨어져 있다. 이것은 동일한 실제 상태·잡음에서 평균 조건화의 제어 효과를 분리한
out-of-distribution 개입이다. 새로운 학습 구조의 성능/Q 검증이나 채택 근거로 바꾸지 않는다.
원래 정지 정렬·물리 검사·PD·sigma는 수정하지 않았고 가상 행동을 학습에 넣지 않았다.

수집한 terminal 두 상태×저장/평균/4표본,12actual1step도 모두 충돌이었다.2,000모델의
현재 평균 Q 오차는KEEP-0.20471/FOLLOW-0.38434로,1,000모델+3.28041/+4.32534보다
실제 terminal 보상에 가까워졌다. 반면 앞선 KEEP상태의 완료된4개 확률 종료 표본에서는
Q=-0.35950, soft-return평균=-14.14479, 차이13.78529가 남았다. FOLLOW4표본은 아직 진행
중이므로2상태 전체 완료로 표현하지 않는다.

다음 수집 후보는 같은 실패 직전 roots에서 최대8step 실제 후속 상태를 round-robin으로
수집하는 방식이다. 기존6roots×8개 독립1step과 새6roots×1경로×최대8step을 비교해
동일한 실패당 최대48전이·전체 학습 예산을 유지한다. 실제 조기 종료 시 해당 경로를 끝내고
남은 예산은 원래 main 수집에 사용한다. 길이/예산 한계에서 가짜 terminal이나 truncation을
추가하지 않으며 모든 실제 거절·실패 전이도 원래 replay에 넣는다. 보상·검사·PD·원래
Gaussian·uniform replay·전이당update1·BC/teacher0은 유지한다. 효과는 아직 미검증이다.

현재 미커밋 `sac_failure_rollouts.py`와 `sac_branch_reconstruction.py`에 실제 후속 상태
연결·종료·예산·계보 복원 helper를 준비했고 새10pytest를 CPU로 실행해 통과했다.
기존 실행 중인 MC 진단이 사용하는 파일은 변경하지 않았다. **학습 collector 연결·저장
감사 확장·실제 새 수집/학습·주행 검증은 아직 미실행**이며 현 모델의 연장이 아니다.

## 20:34 KST 앞선 상태 비용 검사 완료와 실제 후속 수집100회 시작

2,000모델의 두 고정train상태×4seed soft-return 검사가 모두 끝났다. 원본8trace의
hash·tick·첫 평균 행동의 entropy 제외·실제 종료·할인합을 독립 재계산했고,1,000모델과
최초 관측 fingerprint·seed를 대조했다. 원본 보고서 SHA
`df9e75030f8d6850dc234260b939ba5ad446432a6ab4c4c13aae596232f2b092`.

| 상태 | 2,000모델 Q | 실제 soft return 평균±표본SD | Q−표본평균 |
|---|---:|---:|---:|
| KEEP77ae… tick0 | -0.35950 | -14.14479±1.09239 | 13.78529 |
| FOLLOWbaac… tick4 | -0.16899 | -10.74943±0.38338 | 10.58044 |

8/8 모두 충돌했으며 고유2상태의 반복이다.1,000모델 대비 paired return 차이 평균은
KEEP-0.75024/FOLLOW+0.13363이고, 작은 표본에서 전체 주행 개선이나 정확한 기대 Q로
해석하지 않는다. terminal 적합만 좋아지고 앞선 상태의 미래 실패 비용 차이는 남았다.
기존2,000모델은 추가 연장하지 않는다. 평균/g도 단순 제어 개입 결과만 보존하고 채택하지 않는다.

기존 진단 runner와 child가 모두 종료된 것을 확인한 뒤 후속 상태 수집을 연결했다.
`--failure-branch-rollout-horizon 8 --failure-branch-samples 1` 옵션은 같은 최대6roots를
한 경로씩 최대8step 이어간다. 원래 horizon1/독립표본 경로는 그대로 유지한다. 새 helper는
실제 collector 결과와 실제 수집 ID를 받고, 경로별 직전 수집 ID·root fingerprint·현재
관측 fingerprint를 기록한다. 실제 terminal 경로는 종료하고 길이/전체 예산 종료를
가짜 terminal로 바꾸지 않는다. 한 실패당 상한48전이·전체100update 예산은 유지한다.

`audit_sac_branch_replay.py`는 root 관측을 원래 main과, 이어지는 관측/field/item mask를
직전 실제 next-state와 정확히 비교한다. `sac_branch_reconstruction.py`는 후속 Q 진단에서
같은 tick의 main 행동으로 대체하지 않고 실제 branch raw 행동 이력을 복원한다.
continuation 검사는 수집 길이/구현 hash 변경을 거절하며 과거1step 기록과 호환된다.

새/기존 관련26pytest 통과(0.36초), 구문 검사·diff whitespace 검사 통과. 새 감사기로
기존2,000개 실제 replay를 다시 검사해1347main/653branch·계보·raw·보상·종료·train-only·
teacher0의 이전 결과가 유지됨을 확인했다. 출력은 기존 continuation폴더의
`audit2000-rollout-compatible.json`이다. 새 실제 후속 수집 검증은 다음 pilot에서 확인한다.

20:33:57 KST 로컬 `outputs/sac_failure_rollout_collection_20261119/run_pilot.py`를 실제
시작했다. runner PID2355358·train PID2355359, `launch.json`·`pilot-status.json`·`sources/`에
명령·동결 source hash·사본을 보존했다. seed0 fresh learner+원래 자체warmup5,000개에서
100실제전이/update→저장 감사→같은2개개발장면×KEEP/FOLLOW4회 평가 순서다.
현재 actor/Q 초기화·mean 배율·Gaussian·보상·안전검사·PD·uniform replay·BC/teacher0은
직전1,000 비교와 같으며 실패2,000모델의 가중치는 이어받지 않는다.
새100회의 결과는 아직 미확정이며1,000회로 자동 확대하지 않는다.

## 20:50 KST 실제 후속 수집100회 실패·동일 예산 대조와 수집량 통제 보완

첫 실제 후속 수집100회가 완료됐다. main59/branch41,6roots에서후속35전이를 포함하며
실제terminal1/nonterminal40,branch거절40/통과1이었다. train 고유2장면·KEEP만 포함했다.
원본 관측/field/item mask와 실제 후속next-state 연결, raw/보상/종료/계보/train-only/teacher0
저장 감사를 완료했다. 모델SHA `7871bd972f892f590525eec4bc5bb7d3d95f077a8d68852fa348174736c5ffbf`,
replaySHA `abd0dde127816300a6401e7d811cb6f43f1ac89b7de14f06482dc6e19f40c4e9`.

기존 predecessor100은 과거에 저장 검사만 했으므로 같은2개개발장면×KEEP/FOLLOW4회의
주행을 이번에 새로 평가했다. 각 모델의 원본 frame에서 지표를 재계산하고 최초frame/지도/
seed4쌍을 대조했다. 초기평균0도 같은4쌍만 골라 비교했다. 고유장면은2개다.

| 정책 | 목표/충돌 | tick | 거절 | 제동 | 정지 | 평균보상 |
|---|---|---:|---:|---:|---:|---:|
| 초기평균0 | 2/4,2/4 | 216 | 58 | 46 | 22 | -10.01539 |
| 기존 독립1step100회 | 0/4,4/4 | 224 | 106 | 94 | 44 | -20.25450 |
| 후속 경로100회 | 0/4,4/4 | 224 | 108 | 96 | 44 | -20.25984 |

새 후보의 빠른 KEEP/FOLLOW는 각76tick, 저속은 각36tick에 충돌했다. 두100회 모델은
모두 초기 성공을 잃었고 새 방식의 주행 개선은 없다. 같은 모델 설정·자체 warmup·seed·
100update를 확인했고 첫 분기 전29main의 검사·제어 source·보상은 같았다. raw 최대 차이는
1.45e-8로 작지만 bit-identical이라고 주장하지 않는다. 저장 severe44개의 Q MAE는
기존18.38788/새18.48035로 둘 다 크다. 이 초기 결과를 수렴한 정책으로 취급하지 않는다.

추가로6roots의 가장 깊은 분기 이력을 각각 실제 Waymax로 다시 실행했다. 원래 main
prefix→실제 branch raw 이력과 마지막 행동까지79actualstep을 재현해41개 모든 분기 전이의
관측·mask·보상·검사·제어 source·종료가 저장 자료와 일치했다. 최초 root 상태로 매번 잘못
복귀하거나 같은 tick의 main 행동으로 대체한 것이 아님을 확인했다. 반복 prefix를 새로운
독립 장면으로 세지 않는다.
로컬 `outputs/sac_failure_rollout_collection_20261119/`의 `paired100-review.json`,
`actual-suffix-reconstruction.json`, `fixed-q100.json`, `review-status.json`에 근거가 있으며
pilot/review 실행기와 child는 종료됐다.

자가검토에서 첫 후속 방식은 terminal root 경험이 기존8개→1개, main 수집이52→59로
달라지는 추가 분포 변화를 확인했다. 후속 상태 효과를 더 잘 분리하려고 root당 실제8전이를
유지하는 `--failure-branch-restart-terminal` 비교 옵션을 추가했다. nonterminal이면 실제
다음 상태를 계속 밟고, 실제 종료되면 그 경로를 끝낸 뒤 남은 quota만큼 원래 불변 root에서
새 경로를 시작한다. 매 tick 최신 정책으로 새 경로를 제안하고 원래 SAC 업데이트를 한다.
종료 이후를 이어붙이지 않고 새 branch_sample과 별도 이력을 기록한다. 실패당 최대48실제
전이, 원래 정책 표본·보상·검사·PD·uniform replay·전이당update1·BC/teacher0은 그대로다.

`sac_failure_rollout_budget.py`와 CLI·저장 감사·continuation 호환 검사를 연결했다. 새3개를
포함한 관련29pytest 통과(0.38초), 구문/diff 검사 통과. 새 감사기로 앞선100개 replay의
호환성도 다시 확인했다. 실제 새 수집의 개선은 아직 미검증이다.20:49:05 KST 새 fresh100
pilot을 `outputs/sac_failure_rollout_quota_20261119/`에서 시작했다(runner2364007/train2364008).
`launch.json`·`pilot-status.json`·`sources/`에 명령/소스를 보존했다. 같은100update→저장 검사→
같은4주행이며1,000회 자동 확대는 없다. 현재 코드는HEADa6092f5 이후 미커밋이다.

20:53 KST 추가 확인: quota pilot의100실제전이/update와 저장 감사를 완료했다. 실제로
main52/branch48,6roots 각각8전이, terminal8/nonterminal40으로 기존 독립1step100회와
수집 수를 맞췄다. 후속 상태35개, 분기 거절47/통과1이다. 원본/후속 관측·mask·실제
종료 후 새 경로 구분·raw·보상·train-only·teacher0 검증을 통과했다. 새100개의 데이터는
여전히 KEEP2장면이며 FOLLOW 데이터 확보로 해석하지 않는다. 현재4주행 평가 중이다.
모델 SHA `a92376abdc651584a9f09fa3b26056af9d24f1c2eebcd230ad4e3f2f5114f11e`,
replay SHA `a943837cc5e75d54b22577aec5a7eecbe34135e80418cde421648c66fab71505`.
같은 폴더의 `rollout100/report.json`과 `audit100.json`에서 확인했다.

## 21:03 KST 수집량 통제100회 검토 완료와 같은1,000회 예산 비교 시작

quota100의4주행도 모두 충돌했다. 빠른KEEP/FOLLOW 각76tick·저속 각36tick, 총224tick에서
거절108·제동96·정지44·평균보상-20.25984068이다. 기존 독립1step100회도4/4충돌이지만
거절106·제동94였으므로 개선으로 판정하지 않는다. 초기 평균0은 같은4회에서2회 성공이다.
`review_paired100.py`로 같은 최초frame/지도/seed와 전체frame 지표를 대조했고, main52/
branch48·terminal8의 같은 수집 수와 모델 설정·seed·warmup·100update도 확인했다.

6roots의13개 실제 경로를 독립 재실행했다. 반복 main prefix를 포함한282actualstep으로
48개 분기 전부의 관측/mask·보상·검사·제어source·실제 종료가 재현됐다. 즉시 종료 root의
8개 새 경로와 다른5roots의 연속 경로를 구분했다. 고유 상태는6개이며13개 독립 장면이 아니다.
새 Q 감사에서도 저장 severe44개 MAE는18.39707(기존100회18.38788)로 컸다. 두 고정
terminal 현재 평균 Q는-1.70735/-1.65662, 실제 보상은-20.014/-20.017이었다.100회 결과는
주행 실패와 큰 Q 오차이며 성능 gate 통과로 바꾸지 않는다.

로컬 quota폴더의 `paired100-review.json` SHA
`b9ac2a89c23658dcd781a32a40af21d95e2feab21ed50f3e025f22b72649e07f`,
`actual-suffix-reconstruction.json` SHA
`7b01872aa41080e5e3d251be786600ba2be5c496284ca460ac6f4c146168d9cc`,
`fixed-q100.json` SHA `ff375fd38981149ce058d51d92a4cc9d91470eae4a64dc1e1f7323777385ada4`.
pilot/review 실행과 모든 child가 종료된 뒤 검토했다.

같은 설정의 기존 독립1step1,000회 모델은 고정4주행에서2회 성공이었고,100회보다
종료 Q 적합도 나았다. 따라서 양쪽 모두 크게 미학습된100회 결과만으로 수집 효과를
단정하지 않고 **기존과 같은 fresh1,000회 예산의 비교를 한 차례 실행**하기로 했다.
100회 실패를 보존한 별도 제한 진단이며 기본 채택·전체 평가 확대가 아니다. 기존의
1,000회 모델과 동일 설정·초기 자체warmup을 사용하고 root당 실제 수집 quota만 후속 경로로
배분한다. 실제 종료 결과를 강제해 개수를 맞추지는 않는다.

21:02:05 KST `run_reviewed1000.py`를 시작했고 runner2371489/train2371490이 살아 있다.
21:03에는10update를 확인했다. 원본100회 learner나 실패2,000회 learner를 이어받지 않는
fresh seed0 실행이다. `review-decision1000.json`에 실패 판정·검토 근거·원본 hash,
`reviewed1000-launch.json`·`reviewed1000-status.json`·`full-sources/`에 명령/PID/동결 소스를
보존했다. 실제1,000전이→저장 감사→같은4주행→고정Q→replay 노출→새 분기Q 순서이며
이후 자동 학습 연장은 없다. source는90c15a4와 동일하다. 안정적인 정책 확보는 미완료다.


## 21:27 KST 수집량 통제1,000회 완료: 주행 지표 개선 없음

실행 코드90c15a4와 모델/보상/검사/PD/원래 Gaussian/uniform replay/전이당update1/
BC 및 teacher replay0 조건을 유지했다. seed0 fresh learner + 원래 자기 warmup5,000에서
실제1,000회 수집·업데이트와 저장 감사, 같은 개발2장면×KEEP/FOLLOW4주행을 완료했다.
새 수집은42개 root에서336개 분기 경험을 만들었고237개는 실제 후속 상태였다.
전체는 main664/branch336, 고유 train10장면이다. root당8개 예산은 같지만 실제 종료를
강제하지 않으므로 분기 terminal은 기존56→66개로 늘었다. 그 밖의270개는 실제
nonterminal로 bootstrap을 보존했다. 관측/mask/계보/행동/보상/종료/train-only/teacher0
저장 대조를 통과했다.100회 pilot의 실제 경로 재현과1,000회 저장 감사를 혼동하지 않는다.

| 같은1,000회 수집 | main 통과/양의 보상/사전 정지 | branch 통과/양의 보상/사전 정지 |
| --- | --- | --- |
| 기존 독립1step | 333/100/359 | 136/81/96 |
| 새 후속 수집량 통제 | 336/100/362 | 107/54/102 |

전이 구성은 주행 성능이 아니다. 원본 trace의 모든 collection ID와 update 순서, 설정,
warmup/seed/actor/Q 하이퍼파라미터의 동일성을 `review_collection1000.py`로 확인했다.
시뮬레이션 방문 경로나 행동이 bit-identical이라는 주장은 하지 않는다.

| 정책 | 목표/충돌 | tick | 거절 | 제동 | post-step 정지 | 평균 보상 |
| --- | --- | --- | --- | --- | --- | --- |
| 초기 평균0 | 2/4·2/4 | 216 | 58 | 46 | 22 | -10.01539017 |
| 기존 독립1step1,000 | 2/4·2/4 | 236 | 44 | 36 | 12 | -9.98113283 |
| 새 후속 수집1,000 | 2/4·2/4 | 236 | 44 | 36 | 12 | -9.98123323 |

`review_evaluation1000.py`를 실행해12개 원본 episode의 frame 지표와 최초 frame/map/seed
4쌍을 다시 대조했다. 고유 장면은2개다. 양쪽1,000회 모델은 고속 KEEP/FOLLOW에서72tick
목표 도달, 저속 KEEP/FOLLOW에서46tick 충돌이다. 충돌2회는 정지 뒤 뒤쪽 접촉 기하를
기록했다. 책임/단일 원인 판정이 아니다. 초기 제안 경로의 두 학습 모델 간 최대점 차이는
각각 약0.389~0.399mm로 작았다. **새로운 성공이나 기존 모델 대비 거절·제동·정지 개선은
확인되지 않았다.** 초기 평균0 대비 비용 지표 개선은 기존1,000회에서도 있던 결과이며
새 수집의 성과로 다시 계산하지 않는다. 추가 전체 개발 평가/다른 seed 확대는 미실행이다.

고정 자체 warmup의 severe44개 저장 행동 Q MAE는 기존5.40649→새4.89673으로 줄었다.
두 고정 terminal의 현재 평균 Q 오차는 KEEP5.98879/FOLLOW3.32578이고 실제 결과는
여전히 EMERGENCY→COLLISION이다. 종료 적합의 소폭 개선과 주행 개선 부재를 함께
보존하며 기본 채택이나 안정적인 정책 확보로 분류하지 않는다.

로컬 `outputs/sac_failure_rollout_quota_20261119/` 증거:

- 모델 `b32586d181f072a69baaa97539069ba056f0613e25594460dff7939b906e2708`.
- replay `3246b1f9d3f5dc23ee680b5c7196878ebc9f7ec70ccf66ac6b27ea437e36c7fd`.
- 학습 report `4537c814d2fddd5675786764774949731ba8c3ba56a356515872b72e9b459e33`.
- `audit1000.json` SHA `688194dc8c78aa8bbcab5bcdf9e3bb8d29af7d2dfc064858a9fd1ed5070eff77`.
- `collection1000-review.json` SHA `1475053927ae23eee6db2e406f6c72031ee10c5f4650f921e4e84b05500ee48a`.
- `evaluation1000/report.json` SHA `d0b78c354d224d4d297f05f26d5dd055cedd001bc143a14741dc2f6bd54fb729`.
- `evaluation1000-review.json` SHA `bede234511561aafb8c8e837a4ba288cfeab6490f0127bed47ca40ca97662285`.
- `q1000.json` SHA `faa39ebf878dde7bc8b0d7229514e147d930ce30ccef11befb8097395d207a73`.

21:27 현재 실행기2371489의 coverage→수집 상태 새 행동 Q가 남아 있다. 학습2371490은
종료됐다. 이후 같은 두 train 상태에서4개 paired seed씩 frozen 정책의 실제 종료까지
soft return을 검사한다. 실행기2376459는 WAITING이며 `soft-return-status.json`에
소스/명령/PID를 남긴다. `run_soft_return_1000.py` SHA는
`2d6f27dcb21f282a0ecd88e86583b6244a453e7c10a709dfeebe8fb7a53e5c4a`다.

그 뒤 실행기2379547가 `profile_saved_bellman_chain.py`를 실행한다. 과거 실제 두 train
충돌 경로에서 terminal까지 남은 tick0/1/3/7/15와 최초 상태를 Q값을 보기 전에 고른다.
기존1,000/2,000 및 새1,000 모델을 같은 저장 행동/관측에서 비교하며 nonterminal마다
동일 seed32개로 다음 원래 정책 행동을 표본화한다. 저장된 실제 보상에 대한 Bellman
목표와 online/Polyak target의 차이를 기록한다. 현재 평균 Q는 별도 기술값이며 저장
행동의 보상으로 평균 행동의 TD 오차를 계산하지 않는다. 실제 soft return 검사를
대체하는 기대가치 정답이나 새 학습은 아니다. `chain-profile-status.json` WAITING,
실행기 SHA `4ee0d5a61c787502056cfc8d80b40c3b21e60fb2068fe93723e0561d35b565bf`.

두 검사는 구문 검사 및 실행기 대기 상태까지 확인했고 GPU 결과는 아직 미확정이다.
원본/후보 모델과 replay를 읽기만 하고 GPU 작업은 순차 실행한다.1,000회 이후 학습은
자동 연장하지 않는다. 다음 예산/수정은 실제 미래 비용과 Q 목표 전달 결과를 검토한 뒤
정한다. 안정적인 KEEP/FOLLOW·새 seed·동일 조건 Rule-based 최종 비교·독립 test 확보
목표는 계속 active이며, 주행 성능 미달을 기록 완료와 혼동하지 않는다.


## 21:45 KST 1,000회 미래 비용 검사 완료와 같은2,000회 시점 비교 시작

직전 후속 수집1,000회의 저장·4주행·고정 Q·coverage·수집 상태 새 행동 Q·frozen 실제
종료 보상·저장 경로 Bellman 검사를 모두 완료했다. 원래 pipeline2371489, 후속2376459와
2379547 및 child는 종료됐다. 코드90c15a4는 그대로이며 새 로컬 읽기 전용 분석기를 썼다.

수집한 첫 terminal 상태의 현재 평균 Q 오차는 기존→새 KEEP3.28041→2.50442,
FOLLOW4.32534→4.59269로 **개선이 섞여 있다**. 각각 저장/평균/4표본6개, 두 상태12개
실제 행동은 모두 충돌했다. replay draw256,000개 중 원래 warmup233,361(91.16%),
새 main13,409/branch9,230이다. 새 online22개는 아직 추출되지 않았고21개는 마지막100개에
있다. 분기336개 중 현재 정책95% 타원체 안의 기록 행동 비율은93.45%다. 이 조건부
분포 겹침은 충분한 학습·정책 성공의 증거가 아니다.

같은 두 실제 train 상태에서 자기 평균 첫 행동 뒤 원래 확률 정책으로 실제 종료까지
seed88019~88022를 실행했다. 이전과 같은 최초 관측/seed, 첫 entropy 제외, 원본8trace의
hash·tick·실제 종료·할인합을 별도 `review_value1000.py`로 다시 확인했다.

| 새1,000회 상태 | 현재 평균 첫 행동 Q | 실제 soft return 평균±표본SD | Q−표본평균 |
| --- | --- | --- | --- |
| KEEP77ae… tick0 | -0.48678 | -13.39445±2.05693 | 12.90768 |
| FOLLOWbaac… tick4 | -0.23445 | -10.97692±0.59091 | 10.74247 |

고유2상태의 반복8/8충돌이며 독립test/정확한 기대 Q가 아니다. 기존1,000 대비 paired
return 평균차이는+0.00009094/-0.09386039로 주행 개선 근거가 없다. 기존 Q−표본평균은
13.04797/10.66209였으므로 FOLLOW는 이 검사에서도 개선되지 않았다.

별도 저장 행동 검사는 과거 physical1,000의 첫 KEEP/FOLLOW main 충돌 경로에서
terminal까지 남은0/1/3/7/15tick과 최초 상태, 총12개 고정 상태를 Q를 보기 전에 선택했다.
기존 independent1,000/2,000과 새 quota1,000을 같은 저장 관측/행동으로 비교하고,
각 nonterminal에서 같은32seed로 다음 현재 정책 표본의 online/target 값을 대조했다.
새 KEEP tick0의 저장 행동 Q/학습목표는-0.560/-0.173, FOLLOW tick0은-0.426/-0.191이다.
두 상태의 다음 online soft value−target soft value는약-0.047/-0.038로 모두 작았다.
반면 terminal Q는약-17.511/-15.431로 큰 종료 비용을 일부 배웠다. FOLLOW tick0의
저장 행동 검사와 위 MC tick4의 현재 평균 행동 검사는 서로 다른 표본이다.

종료 적합과 앞선 상태의 낙관적인 학습 목표가 분리되어 있다. 현재 online을 target으로
즉시 바꾸면 해결된다는 근거는 없으며, 수집 분포·함수 근사·여러 단계 비용 전달을
이 검사 하나로 단일 원인으로 단정하지 않는다. 기존2,000 모델도 최초 상태 Q가 작지만
15tick 앞의 Q는약-15~-17로 내려가 있었다.

보조적인 이상화 계산에서는 매 update 모든 online Q가 목표를 정확히 맞추는 결정적
표 형식 chain, 중간보상0·종료-20·gamma0.99·tau0.005를 사용했다. 28step 앞 비용의
반영 비율은2,000회약0.000206%/5,000회약29.91%다. 별도 binomial CDF 식으로30개
snapshot cell을 대조해 최대절대오차4.93e-12였다. 실제 신경망은 이미15step 앞 비용을
일반화하므로 **이 계산은 실제 수렴 시점·필요 예산·성능 예측이 아니다**. 실제 target
갱신율이나 보상/알고리즘을 변경하지 않았다.

로컬 `outputs/sac_failure_rollout_quota_20261119/`:

- `soft-returns1000/report.json` SHA `4879ebba76009f65cfc8565a2ae6e1b73e067f92d197d77d0f1f10327e1a3d4b`.
- `saved-bellman-chain.json` SHA `3ada0ca376793dff119df1a5cec20668735560edb65448ea6b0aae72cbb42bcf`.
- `value1000-review.json` SHA `7b7737fa8d58a80b70af9b2a9380a38eb6d0be9931adb2318eeeb02dc4738664`.
- `idealized-target-lag.json`와 `idealized-target-lag-verification.json`은 이상화 계산/별도 수식 대조다.

새 후속 수집1,000의 고정4주행은 초기/기존1,000의2성공을 유지했지만 개선은 없다.
실제 Q/목표의 미학습과 초기 경험의 높은 batch 점유를 근거로, **같은 모델·수집 조건에서
추가1,000실제전이/update, 총2,000회까지만** 이어 학습한다. 기존 independent2,000은
0/16으로 회귀했던 실패를 유지하고 같은 시점의 고정4주행에서 비교한다. 안정적인 정책
기본 채택이나 전체개발/seed 확대가 아니며, 정지 평균 조건화 문제가 해결된다고 가정하지
않는다. 단순 target lag나 추가 예산을 확정 원인/해결책으로 취급하지 않는다.

먼저21:41:03 KST actual1001 smoke를 실행했다. 원본 learner 바이트 동일 복원을 확인하고
기존6,000전이의17배열,1,000개 tick/update 파일 prefix를 원본과 정확히 대조했다.
새 actual 전이는1개이고 총 main665/branch336,후속237·terminal66의 저장 감사를 통과했다.
smoke learner는 새 비교에 재사용하지 않고 원래1,000모델 b32586d…에서 다시 시작한다.

새 로컬 `outputs/sac_failure_rollout_quota_continuation_20261119/`:
`resume1001-review.json`에 실제 복원/저장 대조, `review-decision.json`에 근거와 증거hash,
`launch.json`·`status.json`·`sources/`에 명령/PID/동결 source가 있다.
21:43:15 KST runner2390993/train2390994를 시작했다. learner/optimizer/target/learner RNG와
replay RNG를 유지하며, 환경/장면/정책 수집 RNG만1000으로 명시 재시작한다. 가짜 종료를
넣지 않고 exact simulator resume라고 주장하지 않는다. 보상·검사·PD·원래 Gaussian·
actor/Q·수집 조건·uniform replay·전이당update1·BC/teacher0을 유지한다.

runner SHA `03d5122e8d2a2c6f54b6cd1491bcaf0fd0a43165cf0a2aa608e1142043bdb480`,
결정 SHA `62b077b74793bfa6c87ef62b8b7bcf7c585781cea9112f9d5dc55be1ea5ea981`.
학습→저장 감사→같은4주행→고정Q→coverage→새분기Q 뒤, runner2392865가 실제종료8회,
runner2393214가 같은12상태의 quota1,000/independent2,000/quota2,000 Bellman 검사를
순차 실행한다. 후속 두 실행은 현재 WAITING, 구문/대기 상태까지만 확인했다.
각 `soft-return-status.json`·`chain-profile-status.json`에 소스 hash/PID를 보존한다.
총2,000회 후 자동 학습 연장은 없고, 실제 주행과 미래 비용을 다시 검토한다.
안정적인 정책·새seed·동일조건 Rule-based 최종 비교·독립 평가 목표는 계속 active다.


## 22:11 KST 후속 수집2,000회: 같은 예산의 회귀 감소, 전체 개발 평가 대기

같은 quota1,000 learner에서 추가1,000 actual transition/update를 완료했다. 학습2390994와
기본 검증 실행기2390993은 종료됐다. 저장 감사는 총2,000개·main1,349/branch651,
고유 train19장면·84roots·실제 후속456개·branch terminal129/nonterminal522를 확인했다.
새1,000개는 main685(통과409·양의보상39·사전정지514), branch315(통과56·양의보상56·
사전정지102)다. 같은 예산의 독립 수집은 main683(401/37/514), branch317(86/85/93)이었다.
후속 실패 경험 증가와 유효/양의 보상 분기 경험 감소를 함께 기록한다.

고정2개 개발 장면×KEEP/FOLLOW를 같은 초기 frame/map/seed에서 비교했다.
`review_evaluation2000.py`로 초기 평균0·기존 독립1,000·quota1,000·기존 독립2,000·
quota2,000의 원본20episode를 다시 계산했다. 고유 장면은2개이며20개 독립 장면이 아니다.

| 정책 | 목표/충돌 | tick | 거절 | 제동 | post-step 정지 | 평균 보상 |
| --- | --- | --- | --- | --- | --- | --- |
| quota1,000 | 2/4·2/4 | 236 | 44 | 36 | 12 | -9.98123323 |
| 기존 독립2,000 | 0/4·4/4 | 173 | 91 | 79 | 26 | -20.25300687 |
| quota2,000 | 2/4·2/4 | 236 | 48 | 37 | 13 | -9.99233126 |

고속 KEEP/FOLLOW는 새2,000에서도72tick 목표 도달로 초기 성공을 유지했다. 기존 독립
2,000이 같은 두 성공을 잃었던 회귀를 줄였다. 그러나 새1,000과 초기 평균0도2/4성공이었고,
새 성공은 없다. 저속 KEEP은46tick 충돌이며 새1,000 대비 거절11→14·제동10→11·정지6→7,
저속 FOLLOW도46tick 충돌이다. **같은 예산 대조의 개선과 이전 시점 대비 일부 비용 악화를
함께 보존한다. 안정적인 정책·Rule-based 우위·기본 채택으로 분류하지 않는다.**

종료 severe44개 저장 Q MAE는 기존 독립2,000의0.92579보다 낮은0.74283이었다.
고정 warmup terminal의 현재 평균 실제 Q 오차는 KEEP0.57071/FOLLOW0.71981이며
실제 결과는 여전히 충돌이다. 수집한 첫 terminal에서 평균 Q 오차는-0.33674/-0.23194,
저장/평균/4표본×두 상태12actual 행동도 모두 충돌했다. 전체512,000draw 중 원래
warmup430,763/online81,237, 새 online25개는 미추출이었다. 단기 종료 적합과 앞선 상태의
실제 미래 비용 적합은 구분하며 후자는 현재 재검증 중이다.

로컬 `outputs/sac_failure_rollout_quota_continuation_20261119/`:

- 모델 `7c898963fab8aa1b980199398a4dac6a60cfc01a001c428933cfa48eb29c7347`.
- replay `3713625f8979b00d42c460ee018a1407f0eb1b3370d5c0a910d46668318e8724`.
- 학습 report `d85654f5772fb1c010961ee581a8fe1290b8356f6834823f5b9fbd5519c8168c`.
- `audit2000.json` SHA `0f2a13ddcbd87c546d335dc1e0e314e1f3ea64909757f1899087e97bfbc34761`.
- `collection2000-review.json` SHA `23b7487a2f9f001d6090e7bda3f37adebe2a068475c49ec9df978042278e1792`.
- `evaluation2000-review.json` SHA `8274fd2ab04d990ed3bf5df1f4236bca353c14c2bc71ce46da77d97a1eb9aac7`.
- `q2000.json` SHA `77d3ec5274f29f2e3f76dc411215a3028ebf9353eec60d5199ed330f5b0dd0aa`.

같은 고속 KEEP의 기존/새2,000 기록으로 `review/independent-vs-continuation-2000.gif`와
PNG를 만들었다. PNG를 실제로 열어 도로·차량·제안/실제 궤적·거절·제동·held 표시를
확인했고 GIF25frame/1440×900/물리7.2초+최종1.5초도 읽어 확인했다. 기존은4.7초 충돌 뒤
고정 표시, 새 모델은7.2초 목표 도달이다. SAC 대 SAC이며 새 주행/Rule-based 비교가 아니다.
GIF SHA `f660e16829f3c13a17d429b815bb59c18c8604f55e719bf7bb22a1a8571e80e5`.
source hash와 renderer hash는 같은 `review/render.json`에 보존했다.

현재 runner2392865/child2404077가 실제 종료까지의 보상을 재검증 중이며, runner2393214의
12상태 Q 경로 비교가 뒤따른다. 새2,000이 같은 예산 대조에서 두 성공을 유지했으므로
다른 개발 장면의 회귀도 확인하려고 **기존 개발8장면×2행동16회** 평가를 준비했다.
runner2401291·`full-validation-status.json`은 WAITING이다. 아래 reset 자료 준비까지
끝난 뒤 수행하여 원본 스캔과 성능 측정을 겹치지 않는다. 초기 조건16쌍 및 이번 pilot4회의
결정적 결과를 다시 확인한다. 독립 test 성능 평가나 추가 학습/새 seed 실행이 아니다.
전체 개발 및 미래 비용 결과를 검토하기 전 총2,000회 이후 학습은 연장하지 않는다.

### 별도 평가 자료: 메타데이터 후보 봉인 완료, 실제 reset 적격성 미확정

기존 first200 shard의 저장된 screen 메타데이터만 읽어14,166개 고유 scene,
train11,320/validation1,429/test1,417을 재확인했다. 원본200파일의 크기/mtime는 과거
fingerprint와 같았다. 이 단계에서 원본 protobuf를 읽거나 정책을 실행하지 않았다.
모든 test 행은 `HELD_OUT/test_not_screened`, cached proto 없음이었다. scene ID 중복0,
기존 개발 manifest와 scene/group ID 및 **기록된** raw SHA 중복이 없음을 대조했다.
원본 바이트 SHA 재검증은 후속 로딩에서 수행한다. drive/시간구간 독립성은 여전히 미확인이다.

정책과 무관한 `sha256('heldout_reset_order_v1|scene_id')` 순서로1,417개 후보를 고정했다.
공식 Waymo test가 아니라 원래 training_20s source의 프로젝트 hash-test다.
**1,417개는 미검사 후보 수이며 평가 가능한32개 확보가 아니다.** 현재 train/validation
manifest·진행 중 실험 조건은 변경하지 않았고 학습 loader의 test 거절도 그대로다.

로컬 `outputs/sac_heldout_preparation_20261119/test-candidate-order.json` SHA
`806b7341c33fba89b3406a7107a5ea433fc9d8e307e75891f69a8f0d54947beb`.
`audit_sealed_candidates.py`의 실제 메타데이터/중복/hash split 검사는 완료했다.
`prepare_reset_inventory.py`는 같은 기존 route prefilter와 실제 environment reset만
사용해 모든 제외·데이터/reset 오류를 보존한다. KEEP/FOLLOW 각각 reset만 하고 teacher/SAC
경로 생성·제어 step·보상·성능 선택을 하지 않는다. 구문 검사만 했으며 실제 원본 로딩/reset은
아직 미실행이다. 첫200 pool에서32개가 안 나오면 부족 수를 기록하고 별도 확장을 검토한다.

실행기2397055는 현재 WAITING이다. 현 학습의 실제종료/경로Q 검사 완료 뒤 CPU로 실행한다.
예상 산출물 위치는 Storage의
`/mnt/e/Datasets/WOMD/v_1_3_1/derived/training_20s/sac_heldout_20261119/reset_first200/`다.
실제 생성/적격 수/오류는 완료 후 확인한다. 로컬 `status.json`·`launch.json`에 source hash,
후속 명령, 실행 PID와 대기 조건을 남겼다. 원본/기존 개발 자료는 수정하지 않는다.

이번 추가 작업은 실행 코드90c15a4를 유지한 로컬 분석/실험 준비이며 새 PR/병합은 없다.
안정적인 KEEP/FOLLOW, 새 seed, 같은 조건 Rule-based 최종 비교, 별도 성능 평가와32개
적격 장면 확보는 여전히 미완료다. 전체 목표는 계속 active다.

## 22:23 KST 후속 수집2,000회 미래 비용 검토 완료, 전체 개발 평가 실행

실제 종료 보상 실행기2392865와 저장 경로 Q 검사2393214는 완료 후 종료됐다.
코드90c15a4와 모델7c898963…를 유지한 읽기 전용 검사다. 새 `review_value2000.py`로
8개 원본 trace hash, 최초 평균 행동·첫 entropy 제외, 실제 terminal, gamma0.99 할인합,
기존 quota1,000과 같은 관측/seed,12개 고정 상태의 저장 행동 TD 항등식을 재확인했다.
분석기 첫 실행은 `rollout2000`이라는 잘못된 로컬 경로로 실패했고, 실제 `online2000`으로
수정한 뒤 통과했다. 학습/정책 코드나 실행 중 source에는 변경이 없다.

| 고정 train 상태 | 새2,000 평균 첫 행동 Q | 실제 soft return 평균±표본SD | Q−표본평균 |
| --- | --- | --- | --- |
| KEEP77ae… tick0 | -0.473294 | -13.061026±2.165930 | 12.587732 |
| FOLLOWbaac… tick4 | -0.281600 | -10.649584±1.000974 | 10.367984 |

두 상태 각각4seed88019~88022, 총8/8충돌이다. quota1,000 대비 같은 seed의 return 차이는
KEEP 평균+0.333425/표본SD0.431348, FOLLOW+0.327337/0.787263이다. 표본 수가 작고
충돌을 피하지 못했으므로 안정 주행이나 유의한 성능 개선으로 채택하지 않는다.
현재 자기 평균 첫 행동 이후 원래 확률 정책의 실제 return 표본이며 정확한 기대 Q가 아니다.

수집한 첫 terminal 상태 현재 평균의 정확한 Q−reward는 quota1,000→2,000에서
KEEP2.504419→-0.336740, FOLLOW4.592690→-0.231936으로 개선됐다. 그러나 초기 미래 비용
과대평가10~13점은 남았다. 저장 행동 기준 최초 KEEP Q/TD목표는-0.627577/-0.279360,
FOLLOW tick0은-0.514988/-0.235144다. 다음 online−target soft value는-0.042101/-0.057151로
작아, target 지연 하나만 고치면 된다는 증거는 없다. 저장 행동 TD 적합과 현재 평균의 실제
미래 보상 오차를 섞지 않는다. 종료15tick 앞의 저장 Q는 KEEP-13.944/FOLLOW-16.801이다.

로컬 `outputs/sac_failure_rollout_quota_continuation_20261119/`:

- `soft-returns2000/report.json` SHA `54fd7ec54304a4251749f4127b22cdbbebf4aa87ad3aaf22a469c0cbe7ab3135`.
- `saved-bellman-chain.json` SHA `60f5d3edf5f1f9c4b313872c345fcbdbef2346996e7ab4fe63813e4fe9688bf4`.
- `value2000-review.json` SHA `bee059d9ef16e3eb82a55bf1ca873b4ff445096e52ec29f05feb282753a6583d`.

### 별도 평가 reset 검사 실제 완료:3장면 확보,29장면 부족

준비 실행기2397055/child2408542는 실제 raw 로딩/초기화 뒤 종료됐다. 봉인된1,417개
후보의 원본 SHA와 ID/hash split을 대조했다. 기존 route prefilter에서1,413개 제외,
후보4장면의 KEEP/FOLLOW8회 reset 중2회거절/6회수락으로 **고유3장면·6쌍**이다.
데이터/reset 오류0이며 정책 실행0, 장면 선택에 주행 성능을 사용하지 않았다.
목표32장면에는29개가 부족하다. 약127초의 실제 처리 결과로 미실행 준비 기록을 보완한다.

Storage의 `.../sac_heldout_20261119/reset_first200/`에 report와 test_reset_manifest,
4개 도로 후보의 원본 proto를 보존했다. `review_reset_inventory.py`를 실제 실행하여
전체 후보 집합/고정 순서, 제외·reset 분모, manifest6쌍, cached proto SHA, 기존 개발
manifest와 scene/group/raw SHA 분리를 다시 확인했다. 결과는 로컬
`outputs/sac_heldout_preparation_20261119/reset-inventory-review.json`,
SHA `1e1eb16c38083c6be51ba0a08630af476ae099f333a6186b90781112c32ae457`이다.
공식training 원본의 프로젝트 hash-test이며 drive/시간구간 독립성은 미확인이다.
32개로 부풀리거나 실패 장면/검사 기준을 제외하지 않는다. 향후 성능과 무관한 추가 원본
범위 확장이 필요하며, 현재 학습 loader의 test 거절과 train/validation 자료는 그대로다.

후속 전체 개발 평가 runner2401291/child2410243은 실제 RUNNING/EVALUATING이다.
기존8개 장면×KEEP/FOLLOW16회이며 추가 학습/held-out 성능 평가가 아니다.
`review_full_validation.py`의 전체16쌍·4정책 비교와 provider/초기속도별 재계산을 준비해
구문 검사는 통과했으나 전체 결과가 아직 없어 실행하지 않았다. 완료 후 초기mean0,
기존independent1,000/2,000과 모든 실패를 포함해 대조한다. 전체 결과를 검토하기 전
새 seed·추가 학습·기본 채택으로 확대하지 않는다. 안정적인 정책 목표는 계속 active다.

## 22:36 KST 전체 개발16회 검토 완료, 같은 수집 조건의 총3,000회 진단 시작

전체 개발 평가 runner2401291/child2410243은 완료 후 종료됐다. 원본16개의 초기 frame/map/
seed와 pilot4회 결과를 대조했고, `review_full_validation.py`로 초기mean0·기존독립1,000/
기존독립2,000·새quota2,000의64episode 원시 frame에서 분모·보상·종료·거절·제동·정지·
접촉을 다시 계산했다. 고유 개발 장면은8개이며64개 독립 장면이 아니다.

| 정책 | 목표/충돌 | tick | 거절 | 제동 | post-step 정지 | 평균 return |
| --- | --- | --- | --- | --- | --- | --- |
| 초기mean0 | 2/16·14/16 | 886 | 356 | 334 | 434 | -17.767363 |
| 기존독립1,000 | 2/16·14/16 | 908 | 340 | 322 | 424 | -17.757195 |
| 기존독립2,000 | 0/16·16/16 | 814 | 388 | 364 | 436 | -20.329688 |
| 새quota2,000 | 2/16·14/16 | 950 | 358 | 333 | 437 | -17.764623 |

새 KEEP/FOLLOW 각각1/8목표·7/8충돌이다. 초기 성공2건은 유지했지만 새 성공은 없다.
새 거절/제동/정지 비율37.68%/35.05%/46.00%는 초기40.18%/37.70%/48.98%보다 낮다.
그러나 tick 분모886→950 증가가 있으므로 절대 거절356→358·정지434→437 증가를 숨기지
않는다. 초기 대비 중속83c…의 충돌은KEEP27/FOLLOW23tick 늦어졌고, 저속fd56…는 둘다
3tick 일찍 충돌하면서 거절+6/+5·제동+4/+3·정지+4/+3으로 악화됐다. 정지 출발2장면×
2행동은328/328tick정지 후4/4충돌이다. 전체14충돌에서 정지 중 뒤쪽 접촉 기하가 관찰됐으나
책임/단일 인과를 판정한 것은 아니다. 목표 달성·기본 채택·새seed 확대 기준은 충족하지 않는다.

로컬 continuation 폴더의 `full-validation/report.json` SHA
`82b419e9f6ae20d7fbaa3fa37fbd86fa4421d1322fbd1e734808f9a333c64bfe`,
`full-validation-review.json` SHA
`e8d8d5880a97a041a7275ec40f94f046499ed77c284dc1177dcd52f76a05ccf1`.

### 실제 replay 추출과 새 정책 출력의 동결 시점 비교

기존 coverage의 실제 saved-RNG 대조가 확인된6,000/7,000행 CSV를 CPU에서 다시 읽어
동일 prefix의 장면/tick/보상/속도/lineage와 누적 draw 차이를 확인했다. 추가1,000update의
256,000draw 중 원래warmup197,402(77.11%), main39,502/branch19,096이다. 정지 상태
161,364(63.03%), 이동 중 양의 즉시 보상31,969(12.49%)이다. 반복 추출 횟수이지 독립 전이 수나
성공 사례 수가 아니다. 누적2,000회에서는512,000draw 중 정지325,296(63.53%)이다.

동결2,000정책의7,000저장상태 모두 sigma두축이0.02999보다 컸고,1,000정책에서는
6,000중4,423개였다. 현재 mean의 초기sigma단위 RMS 이동 중앙값은0.2471→0.6008이다.
이는 정책 변화/상한 근처 분산의 기술 통계이지 실패 원인 확정이나 원래 Gaussian 변경
근거가 아니다. `sampling-exposure-review.json` SHA
`19d376203c67ab92feb7dd7530ba21b33a5b2b0135e6783e27fada4a147a2e7e`.

### 다음 제한 실험: 수집·모델 그대로, 실제 경험1,000개 추가

초기 성공은 유지했고 종료 Q 적합은 좋아졌지만, 초기 Q/target은 여전히 낙관적이며
새 경험보다 초기warmup 추출이 많은 상태다. 추가 자기 경험 노출과 비용 전달의 영향을
분리하려고 같은 모델에서 **총3,000회까지 추가1,000 actual transition/update**를 수행한다.
단순 배율/보상/검사/PD/원래Gaussian/actor/Q/수집quota/uniform replay/update비율/BCteacher0은
바꾸지 않는다. 추가 예산이 문제를 해결한다거나 이상화 target 계산이 실제 수렴 시점이라는
가정은 하지 않는다. 다음 결정 전에 실제 주행과 미래 비용을 다시 확인한다.

저속 악화를 놓치지 않도록 다음 고정 개발 검사는 기존8b4c…/8a4f…에fd56…를 추가한
3장면×KEEP/FOLLOW6회다. 이전4회와 분모를 섞지 않고, 이미 완료한2,000 전체 평가의
동일6쌍에서 비교한다. 독립 test나 새 learner seed 확대가 아니다.

새 로컬 `outputs/sac_failure_rollout_quota3000_20261119/`의 `review-decision.json` SHA
`6f634c36ff6411e85b7b5dd4efc9bff528d7df0345d34c9fcdf60b47f2db40fa`에 완료된 근거 hash와
제한·실패를 보존했다.22:35:40.718598 KST runner2414118/train2414120을 실제 시작했다.
원본2,000모델7c898963…의 learner/optimizer/targets/learner RNG/replay RNG를 이어받으며,
환경/장면/정책 수집 RNG는2000으로 명시 재시작한다. exact simulator resume가 아니다.
runner SHA `2c5084124169e66fe925759fa1fbab294145de8ecf374c23be23f5decaff4fc9`.

학습→저장 감사→고정6주행→고정Q→coverage→새분기Q 뒤 runner2414681이 같은 두 train
상태×4seed 실제 종료까지 보상을 재측정하고, runner2414683이 같은12상태에서
quota1,000/2,000/3,000의 저장 경로 Q·TD목표를 비교한다. 후속 실행기는 실제 WAITING이며
3,000결과는 미확정이다. 실행 코드90c15a4 유지, 새 PR/병합 없이 목표는 계속 active다.
총3,000회 뒤 자동 학습 연장·기본 채택·새seed/독립성능평가 승격은 없다.

22:37 실행 확인에서 실제 report는 RUNNING/누적2,002update였고
`restored_learner_bytes_identical=true`, `inherited_online_updates=2000`을 확인했다.
학습 PID와 두 후속 대기 실행기도 살아 있음을 확인했다. 단순 실행기 생성만을 학습 시작으로
대신하지 않았으며 완료 성능은 아직 주장하지 않는다.

## 22:46 KST 학습 중 읽기 전용 관측 점검과 별도 자료 확장 준비

총3,000회 학습 runner2414118/train2414120은 실제 살아 있으며22:45:49 KST에 누적2,604
actual transition/update를 확인했다. source/배율/보상/검사/PD/모델/수집을 변경하지 않았다.
이번 작업은 이전2,000의 동결 replay와 자료 준비 코드만 다룬다.

### 관측 단위 누락 가설은 현재 근거로 지지되지 않음

`src/environment/womd_control/observation.py`는 기하를 현재 ego 좌표로 변환하고, v1.2
관측 단위별 scale로 나눈 뒤[-5,5]로 clip하며 mask를 전달한다. 전역 좌표가 정규화 없이
입력된다는 가설은 코드와 맞지 않는다. 실제7,000행 replay의 hash를 확인하고 CPU에서
유효/무효 필드와11개 관측 그룹의 범위를 독립 계산했다. 무효값의 비영0개, 전체 최대절대값5,
모든 유한성 검사를 통과했다. ego speed 역변환은 기존 coverage7,000행과 완전히 일치했다.

같은 raw 행동을 규약 B로 역변환하고 다시 복원해 최대오차<1e-12를 확인했다. 현재
physical Q 입력 사본의 RMS 중앙값은 정지4,427행에서0.000499625, 이동2,573행에서
0.249813이다. 원래 entropy 좌표 사본은 각각0.249812/0.258857이다. 정지에서g=0.002인
현재 특징식의 결과이며 잘못된 전역 단위·실제 신경망의 gradient 소실·추가 정규화의 필요성을
입증하지 않는다. NN 추론/학습/환경step/원본행동·replay 변경은 없었다.

로컬 continuation 폴더의 `review_observation_action_scale.py` 실제 실행이 통과했고
`observation-action-scale-review.json` SHA는
`31485adbbf5b12a89d6fd749e371d833603400fe6131631ec19742f9d58f744f`다.
이 근거로 불필요한 관측 정규화 변경을 추가하지 않았다.

### 남은800원본 파일의 정책과 무관한 후보 봉인·reset 검사 대기

현재 별도 reset 적격은3장면으로32목표에29개 부족하다. 같은 raw training_20s의 아직
검사하지 않은 shard200~999에서 ID/raw SHA만 읽어, 기존 first200 메타데이터와 합친
전체1,000파일의 hash-test 후보를 같은 고정 hash 순서로 봉인하는 로컬 도구를 만들었다.
이 단계는 도로/초기화/정책 성능을 검사하지 않으며 train/validation manifest를 수정하지 않는다.
그다음 동일한 기존 route prefilter와 environment reset만 적용한다. 모든 제외·오류를 보존하고
모델 결과로 장면을 선택하지 않는다. 전체32개가 확보된다고 가정하지 않는다.

원본 shard00200 한 개로 메타데이터 코드를 실제 검증했다. CPU/TF thread1로75개 record의
ID/바이트SHA를 읽고 새 test13개를 확인했다. 기존1,417후보와 합집합은1,430개, source201개,
정책/route/reset 실행0이다. 고정순서·기존 후보 보존·중복·개발 manifest 분리를 대조했다.
약0.956초의 metadata pass이며, 실제 학습과 별도의 데이터 읽기 검증이다. 새13개는
reset 적격/성공 장면13개가 아니다. 스모크 산출물은 Storage의
`.../sac_heldout_20261119/metadata_smoke_00200/`에 있다.

로컬 `outputs/sac_heldout_expansion_20261119/metadata-smoke-review.json` SHA
`7f2b5d90d70055c32f4803b9da985d92ad2d09004b77224f6bf9e867f1f104e3`.
`freeze_expanded_candidates.py` SHA
`5046c1c50d0172306f40f35a6c5d56113e50ca520e2bad48ee3c4409ad236ab1`,
`prepare_reset_inventory.py` SHA
`c142642064447c5d3b81bff726b7a4ff516fc7b41bb4d7d232ccd92a86794d9d`.
후자는 기존 정책과 무관한 reset 검사와 같고 보고서 저장만50후보/도로후보마다 모아
반복 대용량 쓰기를 줄인다. 구문 검사는 통과했으나 확장 pool의 실제 reset은 아직 미실행이다.

runner2419579를22:42:55.130201 KST 시작했고 실제 WAITING을 확인했다. 현재3,000학습과
실제 미래 보상·저장 경로Q 검사가 모두 끝난 후 metadata800→reset1000을 CPU로 순차 실행한다.
실행기 SHA `604f85b579c45146114a5bcc75a19b12f8756198ef74c1188b34c9fd953d3c7b`.
Storage 예상 출력은 같은 상위 폴더의 `metadata_first1000/`와 `reset_first1000/`이다.
전체 실행은 아직 미실행이며, 다음 학습/성능 평가를 시작하기 전에 이 준비 작업 상태를 확인해
원본 스캔과 겹치지 않게 한다. 현재 소스도 준비 검사 종료까지 동결한다. 부족 수·오류·실제
적격 수는 완료 후 별도로 검토한다. 프로젝트 hash-test와 공식Waymo test를 구분하고
기존 drive/시간구간 독립성 미확인도 유지한다. 전체 안정 정책 목표는 계속 active다.

## 22:59 KST 총3,000회 학습·저장 검토·6주행 완료: 성공 회귀로 후보 미채택

총3,000actual transition/update 학습은 완료됐고 train2414120은 종료됐다. 모델 SHA
`77882fc7745de6ab3160527c049a5b8301e4177fd64a5fede556aff795780382`, replay SHA
`dc9afc57f7b0a1bd6676c35253dcef4d41198431a1232751192eca4250e630d7`.
학습 report SHA `6dbcdc2ee494b84c4ee0cc389ce66a6756aa7e71bb5295aa0952306572553ccb`.

`review_collection3000.py`를 실제 실행해 이전 warmup+online7,000행이17개 replay 배열
모두 바이트 단위로 동일한 prefix임을 확인했다. 이전2,000tick/update 파일 prefix도 정확히
유지됐다. 모델 하이퍼파라미터와 원본 learner 복원을 대조했다. 첫 분석 실행은 arm의
`updates` 메타데이터까지 동일하게 비교해 실패했으며,2,000→3,000을 별도 검증하고
하이퍼파라미터 비교에서만 제외하도록 수정한 뒤 통과했다. 실제 학습 설정 변경/재실행이 아니다.

저장 감사는3,000개 중 main2,013/branch987·고유train25장면·126roots·실제후속701개,
branch terminal185/nonterminal802를 확인했다. 새1,000개는 main664(통과385·양의보상67·
사전정지379), branch336(통과78·양의보상48·사전정지81)이다. 새 수집의 사전정지460/1,000은
직전추가1,000의616/1,000보다 작지만 수집RNG/방문장면도 달라 정책 개선으로 단정하지 않는다.
검사 통과와 양의 즉시보상도 주행 성공이 아니다.

저장 감사 `audit3000.json` SHA
`c04f0bbfe4d9cd50e2ca35381fe28505920307fc04b6f87f3c967c99cb024f88`,
별도 prefix/수집 검토 `collection3000-review.json` SHA
`8ebf7711ca0c36c98a3b4fd33a4b611c7512ef1f3ce2856d84d08a055f4b3749`.

### 같은3개 개발 장면×KEEP/FOLLOW6회: 두 성공을 잃고 모든 비용 악화

기존8b4c…/8a4f…와 악화된저속fd56…의6회 검사를 완료했다. 같은 초기frame/map/seed6쌍을
대조하고 초기mean0·기존독립2,000·quota2,000·quota3,000의24episode 원시 frame에서
모든 분모/보상/거절/제동/정지/접촉을 다시 계산했다. 고유장면3개이며 독립test가 아니다.

| 정책 | 목표/충돌 | tick | 거절 | 제동 | post-step 정지 | 평균 return |
| --- | --- | --- | --- | --- | --- | --- |
| 초기mean0 | 2/6·4/6 | 290 | 84 | 70 | 40 | -13.414949 |
| quota2,000 | 2/6·4/6 | 304 | 85 | 68 | 38 | -13.419059 |
| 기존독립2,000 | 0/6·6/6 | 247 | 117 | 103 | 44 | -20.240019 |
| quota3,000 | 0/6·6/6 | 292 | 140 | 122 | 64 | -20.263320 |

새3,000의 거절/제동/정지 비율은47.95%/41.78%/21.92%로quota2,000의27.96%/22.37%/
12.50%보다 높다. 고속 KEEP/FOLLOW는기존72tick목표에서70tick충돌로 바뀌었다. 고속KEEP의
첫 거절tick2는기존과같지만 terminal연속거절10→32,첫제동65→41,새첫정지63·충돌70이다.
중속두행동도기존46→43tick충돌,저속두행동34→33tick충돌이며 비용이모두나빠졌다.
최초 raw경로의 최대차이는약6.59~8.24mm이나 이후폐루프주행결과는크게달랐다. 이초기차이만을
유일원인으로단정하지않는다.6충돌모두정지중뒤쪽접촉기하이며책임/단일인과판정은아니다.

`evaluation3000/report.json` SHA
`5b6e831bee94af5e6a44266cdc4a5478c0738142571d0a2907ff81b72a95660b`,
`evaluation3000-review.json` SHA
`a3ab1400d325b2157c52c623a0eb0a107116d4d36d2831b62df7a63ad000d249`.

**총3,000회 후보는 성공 회귀로 미채택이다. 새seed나 전체개발/독립성능평가로 확대하지 않는다.**
2,000도안정정책이아니므로대신최종채택하지않는다. source/보상/검사/PD는변경하지않았고,
3,000이후추가학습은시작하지않았다. 고정Q→coverage→수집terminal Q→실제미래보상→저장경로Q
검사를계속해종료적합과앞선상태비용전달,정책악화를구분한뒤다음보완을정한다.

22:58:48확인에서runner2414118의q3000/child2423266은실제실행중이었다. 주행child2421684는
완료후종료됐다. runner2414681/2414683은후속미래보상/경로Q대기,자료확장2419579는그뒤대기다.
`review_value3000.py`는구문검사만통과했으며실제후속결과가나오기전에는실행하지않았다.
학습량/저장검증완료를주행성공으로취급하지않고전체안정정책목표는계속active다.

## 23:03 KST 3,000회 종료 Q·coverage 완료, 실제 미래 비용 검사 실행

기본 실행기2414118과 마지막 branch-Q child2426291은 완료 후 종료됐다. 고정62개
terminal의 저장 Q에서 severe44개 MAE는 quota2,000의0.742834→3,000의0.470003으로
감소했다. 원본62행의 twin-min·실제 종료와44개 오차 평균을 다시 계산했다. 고정 warmup
두 상태의 실제 현재 평균 행동 Q−reward는 KEEP+0.570713→-0.554247,
FOLLOW+0.719806→+0.333576이다. 수집한 첫 terminal 두 상태에서는 현재 평균 오차가
KEEP-0.078794/FOLLOW-0.144980이며, 저장/평균/4표본×2상태12actual 행동은 모두 충돌했다.
실제 행동의 reward와 오차 항등식도 대조했다.

이는 종료 적합의 개선이며 앞선 상태의 미래 비용이 정확해졌다는 뜻은 아니다. 같은 후보의
6주행은 이미0/6성공으로 악화됐다. 원본/반례 기준을 구분한 `fixed-terminal3000-review.json`
SHA `8cf0ade29c0443e500ef81b49873cd983c305b2c089685190cc339fb165e35af`.
`q3000.json` SHA `4ec15c600d0bc3c0d758e967035993556ce40e4a2f83385b18603d879793d1d7`,
`branch-q3000.json` SHA `7d30f088712a556b01258518bc26e8567bfd22cd921a9baf9ee355de9b7288f1`.

coverage는 saved RNG를 대조했고 누적768,000draw 중 warmup601,388/online166,612,
미추출 online24개를 확인했다. branch987개의 현재 정책95% 타원체 안 비율은74.16%로
2,000시점87.56%보다 낮다. branch 학습 draw56,075, 미추출 branch4개다. 이 조건부 분포
겹침은 충분한 학습이나 성공의 증거가 아니다.

CPU에서2,000/3,000의 동일 prefix와 draw 차이를 다시 계산했다. 추가1,000update의
256,000draw 중 정지 상태159,619(62.35%), 이동 중 양의 즉시 보상30,202(11.80%)이다.
3,000정책의 sigma두축>0.02999는7,651/8,000으로2,000의 전체7,000에서 일부 변했다.
분산 상한 근처의 값만으로 영구히 회복 불가능하다고 판단하지 않는다. 현재 평균의 초기
sigma단위 RMS 중앙값은 전체0.6724/정지0.9862/이동하며 양의보상0.4900이었다. 저장 상태의
기술 통계이며 반복 draw를 독립 경험이나 성공 횟수로 세지 않는다.

`coverage3000/report.json` SHA
`26ac446bc28007d0969fee6df18e7bf7ca5b9cb7cf949836e13bfd4ab0435b8b`,
`sampling-exposure-review.json` SHA
`57f504e92f6e3db7a5f8e7a1a60116bcc991a8ae1933862c413c78e01d143e72`.

현재 runner2414681/child2427678이 같은2train상태×4seed의 실제 종료 soft return을 실행한다.
runner2414683의12상태 Q/TD목표 비교, 그 뒤 자료 확장2419579가 대기한다.3,000이후 학습·
새seed·채택은 시작하지 않았다. 기존 실패를 보존하고 실제 미래 비용과 국소 TD목표 결과를
대조해 다음 보완을 정한다. 안정적인 정책 목표는 계속 active다.
## 23:23 KST 후속 3,000회 미래 비용 검토 완료와 현재 정책 경로 진단 준비

실행 코드 `90c15a4`, 브랜치 `feature/yuntae-sac-training`, 기록 직전 HEAD
`41b072839fe40317fc44efa92bf654304d25fe59`다. 보상·검사·PD·원래 Gaussian·BC/teacher 0·
uniform replay·실제 전이당 1 update를 유지했고, 총3,000회 이후 학습은 추가하지 않았다.
3,000 후보의 개발6주행은 이미 목표0/6·충돌6/6으로 미채택이다.

같은 두 train 상태에서 현재 평균 첫 행동을 실행한 뒤 원래 확률 정책으로 실제 종료까지
진행한 4개 seed(88019~88022) 검사를 완료했다. 독립 test가 아니고 반복은 고유 장면 수가 아니다.

| 상태 | 현재 첫 행동 Q | 실제 soft return 평균 ± 표본 SD | Q−평균 |
| --- | --- | --- | --- |
| KEEP, 77ae312df21e80b9 tick0 | -0.714215 | -12.760066 ± 2.507637 | 12.045851 |
| FOLLOW, baac7893b025167a tick4 | -0.385449 | -10.654902 ± 0.529056 | 10.269454 |

8회 모두 충돌했다. 2,000 후보 대비 paired return 차이는 KEEP 평균+0.300960이지만
4쌍 중3쌍은 악화했고 한 쌍의+1.540899가 평균을 올렸다. FOLLOW 평균 차이는-0.005318이다.
소표본 실제 return을 정확한 기대 Q나 주행 개선으로 주장하지 않는다.

과거 경로의12고정 상태에서 1,000/2,000/3,000 모델을 대조한 검사도 완료했다.
3,000 KEEP tick0의 저장 행동 Q/TD목표는-0.808517/-0.476721,
online−target 다음 soft value는-0.197494다. FOLLOW tick0은-0.846657/-0.326707,
차이-0.110853이다. FOLLOW의 이 tick0과 위 MC tick4는 다른 상태다.
현재 Q와 target Q가 모두 초반 미래 비용에 낙관적이므로 target 갱신 지연만으로 설명하지 않는다.
저장 행동의 보상을 현재 평균 행동에 잘못 붙이지 않는다.

`outputs/sac_failure_rollout_quota3000_20261119/review_value3000.py`를 실제 실행해 통과했다.
8개 실제 trace hash, 첫 행동/관측/seed, 엔트로피 제외·할인, 실제 종료, profile12상태의
계산 항등식과 terminal 반례를 재계산했다. 로컬 원본/검토 SHA:

- `soft-returns3000/report.json`: `9dadda2d21e119f5c8361847b046df4ade64b2b7fb62b1c1c90623caa5777de3`
- `saved-bellman-chain.json`: `770b185610eba2db2d0bff90936c70341665d6af70636ffd65195c44e6076d81`
- `value3000-review.json`: `d50e168348cbbf48e18c381c6c80bb8da19a73230002e2c027a6745f05a52fe9`

다음 원인 분리는 **현재 정책의 실제 동일 주행에서 매 시점의 행동 Q·즉시 보상·다음 가치·
남은 실제 return을 함께 추적**한다. 기존 MC의 사전 지정 첫 seed88019, 두 train 상태를
그대로 재현하며 trace 바이트 일치를 요구한다. 각 상태의 다음 원래 Gaussian32표본
(99019~99050)을 online/target에 공통 사용한다. 매 suffix에서 현재 행동 엔트로피를 제외하고
미래 엔트로피만 포함한다. 한 실제 suffix는 정확한 조건부 기대값이 아니며, 평균 행동 Q는
별도 기술 통계다. 진단 관측 snapshot은 학습 replay에 넣지 않는다.

로컬 `outputs/sac_current_policy_credit_20261119/`에 진단기4개를 구현·구문 검사했다.
`return-arithmetic-review.json`은 기존8trace의442suffix를 독립 forward sum과 대조했고
최대 차이1.42e-14로 통과했다. CPU 합성1/2상태 검사도 종료 bootstrap0, 다음 Gaussian
엔트로피 목표, 현재 엔트로피 제외, learner 배열 불변을 확인했다. 최초 합성 실행은 두
profile 파일만 남고 완료 review/복구 가능한 stderr가 없어 전체 통과로 취급하지 않았다.
별도 `review_synthetic_profile.py`를 실행하여 기존 산출물과 결과를 대조하고 exit0을 확인했다.
`synthetic-review/review.json`은 합성 수치 검증이며 실제 주행 성능이 아니다.

runner2435604는23:15:02 KST에 시작해 자료 확장 종료를 기다리는 `WAIT_RAW_INVENTORY`다.
23:22:37에 실제 PID와 WAITING 상태를 확인했다. 이 경로 진단의 실제 환경 실행은 아직
미실행이다. 추가 학습·새 seed·기본 채택은 결정하지 않았다. 진단/실행 소스와 모델을 동결했다.

자료 확장에서는 추가800raw 파일·56,375record를 실제 읽어 전체1,000source의
70,541고유 scene·프로젝트 hash-test7,040후보를 확정했다. `review_metadata.py`를 실제
실행해 metadata hash, raw 파일 fingerprint, split·고정 hash 순서·기존1,417후보 포함·
개발 manifest와 ID/group/raw hash 불중복을 대조했다. 중복 scene0, 정책 실행0이다.
이 재검토는 raw 전체를 다시 읽은 검사가 아니며 적격/성능은 metadata에서 추론하지 않는다.
order SHA `c3b1aaf25a909afb415ca7db95886007b0056d02f2520cc9e7715c63f00b3dfc`.

23:22:37 현재 runner2419579/child2437167의 실제 reset 검사는3,750/7,040후보,
고유13장면·24provider쌍 통과이며 진행 중이다. 장면별 양쪽 행동 통과 여부가 다를 수 있다.
32장면 확보를 선언하지 않는다. 완료 후 전체 제외/오류/원본 cache hash와 manifest를
`outputs/sac_heldout_expansion_20261119/review_reset_inventory.py`로 재검토한다(현재 미실행).
공식 training의 프로젝트 hash-test이며 공식 Waymo test와 다르다. drive/시간구간 독립성은
미확인이고 기존 개발 데이터·학습 loader는 변경하지 않았다.

## 23:39 KST 실제 실패 경로 검토 완료와 종료까지 수집하는 제한 비교 구현

추가 자료 검사는 전체7,040후보를 처리했다. prefilter 탈락6,991·도로 후보32·데이터 오류17,
reset64회 중 통과56/탈락8이다. 최종 고유29장면(KEEP29/FOLLOW27), 양쪽 통과27장면으로
최소32개 목표에3개 부족하다. 고유 초기속도 분포는 정지2·0.1~2m/s2·2m/s이상25장면이다.
정책 실행은0이고 학습 loader·개발 manifest는 그대로다.

첫 reset 검토는 오류0을 요구하는 조건에서 실패했다. 원본과 실패 기록을 보존하고,
검토기는 `COMPLETE_WITH_UNRESOLVED_DATA_ERRORS`로 분모·manifest·cache hash를 대조했다.
17오류는 별도 원본 재로딩/SHA 확인에서도 모두 재현됐다. 각 scene의 모든 track과 SDC 상태
길이가 timestamps 길이와 다르고 dynamic-map 길이는 timestamps와 같았다. 단순 빈 객체
하나의 문제가 아니며 원인·시간 정렬은 미확정이다. padding/잘라내기/객체 제거로 임의 복구하지
않았고 일반 ODD 탈락으로 바꾸지 않았다. 공식 training의 프로젝트 hash-test라는 한계도 유지한다.

로컬 `outputs/sac_heldout_expansion_20261119/reset-inventory-review.json` SHA
`a5105287234e6a5564e6e227b7326b24346176d7482f000b67b474f27e4733c0`,
`raw-shape-review.json` SHA `10818f9420917268cfd1bcb250bff81adbb150234ee7f0182d7d849e4138c925`.
자료 실행기2419579/child2437167은 종료됐다. 32장면 확보와 데이터 오류 해결은 미완료다.

### 재현 실패를 보존하고 동일 자기 행동의 실제 경로를 복원

첫 경로 진단2435604/2446002는 KEEP29tick에서 원본과 바이트 동일성 검사를 실패했다.
같은 모델/seed를 재계산했지만 raw action 최대 차이1.851e-7m, reward 최대 차이2.0e-12,
ell 최대 차이1.065e-8이 있었고 다음 관측 hash가 달랐다. 첫 행동 차이는4.83e-8m였다.
모든 이산 검사/출처/종료와29tick충돌은 같았으나 동일 재현으로 취급하지 않았다.
수치 차이의 정확한 발생 원인은 분리하지 않았으며 임의 tolerance로 trace 검사를 통과시키지 않았다.

별도 `outputs/sac_current_policy_credit_replay_20261119/`에서 기존 MC의 seed88019가 만든
원본 자기 raw action/ell을 읽기 전용으로 재실행했다. 실제 reward와 다음 관측 hash 및 전체
trace 바이트 일치를 요구해 KEEP29/FOLLOW60tick을 모두 복원했다. 새 확률 표본 재생성이
아니라 이미 저장된 자기 정책 행동의 실제 환경 재실행이며 학습 replay/learner 갱신은0이다.
runner2447272/child2447275는 완료 후 종료됐다. 현재 모델의 Q와 다음 Gaussian32표본
online/target 값은 복원된 매 관측에서 계산했다. 이 두 경로는 기존 MC에 포함된 train 경로다.

| 경로 | 시작 실제 행동 Q | 한 실제 경로의 남은 soft return | 차이 |
| --- | --- | --- | --- |
| KEEP tick0, 29tick충돌 | -0.714319 | -15.171351 | 14.457033 |
| FOLLOW tick4, 60tick충돌 | -0.385501 | -11.071767 | 10.686266 |

각 suffix의 독립 할인합, 실제 상태/행동/보상, 종료 bootstrap0과 TD 항등식을 재계산했다.
Q(actual action)−한 실제 suffix return은 경로상의 한 단계 TD차이를 할인하여 합한 값과도
일치했다. 이 항등식은 단일 확률 경로의 분해이며 기대 Bellman 오차나 단일 인과 증명이 아니다.
종료15step 이상 전 구간의 할인 TD 합은 KEEP+15.0022/FOLLOW+12.1785였고 가까운 종료
구간은 음의 합이었다. KEEP tick10에서는 Q-16.413/남은return-16.762로 이미 가까워졌다.
두 경로 모두 감속·정지 이후에 큰 비용을 반영하며, 초반 Q와 학습 목표는 낙관적으로 남았다.

batch로 계산한 원본 행동 Q와 scalar로 재계산한 평균 행동 Q는 각각-0.000104/-0.000052
차이가 있어 별도로 공개했다. 서로 다른 추론 문맥의 값을 바이트 동일하다고 주장하지 않는다.
전체 경로/실제 관측 바이트 검사는 그대로 유지했다. 89시점의 검토와 그래프 저장·이미지 육안
검토를 완료했다. `credit-review.json` SHA
`0151a7d389670f6eb0fcbd38eca54ba7ce0c164f37eb72d42a07596ce8c8af2e`,
`capture/report.json` SHA `43eb472db0d88cb5b772be20c9ac6e3ca0eedd124a009471ec42d974be3f6230`,
`credit-paths.png` SHA `81449cd4737bfc000818625eeaebaa67d6e6642f8275f31e141d987e50d0422d`.

### 수집 길이의 구체적인 한계와 다음 비교

3,000수집의987branch에서 실제 연결 후속701개·126roots·종료경로185개·nonterminal 끝101개를
원시 lineage로 대조했다. moving_predecessor51roots/408steps와 moving_rejection44roots/
334steps는 **95roots 모두 실제 종료를 하나도 수집하지 못했고 최대 길이가8**이었다.
terminal21roots는165step을 종료 상태에 사용했다. 이는 명시적인 branch 연결의 한계이며
주변 replay 지원이 전혀 없거나 Q 오류의 유일 원인이라는 뜻은 아니다. 원본 bootstrap은 정상이다.
첫 검토기는 예산 중단이 마지막 root 하나에만 생긴다고 잘못 가정해 실패했다. 기존2,000예산의
round-robin 중단으로1995/1996/1997/1999/2000의5roots가 남는 것을 확인하고 실패 기록을
보존했다. 실제 수집/원본 데이터를 수정한 것은 아니다.
`branch-endpoints-review.json` SHA `7d82c057db98273ea0ae169a0393fd15ad0d5d1547df74e52f7bff13324d4955`.

이 근거로 `scripts/sac_failure_complete_path.py`와 진단 CLI의 opt-in
`--failure-branch-complete-path`를 구현했다. 과거 main 실패 직전의 가장 이른 accepted moving
predecessor 하나를 우선하고, 없으면 가장 이른 moving rejection→recent→terminal 순서다.
기존6개 후보 중 하나를 고르며 새 행동의 통과 여부로 학습 경험을 필터링하지 않는다.
한 실제 경로를 원래 환경 종료까지 이어가고 종료 후 같은 root를 반복하지 않는다.
200step은 기존 환경 상한 확인용이며 가짜 terminal을 만들지 않는다. 전체 수집 예산이 먼저
끝나면 마지막 실제 transition의 bootstrap을 그대로 남긴다. 원래 확률 정책·보상·검사·PD·
uniform replay·1actual step/1update·BC/teacher0·모델 구조/배율은 바꾸지 않았다.

CLI·저장 auditor·continuation 설정 검증에 새 모드를 분리했다. 기존8step/default 동작은
유지한다. 새 mode로 기존3,000을 조용히 이어받지 않고 같은 자기 warmup5,000·새 learner seed0의
별도 비교를 한다. 전체 완료 경로 수집으로 장면 다양성/종료 반복 노출도 달라지는 trade-off를
기록하며 길이 하나만 바뀐 통제 실험이라고 주장하지 않는다.

관련7개 pytest 파일의34개 테스트를 실제 통과했다(새 파일5개 test 중 terminal/truncated
매개변수로6case). 이후 continuation 모드 변경 거절 검증을 추가해 해당6개를 재실행·통과했다.
실제 신규모드 학습/주행은 현재 미실행이다. 다음은100actual step의 저장·연결·종료 smoke와
같은4개 개발 주행, 검토 후 같은예산1,000회 기존 수집 비교다. 구현 통과를 주행 개선으로
취급하지 않으며 새 기본 채택·새 seed·독립 정책 평가는 아직 없다.

## 23:49 KST 종료까지 수집하는100회 검증 완료, 같은예산1,000회 비교 시작

실행 코드는 `55df7a7386053d31b1f5d1465f4fd73c113be1a9`(push/원격SHA 확인)이다.
100actual transition/update는 main71/branch29였다. KEEP train77ae…의 tick0 accepted
predecessor 한 상태에서29step을 이어 실제 충돌 종료까지 수집했다. 통과1·제동26이며
최초8step 이후의 실제 연결과 종료를 확인했다. 초기 자기 warmup5,000개는 기존quota100과
17배열 모두 바이트 동일하고 actor/Q 설정도 같았다. 기존3,000데이터의 새 auditor 호환도
실행·통과했다. 원시 관측/mask·raw action/보상/종료·계보·teacher0 저장 검증을 완료했다.

같은2개 개발 장면×KEEP/FOLLOW의4주행과 원본 frame 재계산을 완료했다. 초기조건/map/seed
4쌍과3모델12episode를 대조했다. 고유 장면은2개이며 독립 test가 아니다.

| 모델 | 목표/충돌 | tick | 거절 | 제동 | 정지 | 평균 return |
| --- | --- | --- | --- | --- | --- | --- |
| 초기mean0 | 2/4·2/4 | 216 | 58 | 46 | 22 | -10.015390 |
| quota100 | 0/4·4/4 | 224 | 108 | 96 | 44 | -20.259841 |
| complete-path100 | 0/4·4/4 | 224 | 110 | 98 | 44 | -20.265190 |

**100회에서는 주행 개선이 없고 새 후보도 실패다.** 초기mean0의 두 성공을 유지하지 못했고,
quota100보다 거절/제동도2tick씩 늘었다. 모든4충돌은 정지 중 뒤쪽 접촉 기하이며 책임/인과
판정은 아니다. 분기 이전29main의 검사/보상은 같고 raw action 최대차이3.2233e-8을 공개했다.
저장 검증 통과와 주행 성공을 구분한다. 첫29step 이후 서로 다른 자기 경험 수집을 수행했다.

로컬 `outputs/sac_complete_failure_path_20261119/`의 `collection100-review.json`,
`paired100-review.json`, `audit100.json`에 실제 검토 결과가 있다. runner2451410과
train2451411/주행2454887은 완료 후 종료됐다.

그 결과를 검토하고 **fresh learner seed0·같은 자기 warmup5,000의1,000회 비교 한 번**을
23:48:14 KST에 시작했다(runner2456547). 100회에서 주행이 좋아졌기 때문이 아니다.
100회는 완료 경로가 KEEP 하나뿐이며, 기존quota도100회0/4에서1,000회2/4로 달라졌으므로
동일1,000예산의 기존 후보와 대조해 수집 변경의 영향을 확인한다. 새 경험 방문·root 다양성·
종료 노출도 함께 달라지는 비교이며 길이 하나의 효과로 단정하지 않는다.

실패한3,000 또는100정책 가중치를 이어받지 않고 모델/배율/원래Gaussian/보상/검사/PD/
uniform replay/1actual step당1update/BC·teacher0는 동일하게 유지한다. 실행기에는
1,000수집→저장 감사→동일4주행→고정 종료Q→coverage→수집 terminalQ를 연결했다.
`review-decision1000.json`에 실패 지표·근거 hash·고정 조건·한계를 사전 기록했다.
소스는 완료까지 동결한다. 추가 예산 자동 연장·새 learner seed·독립 정책 평가는 없으며,
안정적인 정책 목표는 계속 active다. 성능 개선 근거가 나오기 전에는 평가 확대하지 않는다.

## 2026-11-20 00:03 KST 정지 탐색의 조건부 한계와 후속 검토 준비

실행 코드는55df7a7, 기록 직전 HEAD83a1895다. 같은1,000회 비교는 runner2456547/
train2456548의 실제 실행을 확인했으며,00:02경727update까지 진행됐다. 결과를 미리 채택하지
않는다. 고정4주행·Q/coverage가 끝나면 같은 두 train 상태×4seed(88019~88022)의 실제 종료
soft return을 실행하도록 읽기 전용 runner2460323을23:52:30 KST에 연결했다. 현재
`WAIT_MATCHED1000_PIPELINE`이며 학습 추가나 별도 평가 장면 사용이 아니다.

기존 규약에서 **평균 raw action=0, anchor 속도=0, 가속도=0**인 조건의 탐색 크기를 별도로
계산했다. `3*tau^2*B`를 직접 닫힌식으로 계산하고 실제 코드 B/decoder와 대조했다. g=0.002,
sigma=0.02에서3초 마지막 점의 축별 표준편차는4.7816e-5m이며, 허용 상한0.03에서는
7.1724e-5m다. 원래 trajectory의 전체 정지 정렬을 충분히 보장하는 두 조건(모든 future 점이
anchor1mm 이내, 모든 future 점이 첫 future 점1mm 이내)의 실패 확률을2차원 Gaussian
반경 tail과 union bound로 계산했다. 시간 간 독립성을 가정하지 않는다.

초기 sigma0.02의 상계는1.978e-95, 두 축 모두상한0.03에서도1.192e-42다. 즉 이 조건에서는
잡음만으로 전체 정지 정렬을 벗어날 가능성이 매우 작다. 원래 decoder/Trajectory를 사용한
합성100표본(seed20261119)도 모두31개 knot의 정지 경로로 정렬되고 속도0이었다. 최대 원래
waypoint 거리는0.0001955m였다.100표본은 코드 sanity check이며 확률 상계의 증명 자체가 아니다.

이 결과를 비영점 학습 평균·비영점 anchor·전체 학습 과정으로 일반화하지 않는다. SAC 학습이
불가능하다는 주장, 수렴 시간 예측 또는 보상/검사/PD 변경 근거로 쓰지 않는다. 수집 길이를
늘리는 것이 정지 출발도 자동으로 해결한다는 가정에 대한 구체적인 반례 조건이다.

기존quota3,000의 Q감사에 저장된 평균에서도 추가 조건부 계산을 수행했다. 원래 자기 warmup
replay의 SHA를 검증하고, 고정terminal62행 중 valid packed 속도/가속도가 정확히0인50행을
출력값과 무관하게 전부 선택했다. 이 저장 평균을 zero anchor에서 원래 decoder/Trajectory로
계산하면50/50 모두 전체 정지 정렬이며, 최대 평균 waypoint 거리는3.6967e-5m다. 저장 평균과
sigma상한0.03을 조건으로 정렬 조건 실패 확률의 최대 상계는1.3801e-39다.

이50행 계산은 **과거 batch 추론으로 저장한 평균**에 대한 것이며 scalar Policy 재계산과의
수치 차이를 없었다고 가정하지 않는다. 실제 scene 재실행이나 새 NN 추론/학습은 없었다.
비영점 anchor, 이후 바뀌는 평균, 전체 정책의 성능/불가능성에 대한 상계가 아니다.
보상·검사·PD·잡음·평균 구조는 변경하지 않았고 진행 중인1,000회 조건도 그대로 유지한다.

로컬 `outputs/sac_complete_failure_path_20261119/`의
`standstill-exploration-review.json`(조건부 수학·합성 코드 확인),
`saved-standstill-means3000-review.json`(저장50평균 조건부 계산)에 가정·source hash·수치가 있다.
이 검토와 별개로 실제 주행이 개선됐는지는 완료될 동일예산 평가로 판단한다.

완료 후 사용할 저장/주행/Q 검토기도 준비했다. `review_collection1000.py`는 원시main의
accepted/rejected 이력에서 root 후보와 우선순위를 독립 재구성하고, 새100회 자료에서 실행해
기존17배열 동일·선택 root·29step 실제 종료를 확인했다. 이때100자료는 main71/branch29로
모두KEEP이고, 사전정지 main35/branch20이며 고유train2장면이었다. 완료1,000의 전체
분모는 아직 미확정이다. `review_evaluation1000.py`와 `review_value1000.py`는 구문검사만
완료했고 실제 최종 원본이 나온 뒤 실행한다. 다른 분기 방식의 서로 다른 terminal 상태를
동일 상태의 paired Q로 비교하지 않는다.

## 2026-11-20 00:20 KST 종료까지 수집한1,000회도 실패 — 종료 노출 감소 확인

실행 코드55df7a7, 기록 전 HEAD ea5fb39의 조건을 그대로 사용했다. runner2456547의
학습·저장 감사·동일4주행·고정 종료Q·coverage·수집 terminalQ는 모두 완료됐고 실제
프로세스도 종료됐다. 학습1,000회 완료는 안정 정책 달성이 아니다. 모델 SHA는
`2db717baf1e9b323e9267727ffe27100ddf164659d2007eeea75b01dcbd49054`, replay SHA는
`451640b6db13ae6da5cdc7a5b83dab29c11e90b666b5c9fdcce8a2c2d63fc12e`다.

저장 자료를 독립 검토해 자기 warmup5,000의17배열 바이트 동일, 과거main의 수락/거절 이력에서
root 선택 우선순위 재구성, 실제 후속 관측/mask 연결, 원래 종료/보상/raw action, teacher0를
확인했다. main589/branch411, 고유train9장면이며, 선택한7개 분기는 각각29/31/47/56/120/55/73
실제step 후 모두 충돌했다. 가짜 종료나8step 이후의 연결 누락은 없었다. KEEP/FOLLOW별
분기step은235/176이다. main의 마지막 예산 경계는 성공/종료로 바꾸지 않았다.

같은2개 개발 장면×KEEP/FOLLOW4회와 기존3모델을 합쳐16episode의 원본 frame을 재계산했다.
초기 frame/map/seed4쌍은 같다. 고유 장면2개이며 독립 test가 아니다.

| 모델 | 목표/충돌 | tick | 거절 | 제동 | 정지 | 평균 return |
| --- | --- | --- | --- | --- | --- | --- |
| 초기mean0 | 2/4·2/4 | 216 | 58 | 46 | 22 | -10.015390 |
| quota1,000 | 2/4·2/4 | 236 | 44 | 36 | 12 | -9.981233 |
| independent1,000 | 2/4·2/4 | 236 | 44 | 36 | 12 | -9.981133 |
| complete-path1,000 | 0/4·4/4 | 186 | 96 | 84 | 32 | -20.254178 |

새 후보의 거절51.61%·제동45.16%·정지17.20%는 quota의18.64%·15.25%·5.08%보다 나쁘다.
고속 두 주행은72tick 목표 도달에서57tick 충돌로, 중속 두 주행은46tick 충돌에서36tick
충돌로 바뀌었다. 모두 정지 중 뒤쪽 접촉 기하이며 책임/단일 인과 판정은 아니다.
첫 제안의 최대 World waypoint 차이는 quota 대비 약3.02~3.04mm지만, 이 작은 초기 차이
하나가 전체 실패의 원인이라고 단정하지 않는다. 이후 폐루프 상태/행동도 달라진다.
**새 후보는 미채택이며 전체16·다른 learner seed·별도 정책 평가로 확대하지 않는다.**

긴 경로 수집에는 부작용도 있었다. quota는42root/분기336step/분기종료66개였지만 새 후보는
7root/411step/종료7개다. 전체 사전정지 경험도464→486/1,000으로 늘었다. 고정62개 저장 종료
중 큰 음의 보상44개의 twin-min Q MAE를 원시 Q에서 다시 계산하면4.896726→8.799604로
악화됐다. 같은 실제 종료 두 상태에서 현재 평균 행동을 실행해도 충돌이고, Q−보상 차이는
KEEP5.988793→8.973167, FOLLOW3.325783→7.728353이었다. 별도 수집 terminal 상태들은
서로 다르므로 같은 상태의 paired 비교로 합치지 않는다.

`review_sampling1000.py`는 독립 정수 추출/누적으로 총256,000개 인덱스, 저장 RNG와
6,000행 CSV/replay를 대조했다. 두 실험의 warmup/online 추출 수는233,361/22,639로 같다.
큰 실패 종료 경험의 배치 노출은 아래와 같다. 동일 warmup의 노출은 그대로이며, 새 경험의
구성이 바뀌었다. 이는 종료 노출 감소를 확인한 것이지, 다른 상태/행동 분포 효과를 제거한
인과 실험은 아니다.

| 추출된 큰 실패 종료 경험 | quota1,000 | complete-path1,000 |
| --- | --- | --- |
| 동일 자기 warmup | 2,018 | 2,018 |
| 새 main | 220 | 212 |
| 새 branch | 1,488 | 189 |
| 합계 | 3,726 | 2,419 |

검토기 첫 실행은 저장 metadata 키를 `rng_state`로 잘못 읽어 KeyError가 났다. 원본은
`rng`이며 실험/데이터를 바꾸지 않고 검토기 키만 수정한 뒤 위 전체 대조를 통과했다.
초기 검토기/오류는 로컬에 보존했다. 실행 중인 MC 조회에서도 아직 summary가 생성되지
않은 중간 항목을 바로 읽어 KeyError가 났으나 실제 진단 프로세스는 정상 진행했다.

이번 모델의 저장된 zero packed velocity/acceleration50평균도 원래 zero-anchor 계산에서
50/50 전체 정지 정렬됐다. 최대 평균 waypoint 거리는1.2684e-5m, 최대 sigma0.03 조건부
정렬 이탈 상계는1.3838e-41이다. 이전 절과 같은 조건부 계산이며 실제 장면 출발 검사,
미래 학습 불가능성 또는 새로운 NN 추론의 결과로 일반화하지 않는다.

로컬 `outputs/sac_complete_failure_path_20261119/`의
`collection1000-review.json`(SHA `b3f652191648ffe6cb533072fb4c53a2c6a3197ff750485bafd48d57916b1473`),
`evaluation1000-review.json`(SHA `d06a2a0a6f0b541aed2676d6c604a65f2ae833d5440a7df2dec766b8043761b7`),
`sampling1000-review.json`, `saved-standstill-means1000-review.json`에 검토 근거가 있다.
실행 명령은 각각 같은 디렉터리의 `review_collection1000.py`, `review_evaluation1000.py`,
`review_sampling1000.py`이며, 마지막 계산은 `review_saved_standstill_means.py --q
outputs/sac_complete_failure_path_20261119/q1000.json --output
outputs/sac_complete_failure_path_20261119/saved-standstill-means1000-review.json`으로 실행했다.

같은 두 train 상태×4seed의 실제 미래 보상 대조는 runner2460323/child2468917로 진행 중이다.
현재 소스 동결은 유지한다. 다음 후보는 기존 여러 root의 수집/종료 반복을 유지한 뒤
선택한 이동 경로의 실제 후속 경험을 추가하는 방안을 검토한다. 아직 구현·실행·채택하지
않았으며 전체 예산 안에서 장면 수/정지 비중/종료 노출이 다시 달라질 수 있음을 명시한다.

## 00:26 KST 실제 미래 보상 검토 완료와 기존 quota 보존 옵션 구현

같은 두 train 상태×4seed(88019~88022)의 실제 종료 진단은 완료됐고 runner2460323과
child2468917의 종료 및 마지막 소스 hash 일치를 확인했다. 보상·entropy·discount·terminal,
trace hash·상태/seed 동일성을 `review_value1000.py`로 독립 대조했다. 첫 행동은 각 정책의
현재 평균, 이후는 원래 Gaussian이며 첫 행동 entropy를 제외한 실제 soft return이다.
8/8 모두 충돌했다. 표본 수가 작고 train 상태에 대한 조건부 비교이며 독립 평가가 아니다.

| 상태 | 새 Q | 새 실제 return 평균±표본SD | Q−평균 | quota 대비 paired return 차이 |
| --- | --- | --- | --- | --- |
| KEEP77ae… tick0 | -0.250134 | -14.052836±1.083310 | 13.802702 | -0.658385±1.042408 |
| FOLLOWbaac… tick4 | -0.159687 | -10.801195±0.799995 | 10.641508 | +0.175726±0.933531 |

FOLLOW의 작은 표본 평균 개선을 성공으로 해석하지 않는다. 네 paired 차이는
-0.773574/-0.477651/+0.944823/+1.009308로 혼재하고 모두 충돌이다. KEEP도 앞의 두 차이는
약+0.00044지만 나머지 둘이-2.190034/-0.444385로 나빠졌다. 실제 Q 과대평가는 양쪽 모두
남았고, 별도 개발4주행도 모두 실패했으므로 complete-path1,000은 미채택이다.

최종 검토 SHA는 `value1000-review.json`
`07b4186bdeeae0206728cb0febad887f2a75bc478ba148a7186102eb4c786f2e`,
MC 보고서는 `c2346ac4114981285f358ef534aea6a84ee4f851dcfbe9c4dfde6c38d7c2dbd2`,
추출 재검토는 `4c9a8a246e084a7907a56eefa810b7231608098d1ff90df57621afa12bb0f771`이다.

그 뒤에만 후속 수집 옵션을 구현했다. `scripts/sac_failure_quota_extension.py`는 기존
`collect_fixed_budget_rollouts`를 그대로 호출해 선택된 모든 root의8step 수집을 먼저 마친다.
이 과정의 종료 후 root 재시작도 원래 동작과 같다. 그다음 과거main의 accepted predecessor
우선/이른 tick 규칙으로 한 root를 골라, 마지막 quota 전이가 아직 nonterminal인 경우에만
그 실제 다음 상태에서 원래 종료까지 잇는다. 마지막 quota가 terminal이면 추가 재시작하지
않는다. 기존 quota 안에서 이미 재시작했다면 현재 열려 있는 path를 그대로 잇는다.
새 행동의 통과 여부로 데이터를 선별하지 않는다.

CLI `--failure-branch-extend-selected`는 samples8/window6/moving_predecessors/horizon8/
restart-terminal 조합에서만 허용한다. main/분기 모두 원래 Gaussian·collector·uniform
replay·실제step당 update1을 사용한다. 200step 경계는 가짜 terminal이 아니라 원래 환경이
종료하지 않은 경우의 오류 검사다. 전체 예산 경계에서는 원래 bootstrap을 유지한다.
새 flag와 quota 소스 hash를 continuation 설정 검증에 추가했으므로 다른 수집 모드의
학습 가중치를 조용히 이어받을 수 없다. 새 모드는 별도 진단이며 기본 채택하지 않는다.

저장 auditor는 실제 선행 quota 완료, 선택한 한 root만 확장, 동일 path의 실제 next obs/mask/
이전 collection ID, terminal 이후 확장 금지, global budget 외 nonterminal 확장 중단 금지를
검사한다. 기존3,000quota와 새1,000complete 자료를 새 auditor로 실제 재검증해 모두 통과했다.
로컬 `outputs/sac_quota_extension_20261120/*-backward-audit.json`에 근거가 있다.

관련6개 pytest 파일34case를 실제 통과했다. 새8case는 기존 quota prefix의 완전 동일,
실제 terminated/truncated 종료, 예산 전/중간 경계, quota 마지막 terminal의 재시작 금지,
quota 내부 재시작 뒤 열린 path 연속성, 잘못된 설정/200step 미종료 오류를 다룬다.
추가로 실제 분기 재구성과 Q 진단 관련2개 파일8case도 통과해 관련 총42case를 확인했다.
아직 새 옵션의 실제 학습/주행 성능은 미검증이다. 다음은 같은fresh seed0·자기 warmup5,000의
100actual-step 저장 감사와 같은4개 개발 주행이다. 이100회는 실제 연결 구현 검증이며,
짧은 실패만으로 장기 학습의 불가능성을 주장하지 않는다. 같은예산1,000실험이나 예산 연장은
실제100자료 검토 후 별도로 판단한다. 긴 경로 추가에 따라 전체 예산 내 main/장면 수가 줄 수
있으므로 초기 quota 보존을 전체 terminal 노출 보장으로 오해하지 않는다.

## 00:34 KST quota 보존100회 검증 완료, 같은예산1,000 비교 실행

실행 코드는 `c3fe6a746762a7fc12c96b44a68aeee8920cff79`이며 push/원격SHA를 확인했다.
100회 pilot은00:26:52 KST runner2472555/child2472556으로 시작해 수집·저장 감사·동일4개
개발 주행을 마쳤다. 두 PID의 종료를 확인했다. main31/branch69, 모두KEEP이며 고유train
2장면이다. 같은 장면의 root0/1/2/3/4/28에서 기존8step quota 총48개를 유지하고, root0의
실제 열린 경로에21step을 더해29step 후 충돌 종료를 수집했다. 분기 종료9개다.

독립 검토로 자기 warmup5,000의17배열 바이트 동일과 과거main 이력의 전체 root 선택/순서를
확인했다. 최초77수집의 검사·source·종료·root/path 순서는 기존quota100과 같았다. 다만
GPU 학습을 포함한 별도 실행의 수치는 바이트 동일하지 않았다. 최대 raw-u 차이3.7143e-7,
보상 차이2.8971e-11, packed 관측 차이9.1642e-7를 그대로 기록했다. 원인을 단정하거나
숫자까지 동일한 실험이라고 부르지 않는다. 함수 자체의 주어진 callback에 대한 quota
prefix 동일성은 앞서 unit test로 확인한 별도 사실이다.

연장한 path 하나를 CPU에서 원래 자기 raw action 그대로29step 재실행했다. 그중21step이
추가 extension이다. 관측/다음 관측은 기존 재현 허용오차1e-6, 보상은1e-8 이내이고 mask는
정확히 같았다. 검사 valid/reason, source, 실제 제어 dict, terminated/truncated/bootstrap을
대조했다. 다른 quota path까지 전부 새로 재실행했다고 확대하지 않는다. 모든69분기의
원시 저장 연결 일관성은 별도 auditor로 검사했다. 학습이나 새 NN 추론 없이 수행한 검증이다.

`review_paired100.py`로 같은 초기조건/map/seed4쌍과3모델12episode의 원본 frame 지표를
재계산했다. 고유 개발 장면2개이며 독립 test가 아니다.

| 모델 | 목표/충돌 | tick | 거절 | 제동 | 정지 | 평균 return |
| --- | --- | --- | --- | --- | --- | --- |
| 초기mean0 | 2/4·2/4 | 216 | 58 | 46 | 22 | -10.015390 |
| quota100 | 0/4·4/4 | 224 | 108 | 96 | 44 | -20.259841 |
| quota-extension100 | 0/4·4/4 | 224 | 108 | 96 | 44 | -20.259844 |

**새100회 후보도 성능 실패이며 주행 개선은 없다.** 모든 접촉은 정지 중 뒤쪽 접촉 기하로
분류됐지만 책임 판정은 아니다. 구현/저장/재현 검증과 정책 성능을 구분한다.

로컬 `outputs/sac_quota_extension_20261120/collection100-review.json` SHA
`869013d890524ede0600674984112155ac7660b0b67211ababab5613cb69760a`,
`actual-suffix-reconstruction.json` SHA
`986bfa1f85ee55a3b8382488b660b7302d3b1c1b2a986e0b2839b31d22b948eb`,
`paired100-review.json`, `audit100.json`에 완료 근거가 있다. 각각 같은 디렉터리의
`review_collection100.py`, `reconstruct_actual_extension.py`, `review_paired100.py`를
실제로 실행했다. 새1,000용 검토기들은 구문 검사만 했으며 원본 완료 뒤 실행한다.

위 실제 결과 검토 후00:33:51 KST에 **fresh seed0·같은 자기 warmup5,000의1,000회 비교**를
시작했다(runner2477739/train2477740). 이전100 가중치를 이어받지 않는다. 100에서 수집한
완성 extension은KEEP 하나뿐이고 기존quota도100회0/4와1,000회2/4가 달랐으므로, 같은
1,000예산 대조를 한 번 수행한다.100실패를 성공으로 바꾸거나 더 긴 학습의 개선을 보장하지
않는다. 근거/고정 조건/남은 정지 문제는 `review-decision1000.json`에 사전 기록했다.

자동 범위는1,000수집→저장 감사→동일4개 개발 주행→고정 종료Q→uniform coverage→수집
terminalQ다. 이어 같은 두 train 상태×4seed의 실제 종료 soft return을 대조하는 읽기 전용
runner2478192를00:34:02 KST에 대기시켰다. 두 runner와 train child의 실제 실행을 확인했다.
모델·배율·원래Gaussian·보상·검사·PD·uniform replay·실제step당 update1·BC/teacher0를
유지하며 모든 현재 실행 소스를 마지막 검사까지 동결한다. 결과 검토 전 추가 학습 예산,
전체16/new learner seed/독립 정책 평가는 연결하지 않았다. 안정 정책 목표는 계속 active다.

## 00:45 KST 이전3,000모델의 실제 방문 상태와 replay를 대조

현재 quota-extension1,000과 후속 MC의 실행 소스는 c3fe6a7로 동결돼 있다. 이 분석은 그
미완료 모델의 성능 평가가 아니라, 이미 실패한 quota3,000 모델
`77882fc7745de6ab3160527c049a5b8301e4177fd64a5fede556aff795780382`의 저장 자료를 읽은
별도 진단이다. 학습/새 NN 추론/시뮬레이션/replay 삽입을 하지 않았다.

이전 exact action replay로 복원한 KEEP29/FOLLOW60의 실제89상태(seed88019)를 사용했다.
capture/원본 trace/관측 NPZ/model/replay hash와 기존 credit 검토 출처를 대조했다.
자기 replay8,000행과 같은 seed의 실제 uniform sampling768,000인덱스를 독립 재구성해
저장 RNG와 coverage CSV를 확인했다. 모든 행의 teacher flag는0이다.

### 시작 상태를 보지 못했다는 설명의 범위

KEEP77ae… tick0은 관측 특징과 field/item mask가 완전히 같은 replay 행이3개 있었다.
index0(warmup)/5000(main)/5034(branch)이며 batch 추출은129/126/100, 합계355회였다.
그러나 첫 두 행의 raw action도 동일하므로 **서로 다른 저장 행동은2개**다.355번 추출을
355개의 새 환경 경험이나 충분한 action coverage로 바꾸어 해석하지 않는다.

그 상태의 현재 평균 행동 Q는-0.714215인데, 같은 모델의 기존4개 실제 stochastic suffix
return 평균은-12.760066±2.507637(표본SD)였고 모두 충돌했다. 즉 “이 KEEP 시작 관측이
학습 데이터에 전혀 없었다”만으로 그 과대평가를 설명할 수 없다. 반대로 이것이 충분한
행동/미래 상태 지원, 올바른 bootstrap target, 전체 수렴 또는 단일 실패 원인을 증명하지도
않는다. 기록된 두 행동과 현재 평균 행동의 실제 미래 return을 동일시하지 않는다.

FOLLOWbaac… tick4는 완전히 같은 packed 상태는 없었다. 같은 scene/command/tick 및
동일 field/item mask의 후보7개 중 정규화된 packed 좌표 RMS가 가장 작은 상태의 차이는
RMS3.7379e-7/최대4.7393e-6였다. 이는 입력 좌표의 서술적 차이이며 신경망 표현 거리,
행동 지원 충분성 또는 OOD 임계값이 아니다. 기존 그 상태의 평균 행동 Q와4회 실제 return
차이10.269454점은 그대로 남아 있다.

### 전체 실제 경로와 비교 조건의 한계

같은 scene/command의 replay 후보는 KEEP106/FOLLOW175행이었다. 실제89상태를 각각
같은 simulation tick으로 제한해 비교하면, 동일 mask 후보가 존재하는 상태는
KEEP29/29, FOLLOW38/60이었다. FOLLOW의 나머지22상태는 동일 mask 비교가 불가능하다고
그대로 표시했다. 임의로 거리를0으로 채우거나 근접/안전 상태로 분류하지 않았다.

완전히 같은 상태는 KEEP의 시작1/29뿐이며 FOLLOW0/60이다. 연속 상태에서 정확한 일치가
없다는 사실만으로 모두 학습 분포 밖이라고 판단하지 않는다. 동일 mask 후보 내 최소 RMS
상태와 각 feature block의 RMS/최대 차이를 함께 저장했다. KEEP초기에는 previous_plan
차이가 가장 컸고, 일부 FOLLOW 후기에는 agent feature/slot 변화도 컸다. 전체5,975성분
RMS는 차이를 희석할 수 있고 정규화 배율·slot 순서의 영향을 받으므로, 작은 RMS를 충분한
일반화의 증거로 쓰지 않는다.

로컬 `outputs/sac_current_state_coverage_20261120/`에서 실제 실행한 명령은
`review_current_states.py`, `review_initial_state_exposure.py`, `plot_state_coverage.py`다.
`state-coverage-review.json` SHA
`31a23199f8ef4d65145599fb8e54d246505277987b9c7489635a68f4ef4eb7be`,
`initial-state-exposure-review.json` SHA
`cc8fe32bad71d00bbbcfe5074e10f01f98857c17f3dd7529e3740d41efcc410c`다.

`state-visitation-and-value.png`(SHA
`e08d93618f34c8b73bd6aea414612f575e3fc8b156c502d277900bbfd707b688`)를 생성하고 실제로
열어 글자/범례/축/누락 구간을 확인했다. 위 패널의 return은 하나의 실현된 suffix이며 정확한
기대 Q target이 아니다. 아래는 같은 tick/mask 입력 거리이고, 불일치 구간은 빈 값/별도 음영,
정확히0인 차이는 symlog의0으로 유지했다. 독립 평가나 새 정책 성능 그림이 아니다.

이 결과로 시작 상태 반복 수집만을 충분한 해결책으로 가정하지 않는다. 현재 실행 중인
quota-extension1,000에서 종료 노출을 유지하며 실제 후속 경험을 늘렸을 때, 초기 Q와
실제 미래 보상 차이가 함께 줄어드는지를 확인한다. 그 결과 전에는 수집 변경의 성공,
학습 불가능성, 추가 학습 예산 또는 새 평가 확대를 미리 결론내리지 않는다.

## 00:52 KST 기존96개 장면의 정확한 이동 구간 중복 검사

정책 성능과 별개로 기존train59/개발validation8/별도후보29의 원본을 읽었다. KEEP/FOLLOW
중복을 scene ID로 합치면96개 고유 장면이며, provider 행 수는110/16/56이다. 원래 manifest와
96개 cached proto SHA를 읽기 전후 대조했다. 새 데이터를 받거나 manifest/분할을 바꾸지 않았다.

원본 SDC의 x/y/z·heading·vx/vy·length/width/height를 저장된 좌표 그대로 사용했다.
20개 연속 valid frame이며 양 끝 XY 변위가1m 이상인 구간만 골라, 반올림 없이 float64
벡터 전체가 정확히 같은 window를 찾았다. 같은장면 자체 비교는 제외했다. 중복 후보가
있으면 상대 timestamp 차이도1e-5초 안인지 별도로 검사하도록 했다. 양성 offset 중복,
정지 구간 제외, heading 변경 음성, invalid frame 제외의 합성4case를 먼저 통과했다.

train10,053/validation1,351/별도후보4,922개 moving window, 총16,326개를 검사했다.
96개 장면의 가능한4,560쌍에서 서로 다른 장면의 정확한 일치 구간은0개였다.
모든96원본의 시작 timestamp는0이었다. 이것은 실제 절대 주행 시간이 같다는 뜻이 아니다.
정책 추론/제어/보상 계산은0회이며 평가 결과로 데이터 선택을 바꾸지 않았다.

**이 음성 결과는 drive/시간구간 독립성 입증이 아니다.** 좌표 원점/회전이 다른 동일 주행,
반올림된 상태, 겹치지 않는 같은 drive의 구간, 정지 상태로만 겹치는 구간은 검출하지 못한다.
현재 원본의 populated 필드와 상대 시간만으로 upstream drive 식별자를 확인한 것도 아니다.
따라서 기존 `scenario_fallback_overlap_unverified` 분류를 그대로 유지한다. 추가 목표32개에
3개 부족, 원본 시간축 구조 오류17개 미해결도 그대로이며 별도 후보를 완성된 독립 test로
승격하지 않았다.

실제로 실행한 로컬 `outputs/sac_scene_overlap_20261120/check_exact_sdc_windows.py`와
`exact-sdc-overlap-review.json`(SHA
`a31831fc6383f5e8e8e7bbb22e17795c5b8bebd2fdea6700c323fd668658f432`)에 원본 hash·후보 수·
조건·누락 검출 범위를 기록했다. 표준패키지 import 탐색은 설치가 없어 실패했으며 실제 검사는
프로젝트가 사용하는 vendored Scenario protobuf로 실행했다. 원본이나 실행 환경은 바꾸지 않았다.

## 00:57 KST quota-extension1,000 수집·동일4주행 완료 — 기존 성적 유지, 새 개선 없음

실행 코드c3fe6a7의1,000actual transition/update·저장 감사가 끝났다. 모델 SHA는
`b29adb1a125533dd61a49ef626fa66a5e72e5f4152ce99f39166318e68b14d16`, replay SHA는
`83005d554acd501e022c94fa28ec9bc85fe29866028af664d42d8b5b7c21b004`다.
`review_collection.py --updates 1000`으로 초기 자기5,000의17배열 바이트 동일·모델 설정·
과거main 이력의 root 선택/순서·원본 저장 연결·원래 종료 처리를 독립 검토했다.

고유train8장면, main471/branch529다. KEEP/FOLLOW는 main344/127, branch312/217step이다.
6개 trigger의36roots에서 각8step, 총288quota를 유지했고 선택된6개 path에241step을 더했다.
실제 연결된 path 길이는29/34/47/67/70/42step이며 모두 충돌 종료했다. 분기 terminal54개,
main 종료7개(충돌6·목표1)와 마지막 nonterminal 예산 경계1개를 확인했다. train 목표1회를
고정 개발 평가의 성공으로 더하지 않는다.

| 수집 지표(같은1,000예산) | quota | complete-path | quota-extension |
| --- | --- | --- | --- |
| main/branch step | 664/336 | 589/411 | 471/529 |
| 고유train장면 | 10 | 9 | 8 |
| 선택root | 42 | 7 | 36 |
| branch 실제terminal | 66 | 7 | 54 |
| 사전정지 경험 | 464 | 486 | 375 |
| 검사 통과 경험 | 443 | 370 | 296 |
| 양의 즉시보상 경험 | 154 | 146 | 189 |

긴 실제 연결과 종료 노출은 둘 다 확보했지만, 장면/main 경험과 검사 통과 수는 줄었다.
양의 즉시보상이나 정지 비중 한 지표만으로 수집이 더 좋아졌다고 결론내리지 않는다.
최초77수집의 검사·source·종료·root/path 순서는 quota와 같았으나 별도 GPU 실행의 최대
raw-u 차이7.6607e-7, 보상5.1679e-9, packed 관측2.1607e-6을 기록했다. 첫 quota 이후에는
방문한 상태와 후속 정책도 달라지므로 전체 데이터가 동일한 길이만의 비교라고 하지 않는다.

`review_evaluation1000.py`로 기존3모델과 새 모델의16episode 원본 frame을 재계산했다.
고정2개 개발 장면×KEEP/FOLLOW의4초기 frame/map/seed는 같고, 고유 장면은2개다.

| 모델 | 목표/충돌 | tick | 거절 | 제동 | 정지 | 평균 return |
| --- | --- | --- | --- | --- | --- | --- |
| 초기mean0 | 2/4·2/4 | 216 | 58 | 46 | 22 | -10.015390 |
| quota1,000 | 2/4·2/4 | 236 | 44 | 36 | 12 | -9.981233 |
| independent1,000 | 2/4·2/4 | 236 | 44 | 36 | 12 | -9.981133 |
| quota-extension1,000 | 2/4·2/4 | 236 | 44 | 36 | 12 | -9.981107 |

고속 KEEP/FOLLOW는 각각72tick 목표, 중속 KEEP/FOLLOW는 각각46tick 충돌이다.
거절18.64%·제동15.25%·정지5.08%로 quota1,000과 같다. 작은 return 차이를 성능 향상으로
주장하지 않는다. complete-path1,000의0/4에서 기존 두 성공을 회복했지만 초기mean0과
기존1,000 후보들보다 새로운 성공을 확보한 것은 아니다. 안정 정책/기본 채택은 미달이며
전체16·다른 learner seed·별도 정책 평가로 확대하지 않는다.

로컬 `outputs/sac_quota_extension_20261120/collection1000-review.json` SHA
`1fedb4562ed662eb424f1d85578f3f9116dd3dedd0484632c99fc5bef6d9b10c`,
`evaluation1000-review.json` SHA
`57142671b8453a701217134f699ce6aaa32343e817b7d27bbf6ea888b1b675ab`에 검토 결과가 있다.
train2477740/주행2485021은 완료 후 종료됐고 runner2477739의 고정 종료Q child2486519가
이어 실행 중이다. coverage·수집 terminalQ가 뒤따르고, runner2478192는 같은 두 train
상태×4seed 실제 종료 soft return을 대기한다. 최종 Q/coverage/MC 원본 검토는 미실행이며
완료 후 실행한다. 현재 소스 동결을 유지하며 다음 학습 예산/새 수집 변경은 아직 결정하지 않았다.


## 01:11 KST quota-extension1,000 최종 Q·실제 미래 보상 검토 — 안정 주행 미달

앞 절에서 대기했던 Q/coverage/수집 terminalQ/실제 종료 soft return은 모두 완료됐고
runner2478192/child2490860의 종료도 확인했다. 실행 코드c3fe6a7과 core hash는 검증 종료까지
그대로였다. `review_value1000.py`와 `review_sampling1000.py`를 실제 실행해 원본과 재계산했다.

같은 저장 terminal62개 중 severe44개의 Q MAE는 quota1,000의4.896726에서 새5.584352로
악화됐다. 같은 실제 terminal 상태의 현재 평균 행동도 KEEP6.134260/FOLLOW4.633496점의
오차가 남았다(기존5.988793/3.325783). 다른 수집 terminal 상태의 오차와 섞지 않는다.
새 수집 KEEP tick28/FOLLOW tick55에서 현재 평균·저장 행동·4확률표본은 모두 실제 충돌했으며
현재 평균의 오차는3.464533/6.392146이었다. 이 두 상태는 다른 후보와 같은 상태라는 보장이 없다.

동일2train상태×4paired seed(88019~88022), 첫 행동은 각 모델의 현재 평균, 이후 원래 Gaussian을
사용했다. 실제 종료까지 reward와 규약 entropy를 재합산하고 첫 행동 entropy 제외·gamma0.99·
마지막 bootstrap false·8trace SHA를 검증했다. 모두 충돌했다.

| 상태 | quota1,000 Q−실제 soft-return 평균 | extension1,000 Q | 실제 soft-return 평균±표본SD | 새 Q−실제 평균 |
| --- | --- | --- | --- | --- |
| KEEP `77ae312df21e80b9` tick0 | 12.907675 | -0.432714 | -13.394224±2.056887 | 12.961510 |
| FOLLOW `baac7893b025167a` tick4 | 10.742470 | -0.258815 | -10.766693±0.764925 | 10.507878 |

KEEP의 paired return 차이는 평균+0.000227, FOLLOW는+0.210228±0.498768이었다.
FOLLOW 차이는 [+0.000161,-0.113486,+0.954051,+0.000186]이며8/8충돌과 큰 Q 과대평가는
그대로다. 소수의 반복 표본을 정확한 기대값·독립 장면·안정 주행 개선으로 해석하지 않는다.
고정 개발4주행도 앞 절의2/4성공·2충돌,44거절/36제동/12정지로 기존과 같다.

uniform batch256의256,000개 추출을 RNG부터 다시 계산했다. warmup233,361/online22,639가
기존과 같고 saved replay RNG도 일치했다. severe terminal 추출은 quota3,726회 대비 새3,590회:
warmup2,018동일, main220→187, branch1,488→1,385다. 긴 경로만 수집했던 complete-path의
branch189회보다는 회복됐지만 Q/주행 개선은 없었다. terminal 노출만의 인과 실험은 아니다.

로컬 `outputs/sac_quota_extension_20261120/`의 최종 근거:

- `value1000-review.json`: `be15ea3c579509027377d1c80515c3a440d5b766e84e730988d182803de67173`.
- `sampling1000-review.json`: `a25f78d4248038f63a6d0f83a624d39177d8a24464e794ef169bf15d43a5894e`.
- `soft-returns1000/report.json`: `4ce4fe4a764436abf9d9d81745dbaddf19019f2580cf2f407c40a54d5bf74d5c`.

## 01:12 KST 같은 수집·모델을 총2,000회까지 잇는 제한 비교 시작

**새 수집을 채택하지 않는다.** 긴 실패 경로와 상당한 terminal 노출을 함께 확보해도1,000회에서
주행/초기 Q 개선은 없었다. 기존 quota2,000 결과와 같은 예산으로 한 번 비교하기 위해
추가1,000actual transition/update를 실행한다. 학습 부족이 원인이라고 확정하거나 이 예산으로
수렴한다고 예측하지 않는다. 모델/수집 규칙을 동시에 다시 바꾸지 않는다.

`outputs/sac_quota_extension_continuation_20261120/run_matched2000.py`가01:11:40.586KST에
runner2493589/train2493590으로 시작했다. source는 위 extension1,000의 모델 `b29adb1a…`와
replay `83005d55…`다. fresh 초기화가 아니라 learner/optimizer/target/learner RNG/replay RNG를
복원하며 runtime이 전체 learner serialization 바이트 동일과 BC0/teacher0/counter를 강제한다.
환경과 장면/행동 수집 RNG는 기존 quota2,000과 같은 collection_seed1000으로 명시적으로
재시작한다. 정확한 중간 환경 resume이라고 하지 않고 이전 예산 경계를 가짜 종료로 만들지 않는다.

규약의 보상·검사·PD·원래 Gaussian·uniform replay·실제전이당1update와 별도 비교로 허용된
기존 B-mean/초기화/배율을 유지한다. main runtime/수집 소스 수정은0개다. seed0, 같은
train manifest59고유장면을 사용하며 실제 이번 방문 장면 수는 완료 후 집계한다.

연결 범위는 총2,000수집→원본 저장 감사→같은2개 개발 장면×KEEP/FOLLOW4주행→고정 종료Q→
coverage→수집 terminalQ다. 별도 runner2493877(01:11:46.940KST)은 이 pipeline 완료 뒤
같은2train상태×4seed 실제 종료 soft return을 기존 quota2,000과 비교하도록 대기한다.

근거를 대조한 `review-decision.json` SHA는
`a83372bb0713b2979cb12eaeaf2efde46d7b80d4ed973c5fcc2dd1717d92e06d`다.
원본 완료 후 실행할 collection/evaluation/value/sampling 독립 검토4개를 준비했다.
현재는 아직 미실행이며 이전6,000 replay행의17배열·tick/update prefix·실제 주행 frame·
동일 terminal/MC·512,000추출을 확인해야 한다. 총2,000 이후 자동 추가 학습은 없다.
주행 개선이 없으면 안정 정책/평가 확대/기본 채택으로 처리하지 않는다. 전체 목표는 active다.


## 01:16 KST 작은 soft-return 상승의 구성 분리

로컬 `review_return_components1000.py`를 실제 실행해 기존quota1,000과 새extension1,000의
2train상태×4paired seed, 총16실제 trace의 return을 terminal/그 이전 보상/entropy로 재계산했다.
모든16주행은 충돌이며 원시 trace SHA와 총합을 대조했다. 추가 시뮬레이션/학습은 없다.

KEEP의 평균 soft-return +0.000227은 terminal 기여0, 그 이전 보상 -0.000032,
entropy +0.000259였다. tick/거절/제동/정지 수는 같은4seed 모두 그대로이고 할인 전 누적보상은
평균 -0.000043으로 약간 나빴다.

FOLLOW의 평균 soft-return +0.210228은 할인된 terminal 보상 +0.191172,
그 이전 보상 +0.002377, entropy +0.016679였다. 약91%가 terminal 시점 차이의 산술 기여다.
전체 tick 평균 +1.75, 거절 +1, 제동0, 정지 -0.25이고 할인 전 누적보상은 평균 -0.004829였다.
차이가 가장 큰 seed88021은58→66tick 뒤 같은 -20.017 terminal 보상으로 충돌했다.
soft-return +0.954051 중 terminal 기여 +0.872034, 나머지 보상 +0.006332,
entropy +0.075686이다. 거절49→54/제동44→45/정지25→24이며 할인 전 보상은 -0.027810 나빴다.

이 결과는 충돌 회피나 새 목표 성공이 아니다. terminal 시점 차이가 왜 발생했는지에 대한
인과 개입도 아니며 의도적인 reward hacking이나 보상 변경 근거로 단정하지 않는다.
기존 규약을 유지하면서 성공/충돌/거절/제동/정지와 보상을 함께 판단한다.
`return-components1000-review.json` SHA `a1ef8b79d50c7e4a72945c85060764fd96563640efcdfe8929eb35d5d40e1e75`.


## 01:24 KST CPU 관측별 정책 응답 점검 — 상수 정책은 아니지만 평균 변화가 집중됨

총2,000회 continuation을 유지하면서 완료된 quota1,000/extension1,000/quota2,000을 CPU로
읽었다. 현재 학습 중인 extension2,000의 결과가 아니다. extension1,000 replay6,000행에서
scene/command별 서로 다른 tick 최대5개를 균등 선택했다. Q·보상·검사 결과로 고르지 않았고
같은 tick의 첫 저장 행을 사용했다. 고유train47장면의312상태(KEEP165/FOLLOW147)이며
속도2m/s이상135·0.1~2m/s21·0.1m/s미만156상태다. 모든 ID가 현재 manifest의 train임을
확인했으며 동일 상태를 각 모델에 넣었다. 모든 모델이 이 상태를 직접 방문했다는 뜻은 아니다.

`audit_actor_state_response.py`로 실제 Encoder·마지막 hidden·mu head를 추출했다.
mu head의 affine 출력과 이후 B/g/gain 적용을 독립 식으로 대조했다. CPU만 활성화했고
2개 CPU에 제한했으며 실제 환경·정책 평가·gradient/update·teacher·replay write는 없다.
`review_response_arrays.py`로 저장 배열 hash·선택 인덱스·train 포함·SVD와 속도별 통계를
독립 재계산했다. plot을 생성하고 실제 이미지를 열어 축/범례/분모를 확인했다.

| 같은312상태의 관측별 응답 | quota1,000 | extension1,000 | quota2,000 |
| --- | --- | --- | --- |
| Encoder 중심화 에너지/전체 에너지 | 0.143064 | 0.147689 | 0.152270 |
| mu head 중심화 변화의 첫 주성분 기여 | 99.9334% | 99.9592% | 99.7314% |
| 이동135상태 평균 경로 보정 최대크기의 중앙값 | 0.008503m | 0.009022m | 0.021141m |
| 저속21상태의 같은 중앙값 | 0.001044m | 0.001148m | 0.002176m |
| 정지156상태의 같은 중앙값 | 0.00001224m | 0.00001958m | 0.00001044m |

관측/hidden/mean head가 전부 상수인 정책은 아니다. 다만 이 실제 상태 표본에서 mu head의
중심화 변화는 거의 한 방향에 집중된다. KEEP/FOLLOW를 나누거나 이동 상태만 골라도
대부분 첫 주성분에 집중된다. 이것은 표본 내 관측별 출력 변화의 통계이며 60개 출력의
수학적 자유도가 사라졌다거나 유용한 제어가 불가능하다는 증명이 아니다.

경로 보정은 `max_t ||3*t^2*mu||`만 측정했다. 속도/가속도 anchor를 제외했으므로 전체
경로 길이·PD 추종 오차·안전 여유가 아니다. PCA도 B/g/gain 이전 head 좌표의 통계이며
물리 경로 분산과 동일하게 해석하지 않는다. 한 방향의 변화·작은 보정만으로 수집/모델의
유일 원인을 정하거나 배율·Gaussian·구조를 바꾸지 않는다. 완료될 extension2,000과 같은
312상태의 응답 및 실제 주행을 함께 비교할 후속 근거로 사용한다.

첫 로컬 진단은 command가 packed tensor 맨 앞이라고 둔 검증 assertion에서 실패했다.
신경망 추론 전에 중단됐으며 실패 스크립트/log를 보존했다. 실제 `CORE_SHAPES` 순서로
19개 command block의 offset을 구하고 앞5개 one-hot 및 field mask를 검증하도록 고쳐
완료했다. 학습/계약/관측 runtime에는 변경이 없으며 실패한 결과를 통과로 간주하지 않았다.

로컬 `outputs/sac_actor_state_response_20261120/` 근거:

- `report.json`: `bb110422bb1a6dfde4106e3c264b5da64ead9b8df6a8c9e34176d2bd6568dfda`.
- `array-review.json`: `040038339e83a8b4ec62cf123a4a5dd5db02bf1ed6886243dc07f1f8714870e9`.
- 실제 확인한 `actor-state-response.png`: `23704ad9dd81dc06da8522110f7e121e898997536ca9b413043de5174ac5412c`.

이 결과는 안정 정책 달성이나 평가 확대 근거가 아니다. 실행 중인 총2,000 비교와 runtime
소스 동결을 유지한다.


## 01:34 KST quota-extension2,000 수집·4주행 검토 — 기존 두 성공을 잃어 미채택

학습2,000actual transition/update와 저장 감사가 완료됐다. 모델 SHA
`cce148bb17d6afc54cbfd648445a3a955d6844882cae32ba682a933e461955d9`, replay SHA
`e6857943447874bd3ba18bf82ec1302abd3e4c4eb045f2619556d31d6479d862`다.
`review_collection2000.py`를 실제 실행해 이전6,000행17배열과 tick/update prefix 바이트 동일,
복원 learner/optimizer/target/RNG serialization 동일·source lineage·추가1,000update를 확인했다.

| 같은2,000예산의 실제 수집 | quota2,000 | extension2,000 |
| --- | --- | --- |
| main/branch step | 1,349/651 | 1,117/883 |
| 고유train장면 / root | 19/84 | 15/66 |
| branch terminal | 129 | 108 |
| 사전정지 경험 | 1,080 | 972 |
| 검사 통과 경험 | 908 | 791 |
| 양의 즉시보상 경험 | 249 | 297 |

추가1,000은 main646/branch354, 고유7train장면이다. 전체main episode는충돌11·목표1·
RECORD_END2·상속된 nonterminal 예산 경계1이다. 목표1은이전1,000의train수집에서 발생했고
개발 평가 성공에 더하지 않는다. 긴 경로/정지 비중만 좋아졌다고 전체 수집 효과를 단정하지 않는다.

`review_evaluation2000.py`로 초기mean0·quota2,000·extension1,000·새extension2,000의
16episode 원시 frame을 재계산했다. 동일2개 개발 장면×KEEP/FOLLOW의4초기상태/map/seed를
검증했으며 독립 test가 아니다.

| 모델 | 목표/충돌 | tick | 거절 | 제동 | 정지 | 평균 return |
| --- | --- | --- | --- | --- | --- | --- |
| quota2,000 | 2/4·2/4 | 236 | 48 | 37 | 13 | -9.992331 |
| extension1,000 | 2/4·2/4 | 236 | 44 | 36 | 12 | -9.981107 |
| extension2,000 | 0/4·4/4 | 220 | 87 | 77 | 25 | -20.222395 |

새 거절39.55%·제동35.00%·정지11.36%이며 기존2,000의20.34%/15.68%/5.51%보다 나쁘다.
고속 KEEP/FOLLOW가각67/61tick에 충돌했고 중속두주행은각46tick 충돌했다. **기존 두 성공을
잃었으므로 새 후보 미채택, 추가 예산/전체개발16/새seed/별도정책평가로 자동 확대하지 않는다.**

`review_regression_paths.py`로 실제제안/검사/source/속도/접촉 시간도 대조했다.
고속KEEP는tick34, FOLLOW는tick32에 기존NEW/OK 대 새REUSED/PREDICTED_OFFROAD로 처음
검사/source판정이 달라졌다. 그tick의 두차 위치 차이는각약0.049/0.043m이며 첫 제안30점의
최대차이는약6.80/6.39mm다. 작은첫차이만을 단일원인으로 단정하지 않는다.

새고속KEEP/FOLLOW의 첫제동tick36/34, 최초정지tick61/56, 충돌tick67/61이다. 각각마지막
30tick동안새제안이계속거절됐으며 terminal에는자차가정지한채뒤쪽객체4877과접촉했다.
기존두주행은tick65에제동을시작했어도정지하지않고tick72에목표를달성했다. 새중속두주행도
정지상태에서뒤쪽객체504와접촉했다. 이기하/시간이력은사고책임이나유일한인과판정이아니다.

저장된고속KEEP원본만으로기존SAC2,000 대 새SAC2,000 GIF를렌더했다. 새환경주행/추론0이며
Rule-based비교영상이아니다.25frame·1440×900·실제7.2초+최종hold1.5초(총8,700ms)를읽어검증했고
PNG를실제로열어도로/차량/제안/궤적/거절/제동표시를확인했다.
로컬 `outputs/sac_quota_extension_continuation_20261120/review/quota-vs-extension-2000.gif`,
SHA `bbafc3d73e989ad92c7783ef31873f107a495539c6dbbf4ffaf2aa594c6bac9d`.

새2,000도앞절과동일한312actual train상태로CPU추론하고인덱스/속도/전체선택행동일을확인했다.
mean head첫주성분기여99.9082%, 이동135상태보정중앙값0.021054m, 저속0.002332m,
정지0.00004469m다. 보정크기증가를성능향상으로판단하지않으며전체경로/물리분산과구분한다.
`outputs/sac_actor_state_response_20261120/four-model-comparison.json` SHA
`c7a68327e3429c778a9317bebaa3f612755ecd1e9fe309495d0b2a14b0146a8e`.

로컬 continuation의완료근거:

- `collection2000-review.json`: `1185e07c16cb7e5b1d750275c4af2236f207f283648552b0552d71e8e0b3e0d5`.
- `evaluation2000-review.json`: `b36fab5f2238578343811929a91b9d2b09edefb80f333d0519ae749b150ce68e`.
- `regression-paths-review.json`: `8e19ab43569c12a78f6ca14f645efde325ba7345b88ab969e84e9b418fd5b786`.

01:33:55KST 기준train2493590/주행2501112는완료·종료했고고정Q child2502570은실제실행중이다.
coverage→수집terminalQ와runner2493877의같은2train상태×4seed 실제종료MC가남았다.
최종 value/sampling독립검토는미실행이다. 실행중부분Q수치를최종결과로채택하지않고완료후
검토한다. 새로운수집/학습예산은아직결정하지않았다. 보상·검사·PD·원래Gaussian·BC/teacher0와
현재소스동결을유지하며안정정책목표는active다.


## 01:39 KST 종료 Q·uniform 노출 검토 완료, 실제 미래 보상은 대기

같은62terminal/44severe 저장 행의 min(Q1,Q2)와 MAE를 독립 재계산했다.
extension2,000의 severe MAE는0.847492로 extension1,000의5.584352보다 줄었지만
같은예산quota2,000의0.742834보다 컸다. 같은실제 terminal 상태에서 현재평균의
Q−보상은 KEEP0.673838/FOLLOW1.217628점이며 둘다COLLISION·bootstrap false다.
앞선상태의미래비용이나주행안정을검증한수치가아니다. 실제주행은앞절의0/4성공·4충돌이다.
로컬 `terminal-fit2000-review.json` SHA
`6dbfbc0e937e70177385bf43671db4447cdb97efd71dbf7d1f54c634dedb1999`.

`review_sampling2000.py`를실제로실행해512,000uniform추출의전체인덱스와savedRNG,
CSV와원시replay의ID/tick/reward/bootstrap을대조했다. warmup430,763/online81,237은
quota와같다. severe terminal 추출은quota9,680 대비새8,754:
warmup3,767동일/main606→496/branch5,307→4,491이다. branch전체추출은28,326→42,029로
늘었어도terminal노출은줄었다. 새경험의상태/행동/방문도함께달라단일인과로주장하지않는다.
로컬 `sampling2000-review.json` SHA
`6ac3926a94f6e90bf5955042fbc076995f71d60cf4764ed22657df03fe21c6d7`.

수집terminalQ와그뒤실제종료MC가남았고 `review_value2000.py`는MC완료후실행한다.
추가학습은연결하지않았고소스를그대로유지한다. 안정정책목표는미달이며계속active다.


## 01:48 KST extension2,000 최종 미래 보상 검토 — 비용 과대평가 지속

runner2493877/child2506987은 실제 종료했고 `review_value2000.py`를 실행해 같은 두 train
상태×4paired seed(88019~88022)의 원본8trace SHA·reward·규약 entropy·첫 행동 entropy 제외·
gamma0.99·원래 terminal/bootstrap을 재계산했다. 각 모델의 현재 평균 첫 행동 뒤 원래 Gaussian으로
실제 종료까지 실행한 표본이며, 기존quota2,000과 새extension2,000 모두8/8충돌이다.

| 상태 | 기존2,000 Q−실제 평균 | 새2,000 Q | 새 실제 soft-return 평균±표본SD | 새 Q−실제 평균 |
| --- | --- | --- | --- | --- |
| KEEP `77ae312df21e80b9` tick0 | 12.587732 | -0.313643 | -13.380463±1.966190 | 13.066820 |
| FOLLOW `baac7893b025167a` tick4 | 10.367984 | -0.204530 | -10.257849±0.641258 | 10.053319 |

paired return 차이 평균±표본SD는KEEP -0.319437±0.495077, FOLLOW +0.391735±0.853888이다.
FOLLOW 차이4개는[-0.000017,-0.111177,+0.008173,+1.669961]이며 이 소수 표본의 증가를
안정 주행 개선으로 해석하지 않는다. 고정 개발 평가0/4성공·4충돌과 기존 두 성공의 상실은
그대로다. 종료 Q 적합 개선이 초기의 실제 미래 비용 적합으로 충분히 이어지지 않았다.
수집 terminal 상태들은 다른 수집 방식과 같다고 가정하지 않고, 실제 terminal인 행동만
Q−즉시 보상 오차로 검증했다.

로컬 `outputs/sac_quota_extension_continuation_20261120/value2000-review.json` SHA
`c2e937f550747a0c34a23e63b8dd1572d38b6cb80ec2c5dec87148cdaf0896a9`,
`soft-returns2000/report.json` SHA
`28a4a095502307ba25d7d19449cb4c19e15f9df315c37884eda8026114ff10aa`.
새 수집의 우월성·안정 정책·기본 채택은 입증되지 않았다.

## 01:49 KST 실제 저장 상태의 Q→평균 head 기울기 — 0은 아니며 정지 조건화가 남음

기존quota2,000과 새extension2,000을 동일312actual train상태(47장면)에서 CPU로 미분했다.
현재 평균과 원래 Gaussian8seed(44019~44026)를 사용했고 actor/Q update·optimizer step·
환경 주행·replay write는0이다. 각 모델의2,496확률표본에서 학습된 minQ의 행동 기울기는
모두 유한하고 0이 아니었다. 실제 환경 보상/검사 함수의 미분을 구한 것은 아니다.

현재 physical critic 좌표를w=B^-1*u/c, 평균 head의B/g/gain 적용 전 출력을z라고 하면
head 방향의 Q 기울기는 `mean_gain*g*∂Q/∂w`다. 현재 mean_gain은0.0012098536225957168이며
정지 g=.002에서 비례계수는2.419707245191434e-6이다. 이는 같은물리좌표 Q기울기에 대한
선형 좌표변환 계수의500배 차이이며 서로다른상태에서실측한 기울기중앙값의500배 차이는아니다.

| 모델/속도 집단 | 상태 수 | 물리좌표 Q기울기 norm 중앙값 | mean head Q기울기 norm 중앙값 |
| --- | --- | --- | --- |
| quota2,000 / ≥2m/s | 135 | 0.169261 | 0.000204781 |
| quota2,000 / <0.1m/s | 156 | 0.627997 | 0.000001520 |
| extension2,000 / ≥2m/s | 135 | 0.143131 | 0.000173168 |
| extension2,000 / <0.1m/s | 156 | 0.809839 | 0.000001960 |

실제 head 기울기 norm 중앙값 비율은 이동/정지 quota134.76배·extension88.37배다.
공유 신경망 parameter 전체 기울기·clip·Adam 보정·실제 정책 변화는 이 검사에서 계산하지
않았으므로 학습 실패의 유일 원인이나 g 변경의 개선을 입증하지 않는다. 이미알려진조건부
작은 정지 탐색과 연결되지만, 기울기가 0으로 끊겼다는 설명은 이 표본에서 지지되지 않는다.

처음 진단은 frozen 관측/parameter를 JIT closure로 넣어 constant-folding 경고가 발생했고,
서로 다르게 반올림된 두 float32 Q계산 그래프의 end-to-end gradient를1e-12 수준으로
비교한 assertion에서 실패했다(최대 해당절대차2.821e-10). 실패script/log/미완료report를보존했다.
관측/parameter를동적인JIT인자로넘겨불필요한상수접기를제거했고, 동일Q cotangent에대한
선형 head→raw좌표 변환의 VJP만 독립 검증하도록 검사범위를 명시했다. 신경망 Q 미분의
독립 finite-difference 검증을 통과했다고 주장하지 않는다. production 검사/허용치는바꾸지않았다.

`audit_mean_gradient.py`와 `review_gradient_arrays.py`를 실제 실행했다. 원본모델/관측 hash,
선택인덱스, 이전CPU mean출력 동일, 좌표 VJP, 원래Gaussian 표본 재구성을 확인했다.
Gaussian 재구성 최대절대차는quota3.4694e-18/extension2.6021e-18이다.
로컬 `outputs/sac_actor_mean_gradient_20261120/report.json` SHA
`181306d61eadeeeea9b9086c762355fca445b6d7111140534d92b0bf7a1b329d`,
`array-review.json` SHA
`c20258d3b49f24f91e3d661f1dfc0b3f7e3e2f52e60da4903d4b0137c629f076`.

## 01:51 KST 두 수집 방식의 동일 총5,000회 제한 비교 시작

**실패한extension2,000을 채택하지 않는다.** 종료직전 적합과 초기비용 사이의 큰차이가
남고, 현재 Q기울기가 0인 것은 아니었다. 수집방식의1,000/2,000 결과만으로 수렴이나
학습부족을 확정할 수 없으므로, 모델/수집 규칙을 고정한 같은 총5,000예산을 한 번 비교한다.
학습부족이 유일 원인이라거나5,000이면 충분하다고 예측하지 않는다. 과거 회귀를 지우지 않는다.

각 arm은자신의2,000 checkpoint에서 추가3,000actual transition/update를 이어받는다.
순서는 quota→extension, 총추가6,000이다. 같은train manifest·seed0·최초자기warmup5,000·
모델설정/배율·보상/검사/PD·원래Gaussian·uniform batch256·실제전이당1update·BC/teacher0를
유지한다. 해당arm의learner/optimizer/target/learner RNG/replay RNG를복원하고 환경/장면·행동
수집 RNG는두arm 모두collection_seed2000으로명시적으로재시작한다. 정확한환경resume은아니다.
새 학습 알고리즘/모델/수집 규칙 구현은0개이며 기존 continuation의총예산상한5,000 안에서 실행한다.

`prepare_comparison.py`로두완료원본의버전·설정·ARM_FIELDS동일, 각collection mode/소스hash,
checkpoint/replay/이력·유효continuation을 실제검증했다. quota는기존quota 모드를그대로사용하며
새 extension 플래그는활성화하지않는다. 원본근거와결정은로컬
`outputs/sac_collection_budget5000_20261120/review-decision.json`, SHA
`9ce6247b5540c7bdf4ded05f14eda95aa3cf3a80d9cdf92e3c7e4cc3d4add316`.

01:50:35.571KST runner2510498/첫quota train2510518으로시작했다. runner SHA
`5ab84168a7e7a7cf25641ad7dda0fca774b4ceca3b4f21b5face1a292bd1edbd`.
quota source모델7c898963…/replay3713625f…, extension source모델cce148bb…/replaye6857943…다.
현재 첫arm만실행중이며 extension은대기다. 실제report에서 restored_learner_bytes_identical=true,
상속2,000·총2,021update·추가21actual step·collection_seed2000을확인했다.

각arm은학습5,000→저장감사·이전7,000행17배열/prefix검사→같은4개개발주행→고정종료Q→
uniformcoverage→수집terminalQ→같은2train상태×4seed실제종료MC를순서대로실행한다.
두arm을동시에GPU에올리지않는다. 결과원본완료후독립재계산하며실패/회귀도그대로남긴다.
오류/해시/비유한/lineage불일치는batch를중단한다. 총5,000초과추가학습·전체validation/
새seed/독립정책평가는자동연결하지않는다. 모델선정/평가확대는실제다중주행지표를검토한뒤판단한다.
현재 안정정책은미확보이며전체목표는active다.

## 02:05 KST 같은2,000회 실제 미래 보상의 구성과 후속 독립 검토

같은 두 train 상태×4 paired seed88019~88022의 quota2,000/extension2,000 원본
MC trace를 다시 읽어, soft return을 종료 보상의 할인값·종료 전 보상의 할인합·
entropy 기여로 분해했다. 첫 평균 행동의 entropy 제외, gamma0.99와 실제 terminal을
그대로 사용했고 추가 시뮬레이션이나 학습은 하지 않았다. 모델/관측 식별자는 앞선
최종2,000 검토와 같으며 두 모델 모두8/8충돌이다.

| extension − quota, 같은2,000회 | KEEP 평균 차이 | FOLLOW 평균 차이 |
| --- | --- | --- |
| 종료까지 tick | -2.50 | +3.25 |
| 할인하지 않은 실제 reward 합 | +0.017722 | -0.008112 |
| 종료 reward의 할인값 | -0.310483 | +0.362641 |
| 종료 전 reward 할인합 | +0.007664 | +0.004326 |
| entropy 기여 | -0.016618 | +0.024767 |
| soft return | -0.319437 | +0.391735 |

FOLLOW의 soft return 개선 중 약92.6%는 종료 보상의 할인값 차이이며,
할인하지 않은 reward 합은 오히려0.008112 나빠졌다. 따라서 작은 FOLLOW soft-return
개선을 안전한 주행 확보나 실패 해결로 보지 않는다. 같은2개 train 상태의 소수 표본이며,
정책이 의도적으로 보상을 악용했다거나 할인율 자체가 유일 원인이라는 주장은 하지 않는다.
보상/discount/검사 조건은 변경하지 않았다. 개발4주행0성공·4충돌과 미채택 판단도 유지한다.

근거는 로컬 `outputs/sac_collection_budget5000_20261120/reviewer-paired2000-components.json`,
SHA `53b7f1bfcd708141468c35f3f7bff78dab1bd4870e2ca965402554d176b2004d`다.
각 모델의 실제8trace SHA·reward·bootstrap·최종 outcome과 합계/표본표준편차를 재검산했다.

진행 중인5,000회 실험의 결과를 단계별로 검토하는 로컬 `review_arm.py`도 준비했다.
이미 완료된 extension2,000에 collection/sampling/evaluation/value 네 단계를 실제 실행해
기존 독립 검토와 주행 지표·수집 구성·512,000 batch draw·종료 Q/MC가 일치함을 확인했다.
초기 reviewer가 행동 이름을 KEEP/FOLLOW로 가정해 빈집단 나눗셈 오류로 종료한 기록은
`review_arm_initial.py`와 초기 log에 보존했다. 실제 fixed_keep/fixed_follow 이름을
원본에서 읽고 각2episode인지 확인하도록 수정했다. 학습/환경 runtime 변경은 없다.

최종 reviewer SHA `91727c50ad71b1f7ccc35a0b0cca44c4ad3ddafdce79e5c55ad67ed1de4507b9`,
실제 과거 자료 검증 `reviewer-ready.json` SHA
`850d667ef525af364d6c570dbbe125d5321cda066459c540289765e69408ec4a`.
5,000회 결과는 이전7,000행17배열과 이력 prefix, 실제4주행 frame,
1,280,000 uniform draw와 저장 RNG, 같은 train 상태의 실제 종료 return을 확인한다.
추가3,000 update 동안 새로 모은 severe transition이 실제 몇 번 추출됐는지도 별도로 계산한다.
이는 검토 준비 완료이며 아직 끝나지 않은5,000회 결과를 검증했다고 기록하지 않는다.

02:04:55.314KST 독립 CPU 검토 runner2517850을 시작했다. 로컬 `run_reviews.py` SHA
`0be0609702b06278ae5e35fa06efd05a9431a96b8f6611b08ed1882e659cb363`.
기존 학습 runner2510498/child2510518은 계속 실행 중이고 extension arm은 대기다.
독립 검토 runner는 완료된 원본 phase만 읽으며 오류/해시 불일치 시 자체 검토를 멈춘다.
모델/수집 코드·학습 예산·채택 조건을 변경하지 않고 새 GPU 작업도 만들지 않는다.
진행 상태는 `status.json`, `reviews-status.json`에서 각각 구분한다.
현재 코드 기준은 `feature/yuntae-sac-training`의 HEAD `9f92284`이며 이 절은 문서 추가다.
목표는 active, 안정 정책과 전체 validation/새 seed/Rule-based 최종 비교/별도 평가는 미완료다.

## 02:12 KST 복원된 Adam까지 포함한 격리 1회 업데이트 진단

앞선 Q→mean head 기울기만으로 실제 optimizer의 정책 변화까지 판단하지 않기 위해,
완료된 quota2,000/extension2,000 snapshot을 CPU에 바이트 동일하게 복원했다. 각자 저장된
replay RNG의 복사본으로 기존7,000행에서256개를 uniform 추출하고, 기존 SAC의
Q→actor→alpha→target 순서와 clip·Adam 상태를 유지한 **격리된 가상 update 1회**를 계산했다.
새 transition을 넣지 않았으므로 다음 실제 online update의 재현이나 학습 연장이 아니다.
새 가중치는 저장/채택하지 않았고 원본 가중치·replay·진행 중인5,000회 실행은 바꾸지 않았다.

평균 경로 변화는 이전과 같은47개 train 장면의312상태에서 측정했다. 이동≥2m/s135개,
0.1~2m/s21개, 정지<0.1m/s156개이며 실제 저장 관측과 마스크를 사용한다.
고정 anchor에서 `max_t ||3*t²*(mu_after−mu_before)||`를 계산한 값으로,
차량이 실제로 이동한 거리나 안전 검사·PD 추종 오차가 아니다.

| frozen snapshot | actor 전체 gradient norm | actor clip 배율 | mean head parameter Δ L2 | 이동 경로 변화 중앙값(m) | 정지 경로 변화 중앙값(m) |
| --- | --- | --- | --- | --- | --- |
| quota2,000 | 2.048942 | 1.0 | 0.037207 | 2.721501e-5 | 3.258043e-7 |
| extension2,000 | 0.063141 | 1.0 | 0.032303 | 2.228744e-5 | 7.503245e-8 |

이 두 batch에서 actor clipping은 작동하지 않았고 mean head parameter가 실제로 변했다.
Adam 이후에도 정지 상태의 물리적 경로 변화가 작았지만, 서로 다른 상태/batch의 단회
진단이므로 원인 분리나 장기 학습 불가능성을 입증하지 않는다. 앞서 본 기울기 크기의
비율을 실제 parameter/경로 변화의 비율로 그대로 해석해서는 안 된다.

새 gradient를0으로 놓고 저장된 actor Adam momentum만 한 번 적용한 별도 대조값도
계산했다. 정지 상태 경로 변화 중앙값은 quota4.282407e-8m/extension7.299836e-8m이며,
full update와 이 대조 출력의 차이 중앙값은 각각3.562259e-7m/9.845817e-9m다.
이는 optimizer 기억과 새 gradient의 차이를 관찰하는 가상 비교이며 가산적인 원인 분해나
독립된 학습 정책이 아니다. 실제 sample/보상/안전검사/PD/BC·teacher0 조건을 변경하지 않았다.

실제 실행 명령은 로컬 `outputs/sac_actor_optimizer_response_20261120/`의
`audit_snapshot_update.py`, `review_arrays.py`다. 두 모델의 source SHA·복원 바이트·원본
in-memory state 불변·copied batch indices·312선택 관측·출력 배열 통계를 확인했다.
기존 eager CPU와 이번 JIT CPU의 초기 mu 최대 차이는 quota2.3283e-10/
extension2.9104e-10이며 명시한 atol2e-9/rtol2e-5 이내다. 정지 그룹의 해당 경로 차이
최댓값은1.2278e-11m/1.8984e-11m로 측정한 update 변화보다 작다. 이 검토는
parameter gradient를 별도 수식으로 다시 미분한 검증은 아니며 기존 canonical update를 썼다.

`report.json` SHA `f6303cb1f1dacb917c4294fda3d9669682c80a108f1d25a1817b36a687415787`,
`array-review.json` SHA `8a0d88778bca25049fb3c53c05433efa92631f22332c9d2ffedfbf7164210a6e`.
diagnostic 프로세스2519233은 exit0으로 종료했다. 진행 중인 matched5,000 비교는
02:12:20에 첫 quota3,537/5,000, extension대기이며 학습 runner2510498/child2510518과
독립검토 runner2517850의 실제 생존·모든 고정 소스 SHA 불변을 확인했다.
다음 결정은 현재5,000회 두 arm의 실제 주행/실패 경험/미래 보상 검토 후 내린다.
현재 코드 기준 HEAD `bca1f12`, runtime c3fe6a7 유지. 안정 정책은 미확보이며 목표는 active다.

## 02:40 KST 첫 quota5,000회 완료 — 저장 검토 통과, 실제4주행 모두 충돌

동일 총5,000회 비교의 첫 quota arm이 학습을 완료했다. source2,000에서 실제3,000개의
자기 경험과3,000update를 추가했으며 총 online5,000/replay10,000이다. full learner 복원,
상속7,000행의17배열과 tick/update prefix 바이트 동일, collection RNG restart2000,
BC/teacher0을 기존 감사와 독립 `review_arm.py collection`에서 실제 확인했다.
모델 SHA `4d30fbb35b9201a30cfeed45c05c076ea4544ac6d176ae5d0b91522109940fbd`,
replay SHA `166a27a86a0abd2429375e838689cd6c9b792c2153b354928b8cb7574e4fe5ac`.
보상·검사·PD·원래Gaussian·모델/배율·uniform replay·실제전이당1update는 유지했다.

수집은 main3,456/branch1,544, 고유train37장면·198roots·branch terminal296개다.
main은 통과2,152/양의reward349/pre-step정지2,299이며, branch는 통과391/양의reward286/
pre-step정지477이다. 추가3,000분은 main2,107/branch893, 고유24장면·114roots·
branch terminal167개다. 전체 수집전이의 pre-step정지는2,776/5,000(55.52%)이다.

main 실제종료44개는 COLLISION33/ROUTE_GOAL2/RECORD_END8/TASK_LOST1이다.
보고서의 별도 DIAGNOSTIC_BUDGET1개는 원래 bootstrap을 유지한 과거 예산경계 기록이며
실제 terminal로 세지 않는다. 새3,000분의 main 종료는 COLLISION19/RECORD_END5/
ROUTE_GOAL1/TASK_LOST1이다. 새 FOLLOW train 성공은946ee2be6c924b68의update2627,
tick55에서 수집됐다. **학습 중 계속 변한 정책의 자기 경험이며 고정 모델 평가 성공이 아니다.**

같은 개발2장면×KEEP/FOLLOW의 초기 frame/map/seed4쌍과3개 모델의 원본12episode를
독립 재계산했다. 이전quota2,000의 고속 KEEP/FOLLOW 두 성공을 모두 잃었다.

| 같은4개 개발 주행 | quota2,000 | quota5,000 |
| --- | --- | --- |
| 목표 / 충돌 | 2/4 · 2/4 | 0/4 · 4/4 |
| 전체 tick | 236 | 210 |
| 경로 거절 | 48(20.34%) | 89(42.38%) |
| 제동 | 37(15.68%) | 78(37.14%) |
| post-step 정지 | 13(5.51%) | 25(11.90%) |
| 평균 training return | -9.992331 | -20.233012 |

고속KEEP/FOLLOW는58/62tick에충돌했고 중속KEEP/FOLLOW는46/44tick에충돌했다.
네번 모두 정지 상태에서 뒤쪽 객체와 접촉한 기하 기록이 있으나 귀책/유일원인으로 해석하지
않는다. 초기 mean0도같은4주행에서2성공이므로 새성공확보가아니며 현재후보는미달이다.
고속KEEP 원본에서 첫제동31/첫정지54/최종충돌58tick, 종료직전30tick 연속새경로거절을
확인했다. 이 성공 회귀를 Q 적합 개선으로 상쇄하거나 감추지 않는다.

로컬 `outputs/sac_collection_budget5000_20261120/quota/`의
`collection-independent-review.json` SHA
`8e60cbdffeb795bd2b49e390767d25eca1ddb95098b8ede3660b7212f1c223ab`,
`prefix-review.json` SHA
`ca00a67ebfefb8f410d08210cc616afcb16c62b81b46066528bc0c3349535107`.
`review_arm.py evaluation`도 실제완료됐으며 provider별 지표와 원본 frame 해시를 보존했다.
`evaluation-independent-review.json` SHA
`2305fbbc386d1efe97a4b269e236ce7dddce5f827a6264344b97528d0acb88e8`.

02:40:13KST 학습 child2510518/평가 child2524893은 종료하고 첫Q child2526236으로
전환했다. runner2510498과 독립검토2517850의 실제생존을확인했다. 현재 Q→coverage→
branch Q→동일2train상태×4seed 실제종료return 검토가 남았고 extension5,000은대기다.
이 시점에 두 arm 전체 비교나 Q/MC 검토를 완료했다고 주장하지 않는다. 그 결과 뒤에
다음 조건을 결정하며 전체validation/새seed/별도평가/기본채택은 진행하지 않았다.
문서 기준 HEAD `b7aec2e`, runtime c3fe6a7 유지. 안정정책 미확보·전체목표 active다.

## 02:54 KST quota5,000 최종 Q/미래보상 검토와 extension 실행 전환

첫 quota arm의 예정된 학습·저장·4주행·Q·coverage·실제 종료 MC와 독립 검토를 마쳤다.
`quota/reviewed-status.json`은 REVIEWED_DRIVING_REGRESSION/qualified=false로 기록했다.
이는 첫 arm의 근거 검토 완료이며 안정 정책이나 두 arm 전체 비교 완료가 아니다.

고정된 초기 자기 경험의 terminal62행 중 severe44행 Q MAE는2,000회의0.742834에서
5,000회의0.293067로 감소했다. 실제 고정 종료 상태에서 현재 평균을 실행한 KEEP65tick은
Q−19.877012/reward−20.014/error+0.136988, FOLLOW72tick은Q−20.272860/
reward−20.017/error−0.255860이며 둘 다 COLLISION/bootstrap=false였다.
수집 branch 상태에서도 현재 평균의 terminal 오차는 KEEP28tick+0.308922/
FOLLOW63tick+0.212333이다. `review_terminal_fits.py --arm quota`를 실제 실행해
고정62행 정체성·raw 분기 행동/reward/terminal과 실제replay 인덱스를 대조했다.
서로 다른 수집 모드의 분기 상태를 동일 상태라고 비교하지 않는다.

그러나 같은 두 train 상태×4 paired seed88019~88022의 실제 종료까지 실행은8/8충돌이다.
원래 Gaussian 후속 행동을 사용한 원본 trace의 SHA·첫 평균 행동 entropy 제외·gamma0.99·
terminal/bootstrap·return 구성과 표본 통계를 `review_arm.py value`에서 대조·재계산했다.

| 현재5,000회 | KEEP tick0 | FOLLOW tick4 |
| --- | --- | --- |
| 현재 평균 첫 행동 Q | -0.722204 | -0.437663 |
| 실제 soft return 평균±표본SD | -14.123892±1.501068 | -11.909348±0.173563 |
| Q−실제 평균 | 13.401688 | 11.471685 |
| 이전2,000회의 Q−실제 평균 | 12.587732 | 10.367984 |
| paired return 차이 평균±표본SD | -1.062866±0.702300 | -1.259763±1.070685 |

각행동4개의paired차이가모두음수이며 주행 개선이 없다. 평균종료시점은KEEP5tick,
FOLLOW5.25tick 빨라졌고, return 차이는 각각 종료보상 할인−0.622182/−0.557434,
종료전보상+0.006949/−0.026679, entropy기여−0.447633/−0.675651로 구성된다.
할인하지않은reward합차이는+0.026732/−0.007181이다. 이러한 작은 보상 성분 차이를
개발0/4성공·4충돌이나8/8 MC충돌보다 우선하지 않는다. 소수 train 표본이며 독립 test가 아니다.

uniform replay1,280,000draw도 원시배열/CSV와 저장 RNG를 독립 재현했다. severe
terminal 추출은34,064회(warmup7,734/main2,729/branch23,601)이며 이전2,000은9,680회다.
추가3,000update의768,000draw는 warmup455,945/main214,098/branch97,957이다.
이 중 새로 수집한 severe186행(새main19+branch167)은7,302회추출됐고8행은한번도
추출되지 않았다(main1/branch7). 새row는시간이늦게추가되어노출기회가다르며 표본횟수가
충분한 상태/행동 coverage나 인과적 해결의 증거는 아니다. 실패 경험과 종료적합이 늘어도
초기 미래비용과 실제주행은현재예산에서미달이라는범위로해석한다.

로컬 `outputs/sac_collection_budget5000_20261120/quota/`의
`terminal-independent-review.json` SHA
`fe34a5919eec5fbc69f38d0a907e063902df83c7d2eb81fbd97dc7b3406639a4`,
`sampling-independent-review.json` SHA
`e783e15a86f1d3a6736967b970cffbea61212101f1922d30c5b68868a865f5b9`,
`value-independent-review.json` SHA
`dd18b5b9759119d2ccabe29dd3b6688cbd50e443eb3e1dcf5740ecd9fd9fc344`,
MC `soft-returns5000/report.json` SHA
`1f9220052cb2f47c188b69a71a6abf09be8d7b53e03327c25c33545f6c94b7f3`.

### 추가 CPU terminal 진단의 수치 대조 실패 — 별도 한계 보존

초기warmup62행만으로새terminal행의적합을대표할수없어 저장된모든실제terminal행을
CPU에서평가하는 `audit_all_terminal_values.py --arm quota`를추가했다. stored행동에만
그행의실제reward를대응시키며 현재평균의반사실적reward로사용하지않는다. 환경/학습은0이다.

첫시도는CPU와기존GPU Q의atol3e-5/rtol3e-5 대조에서실패하고exit1로종료했다.
첫불일치절대차0.00010729이며 initial script/log/미완료디렉터리를보존했다. 허용치를넓혀
통과처리하지않았다. 실제62행obs/field/item/raw_u입력을원본warmup과바이트대조하고,
CPU에서도같은62배치로재계산해배치크기차이와플랫폼차이를분리해서기록했다.

최종CPU계산에서원래엄격한기준은124개Q값중41개미달이다. 같은62배치CPU/GPU 최대차는
0.004009247이며 CPU전체terminal배치와CPU62배치간최대차는1.907349e-6이다.
가중치와입력동일성을확인했지만차이의구체적인연산/하드웨어원인을확정하지않았고,
새online행의GPU차이는아직미측정이다. **CPU추가통계는잠정자료로보존하고 이번주행/
가치판정의확정근거로채택하지않는다.** 현재GPU pipeline이끝난뒤같은GPU/62배치의
기준재현과새terminal행검토를수행해야한다. 학습/물리검사 허용치는변경하지않았다.
CPU추가 `all-terminal-values/report.json` SHA
`a4847a981a44a0556844a1acaa003a04036743f67cfd699e6d55a0416cd7ec9a`.

02:53:55KST 기존runner2510498은extension/train5000 child2532783으로전환했다.
실제총2,004/추가4update·learner복원바이트동일을확인했고,독립검토2517850은
extension/audit5000을기다린다. 첫MCchild2530380은정상종료했다. 고정source hash전부유지,
두번째arm은원래정한같은조건/예산으로진행한다. 두arm결과를검토한뒤다음조건을결정한다.
현재문서기준HEAD `4c0b1fe`,runtime c3fe6a7 유지·안정정책미확보·전체목표active다.

## 03:02 KST 같은 GPU/62배치 재검증 준비와 대기 실행

추가CPU terminal진단의수치차이를덮어두지않고, 현재GPU실험과겹치지않는후속검증을
로컬 `outputs/sac_collection_budget5000_20261120/`에준비했다. 이것은학습조건변경이나
새학습실험이아니며 현재모델을고정한stored terminal행동의읽기전용추론이다.

`audit_terminal_gpu62.py`는두arm의원래pipeline완료를먼저요구한다. 같은GPU·동일한
62개배치에서기존warmup62개Q를원래atol3e-5/rtol3e-5로재현해야새terminal행통계를
보고한다. 기준재현실패시FAILED로기록하며허용치를늘려통과시키지않는다. 모든행을
같은62개씩계산하고마지막배치의반복padding은결과에서제외한다. 현재평균에stored reward를
대응시키지않고각각실제로수집된raw행동에만그reward를사용한다.

두script를AST검사하고완료된quota의실제402terminal행과기존62행prefix를CPU에서
대조했다. 7배치·padding32행제외후실제402인덱스가중복/누락없이원래순서로한번씩
남는것을확인했다. 이는실제입력분할검토이며 **GPU실행과수치일치검증은아직미실행**이다.
extension최종모델의입력검토는해당모델이완료된뒤수행한다.

preflight 상태는PREPARED_INPUTS_CHECKED_GPU_NOT_RUN,
`terminal-gpu-preflight.json` SHA
`478753636c8abd4671897dd9f80b5e360c5dcba6958afcfa4c47eb14fb5fbf3a`,
`audit_terminal_gpu62.py` SHA
`540a29d2a1a9fc2b82288a78e2d0c64e3a64ff35669c815c56997d4399943b5b`.

03:01:51.891KST `run_terminal_gpu_review.py` PID2538421을시작했고실제생존과
WAITING/MATCHED_PIPELINE_COMPLETION을확인했다. runner SHA
`341b2d2086cff4a57077ceea5e833098417a358355c1dd3afcb758d1ae4d3a12`.
두arm원래pipeline이끝나고 `nvidia-smi --query-compute-apps=pid`의살아있는CUDA
계산process가없을때quota→extension순서로추론한다. 자동재시작/추가학습/정책채택은없다.
기준재현·source hash·실행오류가나면실패기록을보존한다. 이추가검토가끝나기전까지
기존core와GPU검증script를고정한다. 결과완료후원시배열/구성통계의독립검토가필요하다.

현재학습runner2510498/extension child2532783·기존독립검토2517850은계속실행중이다.
03:01:51에는extension2,579/5,000을확인했다. 상태는원래 `status.json`,
`reviews-status.json`과새 `terminal-gpu-review-status.json`을구분해확인한다.
문서기준HEAD17957b2·runtime c3fe6a7 유지. CPU수치문제해결/안정정책확보는아직아니며
전체목표는active다.

## 03:18 KST 과거 온라인 설정 중복 점검

현재 두 수집 방식의 비교가 끝나기 전에 다음 설정을 결정하지 않도록, 완료된 과거
온라인 진단의 설정만 목록화했다. `rg --files --no-ignore outputs -g report.json`의
141개 경로에서 sources 사본을 제외하고, 온라인 완료 상태·모델 파일 존재·기록된 모델
SHA로 중복을 제거한 범위는 33개다. 전체 연구 이력이나 모든 오프라인 진단을 포괄하는
검색은 아니다. 누락된 설정 필드는 ABSENT로 남겼으며 현재 기본값으로 소급하지 않았다.

이 목록에서 entropy 평균 좌표가 명시되지 않은 과거 두 모델의 실제 가중치 SHA를
재계산해 기록과 일치함을 확인했다. `sac_short_updates_20261118/online100-seed0`는
integrated/gain0.001, `sac_action_coordinates_20261118/online100-seed0`는
integrated/gain0.01/whitened Q다. 당시 문서의 raw B 평균 실험·주행 실패와 대조했지만
과거 주행을 재실행하거나 모든 33개 모델의 가중치를 재검증한 것은 아니다.

이 제한 목록에는 physical_whitened Q와 비entropy 평균의 조합이 명시된 완료 온라인
모델이 없었다. 이는 해당 조합의 효과나 실험 필요성을 입증하지 않는다. 과거 raw B 평균
실패를 현재 physical Q 설정과 같은 실험으로 취급하지 않으며, 새 실험 선택·구현·실행은
하지 않았다. 승인된 B 평균은 여전히 별도 비교이고 기본 정책 채택이 아니다.

로컬 `outputs/sac_collection_budget5000_20261120/`의
`completed-online-setting-inventory.json` SHA
`5b0165c06709bce9d400d0d17c6fea0a600e50fd479b6ccd833cc0a1cc18730f`,
`selected-historical-model-check.json` SHA
`f30a0e39e13d79da092cc341b8e5e663382cc67a18354e423b6bae30b12b913a`.
03:16:44 extension은3,706/5,000, 실제 학습 child2532783과 runner2510498,
독립 검토2517850, 대기 GPU 검토2538421의 생존을 확인했다. 두 arm 비교와 후속 GPU
검토 완료 전까지 runtime/script 고정을 유지한다. 문서 기준 HEAD1824ebe, 실행 코드
c3fe6a7이며 이번 변경은 기록뿐이다. 안정 정책은 아직 미확보다.

## 03:27 KST terminal 저장 배열의 별도 통계 검토

후속 GPU 결과를 검토할 로컬 `review_terminal_arrays.py`를 준비하고, 먼저 완료된
quota CPU 자료에 `--arm quota --cpu-statistics-only`로 실제 실행했다(exit0).
원본 report/source/model/array 해시를 대조하고 replay의 실제 terminal 인덱스402개,
scene/tick/reward, teacher0, coverage의 종류/추출 횟수를 확인했다. Python statistics와
직접 percentile 보간으로 기존 그룹 통계를 다시 계산해 절대오차1e-12 이내로 일치했다.
severe는373개다. 기존 provisional CPU 수치를 바꾸거나 GPU 일치로 재분류하지 않았다.

검토는 저장된 Q 배열의 통계·행 대응 검토이며 신경망 추론의 독립 재실행이 아니다.
GPU용 분기는 기존62행의 원래 atol3e-5/rtol3e-5와 전체 terminal 배열 통계, CPU 차이를
확인하도록 준비했으나 GPU 결과가 아직 없어서 미실행이다. 새 학습/시뮬레이션0,
현재 pipeline의 core/source hash 변경0이며 추가 reviewer는 기존 runner에 연결하지 않았다.
두 arm GPU 검증 완료 후 `--arm quota`, `--arm extension`을 각각 수동 실행해야 한다.

로컬 `outputs/sac_collection_budget5000_20261120/`:
`review_terminal_arrays.py` SHA
`ad08716ee15b8da66e5457730501ad09427d78b0afc1dfcd92f40ad7c341bebf`,
`quota/all-terminal-values/independent-array-review.json` SHA
`08ddbbfb1b6400e05cfb02cad5764b5b0d0edbb0cb184d052d63a24f0ff4944f`.
03:26:22 extension 학습4,323/5,000·child2532783 생존 확인. GPU 수치 차이 해소와
안정 정책 확보는 아직 미완료다. 문서 기준 HEAD1d47763, 실행 runtime c3fe6a7 유지.

## 03:43 KST extension5,000 학습 완료·검사 오류 수정·평가 복구

두 번째 extension도 총5,000 online update를 완료했다. 새 모델 SHA
`2dfc45c7fbc3777225d49b05864268c88e43998235b6ec44bf7139a7db9512ab`,
replay SHA `819d68a8e9a15a26101dfa5d090c02e23df3113392f980fab32f0c54acb26c3e`,
학습 report SHA `3983e7c680b80e1d9ffc9e0506263f7f4736ae533e0ab3a12680382c9bc071fd`.
하지만 최초 `audit5000`이 exit1로 실패해 주 runner와 두 대기 reviewer가 실제 종료했다.
03:37:49 로그·PID 부재·FAILED 상태로 확인했으며, 단순 관찰 timeout으로 재시작한 것이 아니다.

오류는 `audit_sac_branch_replay.py`의 extended root 마지막 상태 검사였다. 실제 전이 중
비최종 위치에서 열린 경로는 collection_id2,000 단 하나다. 이는 이전2,000회 예산에서
bootstrap=true로 끝난 정상 분기 경로이며, 이어지는2,001행은 명시적 환경 재시작의
새 main tick0이다. 당시 파일 마지막을 허용하던 검사 조건이 continuation 뒤에는 그
검증된 과거 예산 경계를 허용하지 못했다. terminal을 조작하거나 수집 모델을 바꾸지 않았다.

검사 코드에 `validated_budget_end_indices`를 추가했다. continuation의 모든 원본 파일
SHA·모델/수집 설정·완료 상태·누적 count·중첩 이력과 tick/update byte prefix를 확인하고,
다음 행이 다른 episode의 main tick0인 경우에만 과거 예산 경계를 허용한다. 임의의
미종료 분기나 확인되지 않은 경계는 계속 거절한다. bootstrap/실제 종료 검사는 그대로다.

관련 continuation/quota-extension/quota-budget/branch-reconstruction pytest30개 통과.
새5,000 실제 audit와 독립 collection 검토 모두 exit0이고, 기존extension2,000와quota5,000도
수정 검사로 재실행해 통과했다. 새 extension의 검증된 budget collection_id는1,000/2,000/5,000.
이전7,000행17배열과 tick/update prefix가 바이트 동일하고 learner 복원·seed2000 환경
재시작을 확인했다. 새 audit SHA
`88379e7a2bb31baac5ccead7bf645c094c7eae95f5b1cf1c177de1542317c868`,
독립 collection SHA `4d5d09efe2dd1dcb4e12e312dada4817f5fe74976f3529ff4978bda8379ee52f`.

전체 main2,908/branch2,092, 고유 train29, roots150, branch 실제 terminal233개다.
선택 경로 연장25roots·추가892step이며 pre-step 정지는 main1,785+branch844=2,629/5,000.
추가3,000회만 보면 main1,791/branch1,209, 고유 train20, branch terminal125개다.
이 경험량이나 audit 통과는 주행 성능 통과를 의미하지 않는다.

로컬 `outputs/sac_collection_budget5000_20261120/audit-failure-20261120T0337/`에
원래 FAILED 상태3개·로그·검사 소스를 보존했다. 원래 주 status SHA
`3b7a2ed195f381f8060251a6c45ec4e653e21232f3f666d710f9f98731630014`.
새 `resume_validation.py` SHA `49d5aaf7bde0417962ef6642156828b092cf39b6a234211d4c509593145ba0f5`는
원래 실행의 검사 파일 한 개만 달라졌고 저장 모델/replay가 그대로임을 확인한 뒤,
미실행 평가부터 순서대로 복구한다. 학습 재실행·추가 예산은0이다.

03:42:57 recovery runner2547534/평가 child2547542 실제 생존·evaluate5000 실행 확인.
현재 상태는 `status.json`과 `validation-recovery-status.json`이다. 원래 두 reviewer의
FAILED status는 보존하며 현재 작업은 recovery runner 하나가 평가→고정Q→coverage→
branchQ→실제종료MC 및 단계별 독립 검토→두 arm GPU62/배열 검토를 수행한다.
주 status에는 원래 failed execution/원본 source hash와 수정한 validation SHA를 별도 보존한다.
현재 검사 SHA `73aa87b9d555de53441d252591888e8188bbfab1be87cedc3e799e3eaccec3f1`.
복구 검증이 끝나기 전까지 현재 core·검사·recovery/reviewer/GPU 스크립트를 고정한다.
학습 runtime c3fe6a7은 유지하며 기록 기준 HEADbb64ec4에서 검사/테스트 수정은 미커밋이다.
안정 정책 미확보·두 arm의 최종 주행/가치 비교 미완료·전체 목표 active.

## 03:47 KST 두5,000회 모델의 같은4주행 비교 — 모두 미달

복구한 extension의 평가와 독립 원시 frame 검토가 완료됐다. 같은 두 개발 장면×
KEEP/FOLLOW의4주행이며 독립 test가 아니다. 이전2,000·현재5,000·초기mean0의
각12episode 검토에 더해 quota5,000/extension5,000의 초기frame·map·seed가
4쌍 모두 정확히 같음을 별도로 대조했다. 각 review의 원본 source SHA도 재확인했다.

| 지표 | quota5,000 | extension2,000 | extension5,000 |
| --- | --- | --- | --- |
| 목표 / 충돌 | 0/4 · 4/4 | 0/4 · 4/4 | 0/4 · 4/4 |
| tick | 210 | 220 | 171 |
| 거절 | 89(42.38%) | 87(39.55%) | 98(57.31%) |
| 제동 | 78(37.14%) | 77(35.00%) | 86(50.29%) |
| post-step 정지 | 25(11.90%) | 25(11.36%) | 32(18.71%) |
| 평균 training return | -20.233012 | -20.222395 | -20.266951 |

extension5,000은 고속 KEEP/FOLLOW52/51tick, 중속34/34tick에 충돌했다. 고속의
첫 제동26/25·첫 정지49/48tick이며 마지막29tick 연속 새 경로 거절이다. 중속은
둘 다 첫 제동19·첫 정지23tick, 마지막18tick 연속 거절이다. 모두 정지 중 뒤쪽 객체와
접촉했지만, 이는 기하/시간 기록이며 귀책 또는 유일 원인 확정은 아니다.
quota5,000과 자신의2,000 대비 모두 aggregate 거절·제동·정지 비중이 늘었다.
**실패 경로 연장만으로 현재 설정/예산에서 주행 개선을 확보하지 못했으며 미채택이다.**
초기mean0의2/4성공도 회복하지 못했다. 전체 validation/새seed/별도정책test로 확대하지 않는다.

로컬 `outputs/sac_collection_budget5000_20261120/`의
`extension/evaluation5000/report.json` SHA
`9d9244d6bfa665e5734740588bba92b3a392b323246d2d73204943d75131a53c`,
`extension/evaluation-independent-review.json` SHA
`00fde9afa9a14c4d68107ace761ae05e0de2f09a10e16d990b29e663f0db19ae`,
`driving-paired5000-review.json` SHA
`363095e8236a93ffa2a5e0522bcf66e3b2d10f0201f3450c87eff2f53b3eff98`.
03:47:15 recovery runner2547534 및 q5000 child2548836 실제 생존 확인.
extension의 Q/coverage/branchQ/MC와 두 arm GPU62 검토는 아직 남았다.
다음 학습 조건은 미선택이며 현재 소스 고정을 유지한다. 기준 HEADf4ce66c.

## 04:01 KST 두5,000회 비교의 최종 가치/GPU 검토 완료

복구 runner2547534는 모든 누락 평가·독립 검토·두 arm GPU62/배열 검토를 마치고
`COMPLETE_PENDING_SCIENTIFIC_REVIEW`로 정상 종료했다. 이후 실제 process 부재와
동결 source/model/report hash를 확인하고 최종 비교를 작성했다.
`scientific-review.json`은 `REVIEWED_BOTH_DRIVING_FAILURES/qualified=false`다.
두 모델 모두 개발0/4성공·4충돌이고 안정 정책이 아니다.

extension의 고정62행 중 severe44 Q MAE는2,000회의0.847492→5,000회의0.256212다.
현재 평균을 실제 실행한 고정 terminal KEEP65/FOLLOW72tick의 Q−reward는
-0.160065/+0.119806, 수집 branch KEEP28/FOLLOW55tick은-0.133846/-0.035326이다.
모두 실제 충돌/bootstrap=false이며, 저장 행동의 reward를 다른 평균에 붙이지 않았다.

같은2train상태×4seed88019~88022의 원래Gaussian 후속 행동은 다시8/8충돌했다.

| extension5,000 | KEEP tick0 | FOLLOW tick4 |
| --- | --- | --- |
| 현재 평균 첫 행동 Q | -0.698844 | -0.688830 |
| 실제 soft return 평균±표본SD | -12.532541±2.215768 | -10.564603±0.514426 |
| Q−실제 표본평균 | 11.833697 | 9.875773 |
| 자신의2,000 대비 paired return 차이 | +0.847921 | -0.306754 |

KEEP의 return 증가에는 종료보상 할인+1.089006이 기여했다. 할인하지 않은 reward합은
-0.051920으로 악화됐고, 종료전보상-0.016765·entropy-0.224320이다. FOLLOW의 할인하지
않은 reward합 차이는-0.00000342, 종료보상 할인+0.056107·종료전+0.002774·entropy-0.365634다.
quota5,000 대비 extension의 soft return은KEEP+1.591350/FOLLOW+1.344745지만 종료보상
할인 차이가+1.400705/+0.976182이고 양쪽 모두8/8충돌이다. 평균 보상 증가를 안전 주행으로
해석하지 않는다. 두 arm의 동일 관측 hash·paired seed·return 구성·원본trace hash를 대조했다.

extension의 uniform1,280,000draw와 saved RNG를 재현했다. severe terminal 추출29,448회
(warmup7,734/main2,045/branch19,669). 새3,000회에 수집한 severe139행은5,705회 추출됐고
5행은 미추출이다. 저장 경험 노출과 terminal 적합이 있어도 현재 앞선 상태의 Q/주행은 미달이다.

같은 GPU/62배치 기준은 원래 atol3e-5/rtol3e-5를 두 모델 모두 통과했다. 기준 최대절대차는
quota0.000101089/extension0이며 상대허용치를 포함한 원래 조건 그대로다. 허용치를 넓히지 않았다.
quota402/extension330 terminal행을 padding 제외 후 모두 검토하고 별도 통계 재계산도 통과했다.
severe373/302행 MAE는각0.231766/0.190450이다. **모든 terminal Q가 정확하다는 뜻은 아니다.**
extension의 main RECORD_END(replay index7,600,scene24aee88cb4ba196a,tick195)는
reward-0.013391/Q-12.540760/error-12.527369이며 실제 학습batch63회에 포함됐다.
severe subset 평균만으로 이 비충돌 종료의 큰 과소평가를 숨기지 않는다.

quota CPU/GPU의 같은 원본행 차이는 전체 Q1최대0.004009247/Q2최대0.002593994,
새online행 min-Q최대0.001495361이다. GPU 기준 재현과 새행 통계를 확인한 것이며
CPU 차이의 구체적인 연산/하드웨어 원인을 확정하거나 CPU 엄격대조 실패를 소급 통과시키지 않았다.

로컬 `outputs/sac_collection_budget5000_20261120/`:
`scientific-review.json` SHA `c5172855bad608fe6417d91fde941612556f41689084634dfb6cadc9d8550ef8`,
`extension/value-independent-review.json` SHA `ee7135745e5c0749320f0c654e12788ad095bb546c112b2a1c1c0a36a7680896`,
`extension/soft-returns5000/report.json` SHA `ed635a956b70bddeba8b88f7b55b0e23699a124aecfc5aef4f3d7ac179761c13`,
GPU reports quota `7982e22c86f2f19f94ba2ff031f168dcd26e65db07ada4eae55746fd1ac69cd6`,
extension `f8b5a02e1b24dc2057ab6f868f7a891e8a7f357fd25bbb3b143b64650d198b0c`,
GPU independent arrays quota `6f643c29340c26e5759aa58c85a9d22efd14483d7b4029aeaac392ddf60bee2c`,
extension `39e137f3ddf2235ea406cf588bf7f1bf904fe937dc4ee6521028d3199b1a3dd2`.
학습 수집/보상/안전검사 규약은 유지했다. 두 수집 대안의 이 예산 비교는 끝났고,
장기 수렴 불가능이나 수집이 무의미하다는 일반 결론은 내리지 않는다. 안정 정책 목표는 active다.

## 04:12 KST 저속 평균의 명시적 g 분리: fresh1,000회 비교 시작

수집만 바꾼 두5,000회 모두 미달이므로, 앞서 측정한 작은 평균 변화의 구조적 요인을
승인된 B 평균 비교 범위에서 분리한다. [평균 좌표 실험 문서](../architecture/sac-mean-basis-experiment.md)에
조건·한계·추가 개발 평가를 명시했다. 기준은 entropy 평균 gain0.0012098536225957168,
후보는 raw B 평균 gain0.0001이다. gain*c=0.0001로 속도2m/s 이상 기준 배율을 맞추고
평균에만 명시적으로 곱하던 저속 g를 제거한다. 탐색Gaussian의 g/sigma/ell/log_pi는 유지한다.
단순500배 성능 향상이나 유일 원인 해결을 가정하지 않으며, 이전 사후 mean/g 실패와 달리
새로 초기화해 실제 자기 경험으로 학습한다. 기본 설정·사용자 채택 결정은 바꾸지 않는다.

`scripts/diagnose_sac_online_updates.py`에 명시적 `--mean-coordinate-scale` override만
추가했다. Actor/critic/learner 코어는 변경0이다. 같은 seed0·자기warmup5,000·physical Q와
zero-action 초기화·quota8/window6/moving_predecessors/restart-terminal·전이당1update·
uniform replay·보상/검사/PD·BC/teacher0을 유지한다. extension 옵션은 사용하지 않는다.
기준1,000 실제 모델 SHA `b32586d181f072a69baaa97539069ba056f0613e25594460dff7939b906e2708`를
재확인했으며 새 후보는 두5,000 모델을 이어받지 않는다.

실제 초기 자기 경험에서 정책과 무관하게 균등 간격256행을 선택했다(정지171/저속13/이동72).
두 fresh learner의 파라미터·optimizer·target·RNG 직렬화 바이트와 초기mu0/rho/표본/ell/log_pi가
정확히 같음을 GPU에서 확인했다. 이는 초기 동등성만 검증하며 학습 후 성능 증명이 아니다.
관련 action-coordinates/continuation/diagnostic-critic pytest27개 통과.
초기 테스트1개는 서로 다른 평균을 뺀 noise 차이2.1684e-19가 임의 atol2e-19를 넘어서 실패했다.
허용치를 키우지 않고 공통 zero-mean noise에 각 평균을 더한 표본과의 정확한 일치 검증으로
바꿔 재실행했다. `initial-test-failure.json`에 보존했으며 production/물리검사 허용치 변경0이다.

기존 고속/중속2장면×KEEP/FOLLOW4주행은 유지한다. 별도 사전 지정 개발 cohort로
정지 출발 `8dd5939855c15b3f`, `a62f5a7d2cd9b3d9`의 KEEP/FOLLOW4주행을 기준1,000/후보
모두 실행하고 실제 이동·정지·성공·충돌·거절·제동·보상을 검토한다. 기존 평가를 소급 변경하거나
새 독립test라고 부르지 않는다. 그 뒤 고정Q/coverage/branchQ/같은2train상태×4seed MC를 연결했다.
주행/가치 미달 시 다음 조건을 다시 판단하며1,000 이후 자동예산 확대·새seed·기본채택은 없다.

로컬 `outputs/sac_mean_decoupled_g_20261120/`의 preflight SHA
`d6370e99efff63e585039438e617e38a2cff018cf2869cf9a10b1dcc82e47079`,
decision SHA `0ec4fe614523dcbabbd8afba872c4ca6c50b0f3655cf67bc98981d28c5a7a390`,
runner SHA `ce102fa1e5b933a92c11588944df0e8bbc4c57d27ba5913fb7a10aaaf9bd114a`.
04:12:03 runner2560409 생존·train child2560430 시작 기록과RUNNING/train1000을 확인했다.
source snapshot/hash/정확한 명령은decision.json·status.json·sources/에 보존한다.
기준 HEAD35dd60e에서 이번 진단CLI·테스트·문서 수정은 미커밋이며 실제 학습 성능은 아직 미검증이다.


## 04:33 KST fresh1,000회 수집 완료와 실제 학습 노출 검토

저속 평균 g 분리 후보는 새 전이/업데이트1,000회와 저장을 완료했다.
완료 학습 모델 SHA `9cdced951a997905c32ab8ea18c7b9da9866b27c81e8f8de23da06728bd6883c`,
report SHA `2094681254b5f02287baea6f92212a35b4d47e777355b9904ea1c8d599394697`,
replay SHA `792420fef4bb6052c52b158772609bc51338aa9236f99ad65635bef0534a86cf`.
같은 seed0/자기warmup5,000/BC·teacher0/보상·검사·PD·원래Gaussian 조건을 유지했다.
실제 train9장면에서 main712/branch288전이, branch36개 시작상태×8step과 실제충돌48개를
수집했다. 기존 auditor는 전이·행동·보상·종료·부모/연속 관측·train manifest를 대조해
통과했고, warmup17배열5,000행 byte-prefix도 동일했다. 학습 완료는 정책 성능 입증이 아니다.

별도 CPU 검토 `review_fresh_sampling.py`에서 매 삽입 후 uniform256 추출을 다른
NumPy index 연산으로 재현하고 최종 saved RNG와 전이별 scene/pre_tick/reward/bootstrap을
대조했다. 두 fresh1,000의 총256,000draw 중 warmup233,361/new22,639는 동일하지만,
각 시점에 실제 수집한 경험은 서로 다르다. 기존 continuation 전용 검토기를 잘못 재사용하지 않았다.

| 실제 경험·추출 | 기존 entropy 평균1,000 | raw 평균 후보1,000 |
| --- | ---: | ---: |
| main 행 / branch 행 | 664 / 336 | 712 / 288 |
| main 추출 / branch 추출 | 13,409 / 9,230 | 13,909 / 8,730 |
| main severe 종료 / branch severe 종료 | 7 / 66 | 6 / 48 |
| main severe 추출 / branch severe 추출 | 220 / 1,488 | 190 / 1,416 |
| branch severe 미추출 행 | 2 | 0 |
| 새 경험의 step 전 정지(<0.1m/s) | 464/1,000 | 440/1,000 |

두 모델 모두 기존warmup severe44행은2,018회 추출됐다. 새 경험 노출은 전체추출의
8.84%이며 warmup 비중91.16%다. 이는 현재1,000회 uniform replay의 계산 결과이며
경험 재가중을 시행하거나 수집비율만으로 주행 개선을 주장하지 않는다.
독립검토의 첫 시도는 replay의 행동 전 tick을 trace의 행동 후 tick과 비교해 실패했다.
기존 auditor와 필드를 대조해 pre_tick으로 검토기만 수정하고 다시 실행해 통과했다.
첫 실패와 원본 script는 `fresh-sampling-initial-review-failure.json`에 보존했다.
실제 replay/학습 코드/검사 허용치는 바꾸지 않았다.

로컬 `outputs/sac_mean_decoupled_g_20261120/fresh-sampling-review.json` SHA
`af0f42011fb8bfadb866e367c4e502177161afbee6c6fdfeb470c7cdc217f1e6`.
04:32:45 실제 evaluation child2567386 생존·주행평가3/4완료를 확인했다.
완료한3주행은 모두충돌이며 나머지1주행, 별도정지출발 비교, Q/coverage/MC가 남아 있다.
고정 source hash 변경0. 기준 HEAD556fc3d에서 이번 로컬검토·문서만 추가했으며,
성능 미달 후보 채택/전체validation 확대/새seed 실행은 하지 않았다. 전체목표는 active다.


### 04:34 KST 기존 평가4주행 완료: 기존 두 성공을 잃음

같은2개 개발 장면×KEEP/FOLLOW4주행의 초기frame/map/seed를 기준1,000·후보·mean0와
대조한 독립 검토가 완료됐다. 후보는0/4성공·4충돌이고 기준1,000은2/4성공·2충돌이다.
양쪽총236tick에서 거절44→92(18.64→38.98%), 제동36→84(15.25→35.59%),
step후정지12→30(5.08→12.71%), 평균return-9.981233→-20.225907로 악화됐다.
후보KEEP/FOLLOW는 각각0/2성공·2충돌이며, 원본상4회 모두정지중뒤쪽차량접촉이다.
접촉기하만으로 단독 과실을 판정하지 않는다. 수집·노출 개선과 실제주행 개선을 구분한다.

이 예산의 후보를 채택하거나 전체validation/다른seed로 확대하지 않는다.
평균g제거가 저속이동에 미치는 영향은 사전지정 정지출발cohort와 Q/MC 검토가 남아 있어
아직 별도로 판단하지 않는다. 04:34:09 runner2560409와 stopped-baseline child2568843의
실제 생존, RUNNING/stopped-baseline을 확인했다. 기존 학습·고정설정 변경0.

독립 frame 검토 SHA `d3ccd2c4065b6f7aabccc851df257b59ca11d8b8e1484f828aeafeb0441a2c4a`,
평가 report SHA `0ce4f29b1d3e9b421a66f415c1a8d64b15af615009f0cd7437ff52331eb30ab0`.


## 04:44 KST 정지 출발 비교 완료와 저장 평균 가중치 점검

사전지정 추가 개발2장면×KEEP/FOLLOW를 기존1,000/후보1,000에서 각각4회 평가하고
원본frame/map/seed의4쌍 일치와 source/model hash를 대조했다. 이는 기존고속/중속4주행과
별도 집단이며 독립test가 아니다. 두정책 모두0/4성공·4충돌,328/328tick정지(<0.1m/s)다.
후보 거절80→124(24.39→37.80%), 제동76→98(23.17→29.88%),
평균return-20.429955→-20.540071로 악화됐다. 양쪽4회모두정지중뒤차접촉이다.

| 초기 정지 장면·명령 | 기준 누적이동(m) | 후보 누적이동(m) | 후보 최대속도(m/s) |
| --- | ---: | ---: | ---: |
| 8dd5939855c15b3f KEEP | 0.00004578 | 0.00359347 | 0.00245964 |
| 8dd5939855c15b3f FOLLOW | 0.00004578 | 0.00380452 | 0.00246542 |
| a62f5a7d2cd9b3d9 KEEP | 0.00006104 | 0.00845849 | 0.00253488 |
| a62f5a7d2cd9b3d9 FOLLOW | 0.00006104 | 0.00851493 | 0.00247826 |

기준post-step속도는전부0이며작은초기이동을출발로취급하지않는다. 후보도최대8.52mm의
누적이동을성공으로세지않는다. 평균g축소제거만으로이예산의정지문제를해결하지못했다.
04:43:49 runner2560409/q1000 child2572464실제생존확인. Q/coverage/branchQ/MC검토를
이어가며현재후보채택/평가확대/새학습조건결정은없다.

### 출력 크기 점검: 실제 가중치의 이상적 실수 연산 경계

GPU추론/시뮬레이션/업데이트없이 저장된4모델의 마지막LayerNorm affine과mean head를
읽었다. 이상적인실수연산에서 LN 정규화벡터는||z||≤sqrt(256)이므로
R=sqrt(256)*max|gamma|+||beta||, H_j=R*||W[:,j]||+|bias_j|를계산했다.
ReLU는이L2경계를늘리지않고 B와3t²는원소가음수가아니므로
|경로잔차(t)|≤3t²*B*reshape(H)*gain*(해당entropy설정의c*g)다.
상수head1의3초endpoint 계수4.96m와닫힌형합 i(i+1)(i+2)dt³/6을수치대조했다.

| 현재 가중치 | g=.002일 때 3초 학습잔차 L2 경계(m) | 편집 전 첫 참조속도 잔차 L2 경계(m/s) |
| --- | ---: | ---: |
| 기존 entropy1,000 | 0.000140456 | 0.000000523436 |
| raw 후보1,000 | 0.0629054 | 0.000234703 |
| quota entropy5,000 | 0.000916476 | 0.00000331124 |
| extension entropy5,000 | 0.00106263 | 0.00000344721 |

첫참조속도는PD가t=.1의궤적미분을쓰고Trajectory의v1=(p2-p0)/.2인코드를대조해,
편집전학습잔차r(.2)/.2로계산했다. 후보r(.2)의경계는0.0000469406m다.
**이표는f32의모든입력에대한인증경계나실제주행최대속도가아니다.** LN분산/dot의
반올림오차를엄밀히묶지않았고,anchor속도/가속도·탐색잡음·terminal suffix편집·폐루프누적은
포함하지않는다. 현재고정가중치의학습잔차만점검했으며모델자유도제한/장기학습불가능/
단독실패원인을증명하지않는다. 현재평균g제거후에도PD가보는초기경로보정크기가작을수
있다는분석근거이며 임의gain채택이나기본구조변경은수행하지않았다.

로컬 `outputs/sac_mean_decoupled_g_20261120/mean_weight_envelope.py`를
`OPENBLAS_NUM_THREADS=1 JAX_PLATFORMS=cpu python`으로실행했고정상종료했다.
새pytest/학습은실행하지않았다. 런타임·동결runner 변경0,기준HEAD41a9c5f.

로컬 `stopped-independent-review.json` SHA `f4fa262c5b74e8dd3c798897f5d3f2fb817d65d784f06c1a9bddd757296ff2b4`.

로컬 `mean-weight-envelope.json` SHA `8aff75d7449c2ea6d62cf3f27c05b58afd54855dfd7571a5dcb88c6a4caae375`.


## 04:59 KST raw 평균1,000회 최종 검토: 주행·가치 미달

runner2560409/마지막검토2579411의실제종료와모든job exit0/source hash를확인했다.
`scientific-review.json`은REVIEWED_DRIVING_FAILURES/qualified=false다. 기존4개발주행과
추가정지4주행각각0성공·4충돌이고,같은2train상태×4seed88019~88022도8회모두충돌했다.

| 현재평균첫행동→원래Gaussian 후속주행 | KEEP tick0 | FOLLOW tick4 |
| --- | ---: | ---: |
| 첫행동 Q | -0.410892 | -0.220348 |
| 실제 soft return 평균±표본SD | -13.576946±1.736406 | -10.421171±0.477574 |
| Q−실제평균 | 13.166054 | 10.200823 |
| 기준1,000대비 paired return평균 | -0.182494 | +0.555750 |

FOLLOW의return증가는종료보상할인+0.501741(약90.3%),종료전+0.008673,
entropy+0.045337로분해됐다. 할인하지않은reward합은-0.008690으로악화됐고4/4충돌이다.
KEEP는충돌tick평균이1.5줄고종료보상할인-0.171887·종료전+0.002129·entropy-0.012737이다.
동일상태/관측hash/paired seed/원본trace와구성합을대조했다. 보상증가를주행개선으로세지않는다.
고정warmup severe44행Q MAE도기준4.896726→후보5.879852로악화됐다.

실제current mean과stored행동을다시실행한고정terminal2개는모두같은reward/next관측hash로
충돌했지만,현재평균Q를KEEP65tick+1.403010/FOLLOW72tick+0.973963높게평가했다.
수집branch의KEEP28tick/FOLLOW69tick에서는저장행동·현재평균·현재Gaussian4표본모두
각상태에서동일reward/next관측으로충돌했다. 현재평균선호는+1.958303/+2.503785,
현재평균Q−reward오차는+5.341227/+6.388299다. stored raw_u와실제replay행도대조했다.
다른실험의서로다른FOLLOW종료tick을동일상태paired비교로섞지않았다.

### 동일warmup상태의분포차이와실제추출을분리

coverage CSV6,000행의scene/pre_tick/reward/bootstrap과256,000draw/savedRNG를별도CPU로
대조했다. 현재후보online1,000행중54severe종료는1,606회추출됐고,
마지막100행은모두정지상태이며223회추출·21행미추출·현재95%Gaussian영역내0행이다.
동일warmup5,000행을두모델간identity/speed/reward/bootstrap/draw까지대조한결과:

| 동일 초기 경험 | 행수 | 기준 현재95%영역내 | 후보 현재95%영역내 | 실제추출수(양쪽동일) |
| --- | ---: | ---: | ---: | ---: |
| 정지(<0.1m/s) | 3,347 | 100% | 0% | 156,027 |
| 저속(0.1~2m/s) | 213 | 100% | 36.15% | 10,091 |
| 이동(≥2m/s) | 1,440 | 100% | 100% | 67,243 |

정지행의stored-action/현재sigma RMS중앙값은0.686471→96.218297,
현재평균의초기sigma좌표변위중앙값은0.212440→144.388756이다.
현재분포95%타원체의조건부겹침이지Gaussian지지집합/신뢰구간/새안전gate는아니다.
같은행에서Q(mean)−Q(stored)평균은0.001570→0.226327이나,5,000개대체행동을모두
실행한것이아니므로그차이를실제이득으로해석하지않는다. 물리적움직임은작아도좁은
저속탐색분포에비해평균변화가크다는점을기록한다. 단독인과관계확정은아니다.
CSV비교첫실행은hash helper에문자열경로를넘겨AttributeError로저장전에실패했다.
Path변환만고쳐재실행했고원본/오류를paired-warmup-coverage-initial-error.json에보존했다.
학습/추론런타임변경0이다.

로컬 `outputs/sac_mean_decoupled_g_20261120/scientific-review.json` SHA `bf9f29ad1505a2b8642f2f988a9b1c3fc4b3da3dae51c73d86198f02a44ccd3d`.

로컬 `outputs/sac_mean_decoupled_g_20261120/value-independent-review.json` SHA `93e601bbb1f5cf54443cb2ad3aa7c9e2c021f1a06c7ffc12eb02b66f1c765996`.

로컬 `outputs/sac_mean_decoupled_g_20261120/fresh-coverage-independent-review.json` SHA `f7ce8b78549c47690f6f19ddbbcae80c251760d2adba25adf5af9aae5501bc02`.

로컬 `outputs/sac_mean_decoupled_g_20261120/paired-warmup-coverage.json` SHA `fb9debd29b038b0da55e46dc1f4f080eb3e6ca1a54fe61563f25a0ab87443d64`.

로컬 `outputs/sac_mean_decoupled_g_20261120/terminal-preference-review.json` SHA `3a6b84a22233df470436b53678d265aab384e6c3063cdd8d7f9fe5446b1ab309`.


## 05:02 KST 동일설정 자기경험 추가1,000회 비교 시작

다음변경은학습/수집예산뿐이다. raw1,000의완료가중치/optimizer/target/RNG와replay/RNG를
이어총2,000(실제추가1,000)까지만진행한다. 새로운gain/보상/안전조건을택하지않는다.
raw gain0.0001·physicalQ/zero-action·quota8/window6/moving_predecessors/restart-terminal·
uniformreplay·전이당1update·원래Gaussian·공통PD·BC/teacher0을유지한다. 환경/장면/행동RNG는
기존entropy1,000→2,000과같은collection-seed1000으로새로시작하며exact resume가아니다.

분포차이·현재행동Q오차와새경험노출을함께확인하기위한제한진단이다. 1,000회실패가
경험부족때문이라고확정하거나추가학습의성공을보장하지않는다. 같은예산entropy2,000과
자신의raw1,000을기존4개발주행/Q/MC에서각각대조한다. 추가정지4주행은baseline2,000의
이미완료된full-development16중사전지정4쌍을재사용하고raw1,000/새raw2,000과비교한다.
후보전체validation확대가아니다. subset검토기는완료된raw1,000자료로실행해기존측정값과
정확한일치·4쌍초기frame/map/seed를확인했다. 새로쓴로컬runner/reviewer py_compile통과.
실행코어수정이없어새pytest/추가GPUpreflight는하지않았으며실제full-state복원검사는
기존continuation런타임이시작시에강제한다.

로컬 `outputs/sac_mean_decoupled_g_continuation_20261120/decision.json` SHA
`1252ccd0b66c9ef84c94a76d1ebfb71bd18992a438acdd6f1e6cacbfdab3bff5`,
runner SHA `a1679509acdb09c58b7a89f640d79ea2c6316430b21f155ce14c448ca2a6940f`.
05:01:33 runner2580230시작,05:02:06 train2580237과실제생존·RUNNING/train2000확인.
이시점학습report는아직생성전이므로복원/추가update완료를미리주장하지않는다.
기준HEAD773cb60의실행코어/CLI는그대로이며이번실험문서개정은미커밋이다.
2,000뒤자동확대/다른seed/기본채택은없다. 안정정책미확보·전체목표active.


05:03:02 실제report에서 restored_learner_bytes_identical=true,승계1,000/추가9/총1,009,
collection_seed1000,raw/gain0.0001/physicalQ/zero-action override를확인했다.
runner2580230/train2580237은둘다실제생존했다. 따라서복원과실제추가학습시작은확인됐고,
총2,000완료/후속주행성능은아직미검증이다.


## 05:11 KST 짧은진단과학습규모를구분하는예산점검

현재2,000회실행을고정한채원저자의
[Soft Actor-Critic Algorithms and Applications, arXiv1812.05905v2](https://arxiv.org/pdf/1812.05905)를
확인했다. Fig.1/§7.1은백만step축의곡선(과제별끝점1/3/10million),5seed와1,000step마다의
평가를보고한다. AppendixD는lr3e-4/γ.99/batch256/τ.005/gradient step1이고,
AppendixC의tanh행동과표의entropy target -dim(A)는우리상관경로분포·정규화ell target0과다르다.
현재구현에서도lr/γ/batch/τ값은대조했지만논문환경과경로·보상·안전검사는같지않다.

**판단:** 1,000~5,000회에서의실패는해당예산의실패근거이며장기수렴불가능의증거가아니다.
원논문의1,000은학습종료예산이아닌평가간격이다. 반대로논문학습량을그대로우리필요량이나
성공보장으로옮기지않는다. 여러fresh후보의step을합쳐한정책이그만큼학습했다고세지않는다.
현재짧은검증에서확인한규약/수치오류와주행실패는유지하면서,동일learner의더긴학습곡선과
검증빈도를구분해다음예산을판단할필요가있다. 현재2,000예산/조건을바꾸지는않았다.

완료raw1,000의보존launch기록04:11:45.213666과완료train report의파일mtime04:29:35.793593을
대조하면1,070.580초(약17분51초)다. 모델기동·실제수집·업데이트·저장을포함하며그뒤의
주행/Q/MC평가는제외한다. 새부하시험이아닌과거파일시각분석이고1.07058초/새전이다.
단순선형산술로20,000→약5.95시간,50,000→14.87시간,100,000→29.74시간이다.
이는새예산선택/완료예정시간이아니며초기화상각·정책별검사비용·하드웨어차이와평가비용이
반영되지않았다. 논문규모와실행가능범위를분리하기위한참고치다.

현진단CLI와continuation 검사는누적5,000을상한으로강제한다. 실제saved replay의capacity는
50,000,size/cursor6,000이다. 현재감사는warmup5,000+모든신규행의prefix보존을전제로하므로,
50,000update로늘리면55,000행이되어ring wrap까지고려해야한다. **상한숫자만올려긴학습을
실행할수있다고주장하지않는다.** 더긴예산을선택할경우먼저상한·용량·저장/복원·lineage검증을
일관되게다뤄야한다. 현재실행중인소스변경/새대규모학습선택은없다.

로컬 `outputs/sac_mean_decoupled_g_20261120/historical-runtime-budget-context.json`
SHA `ca3e559da6725da7e71994f35ec5f02076080fe879d5f73fc2bd955a7ec8fa42`.
05:10:04 실제train2580237생존·총1,446/2,000(추가446)확인. 기준HEADe82b14e에서
이번기록만추가했고현재실험의평가완료나주행개선을주장하지않는다.


## 05:20 KST raw2,000 수집·저장검토와기존4주행완료: 여전히미달

추가실제전이/update1,000개를마쳐누적2,000이다. 학습모델 SHA
`8d8f8c0c8ec27ad0c2e92ac2c7b82db066fdb2dbf7dae0c7381a06cc06fe4601`,
replay7,000행 SHA `757f46e0a89b7eee8b737a881e9538d325871d26dff08eb1292c915d0d89af8a`,
report SHA `10aadae7605d37f9dfe3d6ec22aa3a0e131c93419fa7c24b640933c5b5383931`.
동일learner/optimizer/target/RNG복원과기존6,000행17배열byte-prefix·traceprefix동일,
collection-seed1000·teacher0·실제train전이lineage검사를통과했다. 원본hash도다시대조했다.

전체train18고유장면,main1,393/branch607·branch충돌112개다. 새1,000만보면9고유장면,
main681/branch319·branch충돌64개,step전정지638/1,000이다(기존1,000은440/1,000).
새main수락97/681·branch46/319,합143/1,000이다. 학습중다른상태/다른장면에서수집한
수치이며같은상태의고정정책성능비교로세지않는다.

신규운동한계거절은main484/branch186개,그중step전정지는각387/78개다.
curvature_rate위반은각484/186,curvature는483/181이고가속도/jerk위반도함께발생했다.
한거절의다중위반을서로독립된거절로합하지않는다. 이전1,000의운동한계거절115/86과
단순차이만으로정책변경의단독인과효과를주장하지않는다. 이반복정지/곡률위반편중을
후속고정주행·Q/coverage와함께검토한다.

| 기존2개개발장면×KEEP/FOLLOW | entropy2,000 | own raw1,000 | raw2,000 |
| --- | ---: | ---: | ---: |
| 성공 / 충돌 | 2/4 / 2/4 | 0/4 / 4/4 | 0/4 / 4/4 |
| tick | 236 | 236 | 230 |
| 거절 | 48 (20.34%) | 92 (38.98%) | 100 (43.48%) |
| 제동 | 37 (15.68%) | 84 (35.59%) | 88 (38.26%) |
| step후정지 | 13 (5.51%) | 30 (12.71%) | 36 (15.65%) |
| 평균return | -9.992331 | -20.225907 | -20.247278 |

두비교의4쌍초기frame/map/seed와원본값/해시대조완료. 현재4회모두정지중뒤차접촉이다.
raw2,000은자신의1,000및동일예산entropy2,000대비개선되지않아미채택을유지한다.
05:20:01 runner2580230/stopped-candidate2588386실제생존·정지출발추가4주행실행중을
확인했다. 그뒤Q/coverage/실제branchQ/MC검토가남았다. 새로운예산/gain/구조는미선택이다.
현재소스·보상·검사·PD·원래Gaussian·순수SAC조건유지·전체목표active.

로컬 `outputs/sac_mean_decoupled_g_continuation_20261120/`:
collection-independent-review SHA `710919570aa49d0073c07f2c724cda244fab4d030bee7f2d5b4d132eccd06d92`,
collection-violation-breakdown SHA `0e25a1b40fec35157cd105b15ae483f6f92550a3f17133d3e87c0515070029fa`,
evaluation-independent-review SHA `3b5e0426595ece3a474aefa6c2252b954cfa3be2883f671d633d5abac9903f92`,
own-evaluation-independent-review SHA `29f5ce696972c8309eb57a1d0d4757e179ed49b14fb66469b9fb1ed4363fb511`.
기준HEAD792800d에서소스변경없이이번결과기록만추가했다.

## 2026-11-20 05:39 KST — raw2,000 최종 미달과 동일 설정5,000 후속 비교

기준 HEAD `530322677c8d3b62c83241a1818113ae4ff852f5`, 같은 개발 브랜치에서
런타임 소스 변경 없이 검토·실험 규약을 기록했다. CLI556fc3d/core96fa1d6 유지.
05:38:04 이전 runner2580230과 모든 자식 PID의 실제 종료, 모든 job exit0,
고정 소스 및 검토 입력 SHA를 확인했다. 로컬 `sac_mean_decoupled_g_continuation_20261120/`
아래 `scientific-review.json`은 `REVIEWED_DRIVING_FAILURES`, qualified=false,
SHA `8d6aa470bc18c03998e09ffd61c067cf1d37b22fa60454fa1ca421891070e290`이다.
안정 정책 확보나 전체 목표 완료가 아니다.

### 정지 출발과 실제 종료 행동

같은 정지2장면×KEEP/FOLLOW의 초기 frame/map/seed를 대조했다. 세 모델 모두
0/4성공·4/4충돌·328/328틱 정지였다. 서로 다른 실제 주행의 접촉 위치는 과실 판정이 아니다.

| 정지4주행 | entropy2,000 | raw1,000 | raw2,000 |
| --- | --- | --- | --- |
| 거절 | 80/328 | 124/328 | 264/328 (80.49%) |
| 제동 | 76/328 | 98/328 | 140/328 (42.68%) |
| 평균 return | -20.429955 | -20.540071 | -20.890173 |

raw2,000의 주행별 누적 이동은0.000828~0.001541m, 최대 속도는0.000425~0.001100m/s다.
미세한 이동을 출발 성공으로 해석하지 않는다. curvature/curvature_rate 위반을 포함한
운동 한계 거절이 늘었고, 기존4주행도 이미0성공·4충돌로 미달이다.

고정 종료 KEEP65/FOLLOW72에서 현재 평균의 Q는 저장 행동보다+0.737793/+1.484426,
실제 종료 보상 대비 오차는+2.846850/+1.919857이었다. 수집 branch KEEP28/FOLLOW69에서는
저장 행동·현재 평균·현재 Gaussian4표본이 모두 같은 보상/다음 관측으로 충돌했는데,
현재 평균 선호는+6.129089/+3.890041, 보상 대비 오차는+5.545614/+4.445751이었다.
이 종료 전이는 bootstrap=false이므로 다음 상태 추정치로 차이를 설명할 수 없다.
원본 재실행 결과의 reward/next hash/종료를 CPU로 대조한
`terminal-preference-review.json` SHA는
`31cd63e927257afe7596a6f1dcb033468ba262ff3360dca401c9880290168185`이다.

### 경험 노출과 실제 미래 보상

전체512,000 batch draw 중 warmup430,763, 새 경험81,237이었다. 추가1,000회 구간에서는
새로 생긴 main 충돌7행이117회, branch 충돌64행이1,109회 추출됐다. branch4행은 미추출이다.
최종 online severe125행은5,026회 추출됐고 저장 행동 Q MAE는0.989634였다.
충돌 데이터가 전혀 학습되지 않았다는 설명은 맞지 않지만, 충분한 학습을 증명하지도 않는다.

최종 정책의 조건부95% Gaussian 타원체 안에 있는 행동은 online624/2,000이었다.
속도별로 정지0/1,078, 저속23/163, 이동601/759이며, 최근100행은 정지0/43·저속14/28·
이동25/29였다. 최근 정지43행은72회 추출됐고12행 미추출, 현재sigma RMS 중앙값81.058360이다.
이는 Gaussian의 지지집합·안전 gate·신뢰구간이 아니다. 예산 사이의 실제 상태도 다르다.
읽기 전용 `coverage-speed-stratification.json` SHA
`2572e1a64039c8f6bbec15c9cd2b15c82a310c3e6cc9fdcdb036237f0207df2a`에 분모·입력 SHA를 남겼다.

같은2train상태×4seed88019..88022 후속 주행은8회 모두 충돌했다.

| 현재 평균 첫 행동→원래 Gaussian 후속 주행 | KEEP tick0 | FOLLOW tick4 |
| --- | --- | --- |
| 첫 행동 Q | -0.362388 | -0.239095 |
| 실제 soft return 평균±표본SD | -12.612033±2.576786 | -10.548081±0.290141 |
| Q−실제 평균 | 12.249646 | 10.308986 |
| 자신의1,000 대비 paired return 평균 | +0.964912 | -0.126910 |

KEEP 증가에는 종료 보상 할인 차이+1.050353이 들어갔고, 비할인 실제 reward합은
-0.046333으로 악화됐다. FOLLOW는 종료 할인+0.058358·종료전+0.018837보다
entropy 기여 감소-0.204105가 컸다. 동일 관측·paired seed·원본 trace/구성 합 검토 완료.
고정 충돌44행의 Q MAE는 raw1,000의5.879852에서3.616825로 줄었지만
entropy2,000의0.742834보다 크고, 실제 주행 실패는 그대로다. 보상 증가나 Q 적합만으로
개선 판정하지 않는다. 두 개발 cohort와 train MC는 독립 test가 아니다.

### 다음 실행: 같은 learner로 총5,000회까지

전체2,000 검토를 마친 뒤 총5,000회 제한 비교를 별도 결정했다. 매번 초기화하거나
배율을 다시 바꾸지 않고 raw2,000의 전체 learner/optimizer/target/RNG와7,000 replay행/RNG를
복원해 실제 전이3,000개와 update3,000회를 추가한다. collection-seed2000으로 환경/장면/
행동 RNG를 재시작한다. 이미 완료된 entropy5,000과 동일 예산의 주행/Q/MC를 비교한다.
정지4주행의 기존 참고값은 entropy2,000/raw2,000이며 동일 예산 비교라고 부르지 않는다.

원래 Gaussian·보상·검사·PD·순수 SAC·uniform replay50k·전이당1update·mean gain0.0001·
physical Q·quota8/window6/moving_predecessors/restart-terminal을 유지한다.
더 긴 학습이 해결책이라는 확정이나 후보 채택은 아니다. 실패한 짧은 진단과 충분한
학습 예산을 구분하고 동일 설정의 경과를 확인한다. 전체 validation/다른 seed/독립 test/
5,000 이후 예산은 완료 검토 후 판단한다.

로컬 `outputs/sac_mean_decoupled_g_budget5000_20261120/`의 decision SHA
`e6bf95dcc6f4c90f9a22f2381eb2458472e85cdac555974b0137fc657c9a6f55`, runner SHA
`6c3e1cda8f3811aaea2b54de90db69bd0d8a4a27cd723a785fcae110129b05af`.
새 로컬 runner/reviewer py_compile과 완료 raw2,000을 사용한 정지 검토기 preflight 통과.
런타임 변경이 없어 새 pytest/추가 GPU preflight는 실행하지 않았다. 실제 continuation이
full-state 복원 동일성을 검사한다. 05:39:02 runner2597128 시작,05:39:10 train2597131 실제 생존.
이 시점 학습은 RUNNING이며 새5,000 성능은 미검증이다. 전체 목표는 active다.

05:45:47 실제 report에서 `restored_learner_bytes_identical=true`, 승계2,000·추가416·
누적2,416/5,000·collection-seed2000을 확인했다. runner2597128/train2597131 생존 확인.

## 2026-11-20 06:01 KST — 별도 평가 원본 길이 오류의 외부 근거 확인

5,000 학습을 유지하며 기존 `raw-shape-review.json`과 공식 저장소 자료만 읽었다.
새 데이터 scan/reset/정책 평가/시간 보간/loader 변경은0이다. 기준 HEAD1328990.
17개 원본 모두 장면 내부 track 길이는 서로 같고 dynamic 길이는 timestamps와 같지만,
track 대 timestamps의 길이는 다르다. track−timestamp 차이는
`+1:2, -1:3, +6:1, -3:2, +2:1, -2:5, +3:2, +7:1`이다.

[Waymo의 Scenario 스키마](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/scenario.proto)는
timestamp/track/dynamic 길이와 index 대응을 요구한다. 공식 저장소
[이슈773](https://github.com/waymo-research/waymo-open-dataset/issues/773)은 v1.2.0에서
비슷한 불일치를 보고했으며 조회 시 open이었다. 2024-02-01의
[프로젝트 협업자 답변](https://github.com/waymo-research/waymo-open-dataset/issues/773#issuecomment-1921936778)은
잘라내기 또는 명목0.1초 timestamp 근사를 임시 방안으로 제시했다. 정확히 복구된
timestamp/state 대응표나 원본 수정본을 제공한 것은 아니다.

로컬 v1.3.1 장면 `33e5ac4c35a5af28`은 보고된 장면 ID와 timestamps193/states199가 같다.
그러나 공개 보고의 shard00123과 로컬 shard00285(record16)는 다르다.
로컬152track은 모두199states, dynamic은193이며 원시 SHA는
`db19918b9e1e021e145f5e8e1552ea195de8d07890004145030e2948f0fb71ff`다.
공개 원시 hash는 없어 바이너리 동일성이나17건 전체의 동일 근본 원인을 확정하지 않는다.
첫 대조에서 파일 번호까지 같다고 가정한 assertion이 실패했다. 이를 즉시 정정하고
`upstream-shape-initial-review-failure.json`에 남겼으며 원본/실행 코드는 수정하지 않았다.

또한 [2022년 이슈402 답변](https://github.com/waymo-research/waymo-open-dataset/issues/402#issuecomment-1029696009)은
training_20s를 공식 training 부분의 비중첩20초 구간으로 설명한다. 이는 현재 프로젝트의
hash 분할이 drive 단위로 독립이라는 증명이 아니다.
[2025년 이슈931 답변](https://github.com/waymo-research/waymo-open-dataset/issues/931#issuecomment-2825188051)은
9초/20초 형식의 scenario ID가 다르며 대응 방법을 제공하지 않는다고 설명했다.
다른 형식의 ID가 겹치지 않는다는 이유로 궤적 독립성을 단정하지 않는다.

**판단:** 길이가 다른 배열을 임의로 잘라 목표32개를 채우지 않는다. 보존된17개 입력 오류와
29개 후보/목표32개·drive 독립성 미확인·정책 test0 상태를 유지한다. 외부에 유사 오류와
같은 장면/길이 보고가 있다는 근거를 추가했으며, 정확한 복구는 여전히 미해결이다.
로컬 `outputs/sac_heldout_expansion_20261119/upstream-shape-evidence.json` SHA
`098a0b1d17387dc7cc7e940f3462915f093e7579e2d4abe047588f44eec7526d`에 조회 상태·출처·
원본 검토 hash·일치/불일치 항목을 보존했다. 새 pytest는 실행 코드 변경이 없어 미실행.
06:01:04 runner2597128/train2597131 실제 생존, 학습3,569/5,000 확인. 안정 정책은 미확보다.

## 2026-11-20 06:23 KST — raw5,000 학습·주행 검토 완료, 정지 출발은 전 구간 거절

기준 HEAD5f6816e, 실행 CLI556fc3d/core96fa1d6 및 고정 소스 유지. 런타임 변경0.
`outputs/sac_mean_decoupled_g_budget5000_20261120/online5000` 학습·저장 및 audit/collection
검토를 완료했다. model SHA `7b3352405eb5b4385cb17e97b56c81eb6be479e4c2cebcd46d24cbd16d1142e2`,
replay10,000행 SHA `94887d12610ed684447b959fa5e80553dda8e6a38ad1c2a6a6985e64275abea7`,
report SHA `b961bfeddc6a298a28207b4186f582f6ad48e08da63e3e18d18e1c511147bcce`.
learner 전체 복원 동일·이전7,000행17배열/trace prefix 동일·train-only·teacher0 검토 통과.
추가 전이와 업데이트는 각각3,000개이며 collection-seed2000 재시작을 기록했다.

전체5,000 경험은 train36고유장면, main3,529/branch1,471, branch종료274개다.
추가3,000은24고유장면, main2,136/branch864, branch종료162개였다.
추가 경험의 step전 정지는1,590/3,000, 수락은583/3,000이다. 이전1,000 구간의
정지638/1,000·수락143/1,000과는 장면과 학습 중 정책이 다르므로 통제된 주행 비교가 아니다.

### 고정 기존4주행: 성공 없음

동일 초기 frame/map/seed4쌍과 원본 수치·hash를 두 비교에서 확인했다.

| 기존2장면×KEEP/FOLLOW | entropy5,000 | own raw2,000 | raw5,000 |
| --- | --- | --- | --- |
| 성공 / 충돌 | 0/4 / 4/4 | 0/4 / 4/4 | 0/4 / 4/4 |
| tick | 210 | 230 | 213 |
| 거절 | 89 (42.38%) | 100 (43.48%) | 87 (40.85%) |
| 제동 | 78 (37.14%) | 88 (38.26%) | 78 (36.62%) |
| step후정지 | 25 (11.90%) | 36 (15.65%) | 23 (10.80%) |
| 평균 return | -20.233012 | -20.247278 | -20.226955 |

모두 정지 중 뒤쪽 차량과 접촉했다. 접촉 방향은 과실 판정이 아니다. 일부 지표는 낮아졌지만
종료까지의 길이도 달라졌고 성공은 생기지 않았다. 이를 안정 주행 개선으로 인정하지 않는다.

### 추가 정지4주행: 거절·제동·정지 모두100%

같은 초기4쌍의 과거 entropy2,000·자신의 raw2,000·새 raw5,000을 대조했다.
entropy5,000 정지 평가를 수행한 것은 아니다. 세 모델 모두328틱·0성공·4충돌이었다.
raw2,000의 거절264/328·제동140/328은 raw5,000에서 각각328/328이 됐다.
raw5,000은328틱 모두 KINEMATIC_LIMIT이며 정지328/328, 평균 return -21.049955다.
누적 이동은 장면별0.000045776/0.000061035m이고 최대 속도는 초기 속도
0.000380472/0.000367104m/s였다. 출발 성공이나 의미 있는 이동이 아니다.

### 완료 가중치의 읽기 전용 보정 폭 대조

기존 이상적 산술의 LayerNorm/weight envelope 계산을 같은 raw learner의1,000/2,000/5,000
완료 가중치에 적용했다. CPU에서 msgpack만 읽었으며 NN forward/환경 step/update는0이다.
기존 raw1,000과 entropy5,000 결과를 재사용 대조해 모든 값이 같은지 확인했다.

| 고정 raw 가중치 | 1,000 | 2,000 | 5,000 |
| --- | --- | --- | --- |
| 3초 learned residual norm envelope (m) | 0.062905 | 0.173719 | 0.369869 |
| suffix 수정 전 첫 참조 속도의 learned residual envelope (m/s) | 0.000234703 | 0.000734387 | 0.001576203 |

**전체 float32 입력에 대한 인증된 bound가 아니며 차량 최대 속도가 아니다.** anchor v/a,
Gaussian 잡음, suffix 수정, 폐루프 누적 이동을 제외한 현재 가중치의 평균 보정분만 다룬다.
영구적 모델 용량 제한이나 수렴 불가능의 증명도 아니다. 보정 폭은 커졌지만 실제 정지
출발 실패는 악화됐으므로 이 수치만으로 새 gain/구조 변경을 선택하지 않았다.
로컬 `mean-weight-envelope.json` SHA
`9cb13c63a472ba163d45d7d23ac549130b59d9cde4789001016ed7b1248b63f6`.

검토 산출물 SHA: collection `1945e671684a8cbfabfe14be99334d12091bf9195b64867fc7f87058cfd79122`,
동일 예산 primary `04de50af2a64a38ed7cb6b999ae3214bbf109d761633bbf6e6a41c703fc4d165`,
자신의2,000 primary `f409d9047d561880f8dfc02cc2f1fc7359b7e33aa76e76b71cac10b17d85a72c`,
정지 비교 `4f89c9d283aba5e4dff1160ce11f25606426986dbe80204c8cc029716998205b`.
모두 같은 로컬 출력 폴더 안의 JSON이며 공개 산출물이 아니다.

06:23:34 runner2597128/q5000 child2612309 생존, q5000은 RUNNING이다.
남은 고정Q/coverage/실제branchQ/MC와 가치 비교를 이어간다. 완료 학습을 성능 달성으로
세지 않으며 후보 미채택·안정 정책 미확보·전체 목표 active다. 새 예산/gain/구조는 미선택이다.


## 2026-11-20 06:36–06:39 KST — raw5,000 최종 미달 검토 및 동일 learner 10,000 준비

5,000 runner2597128과 모든15개 child가 정상 종료했고 PID가 없는 것을 확인했다.
변경 전35개 고정 소스 및 검토 입력 hash를 확인한 뒤 `finalize_review.py`를 실행했다.
로컬 `outputs/sac_mean_decoupled_g_budget5000_20261120/scientific-review.json`은
`REVIEWED_DRIVING_FAILURES / qualified=false`, SHA
`998b426116777f165a03e77fbab34253656d2c5f630c59f6635f49d3b6fc5ffb`다.
기존4주행과 정지4주행의 실패는 위 기록 그대로이며 두 cohort를 독립 test로 부르지 않는다.

### 실제 미래 보상과 종료 행동의 잘못된 Q 선호

같은2개 train 상태×seed88019..88022의 원래 Gaussian 후속8주행이 모두 충돌했다.
고정 첫 평균 행동의 Q와 이후 실제 discounted reward/규약 entropy 합을 대조했다.

| 상태 | 첫 Q | 실제 soft return 평균 ± 표본SD | Q−실제평균 |
| --- | --- | --- | --- |
| KEEP 77ae312df21e80b9 tick0 | -0.889010 | -12.505350 ±1.393296 | 11.616339 |
| FOLLOW baac7893b025167a tick4 | -0.776821 | -11.175464 ±0.491882 | 10.398643 |

자신의 raw2,000 대비 KEEP soft return은+0.106684지만 종료 할인 효과+0.432048,
entropy 변화-0.324451이고 undiscounted reward는-0.006300 악화했다.
FOLLOW는 soft return-0.627383, 종료 할인-0.223047/entropy-0.399074이며
undiscounted reward+0.002312다. 동일 예산 entropy5,000 대비 KEEP+1.618542도
종료 할인+1.482401이 대부분이며 undiscounted reward-0.058072다.
성공 없는 return 증가를 안정 주행 개선으로 판정하지 않는다. 소규모 paired MC이며
정확한 기대 Q나8개 독립 장면의 근거가 아니다.

고정44개 심한 종료의 저장 행동 Q MAE는 raw2,000의3.616825→1.152763으로 감소했다.
그러나 같은 충돌 결과·보상·다음 관측 hash·bootstrap=false를 갖는 현재 평균 행동을
저장 행동보다 높게 평가하는 문제가 남았다. 고정 KEEP/FOLLOW 반례 선호 차이는
+3.553111/+0.153849, 별도 실제 branch KEEP/FOLLOW는+4.834799/+0.895765다.
각 현재 평균 Q−즉시 보상은5.168016/0.390264/4.060415/0.434223이다.
branch 원본 행동은 저장 replay행과도 일치함을 확인했다. 이 terminal 오차는
해당 전이의 다음 Q bootstrap으로 설명되지 않는다.
`terminal-preference-review.json` SHA
`40d4a5be4102261429dee28c7a33a7766a42512915a180cc38b3e6791353cfbf`.

### 자기 실패 경험은 실제 학습에 들어갔지만 현재 행동 분포 차이가 남음

총1,280,000 batch draws 중 warmup886,708/online393,292다.
추가3,000 중 새 main 심한 종료18행은786회, branch162행은6,528회 뽑혔고
한 번도 뽑히지 않은 새 심한 종료는0행이다. 단순 저장 누락/전혀 학습하지 않음은
이 결과와 맞지 않는다. 충분히 학습됐다는 인과적 증명은 아니다.

최종 정책의 조건부95% Gaussian 영역 안에 있는 online행은16.96%다.
속도별 online 정지2,668행 중0행, 저속360행 중11행, 이동1,972행 중837행이다.
최근100행에서는 정지27행 중0행, 저속3행 중1행, 이동70행 중30행이다.
이는 Gaussian 지지집합/성공 확률/새 검사 기준이 아니다. 장면과 학습 중 정책이
다른 구간 간 비율만으로 수집 방식의 인과 효과를 주장하지 않는다.
`coverage-speed-stratification.json` SHA
`7447e9603b2f4f85e8e510267514e41636f1cbe4dd69df9d1f444fd8f7b5c88b`.

### 다음 제한 실험과 실행 전 검증

5,000 완료 후 선택한 다음 범위는 **같은 learner 총10,000회**다. 추가5,000 실제
자기 전이로 비용 전달·현재 행동 적합·실제 주행의 학습 경과를 함께 확인한다.
학습 부족만이 원인이라는 확정이나10,000에서 성공한다는 보장은 없다.
raw gain0.0001·physical Q·quota8/window6/moving_predecessors/horizon8/restart-terminal,
uniform replay50k·전이당1update·원래 Gaussian·보상·검사·PD·BC/teacher0은 유지한다.
원래5,000 모델/optimizer/target/learner RNG와 replay10,000행/RNG를 복원하고
collection-seed5000으로 환경/장면/행동 RNG만 명시적으로 재시작한다.
exact rollout resume가 아니며 같은 예산 entropy10,000 비교군은 없다.

`diagnose_sac_online_updates.py`와 `sac_diagnostic_continuation.py`의 continuation
상한만5,000→10,000으로 확장했다. fresh상한1,000은 유지한다.
`5000 + total_budget < replay.capacity`를 두 실제 Replay.load 직후 검사해
원본prefix 소실과 cursor wrap을 막는다. replay를 키우거나 과거 행을 제거하지 않는다.
기준 HEAD e04309e 이후 미커밋 변경이며 core SHA
`96fa1d625b0d8525b1de6de05a12cc132721a03ee47eb51dff3d6e25f02e30ef` 동일 확인.
과거 실험의 CLI 현재파일 hash는 이 변경 이후 달라지며 당시 소스 사본/해시가 기준이다.

검증: `JAX_PLATFORMS=cpu .../python -m pytest tests/training/test_sac_diagnostic_continuation.py -q`
**20 passed**. 초과/비증가 예산·설정/해시/이력 변조·검증된 과거 경계·용량 동일/초과
거절 및 충분한 용량을 확인했다. 실제 CLI의 fresh1,001/continuation10,001은
입출력 생성 전에 exit2로 거절됐다. 실제 완료5,000의 lineage/hash와 replay capacity50,000,
size/cursor10,000을 CPU 검토했다. 예정15,000행은 wrap이 없다. 새 정지 reviewer는
완료5,000을 양쪽에 넣어4쌍 frame/map/seed/수치를 대조하는 preflight를 통과했다.
runner/reviewer py_compile 통과. 새학습의 실제 learner복원은 실행 후 별도로 확인한다.
전체회귀/추가 GPU preflight는 미실행이며 코어 수학·환경·모델은 변경하지 않았다.

로컬 실행 준비: `outputs/sac_mean_decoupled_g_budget10000_20261120/`의 decision/run_trial/
preflight/stopped-review-preflight. 총10,000 학습→저장/audit/prefix→기존4주행→정지4주행→
고정Q/coverage/실제branchQ/같은2train상태×4seedMC를 순차 연결했다.
기존 raw5,000과 대조하며 전체 개발·다른 seed·Rule-based·별도 test 확대는
실제 개선 후 판단한다. 10,000 이후 추가 학습이나 기본 채택은 자동 연결하지 않는다.
안정 정책은 미확보이며 전체 목표는 계속 진행한다.


## 2026-11-20 06:44 KST — 10,000회 후속 실행 복원 확인과 정지 거절 세부 분석

코드 `6b1cfd2feaed7e47bca908c5bf5478a0b8616160` push/원격 SHA 확인.
06:40:35 runner2620438/train2620440으로 실제 시작했다. 2026-11-20T06:44:24.274159+09:00 두 PID 생존,
총5128/10,000·새 전이/update128 확인. learner/optimizer/target/RNG 전체
복원 바이트 동일, collection-seed5000, 고정 소스 hash 동일이다. 런타임 변경 없이 진행한다.
로컬 `launch.json`, `restoration-observed.json`, `status.json`에 실제 실행 근거를 남겼다.

완료 raw5,000의 정지4주행 저장 frame을 읽어 운동한계 세부항목을 대조했다.
모델 forward/환경 step/학습은 추가 실행하지 않았다. 같은 초기4쌍의 과거 결과는 다음과 같다.

| 모델 | 총 tick | 거절 | KINEMATIC_LIMIT | PREDICTED_COLLISION | 곡률 위반 | 곡률 변화율 위반 |
| --- | --- | --- | --- | --- | --- | --- |
| entropy2,000 | 328 | 80 | 8 | 72 | 2 | 8 |
| raw1,000 | 328 | 124 | 57 | 67 | 57 | 57 |
| raw2,000 | 328 | 264 | 252 | 12 | 252 | 252 |
| raw5,000 | 328 | 328 | 328 | 0 | 328 | 328 |

raw5,000의 모든 거절은 곡률과 곡률 변화율이 함께 위반된 것이다. 이 cohort에서는
가속도/jerk 위반이 기록되지 않았다. 한 거절의 다중 위반을 합쳐 거절 수를 늘리지 않는다.
KINEMATIC_LIMIT가 먼저 반환되므로 PREDICTED_COLLISION0을 예측 충돌 위험0으로
해석하지 않는다. 같은 시작 상태지만 후속 정책/관측은 다르며 독립 test나 인과 분리가 아니다.

raw5,000의3초 raw 경로 끝점 변위 중앙값은4쌍별0.05546/0.05510/0.05865/0.06924m다.
이동 경로는 매우 짧지만 이를 곡률 검사 면제로 삼지 않는다. ego heading 방향의 끝점이
뒤쪽인 제안은101/328개였다. 이는 raw제안 끝점의 기하학이지 실제 후진, 도로 Frenet방향,
PD 참조 속도나 학습 잔차만의 값이 아니다. 원래 decoder는 ego-local residual을 yaw로
world XY로 회전하며, 본문 확인에서 그 회전 누락을 발견한 것은 아니다.

최초 분석기는 모든 과거 거절도 KINEMATIC_LIMIT라고 잘못 가정해 assertion으로 실패했다.
이를 숨기지 않고 `stopped-violation-review-initial.py`와 initial-failure.json에 보존했다.
모든 이유를 그대로 집계하도록 읽기 전용 분석기만 수정했으며 입력/런타임은 그대로다.
완료 산출물(로컬5,000 출력폴더) `stopped-violation-review.json` SHA
`10fb4cf1fd0e14c837d5faf4f402368ca22a8fb95f1a226a04ccd2a093281749`.

Notion 첫 화면·03절43·04 현재 작업을6b1cfd2 기준으로 갱신하고 실제 재조회했다.
핵심 수치/새 실행 경로/커밋 링크 및 과거 headings5/44/19개 보존·43절 중복 없음 확인.
이 추가 정지 분석과 실제 복원 확인도 같은 절에 후속 반영한다. 안정 정책은 여전히 미확보다.


## 2026-11-20 07:25 KST — 10,000 후속 검토기 준비·기존 결과 재현

실행 코드6b1cfd2와 모든 동결 조건을 유지한다. 07:25:11 runner2620438/train2620440
실제 생존·총8,072/10,000 확인. 현재 모델의 완료 평가 결과는 아직 없고 안정 정책은 미확보다.

로컬 `outputs/sac_mean_decoupled_g_budget10000_20261120/`에 두 CPU 읽기 전용
후속 검토기를 준비했다. 모델 forward/환경 step/학습을 추가 실행하지 않고,
기존 완료5,000의 원본 JSON/CSV/replay를 읽어 이전 검토 수치와 대조했다.

- `review_terminal_preference.py`: 실제 보상/종료/outcome/다음 관측 hash가 같을 때만
  동일 실패 전이의 Q 선호로 해석한다. 새 평균 행동이 비종료로 바뀌면 Q−즉시보상을
  terminal 오차로 세지 않는다. 저장 행동·scene/tick/reward/bootstrap를 실제 replay와
  대조하며 기존 네 반례의 모든 기존 수치가 정확히 재현됐다.
- `review_speed_coverage.py`: [0,0.1)/[0.1,2)/[2,∞)m/s의 정지/저속/이동으로 전체
  online행과 최근100행을 분류한다. 원본 index/전체행 수/자료 hash를 검증하고 기존5,000
  그룹별 행 수·현재95%영역 포함 수·추출/미추출 수·RMS 중앙값 모두 정확히 재현했다.
  정책/방문 상태가 다른 예산 사이의 값이며 인과적 paired 주행이나 안전 gate가 아니다.

실행은 각 script에 `--run outputs/sac_mean_decoupled_g_budget5000_20261120/online5000`
및 새 출력폴더 내 별도 `--output`을 사용했다. 속도 검토에는
`--previous-review outputs/sac_mean_decoupled_g_budget5000_20261120/coverage-speed-stratification.json`도 지정했다.
두 script와 후속 `finalize_review.py`의 py_compile 통과. 준비 검증의 정확한 source hash는
`review-preparation.json`에 있으며 terminal preflight SHA
`f21906226f20ca0e420552668a7802cbf40d5fc8bf9ff922a9dbb47453cc9b79`,
speed preflight SHA `d1f1f326010dca6df86a74588de89a3541f3319d2d2ba151dc97ad67b8e6f13b`다.

실제10,000 Q/branch/coverage가 COMPLETE일 때만 새 입력에 적용한다. 최종 검토기는
전체 runner job 정상 종료·PID 부재·소스/입력 hash 확인 후에만 결과를 작성하도록
준비했으며 아직 실행하지 않았다. runtime/runner/decision 변경0, 새 pytest/주행 평가0.
이 준비는 새 후보의 주행 성능이나 목표 달성의 증거가 아니다.


## 2026-11-20 07:50 KST — raw10,000 학습·저장 검토 완료, 주행은 미달

실행 코드6b1cfd2·기록 기준 HEAD1589d77, core96fa1d6 및 고정30개 소스를 유지했다.
runner2620438의 train10000/audit10000/collection 검토가 exit0으로 완료됐다.
모델 SHA `f6fafffa758e6b44656f36ca37216361e173ad9a1599b822c2064c7c3d4d1101`,
replay15,000행 SHA `2f7ed86f88e933a6468251891fa20dab0a15c5071d963fb5bfb4396c4399f51a`,
report SHA `6fb3eb7c75fce72f1df4d389d923431244064435bd1789f55ffaf10aa2f0e309`.
실제 추가 전이/update5,000개, learner 전체 복원 byte 동일, 이전10,000행17배열/traceprefix
동일, train-only·teacher0·collection-seed5000을 확인했다. exact rollout resume는 아니다.

총10,000 경험은52개 고유train장면, main6,753/branch3,247·branch종료600개다.
추가5,000은36고유장면, main3,224/branch1,776·branch종료326개,222개 root였다.
추가 전이의 수락945/5,000, step전 정지2,476/5,000이다. 현재/과거 구간의 방문 상태와
정책이 다르므로 이 수집 비율을 통제된 주행 개선으로 해석하지 않는다.
collection 검토 SHA `74072e835ccb06969cf198858f2bf696f906b47380b2979ba4cf6d03032eca82`,
audit SHA `074b95c18a1b41940842c0cf176505481a42fc3d4807bd8249ca0514cb134097`.

### 같은 기존4주행: 모두 충돌하며 주요 tick 비중 악화

두 고유개발장면×KEEP/FOLLOW, 각 초기frame/map/seed4쌍과 원본 수치/hash 대조 완료.

| 지표 | own raw5,000 | raw10,000 |
| --- | --- | --- |
| 성공 / 충돌 | 0/4 / 4/4 | 0/4 / 4/4 |
| 전체 tick | 213 | 182 |
| 거절 | 87 (40.85%) | 104 (57.14%) |
| 제동 | 78 (36.62%) | 84 (46.15%) |
| post-step 정지 | 23 (10.80%) | 27 (14.84%) |
| 평균 return | -20.226955 | -20.287243 |

KEEP/FOLLOW 모두 각2/2충돌이다. KEEP87틱·거절52/제동42/정지14,
FOLLOW95틱·거절52/제동42/정지13이다. 모두 정지 중 뒤쪽 차량과 접촉했으며 과실
판정을 뜻하지 않는다. 종료 시점도 변했으므로 분모를 함께 공개한다. 같은 mean0 참고는
2성공·2충돌이며 Rule-based가 아니다. 후보를 채택하거나 평가 확대 근거로 삼지 않는다.
주행 검토 SHA `78b11d13b7d38a1c2d8351b0f432a556fdd364e6305f937a4ccb998822949c90`.

### 추가 정지4주행: 계속 전 구간 거절

같은2개 정지장면×KEEP/FOLLOW에서 후보와 자신의5,000 모두0성공·4충돌,
328틱 전부 거절·제동·정지, 평균return -21.049955다. 같은 초기4쌍과 모델·원본hash를
대조했다. 모두 곡률과 곡률 변화율이 함께 위반됐고 가속도/jerk 위반은 기록되지 않았다.
운동한계가 먼저 반환되므로 예측 충돌 reason0을 위험0으로 해석하지 않는다.

3초 raw 끝점 변위 중앙값은4쌍별0.08663/0.09038/0.06953/0.07577m다.
직전ego heading 뒤쪽 끝점은5,000의101/328→10,000의287/328이다.
이 기하학은 실제 후진·도로 Frenet방향·PD속도·학습 잔차만의 값이 아니다.
완료 저장 frame만 CPU에서 읽었고 추가 NN forward/환경 step/update는0이다.
기존 entropy2,000/raw1,000/2,000/5,000의 모든 그룹·episode 값 재현도 확인했다.
정지 검토 SHA `3e10bc32ded3dc8b4af432e001b2d57ea6aaf2af9e33d0fdf7b92f6742640df4`,
위반 분석 SHA `3c44b52698035515dfc87f519cbaed7e5ced95402c4439940b47a14f81ddddf8`.

### 현재 가중치의 평균 보정 범위 참고

동일 이상적 산술의 LayerNorm/weight envelope를 완료10,000에도 적용했다.
이전 모든 모델의 수치가 정확히 재현됐고3초 learned residual norm envelope는
5,000의0.369869→10,000의0.971213m, suffix 수정 전 첫 참조속도 잔차 envelope는
0.001576203→0.003976395m/s다. **모든f32 입력에 대한 인증이나 차량 최대 속도가 아니다.**
anchor v/a·Gaussian·suffix 수정·폐루프 누적 이동을 제외한다. 영구적 용량 한계나
수렴 불가능의 증명도 아니며 폭 증가가 실제 주행 개선을 뜻하지 않는다.
로컬 `mean-weight-envelope.json` SHA
`5fb97efef96cfaed8db90f1935d751d42fab0fbfb22a23b8506d96e17c5a56ff`.

2026-11-20T07:50:03.151111+09:00 runner2620438 생존, 현재 작업 [('q10000', 2641718)] 실제 실행 중이다.
고정Q/coverage/실제branchQ/동일2train상태×4seedMC·가치 비교가 남아 있다.
완료 후 terminal/속도 검토기와 최종 검토기를 적용한다. 전체개발·다른seed·Rule-based·
별도test 확대나 새로운 예산/구조/손실 변경은 아직 선택하지 않았다. 이번 예산의 실패를
장기 수렴 불가능으로 단정하지 않으면서, 현재 후보 미채택·안정 정책 미확보를 유지한다.

## 2026-11-20 08:05 KST — raw10,000 전체 검토 종료, 현재 행동의 Q 오차 유지

실행 코드6b1cfd2·기록 기준 HEAD ea829bf, core96fa1d6 유지. runner2620438과
13개 child의 정상 종료/실제 PID 부재, 고정30개 소스·decision·runner hash 및 모든
후속 검토의 원본 hash를 확인하고 `finalize_review.py`를 실행했다.
로컬 `outputs/sac_mean_decoupled_g_budget10000_20261120/scientific-review.json`
SHA `27d54545a21c52375777dc71b00bb47f6af528dc8a7fc2c7c46655743e256e8c`.
별도 `policy-decision.json` SHA
`bb11860d530edefe62d949b54c58c099bfb960c15148d7660f2402f6a1cee5ef`는
`REVIEWED_DRIVING_FAILURES`, qualified=false다. 후보 미채택이며 전체 목표는 미완료다.

앞 절의 기존4주행·정지4주행 실패에 더해 같은2개 train 상태×4seed의 실제 종료 검증도
8/8충돌했다. KEEP Q -1.352793, 실제 soft-return 평균 -12.885103±2.440118(표본표준편차),
차이11.532310이다. FOLLOW Q -0.669760, 평균 -9.083225±1.133343, 차이8.413465다.
자신의5,000 대비 KEEP return -0.379753, FOLLOW +2.092239다. FOLLOW 차이는
종료 보상의 할인 +2.297761·종료 전 보상 +0.157178·entropy -0.362700으로 나뉜다.
평균 충돌 시점이24틱 늦어졌고 비할인 보상 차이는+0.092367이다. 4회 모두 충돌하므로
할인 return 증가를 주행 성공으로 판단하지 않는다. 이2개 train 상태의 소표본을
개발 validation이나 독립 test 결과와 합산하지 않는다.

### 현재 행동과 저장 행동의 차이

고정 severe44개 저장 행동의 Q MAE는 raw5,000의1.152763→10,000의3.109345로 악화됐다.
별도 실제 상태 재실행에서 저장 행동과 현재 평균의 reward/종료/next_obs가 같은4지점
모두 현재 평균을 더 좋게 평가했다. 고정 KEEP/FOLLOW 선호 차이는+6.013400/+2.452528,
실제 수집 branch KEEP/FOLLOW는+7.675698/+3.512333이다. 해당 현재 평균의 Q−실제
종료 보상은 각각+9.223094/+3.599493/+8.669208/+6.796194다. bootstrap=false인
동일 실제 종료를 대조했으므로 이4지점은 미래 가치 항으로 설명되지 않는다.
`terminal-preference-review.json` SHA
`9bffcdf0f34c9eedfbe1cf44f4565c1f461d06f5f96d87ca78e16600093c242c`.

온라인 경험 중 최종 정책의 조건부95% Gaussian 영역 안에 들어가는 비율은
5,000의16.96%→10,000의3.59%였다. 이는 확률분포의 support·안전 기준·성공 확률이 아니다.
누적 정지5,144행은 모두 영역 밖이며 현재 sigma 단위 RMS 중앙값488.644다.
최근100행의 정지20행도 모두 영역 밖이고 중앙값338.207, 누적batch 추출10회·미추출12행이다.
나머지 최근100행 중 저속3행은1행, 이동77행은37행이 영역 안이다. 속도/상태/정책이
다른 구간을 통제된 인과 비교로 해석하지 않는다. 속도 검토 SHA
`b6e3d08c77a9f712d4676613a01f7722b9b029936ceb4bb605d0acf50c129f67`.

추가5,000update의 새 severe main37행은1,955회 추출(미추출1), branch326행은16,732회
추출(미추출5)됐다. 총 실제 수집·uniform replay RNG와 추출량은 확인했지만, 이 수치가
현재 행동에 대한 학습 충분성을 증명하지 않는다.

### 오래된 정책 실행 가능성 점검과 다음 진단

main/branch 모두 `train_actual_step`에서 행동 직전 `policy.actor=state.actor`를 대입한다.
`model_proposer`의 closure는 같은 Policy 객체를 호출하고, `Policy.action`은 현재
`self.actor.params`를 JIT 함수의 인자로 전달한다. 파라미터를 초기 상수로 캡처하지 않는다.
10,000개 trace의 policy_updates_before=i/update=i+1도 모두 일치했다.
이 소스 점검에서는 stale-policy 결함을 찾지 못했다. 과거 각 행동 시점의 실제 파라미터
hash를 기록한 것은 아니므로 counter만으로 모든 역사적 실행을 증명했다고 쓰지 않는다.
마지막 정지 branch 행14991은 행동 이후9update 만에 최종 정책과374.236552 sigma RMS
차이가 났고 학습 batch 추출은0회였다. 소스/trace 검토 SHA
`b69538a5ad6b0023ce16bdaddd312b8b02a64a318bb9fb966788335bcabe302f`.

다음은 로컬 `outputs/sac_raw10000_update_drift_20261120/probe.py`의 읽기 전용 진단이다.
동일10,000 learner/optimizer/target/learner RNG로 매번 돌아가 uniform256 batch4개 각각에
원본 `update()`를 한 번 적용한다. 새 실제 전이는 넣지 않으므로 실제 온라인 다음
update 재현이나 누적4회 학습이 아니다. 마지막100개 실제 관측 전부에서 평균의 변화와
head/feature 기여를 분리한다. 기여 norm은 서로 상쇄될 수 있어 합산 백분율로 쓰지 않는다.
checkpoint/replay 쓰기·환경 step·새 reward label·배포는 없으며 규약10절의 손실/lr/tau,
원래 sampler·보상·검사·PD·BC/teacher0을 유지한다. 단일 변화량이 주행 개선을 증명하지 않는다.
구문 검사는 통과했고 08:05에 PID2649190으로 시작했다. 진단 결과는 후속 절에 기록한다.
결정 SHA `74546826d12fe08704a8e6679a439a58a95367aaef189b66e727e8770fcfb43d`.

## 2026-11-20 08:10 KST — 한 번 update 진단과 평균 B 곱셈의 수치 오차

위 진단 PID2649190은 exit0으로 종료됐다. 원본 learner의 byte 동일성 및 입력 파일 hash가
유지됐다. 4개 독립 batch의 정지20관측 평균 변화는 기존 sigma 단위 RMS 중앙값
32.343/40.898/45.118/49.241, 이동77관측은0.172/0.151/0.166/0.170이었다.
저속3관측은0.941/1.074/1.110/1.015였다. raw head/hidden의 실수 연산 분해와 실제 출력의
차이가 있어 feature 변화만으로 전체 원인을 확정하지 않았다. 이는 저장된 자기 경험에
대한 반사실적 한 번 update4개이며 실제 환경 전이·추가 학습 완료로 세지 않는다.
로컬 `outputs/sac_raw10000_update_drift_20261120/report.json` SHA
`4aa59635d1307432b20db117e2be1868a15a037ff2fbab6503328c5cc4e918ea`.

별도 `basis_numerics.py`에서 **같은 최종 모델의 같은 NN head**를 고정하고 마지막100관측에
대한 B 곱셈만 대조했다. NN의 정밀도나 가중치를 바꾸지 않았다. f32에서 기존 gain을 곱한
head를 기준으로 B의 float64 곱을 독립 NumPy 계산과 대조했다(최대 오차6.51e-19).
실제 actor 출력은 별도로 수행한 default B 곱과 정확히 일치했다.

| B 곱셈 | float64 기준 raw 평균 최대 차이 | 정지20관측 sigma RMS 중앙값 / 최대 | 이동77 중앙값 |
| --- | ---: | ---: | ---: |
| 현재 GPU default | 1.402184e-6 | 279.463767 / 942.256170 | 1.570492 |
| float32 highest | 3.549749e-10 | 0.119780 / 0.209780 | 0.0003053 |
| float64 | 6.505213e-19 | 1.5158e-10 / 3.5973e-10 | 6.4502e-13 |

조건수가 큰 B의 역변환과 작은 정지 g 때문에 작은 raw 차이가 정책 분포 및 critic 입력
좌표에서 증폭될 수 있음을 **현재 모델의 고정 head 비교로 확인**했다. 이전16:37의
target fused/분리 계산 검사는 다른 연산을 다뤘으므로 당시 정밀도 보류 판단과 구분한다.
이 발견만으로 주행 실패 전체가 설명되거나 고정밀 후보가 성공한다고 주장하지 않는다.
기존 coverage는 실제 출력된 평균 기준의 값이므로 삭제하거나 계산 오류로 재분류하지 않는다.

현재 운영/학습 소스·checkpoint·sampler·reward·검사·PD는 그대로다. 다음 보완은 별도
실험 actor에서 NN head를 유지하고 **승인된 B 평균의 곱셈만 float64로 계산**하는 것이다.
60자유도·무제한 출력·동일 B/gain·원래 Gaussian·SAC 목적함수와 lr/tau·BC/teacher0을
유지한다. 이론적 식이 같아도 실제 출력/gradient가 달라지므로 별도 비교로 명시한다.
초기 가중치/직렬화 동일성, sampler/밀도, B 왕복과 gradient를 검증한 뒤 fresh 제한 학습과
동일 주행을 비교한다. 아직 구현/학습/주행은 미실행이며 기본 채택은 아니다.

로컬 `basis-numerics.json`·`basis-numerics.npz`·두 진단 스크립트에 모델/입력/소스 hash와
원본 배열을 보존했다. 규약10절의 손실을 바꾸는 CQL·actor 지연·lr 변경은 선택하지 않았다.

## 2026-11-20 08:17 KST — 고정밀 B 옵션 구현·초기 검증과 fresh1,000 비교 준비

기록 직전 HEAD03e1521에서 `scripts/sac_diagnostic_actor.py`를 추가했다. 원본 Actor의
동일 NN head를 capture하여 gain은기존float32, B 곱만float64로 수행한다. 파라미터·NN·
분산 head·원래 sampler·보상·검사·PD·학습 수식은 바꾸지 않는다. 코어96fa1d6 유지.
기본 factory는 과거 Actor 그대로이며 float64만 명시적으로 선택하고 복원 시 factory
hash를 검사한다. 학습 CLI, 평가/Q/coverage 진단 복원 경로에 이를 연결했다.
continuation에서는 정밀도/actor factory hash 변경을 거절한다. 과거 로컬 MC 스크립트는
보존하고, factory를 사용하는 `scripts/diagnose_sac_soft_returns.py`를 별도 입구로 추가했다.

실행한 검증:

- CPU: mean precision 초기3개·continuation·action coordinates 합계34pytest 통과(25.46초).
- GPU: mean precision4pytest 통과(53.62초). 추가한 전체 learner 초기 직렬화 동일성,
  복원 후 실제 `update_checked`·유한 gradient/update·원본 learner 불변성 검사 포함.
- compileall, git diff --check, CLI help 확인 통과. 전체 테스트 suite는 미실행이다.
- 실제 자기 warmup에서 균등 간격으로 고른256관측(정지171/저속13/이동72)에서 두
  fresh learner의 optimizer/target/learner RNG 포함 byte 동일, 초기mu0·rho·동일key의
  raw표본/ell/log_pi 동일 확인. 학습 후 동등성이나 실제 주행 성능 검증은 아니다.

로컬 `outputs/sac_mean_basis_f64_20261120/preflight.json`과 배열/로그에 입력과 소스 hash를
보존했다. actor factory SHA
`15bbb08bbe17f3e526bab7689be0c1453317170a9c14cee55fb23db40be9a785`.
검사 stdout에서 기록한 `targeted-tests.json`은 요약이며 원본 stdout 파일로 주장하지 않는다.

다음 제한 실험은 같은 seed0·자기 warmup5,000에서 **fresh1,000 실제 전이/update**다.
기존raw1,000 모델 `9cdced951a997905c32ab8ea18c7b9da9866b27c81e8f8de23da06728bd6883c`와
같은 기존2개 개발장면×2명령4주행, 같은 정지2장면×2명령4주행을 대조한다.
정지 baseline은 기존 완료 frame을 재사용하며 초기frame/map/seed를 다시 대조한다.
gain0.0001·physical Q·zero-action 초기화·quota8/window6/moving_predecessors/horizon8/
restart-terminal·uniform replay50k·원래 Gaussian·SAC 목적함수/lr/tau·BC/teacher0 유지.
검토 순서는 실제 저장/audit/자기 warmup17배열prefix→두 주행 cohort→Q/coverage/실제branchQ
→같은2train상태×4seedMC다. 초기1000의 실패만으로 장기 수렴 불가능을 단정하지 않지만
진행 중 조건이나 예산을 바꾸지 않고 완료 결과를 검토한다. 새 PR·merge는 하지 않는다.

## 2026-11-20 08:46 KST — 고정밀 B 1,000회 학습·두 주행 cohort 완료

실행 코드 c4babca·core96fa1d6, 고정32소스 유지. 로컬
`outputs/sac_mean_basis_f64_20261120/`의 train1000/audit1000은 정상 종료했다.
실제 새 전이/update1,000, train10고유장면·teacher0, 기존 자기 warmup5,000행17배열
바이트 동일성을 확인했다. 모델 SHA
`904ca431f8f383e8e2fdaae89543c37abe03ed2f7b4131724e83c1d2f6a5babe`,
replay6,000행 SHA `649325520266255e318cdf34af62764bd8a371c8f6bd33b21870857b16cdf1ff`,
report SHA `5f0cde10d7a587d090e0c6fa3d8cfc6b688caab4eff5624d8537f3532ac8db9f`.

### 같은 기존4주행은 회복됐지만 새 성공은 미확인

초기frame/map/seed와 원본 frame/hash를 독립 대조한 결과다.

| 지표 | default raw1,000 | float64 B raw1,000 |
| --- | ---: | ---: |
| 목표 / 충돌 | 0/4 / 4/4 | 2/4 / 2/4 |
| 전체 tick | 236 | 236 |
| 거절 | 92 (38.98%) | 44 (18.64%) |
| 제동 | 84 (35.59%) | 36 (15.25%) |
| 정지 | 30 (12.71%) | 12 (5.08%) |
| 평균 return | -20.225907 | -9.981245 |

8b4c1e7e09f5f4c3의 KEEP/FOLLOW는 각각72틱 ROUTE_GOAL, 다른 장면
8a4fba3463c6c62d의 KEEP/FOLLOW는 각각46틱 COLLISION이다. 초기 mean0도 같은
성공2/4였으므로 새 성공을 얻었다고 쓰지 않는다. mean0 참고는216틱·거절58·제동46·
정지22로, 현재 후보의 해당 비중은 줄었지만 전체 안정 주행의 증명은 아니다.
주행 검토 SHA `8e677c31acc554f922016ff907201ec8a378a1c1d7aafd2cadb221710a1ec548`.

### 정지4주행은 모두 실패하며 거절·제동 증가

같은2정지장면×KEEP/FOLLOW에서 두 모델 모두0성공·4충돌,328/328틱 정지다.
거절124→146틱(37.80→44.51%), 제동98→108틱(29.88→32.93%), 평균return
-20.540071→-20.595124로 악화됐다. 후보의 최대 위치 변위는 각 주행2.25~3.88mm,
최대 속도는0.00142~0.00145m/s였다. 미세한 움직임이나 부분 경로 수락을 출발 성공으로
세지 않는다. 모두 정지 중 뒤쪽 차량과 접촉했으며 과실 판정이 아니다.
정지 검토 SHA `2c08ffb0fea6455098d26aaec6303da3e436ed824a0e8529bd27272f722a1e66`.

### 수집·실제 학습 노출

main616/branch384이며 branch48개 root에서 각각8step을 실행했다. 분기 후속상태272개,
실제 종료74개·bootstrap310개, main은8충돌/1기록끝/1예산경계다. 기록끝과 예산경계를
목표 성공으로 세지 않는다. 원래raw1,000 main712/branch288과는 방문 상태가 다르다.
현재 main 수락270/정지354, branch 수락104/정지124로 합계수락374/정지478이다.

uniform replay RNG를 재구성해 전체256,000추출 중 warmup233,361·main12,527·
branch10,112를 확인했다. warmup severe44개는2,018회, 새 main severe8개는216회,
branch severe74개는1,619회 추출됐고 branch severe5행은 아직 미추출이었다.
이 노출량은 현재 행동 Q 적합이나 정책 성능의 충분성을 뜻하지 않는다.
fresh sampling 검토 SHA `956607f5ab95cee15fa87740979793089e1e382147f170741c2ee7240329fad4`.

과거raw1,000을 이용한 검토기 사전검증에서는 수치·원본hash를 재현했다. 첫 전체dict
비교는 상대/절대 경로 문자열 차이로 실패했고 resolve한 파일과 hash가 같음을 확인했다.
숫자나 허용치를 바꾸지 않았으며 실패/정정은 sampling-preflight-review.json에 보존했다.
최종 검토기의11개 job 목록과 실행 중 거절 검사를 확인했다. 후보 최종 검토는 미실행이다.

08:45:23 runner2656472/q1000 child2667013 실제 생존, Q→coverage→실제branchQ→
같은2train상태×4seedMC가 진행 중이다. 전부 끝나면 같은 저장 관측에서 실제 Policy
batch1과 진단 batch100/256의 수치 일치도 추가 확인한다. 이는 아직 검증하지 않은
가설이며 현재 분포 차이를 batch 문제라고 단정하지 않는다. 해당 읽기 전용 도구의
구문/active-run 차단 검사만 통과했고 NN/환경 실행은0이다.
후속 검토기 준비 SHA `0489e3b556b82588ea18b93def0e444fede06fc7c2608efa0cfcef8ddf454436`.
**안정 정책 미확보·기본 미채택**을 유지하며, 다음 예산/손실/수집 변경이나 평가 확대는
남은 실제 결과를 검토한 뒤 결정한다. 현재 실행의 source/runner/decision은 바꾸지 않는다.

### 2026-11-20 09:00 KST — 고정밀 B 1,000회 검토 종료, 동일 정책 5,000회 비교 시작

실행 코드 `c4babca1220c41715df2043cab00cae7eaeda70f`, 기록 직전 HEAD
`b1c12a727add00731a9db57795df861858b586c5`, 브랜치 `feature/yuntae-sac-training`.
새 production/학습 소스 변경 없이 로컬 실험 orchestration과 기록만 추가했다.
고정밀 1,000회의 11개 job은 모두 exit0, runner2656472와 모든 자식 PID가 종료됐고
32개 고정 소스·decision·runner·원본 결과 hash를 대조했다. 최종 검토를 실행했다:

```bash
python outputs/sac_mean_basis_f64_20261120/finalize_review.py
python outputs/sac_mean_basis_f64_20261120/audit_actor_batch_shapes.py
```

두 명령 모두 실제 exit0. 첫 검토 SHA는
`fc87375757fb29e9e650d2e5b95e4142d77067442b53a73a8a83965ddf46e1d9`.
**주행 실패·미채택**이다. 기본4건은0→2성공으로 회복했으나 초기 mean0도2성공이었고,
정지4건은 모두충돌·328/328정지였다. 별도 train2상태×4seed의 실제 종료 검증도
8/8충돌이었다. 고유2상태의 반복 실행이며 독립8장면으로 세지 않는다.

| 실제 미래 가치 검증 | KEEP | FOLLOW |
| --- | ---: | ---: |
| 현재 첫 평균 행동 Q | −0.472053 | −0.280522 |
| 실제 soft-return 표본평균 | −13.510993 | −10.758587 |
| 표본 표준편차(각4회) | 1.849483 | 0.379546 |
| Q−표본평균 | 13.038940 | 10.478065 |
| default raw1,000 대비 평균 return 변화 | +0.065952 | −0.337416 |

원본 trace에서 보상·entropy·할인을 다시 합산했다. 가치 독립 검토 SHA
`438ca73f6f9faaf61ce87c62171992eddecb02eeaf65c68b4575fd132bce9299`.
고정 종료44행의 저장 행동 Q MAE는5.879852→5.397112로 조금 줄었지만, 실제 현재 행동의
미래 실패 비용 과대평가가 남았다. 비종료 Q를 즉시 보상과 비교해 오차라고 판단하지 않는다.

새 실제 분기 종료에서는 저장 행동·현재 평균·확률4표본이 같은 next_obs/보상/충돌을
만드는 사례를 확인했다. KEEP `77ae312df21e80b9` tick28은 현재 평균 선호가 저장 행동보다
+1.066748점, FOLLOW `baac7893b025167a` tick64는+8.817206점이다. 현재 평균 Q의 실제
종료 보상 대비 오차는 각각3.356329/12.668435점이다. 서로 다른 모델의 첫 종료 사례는
동일 관측이 아닐 수 있으므로 그 차이를 통제된 모델 성능 비교로 쓰지 않는다.
고정 warmup의 실제 종료2건도 포함한 검토 SHA
`54077b7d4919d7471a89647b54efbd187300b637ea68902ea063c839cffafaf0`.

현재 자기 경험1,000개의 조건부95% Gaussian ellipsoid 내 비율은45.3%, 마지막100개는1%다.
마지막100개 중99개가 정지 상태이며 총223회 학습 추출,21행은 미추출이었다.
이는 표본의 통계적 support·안전성·주행 성공률을 뜻하지 않는다.

배치 계산을 학습 변화로 오해했는지도 별도 검사했다. 후보의 실제 마지막100관측을
두 고정 모델에 공통으로 넣고 원래 `Policy.forward`의 batch1/100/256을 비교했다.
모델별3관측은 실제 `Policy.action`과 정확히 같았다. batch256의 반복 관측은 배치 크기를
맞추기 위한 것이며 독립 데이터가 아니다. 99정지/1이동, 저속0으로 범위가 제한된다.

| 모델 | batch | 정지 평균 차이 σ-RMS 중앙값 / 최대 |
| --- | ---: | ---: |
| default raw1,000 | 100 | 0.009423 / 0.065198 |
| default raw1,000 | 256 | 0.011505 / 0.101131 |
| float64 raw1,000 | 100 | 0.002594 / 0.046342 |
| float64 raw1,000 | 256 | 0.002519 / 0.094754 |

float64 후보 raw 평균 최대차는 batch100에서1.370e−7, batch256에서2.774e−7이다.
정확히 동일하지 않지만, 이 상태 집합에서 마지막100행의 기록된 행동 거리 중앙값
8.115σ-RMS를 설명할 만큼 크지는 않았다. 다른 상태/모델에 대한 수치 동등성 인증이나
전체 실패 원인 확정은 아니다. 배치 감사 SHA
`152923c82488a3f9fd1a35d4255ed6466b941dc3b0cb4dbc8b7ec266a17afbd9`.

**다음 결정:** 같은 f64 learner를1,000→2,000→5,000으로 이어, 기존 default raw의 같은
예산과 비교한다. 장기 수렴 불가능이라고 결론내릴 수 없는1,000회 결과에서 새 설정을
동시에 바꾸지 않는다. collection RNG는1,000/2,000 경계에서 각각1000/2000으로 재시작해
기존 비교의 경계를 맞춘다. learner/optimizer/target/replay RNG는 복원하고 실제 prefix를
검사한다. 환경까지 정확히 이어받는 resume는 아니다. 2,000/5,000에서 기본4주행을 평가하고,
5,000에서 정지4·고정/실제 종료Q·coverage·실제 미래 보상을 함께 검토한다.

최근 경험을 더 자주 뽑는 방법은 **설계 제안·미선택**으로 남겼다. v1.2 §10은 batch256,
replay50k,실제 전이당1update와 손실을 명시하지만 uniform 추출을 명시하지는 않는다.
다만 현재 비교에서 uniform은 고정 조건이다. [Wang·Ross의 ERE 원논문](https://arxiv.org/abs/1906.04009)은
최근 경험을 강조하는 추출과 업데이트 순서를 제안한다. 그 MuJoCo 결과를 우리 경로 정책의
성공 근거로 옮기지 않으며, 이번에는 ERE나 임의 recent 비율/window를 구현·채택하지 않았다.

후속 로컬 경로 `outputs/sac_mean_basis_f64_budget5000_20261120/`.
09:00:14 KST runner2674664를 시작했다. 원래 Gaussian·보상·검사·PD·순수 SAC(BC/teacher0)·
mean gain0.0001·physicalQ·quota수집·uniform replay·손실/lr/tau·전이당1update를 유지한다.
**34개 소스**를 고정하며, 새 runner와 stopped 검토기는 구문 검사 및 과거 명령과
precision/입출력 경로만 다른지 대조했다. 기존 실행을 변경하거나 재시작하지 않았다.
후속 decision SHA `2dccc391e706192af2ade1a3d0c8330e5d1d35035e51b18be244729da23b9342`,
1,000회 정책 판단 SHA `fd238557ddd06b89a2f4358536ce749fa593dca1dda6e9338a79e0288c1b8964`.
이번5,000은 선택된 제한 비교이며, 그 이후 예산·최근 replay 변경·기본 채택·전체 개발/다른seed/
동일 Rule-based/별도test 확대는 미선택이다. **안정적인 정책 확보 목표는 계속 진행 중**이다.

### 2026-11-20 09:07 KST — 최근 경험의 학습 지연과 추출 후보를 CPU로 분리

기준 HEAD `5601b8b107996613de4d43a5fff72630607cb7ba`, 실행 코드는c4babca다.
현재5,000회 비교는 변경하지 않았다. 09:07:02 runner2674664/train2674665 실제 생존,
총1,288/2,000이며34개 고정 소스 hash가 같다.

완료된 f64 fresh1,000의 **고정 삽입 순서**를 CPU로 재현했다. 각 update에는 그 시점까지
실제로 들어온 행만 사용하고, 모든 후보의 추출량은256개다. 원래 uniform256의256,000개
추출 횟수와 저장 replay RNG가 원본 CSV/metadata와 정확히 일치했다.
실행한 명령은 다음과 같으며 exit0이다:

```bash
python outputs/sac_replay_recency_review_20261120/review_sampling.py
```

| 고정 stream의 추출 계산 | uniform256 | uniform255+최신1 | uniform128+최근256행에서128 |
| --- | ---: | ---: | ---: |
| warmup 추출 비중 | 91.1566% | 90.8023% | 51.9102% |
| 새1,000행 중 미추출 | 22 | 0 | 2 |
| 새 severe종료82행 중 미추출 | 5 | 0 | 0 |
| 새 severe종료 추출 횟수 | 1,835 | 1,963 | 9,122 |
| 첫 추출까지 추가 update 중앙값(추출된 새 행만) | 14 | 0 | 1 |
| 같은 지연 p95 | 63 | 0 | 5 |
| 삽입 직후 update에 포함된 새 행 수 | 48/1,000 | 1,000/1,000 | 387/1,000 |
| 전체 추출의 정지 관측 비중 | 64.1316% | 64.1406% | 54.0680% |
| 마지막100update 추출의 정지 관측 비중 | 64.0664% | 64.1719% | 76.7188% |

uniform에서 새 severe종료의 첫 추출 지연 중앙값은18update, 미추출5행에는 각각4~30회의
학습 기회가 있었다. 아직 추출되지 않은 행은 우측 검열된 관측으로 별도 표시했으며,
위 지연 통계에0으로 넣거나 전체 지연의 불편 추정치라고 부르지 않는다. 시간 단위는
추가 learner update 수이며 wall-clock초나 물리 tick으로 바꾸지 않았다.

**판단:** 최신1개를 반드시 포함하면 이 stream에서 미학습 전이는 없어지지만 warmup
지배는 거의 그대로다. 따라서 이 한 변경으로 현재 행동 Q 적합까지 해결된다고 볼 수 없다.
반대로 예시128/128은 더 최근 실패를 보지만 마지막100update의 정지 경험 비중이 크게
늘어난다. 마지막100삽입 중99개가 정지였기 때문이다. 최근성을 강화하면 분포 편중도
강화할 수 있으므로 성공률·충돌률·정지 비중을 함께 검증해야 한다.

이는 **추출 인덱스의 반사실적 계산만 완료**한 결과다. 학습·환경 실행은0이고, 실제로
추출을 바꾸면 가중치와 뒤의 수집 stream도 달라진다. 표의 차이를 주행 성능 예측이나
과대평가 원인의 인과 증거로 쓰지 않는다. 128/128·window256은 단일 민감도 예시이며
하이퍼파라미터 탐색·선택·훈련 적용이 아니다. 산출물은 로컬
`outputs/sac_replay_recency_review_20261120/`, report SHA
`bd015515c4408dc22d27240bc83130a353bd213389385858d88a22250677e530`.

[ERE 원논문 §3.1–3.2와 Appendix B.1](https://arxiv.org/html/1906.04009v1)은 episode 수집 뒤
여러 update의 추출 범위를 순서대로 좁힌다. 매 전이1update 방식에 순서를 적용하는 것은
그 논문에서 후속 연구로 남겼다. 따라서 우리 예시를 ERE 구현이라고 부르지 않는다.
[Zhang·Sutton의 CER](https://arxiv.org/pdf/1712.01275)은 추출 batch에 최신 전이를 더한다.
여기서는 규약 batch256을 유지하려고255+1로 계산했으므로 원논문 그대로의 알고리즘이라는
주장도 하지 않는다. 두 논문의 다른 환경 성능을 우리 SAC 주행 성공 근거로 쓰지 않는다.

v1.2 §10의 손실 수식·batch·전이당 update 수를 유지할 수 있어도, 추출 분포를 바꾸면
손실에 반영되는 상태·행동의 가중치가 달라진다. uniform 목적함수의 동일한 불편 추정기라고
표시해서는 안 된다. 현재는 세 후보 모두 수치 검토이며 **추출 변경 구현·훈련·채택은
미실행**이다. 실행 중인 동일 조건5,000 결과를 먼저 검토한다.

### 2026-11-20 09:22 KST — f64 누적2,000의 주행 퇴행 확인,5,000 학습 계속

실행 코드c4babca, 기록 직전 HEAD `104e9e981cfea2a1ea2c768b4083061b8f8a53c8`,
브랜치 `feature/yuntae-sac-training`. 이미 선택한1,000→2,000→5,000 실험의 중간 결과다.
현재34개 학습/검증 소스는 유지했다. train2000·audit2000·collection2000·evaluation2000·
동일 예산/자체1,000 비교 검토까지6개 job이 모두 exit0으로 끝났다.

누적2,000 모델 SHA `65196cc1183d10a3a61a23f9aca01687cac6b9ab320739791168208a48a9798f`,
replay7,000 SHA `9ab4c6c50ac260f1553369779c100e42950eae0fa7678b87219623bc6696a085`.
이전6,000행의17배열·ticks/updates 기록 prefix가 바이트 동일하고 learner 복원도 동일했다.
수집 RNG 재시작1000은 계획대로이며 정확한 환경 resume를 주장하지 않는다.
audit 파일 SHA `45c62b1912b452e7739ab7477c913a08978ad6afa2d75637338a2346e00989f2`,
collection 검토 SHA `9b52753c2b4e6cd0b6f8a086942265005d4c14ef6fff5db5ddb06aedb2942c61`.

누적19고유 train장면, main1,319/branch681, 분기 실제 종료134개를 저장했다.
추가1,000은9고유장면의 main703/branch297이다. 추가분 중 수락225·정지 전 상태570이며,
분기297개 중 FOLLOW288/KEEP9, 실제 종료60개다. 명령별 수집 편중을 보존한다.
이는 서로 다른 방문 상태의 학습 수집 통계이며 평가 성공률이 아니다.

같은2개발장면×KEEP/FOLLOW의 초기 frame/map/seed와 원본 evidence를 대조했다:

| 지표 | f64 1,000 | default raw2,000 | f64 2,000 |
| --- | ---: | ---: | ---: |
| 목표/충돌(각4회) | 2/2 | 0/4 | **0/4** |
| 실제 tick | 236 | 230 | 208 |
| 경로 거절 | 44(18.64%) | 100(43.48%) | 102(49.04%) |
| 제동 개입 | 36(15.25%) | 88(38.26%) | 86(41.35%) |
| 정지 | 12(5.08%) | 36(15.65%) | 32(15.38%) |
| 평균 reward return | −9.981245 | −20.247278 | −20.262482 |

**기존 성공2건을 잃었고 같은 예산 default 대비 개선도 입증되지 않았다.** 짧아진 episode와
정지 tick 감소를 개선으로 세지 않는다.4회 모두 정지 중 후방 접촉으로 분류됐지만,
사고 책임이나 다른 제어였다면 반드시 피했다는 인과 주장은 하지 않는다. mean0도2/4목표였고
Rule-based와는 다른 참조다. 동일 예산 검토 SHA
`b59b8931852602ccc7315cfde279e4e5b04dbaa7d9bbed7aa28698cc1f3a3cbf`, 자체1,000 비교
`b24244efe5202690f58b844e29736defa40a7a269c9241a0ea9ff3b6d99c97ca`.

CPU 원본 비교 `python outputs/sac_mean_basis_f64_budget5000_20261120/review_regression2000.py`
도 실제 exit0이다. 고속 `8b4c1e7e09f5f4c3`의 두 명령 모두 실행 분류가 처음 달라지는 시점은
post tick30/proposal tick29다. 이전은NEW/OK, 현재는REUSED/PREDICTED_OFFROAD다.
현재는post32 BRAKE→40 EMERGENCY→58정지→63충돌(객체4877)로 이어졌다.
직전1,000은post65부터BRAKE를 썼지만72 ROUTE_GOAL, 정지0·EMERGENCY0이었다.
2,000의 마지막30tick은 연속 거절이었다. 나머지 장면은post26에 실행 분류가 갈라지고
28비상제동→32정지→41충돌로, 직전의37비상제동→41정지→46충돌보다 앞당겨졌다.

같은 초기상태의 첫 제안30점 최대 좌표차는4건 모두 약0.0223m였다. 고속 분류 차이가
나기 직전 실제 차량 위치차는KEEP0.06130/FOLLOW0.06162m였다. 이후 관측은 이미 서로
달라졌으므로 동일 관측의 정책 인과 비교라고 부르지 않는다. 수치와 경로 거절→제동→정지→
충돌의 순서를 기록한 것이며 유일 원인·필요충분조건을 입증한 것은 아니다.
퇴행 검토 SHA `fee3e2b82157387b47336a5e57e413c3d68df757fad9fc940a6acb3e7463cfc9`.

이후 예정한 총5,000 학습으로 넘어갔다.09:22:08 runner2674664/train5000 child2683551,
총2,029/5,000·collection seed2000·learner 복원 동일을 확인했다.2,000의 별도 Q/정지4/MC는
이번 계획에서 실행하지 않으며5,000에서 수행한다. 직전1,000 Q 결과를2,000 결과로 옮기지 않는다.
현재 보상·안전검사·PD·Gaussian·순수 SAC·수집·추출·손실·학습률·배율은 변경하지 않았다.
**안정 정책 미확보·2,000 후보 미채택·전체 목표 진행 중**이며,5,000 이후 변경과 확대는
전체 결과 검토 후 결정한다.

### 2026-11-20 10:31 KST — f64 5,000회 검토 완료, 남은 개발 장면 평가

실행 코드 `c4babca1220c41715df2043cab00cae7eaeda70f`, 기록 직전 HEAD
`b57c83f718a53a321002153ef3e78d0815f1c037`, 브랜치 `feature/yuntae-sac-training`.
로컬 `outputs/sac_mean_basis_f64_budget5000_20261120/`의21개 job이 모두 exit0으로 끝났다.
runner/모든 child의 실제 종료와34소스·decision·runner·산출물 hash를 확인했고
`finalize_review.py` 실행도 exit0이다. 학습 종료와 안정 정책 확보는 다르다.
검토 SHA `303138327b9ee5dd4c66efcadefb1bdfdbcb652e9a2268ba33982acb17653eab`,
미채택 결정 SHA `6a784fce2073f4d3ef8ef61393fb865784f136de520a80f502d9bc20d5cac6e2`.

모델 SHA `04469e643124f5dee4bbb8b25bd07a5b5dd40564287f14c95526ad04b6f2a880`,
replay10,000 SHA `1531d64964ecbabfab4a8e2932f3300ef222cf97ed0d68a63015616e276ef0a9`.
이전7,000행17배열·tick/update prefix와 복원 learner의 바이트 동일성을 확인했다.
추가3,000은24고유 train장면의 main2,184/branch816이며 분기 실제 종료154개다.
누적5,000은37장면 main3,503/branch1,497, 분기 실제 종료288개·수락1,770·전이 전 정지2,737개다.
이는 학습 중 변하는 정책의 수집 통계로, 고정 정책 평가와 구분한다.

| 기본2장면×KEEP/FOLLOW(4회) | default raw5,000 | f64 1,000 | f64 2,000 | f64 5,000 |
| --- | ---: | ---: | ---: | ---: |
| 목표 / 충돌 | 0 / 4 | 2 / 2 | 0 / 4 | **2 / 2** |
| tick | 213 | 236 | 208 | 241 |
| 거절 tick | 87 | 44 | 102 | 55(22.82%) |
| 제동 tick | 78 | 36 | 86 | 47(19.50%) |
| 정지 tick | 23 | 12 | 32 | 12(4.98%) |
| 평균 reward return | −20.226955 | −9.981245 | −20.262482 | −10.010669 |

고속 KEEP/FOLLOW가77/72tick에 목표 도달했고, 나머지 장면은둘다46tick에 충돌했다.
**mean0도 성공했던 동일2건의 회복이며 새 성공이나 안정 정책이 아니다.** mean0는 Rule-based가 아니다.
정지 출발2장면×2명령은0목표·4충돌,328/328tick정지다. 거절159/제동122로 default의328/328보다
줄었지만 f64 1,000의146/108보다 늘었다. 최대 변위는 약2.20mm로 출발 성능을 입증하지 못했다.
초기 frame/map/seed·원본 tick 집계 대조를 완료했다. 후방 접촉 분류를 사고 책임으로 해석하지 않는다.

고정 severe44개 저장 Q MAE는1.5782(f64 1,000:5.3971, default5,000:1.1528)다.
실제 terminal에서 새 평균 행동의 Q−보상은 고정 KEEP4.2901/FOLLOW−0.0835,
분기 KEEP4.7991/FOLLOW7.1524다. 작은 FOLLOW 과소평가를 큰 과대평가로 표현하지 않는다.
저장 행동 적합 개선이 현재 평균 행동과 앞선 상태의 미래 비용 적합을 보장하지 않는다.

같은2train상태×4확률 seed88019–88022의 실제 종료 검사는 **8/8충돌**이다.
KEEP의 Q−표본평균 soft-return은12.7803(Q−0.9999,평균−13.7802,표본표준편차1.9446),
FOLLOW는10.0737(Q−0.7732,평균−10.8470,표본표준편차0.1562)이다.
default5,000 대비 KEEP은 평균9.75tick 빨리 충돌하고 soft-return이1.2749점 나빠졌다.
FOLLOW는 평균2.5tick 늦게 충돌하면서0.3285점 좋아졌지만 성공은 없다.
자체1,000 대비 KEEP/FOLLOW soft-return은−0.2693/−0.0884점으로,
entropy 기여 감소와 terminal 할인 변화도 분리했다. return 차이만으로 주행 개선을 판정하지 않는다.
MC는 train 진단이며 개발 평가16건이나 독립 test 분모에 더하지 않는다.

uniform256의 총1,280,000추출은 warmup886,708/main261,479/branch131,813이다.
자기 신규5,000행의 현재 평균에 대한 조건부95%ellipsoid 포함 비율은0.3068,
sigma RMS 거리 중앙값104.9721이다. 최근100은1장면·정지16·포함0.79·거리중앙값1.0276,
총114회추출/45행미추출이다. 포함 비율은 안전성·확률적 confidence·충분한 학습의 증명이 아니다.
정밀도 수정 후에도 일부 replay 행동과 현재 평균 사이의 거리가 크지만 단독 실패 원인은 미확정이다.

다음은 같은5,000모델의 **빠진4개발장면×2명령8회만 실행**해 기존기본4+정지4와 합쳐16쌍을
한 번씩 검토한다. 로컬 `outputs/sac_mean_basis_f64_full5000_20261120/`의 decision SHA
`e10dd446a0f2310379ea96d2eb9ec93df52384f97a5bb500902673895af48fe0`.
10:30:51 KST runner2707723으로 실제 시작했다. 남은 장면은57eb0840351f6cdc,
22a504c8e79dc7ba,83c142d159860844,fd56befbc8f7e7db다. 모델·manifest·36소스와 기존 증거를 고정하고,
새 경험 저장이나 학습 없이 평가한다. 이미 완료된8건을 중복 실행·분모 중복 계산하지 않는다.

보상/안전검사/PD/원래Gaussian/BC0/teacher0/gain/physicalQ/수집/uniform replay/손실/lr/tau 유지.
최근 replay 추출과 보조 평균 행동 수집은 아직 설계 검토이며 구현·학습 선택은 없다.
추가 학습 예산·다른seed·최종 Rule-based 비교·별도test·기본채택도 미실행이다.
**안정 정책 미확보, 전체 목표 진행 중.** 이번 확대는 기존 성공 회복 범위를 확인하기 위한 개발 평가다.

### 2026-11-20 10:38 KST — 전체 개발3/16 성공, 같은 f64 정책 총10,000회 선택

직전 기록 HEAD `6eb4fee9d7d4f561bc8844774f0b84875df281b2`, 실행 코드c4babca 유지.
남은8회 실행과 원본 검토가 모두 exit0으로 끝났고 runner2707723/child2707733·2709521의
실제 종료를 확인했다.36소스·모델·manifest·기존증거 hash 동일,8고유장면×2명령16쌍의
누락/중복 없음과 초기 frame/map/seed를 확인했다. 새8건+기존8건이며 기존8건을 재실행한 것이 아니다.
로컬 `outputs/sac_mean_basis_f64_full5000_20261120/full-development-review.json`
SHA `165e89505012eafb78034d811972bbc1dc3da3ed8478afdf2f5bfc5f1e656252`.

| 전체 개발16쌍 | 초기 mean0 참조 | f64 5,000 |
| --- | ---: | ---: |
| 목표 / 충돌 | 2 / 14 | **3 / 13** |
| KEEP 목표(8회) | 1 | 2 |
| FOLLOW 목표(8회) | 1 | 1 |
| tick | 886 | 969 |
| 경로 거절 | 356(40.18%) | 415(42.83%) |
| 제동 개입 | 334(37.70%) | 366(37.77%) |
| 정지 | 434(48.98%) | 415(42.83%) |
| 평균 reward return | −17.767363 | −16.534533 |

새 성공은 `83c142d159860844`의 KEEP 한 건이다. 같은 장면의 초기 mean0는71tick 충돌,
후보는110tick 목표 도달했다. 원본 frame에서 초기 정책은39거절→41제동→50비상제동→
61정지→71충돌, 후보는94거절/제동→108비상제동→110목표·정지0이었다.
같은 장면의 FOLLOW는58거절→60제동→67비상제동→78정지→90충돌로 여전히 실패했다.
초기 상태 이후 관측은 달라지므로 이 시간 순서를 단독 인과나 수치 정밀도만의 효과로 해석하지 않는다.
기존 두 성공은 유지했고 새 성공은1개 장면의1명령뿐이다. mean0는 Rule-based가 아니며,
default raw5,000 전체16쌍을 실행하지 않았으므로 수치 정밀도의 전체 성능 효과를 분리한 비교도 아니다.

중속4회는0→1성공, 저속2회와 정지4회는 모두 실패한다. 정지 출발328tick은 모두정지이며
거절80→159·제동76→122로 mean0보다 악화됐다. 전체13충돌은 정지 중 후방 접촉 분류이나
사고 책임을 뜻하지 않는다. 성공률18.75%·충돌률81.25%와 FOLLOW1/8을 안정 정책으로 판정하지 않는다.
개발8장면 반복 사용·단일seed, tracking RMS 미측정·독립test 미실행이라는 한계도 유지한다.

이전 기본4건에서는 새 성공을 발견하지 못했지만 전체16쌍에서 제한된 학습 개선이 확인됐다.
따라서 다른 추출/수집 요인을 바로 추가하기 전에 **동일 f64 learner를 총10,000회까지 이어서**
새 성공의 유지/퇴행과 실패 비용 적합을 확인한다.5,000 모델은 별도 보존한다.
full policy-decision SHA `91cc0f5cd34d8dc5029996cdafcaa8e5b258ab2b38242d765c6d12a9809a6b39`.

로컬 `outputs/sac_mean_basis_f64_budget10000_20261120/`에 새 decision/runner/reviewer를 준비했다.
decision SHA `22d5c8b5b0484ce7f99cabfaf78740f43b9a0f2f694cc0bb1214b143822b01ca`.
준비한 Python3개 AST와 실제 기존5,000의 `validate_source`/hash/기록·용량 검사를 통과했다.
10:37:28 KST runner2709963으로 시작했다. **핵심 학습 소스 수정0**이며40소스와 설정을 고정한다.
보상/검사/PD/BC0/teacher0/원래Gaussian/gain/physicalQ/quota수집/uniform replay/손실/lr/tau/
전이당update1 유지. 실제 신규 경험5,000개를 더 수집하며 총replay15,000은capacity50,000 안에서
wrap 없이 보존한다. collection RNG5000 재시작은 default raw10,000의 경계와 같고 정확한 환경resume는 아니다.

연결 순서: train10,000→저장/collection감사→기본4/정지4/나머지8과16쌍 검토→
고정Q/coverage/추출감사/실제branchQ→같은2train상태×4seed MC/가치비교→실제terminal 평균선호검토.
총16job이다. 전체 개발은 자체5,000/mean0와 초기 조건을 대조하고, 목표·충돌·거절·제동·정지·
보상을 함께 판단한다. 종료 후 검토 전까지 소스/설정/runner 변경·동시GPU실험 없음.
**안정 정책 미확보·전체 목표 진행 중.** 최근 replay·보조 평균 행동 수집·다른seed·기본 채택·
최종 Rule-based 비교·별도 정책test는 아직 선택/실행하지 않았다.

### 2026-11-20 10:42 KST — 성공 종료 직전의 제동 의존 확인

실행 코드c4babca·기록 직전 HEAD `6fddafa6d237960c4551b6742588b1f2065eb636` 유지.
현재10,000회 실험의40소스/설정은 변경하지 않았다. 완료된5,000회 모델의 실제 frame만 읽어
성공3건의 종료 직전 상태를 추가 확인했다. 실행 명령:
`python outputs/sac_mean_basis_f64_full5000_20261120/review_success_endings.py`, exit0.
로컬 결과 `success-endings-review.json` SHA
`c63213be13551d4e4ef50c095ae8bb4be52c00f55b772414b3a87f285cc05425`.
원본 evidence/hash를 검증하고 초기 mean0의 성공2건도 같은 방식으로 집계했다.

| f64 5,000 성공 episode | 종료 tick / 속도(m/s) | 종료 실행 | 종료 직전 연속 거절 | 마지막20tick 실행 |
| --- | --- | --- | ---: | --- |
| 고속 KEEP | 77 / 2.7064 | EMERGENCY | 21 | REUSED1·BRAKE5·EMERGENCY14 |
| 고속 FOLLOW | 72 / 11.0202 | BRAKE | 10 | NEW10·REUSED2·BRAKE8 |
| 중속 KEEP(새 성공) | 110 / 5.2282 | EMERGENCY | 15 | NEW4·BRAKE13·EMERGENCY3 |

**3건 모두 현재 ROUTE_GOAL 정의상 성공이지만, 종료 때 정책의 새 경로를 그대로 추종하는
상태는 아니었다.** 두 KEEP은 최종 실제 가속도−6m/s²의 비상제동 중 목표에 도달했다.
고속 KEEP의 마지막20tick은 모두 거절, 중속 KEEP은16/20거절이다. 중속의 거절 이유는
마지막20tick 기준 PREDICTED_COLLISION14·KINEMATIC_LIMIT2, OK4였다.
초기 mean0의 고속 두 성공은 모두72tick BRAKE·연속거절10·비상제동0이어서, 후보 고속KEEP의
성공 유지가 제동 품질 유지까지 뜻하지 않는다.

실제 충돌 event는 이 성공 episode들에 없었고, **기존 성공3/16·충돌13/16 집계는 유지한다.**
목표 도달 뒤를 실행하지 않았으므로 이후 충돌이나 필연적인 위험을 추정해 결과를 바꾸지 않는다.
마지막20tick은 추가 설명용 집계이며 새 gate/탈락 기준이 아니다. 정지/제동을 포함한 기존
다중 지표 원칙에 따라, 새 목표 도달 한 건만으로 안정 정책을 선언하지 않는 근거로 기록한다.

10,000회 종료 후 검사할 로컬 `finalize_review.py`도 준비했다.16job의 실제 종료·각 결과hash·
40소스·기존증거·replay/복원/추출·전체16쌍·terminal/MC 모델 일치 검사를 묶고,
성공/충돌 개수는 미리 가정하지 않는다. 소스 SHA
`e73364438bae988fb26e4df4afab68bffd37a4cb10e62af845f2897c6897cf6e`.
AST 파싱과 현재 RUNNING 중 검토 거절 확인을 실제 실행했다. 완료된10,000 산출물에 대한
실행은 아직 미실행이며 scientific-review도 생성하지 않았다. 자동 정책채택은 하지 않는다.

### 2026-11-20 10:59 KST — 현재5,000 모델과 과거 Rule-based 기준선 대조

기록 직전 HEAD `69b33bb37b2dbd9ac368e7a1c663d2dd200af553`, 실행 코드c4babca 유지.
학습 중 추가 시뮬레이션 없이 보존된 Frenet v4 기준선과 완료된5,000 모델의16쌍을 대조했다.
`python outputs/sac_mean_basis_f64_full5000_20261120/review_rule_based_reference.py` exit0.
로컬 결과 `rule-based-reference-review.json` SHA
`42a0150d0833cc8e3353c210c20b91a908e47ee35f70441db375f765a33af1c3`.

기준선은 `outputs/sac_stability_20261117/revalidated-final/baseline.json`, SHA
`9981e5dca007764cd7a09df41d28ca8b91c92516ac73b99c4940f1d4f18bcd8e`이며 새 실행이 아니다.
이전 provenance의 baseline revision `1d6ea21b66f2dd51bc0d48d54d0515f968cc2e75`부터
현재 작업 트리까지 src/configs/공통평가 실행부의 변경 경로는 learner.py/networks.py뿐이다.
Frenet 경로 실행에서 사용하는 환경·관측·보상·검사·PD·teacher·scenario·contract는 동일하다.
metadata 차이는 global code hash뿐이며 manifest/contract/dependencies/teacher가 일치한다.
16개 scene/provider·seed0·초기 속도도 일치하고, 기준선 episode 원본 요약으로 지표를 재집계했다.

| 개발16쌍 | 과거 Rule-based 참조 | f64 SAC5,000 |
| --- | ---: | ---: |
| 목표 / 충돌 / TASK_LOST | 7 / 8 / 1 | 3 / 13 / 0 |
| KEEP 목표(8회) | 4 | 2 |
| FOLLOW 목표(8회) | 3 | 1 |
| tick | 1,077 | 969 |
| 경로 거절 | 239(22.19%) | 415(42.83%) |
| 제동 | 223(20.71%) | 366(37.77%) |
| 정지 | 388(36.03%) | 415(42.83%) |
| 평균 reward return | −10.068949 | −16.534533 |

**현재 SAC는 초기 mean0보다 성공1건을 늘렸지만 과거 Rule-based 기준선에도 미달한다.**
SAC에서만 새로 성공한 scene/provider는 없고 Rule-based만 성공한 쌍은4건이다.
Rule-based의 TASK_LOST1건도 성공으로 세지 않는다. Rule-based 자체도8충돌이므로 안정성 인증은 아니다.

이는 소스/metadata가 맞는 **과거 참조 비교**다. 기준선 요약에는 전체16건의 초기 frame/map과
주행 원본이 없어 직접 같은 초기 frame을 대조하거나 궤적을 바이트 단위로 재현한 검증은 아니다.
새16회 Rule-based 재실행·최종 paired 비교·다른 SAC seed·독립test를 완료했다고 쓰지 않는다.
SAC 요약의 tracking RMS도 없으므로 기준선의 RMS와 비교 값을 만들지 않는다.
모든16개 쌍을 유지했고 새 유리한 장면을 고르지 않았다. 이 자료는 학습에 들어가지 않는다.
현재 총10,000회 실험의40소스·설정과 보상/검사/PD/순수SAC/수집/손실은 그대로다.

### 2026-11-20 11:48 KST — f64 10,000 학습·전체 개발 평가 완료,16건 모두 충돌

실행 코드c4babca, 기록 직전 HEAD `ee9516f6116206c8dbbfbfb61d09160c83187eff`,
브랜치 `feature/yuntae-sac-training`. 로컬 `outputs/sac_mean_basis_f64_budget10000_20261120/`의
학습·저장/audit·collection·기본4·정지4·나머지8·원본 검토까지9개 job이 exit0으로 완료됐다.
각 완료 child의 실제 종료,40소스·decision·모델·replay·초기 frame/map/seed의 동일 조건을 확인했다.
누적10,000 update와 신규 자기 경험10,000개, 총replay15,000개이며 BC/teacher0를 유지했다.
모델 SHA `1d809864850cb13ac1fab431b4f3b9270195327e479af985d8c96d78f94aa4e4`,
replay SHA `ce89018e323de0e26885c0375a20b8e8ed057731dc2e5a9f439012b2286434f3`.

이전10,000행17배열·tick/update prefix와 learner 복원은 바이트 동일했다.
collection RNG5000 재시작은 계획대로며 정확한 환경resume는 아니다.
추가5,000은36고유train장면의 main3,264/branch1,736, 분기 실제 종료336개다.
수락1,390/전이전정지2,521, 분기KEEP1,008/FOLLOW728이다.
누적10,000은53고유train장면·main6,767/branch3,233·분기 실제 종료624·414roots,
수락3,160/전이전정지5,258이다. 수집 분모와 고정 정책 평가 분모를 섞지 않는다.
audit SHA `3c996823944fef87780be4acd2a9f1140508d665351891520f6e062eef50e48e`,
collection 검토 SHA `69c9e5a7285ecc6dc46aacbac174714f58610959a7acf0f0567d4b11f5dabb73`.

| 전체 개발8장면×2명령16쌍 | f64 5,000 | f64 10,000 |
| --- | ---: | ---: |
| 목표 / 충돌 | 3 / 13 | **0 / 16** |
| KEEP / FOLLOW 목표(각8회) | 2 / 1 | 0 / 0 |
| tick | 969 | 729 |
| 거절 | 415(42.83%) | 670(91.91%) |
| 제동 | 366(37.77%) | 640(87.79%) |
| 정지 | 415(42.83%) | 459(62.96%) |
| 평균 reward return | −16.534533 | −20.513505 |

전체16건을 새로 실행하고 이전5,000/초기mean0의 동일16쌍과 대조했다. 이전 성공3건을 모두
잃었고 새 성공은 없다. 정지4건은328/328tick 거절·제동·정지, 최대 변위0.061mm 이하이며
모두 충돌했다. 기본4건도0성공/4충돌·118tick,거절94/제동82/정지30·평균return−20.275614다.
모든16충돌은 정지 중 후방 접촉으로 분류됐지만 사고 책임·다른 제어의 회피 가능성 증명은 아니다.
짧아진 episode를 개선으로 세지 않는다. 반복개발·단일seed·tracking RMS미측정 한계도 유지한다.
전체 검토 SHA `7ef7970d55d34b9b8292d6f17d1fee3409ebeb4cbb3ed15ae8aad0275e7a77ba`,
기본4 검토 SHA `99171fec1873a5ea5831c692fdcae38d123f37f497fd1b7a375b8abdd834f18e`,
정지4 검토 SHA `fbd6a596a88c112bcd5f771d0983ea3ef5111fe18c35385b90d2d4c356b023ac`.

추가 CPU 원본 비교 `python outputs/sac_mean_basis_f64_budget10000_20261120/review_regression10000.py`
도 exit0이다. 고속 두 명령의 실행 분류는post10/proposal9에서 기존NEW/OK와현재REUSED/
PREDICTED_OFFROAD로 갈라졌다. 현재post12제동→16비상제동→34정지/충돌(객체4877),
마지막25tick연속거절이다. 이전KEEP는77목표/FOLLOW는72목표였다. 나머지 기본 장면은
post6부터분류차이→8비상제동→12정지→25충돌(객체504)로,이전46충돌보다 앞당겨졌다.
동일초기상태 첫제안의최대점차는고속KEEP0.05439/FOLLOW0.05014m,
나머지KEEP0.05848/FOLLOW0.06012m이다. 이후관측은이미달라졌으므로이순서는단독인과증명이아니다.
퇴행 검토 SHA `5c5fe2808c3aa6ac90fd193acc539235d3ee1c5d51beb9d22cb2bec07ddc7db0`.

**10,000 모델 미채택·안정 정책 미확보.** 5,000 모델은 보존됐지만그모델도3/16성공으로미달이다.
이번 추가5,000은주행을악화시켰다. 이것을장기수렴불가능의증명으로확대하거나보상/검사를완화하지않는다.
11:47:33 runner2709963/q10000 child2732373실제생존. 고정Q→coverage/추출감사→실제branchQ→
2train상태×4확률seed실제미래보상/가치비교→terminal평균선호 검토가 남았다.
진행중40소스/설정은유지하며,전체진단검토전에추가예산·새추출/수집방식·다른seed·독립test를
선택하지않는다. 전체안정정책확보목표는계속진행중이다.

### 2026-11-20 12:00 KST — f64 10,000의 16개 job 검토 종료, 최근 replay 비교 선택

전체 job exit0·실제 PID 종료·동결40소스·결정/결과 hash를 확인한 뒤
`python outputs/sac_mean_basis_f64_budget10000_20261120/finalize_review.py`를 실행했다.
검토 SHA `61e3bbc1df326f1c02b36d7d9981555a61b09562306ed7614645a4aa2dbe55d8`,
미채택 결정 SHA `c3515d3bd1ea363be22e8ebd6431795a20df748ca1613a0613fde238502f73ac`.
이후 아래 진단 소스 수정에 착수했으며 실행 중 소스 변경은 없었다.

개발16건의 목표0·충돌16에 더해 같은 train2상태×4확률seed도8건 모두 충돌했다.
KEEP 초기 Q−실제 soft-return 평균은12.7803→13.3153, FOLLOW10.0737→11.1234로 악화됐다.
실제 평균 soft-return은KEEP−14.73184(표본std0.86295), FOLLOW−11.84528(std0.49462)이다.
고정 종료44개 Q MAE는1.57817→1.04347로 개선됐지만 실제 주행 개선으로 이어지지 않았다.
실제 같은 종료 전이를 만든 저장 행동과 현재 평균의 Q도 달랐다. 평균 Q 오차는 고정
KEEP5.31449/FOLLOW2.88967, branch KEEP1.89845/FOLLOW10.62790이다.
특히 branch FOLLOW는 실제 보상−20.017에 대해 저장 Q−19.83915, 평균 Q−9.38910이었다.

coverage의 신규10,000개 중 현재 조건부95% ellipsoid 안의 과거 행동은14.31%였다
(5,000 당시30.68%). Gaussian support/신뢰구간/성공률로 해석하지 않는다.
마지막100개는2장면·정지41개·총96회 추출·44개 미추출이었다. severe 종료4개 중3개는
9/8/2회의 추출 기회 동안 한 번도 선택되지 않았고 저장 Q 오차가5.22/5.19/5.06이었다.
다른1개는95기회 중1회,11update 지연 후 선택됐고 오차0.615였다.
CPU `review_recent_exposure.py`는15,000행 전체 추출 횟수와 저장 RNG를 재현해 exit0이다.
종료 직전 표본의 관측 기간이 짧다는 한계가 있으며 이것만으로 유일 원인을 확정하지 않는다.

다음 후보는 **최근 경험의 학습 반영 비중** 한 가지를 바꾼다. 같은 f64 fresh learner와
자기 warmup5,000에서 신규1,000개를 수집하며 전체128+최근256구간128 복원 추출을 사용한다.
기존 f64 균등1,000을 보존된 비교군으로 사용한다. 수집 Gaussian/실패 분기 규칙/보상/검사/PD/
SAC 손실식/lr/tau/전이당update1/BC0/teacher0는 유지한다. 배치의 상태·행동 가중치는 달라지며
균등 목적함수의 불편 추정이나 원 논문 ERE의 정확한 재현으로 주장하지 않는다.
기본 채택은 없고 정지 편중 증가와 기존 성공 유지도 검사한다. 상세 개정은 architecture 문서 참조.

구현은 진단용 sampler·CLI·continuation 설정/hash 검사·coverage 배치 hash/RNG 감사다.
기본 Replay/learner와 환경 코드는 수정하지 않았다. 관련 pytest31개 통과,
4개 Python 모듈 compile·CLI help·git diff --check 통과. 전체 회귀·새 GPU 학습 검증은
이 기록 시점에는 미실행이며 다음 제한 실험에서 확인한다. 기준 HEAD d42b2f2 이후 미커밋.
전체 안정 정책·다른 seed·최종 Rule-based 비교·독립 test 목표는 미완료다.


### 2026-11-20 12:10 KST — 최근 replay 1,000회 비교 실행과 초기 동등성 확인

실행 코드 `6455bae1312e28487bd79a27b9abeaa12e921b98` commit/push·원격SHA 확인.
로컬 `outputs/sac_recent_replay_20261120/`의 runner2739758을12:03:49KST에 시작했다.
12:10:35 train2739781 실제 생존, 기록된 update286/1,000.36개 소스·설정은 동결하며
기존10,000 실험 종료 뒤에 시작했고 동시에 GPU 실험을 실행하지 않았다.
결정 SHA `f108b179776ba6e6b9aff3719f3730a88bbeb402dc6606f087ee14435470c324`.

첫 학습 전의 두 실험 raw_u·XY 경로·anchor·시간·모델/분포 버전과 실제 검사/보상/종료는
정확히 같았다. observation_token/proposal_id는 독립 실행의 ID라 다르며 물리 값 비교에서
구분했다. 이후 상태/행동의 동일성까지 주장하지 않는다. 초기 검토 SHA
`aa974e2d896461207c36722455b3f56994bcbb757a00d8b9902c8e7e59abb1d1`.
첫100개 실제 update의 모든 배치 index hash를 재현했다. 총25,600번 중 warmup22,972/
신규2,628 추출이었다. 같은 삽입 index 순서의 균등 계산은25,314/286이다. 이는 CPU의
고정 index 반사실 비교이며 새 균등 학습 실행이나 같은 미래 경험의 증명이 아니다.
모든100개 learner state_finite=1. 주행 성능 및 최종 저장 RNG 검토는 아직 미완료다.

학습→replay 감사→같은 기본4/정지4→Q→coverage/배치hash/RNG→실제branchQ→
train2상태×4확률seed 미래보상→가치·terminal 평균선호의12job을 연결했다.
완료 후 `python -m outputs.sac_recent_replay_20261120.finalize_review`로 검토한다.
준비한 검토기의 AST·실행중 거절 guard 확인, 아직 결과 검토는 수행하지 않았다.
기본 채택·추가 학습 예산·새 seed·독립 test 자동 실행은 없다. 안정 정책 확보 목표는 진행중이다.

```bash
python -m json.tool outputs/sac_recent_replay_20261120/status.json
python -m json.tool outputs/sac_recent_replay_20261120/online1000/report.json
```

노션 첫 화면/02/03/04를12:05KST 갱신하고 다시 읽었다.03의54절은1회만 존재하고
과거53절과 결과가 보존돼 있으며,04의현재진행과 첫화면은 최근replay 비교로 일치한다.
노션 저장 확인은 실험 성능 검증을 대신하지 않는다.


### 2026-11-20 12:32 KST — 최근 replay 1,000 학습과8주행 완료, 기존 성공 상실

실행6455bae, 기록 직전 HEAD `2c596c0813379045ebf54f1663b620789206f552`.
로컬 `outputs/sac_recent_replay_20261120/` 학습·replay 감사·기본4/정지4와 원본 검토6job
exit0. 현재 고정Q 진단 중이며36소스/설정은 유지한다. 신규1,000+자기warmup5,000,
저장 모델 SHA `47192c03eaf2132c83de7951922923d690d6f976aa4c1f0d6bc0ef51dd1f1cf6`,
replay SHA `6b40d6adc25b042bb304b30a8ac0d79ca57905b66b9bf03d508e84bea9531dab`.

1,000개 전체 batch index hash와 최종 저장 RNG를 CPU로 재현했다. 총256,000추출 중
신규경험123,110(48.09%), 기존 균등22,639(8.84%)다. 신규 미추출2개(기존22개),
마지막100행 추출2,604(기존223)·미추출2(기존21)이다. 신규 실제 경험은9장면,
main712/branch288·수락477·전이전정지443·실제종료56이다. 기존은10장면,
main616/branch384·수락374·전이전정지478·실제종료83이다. 학습 후 경험 분포가 갈라졌으므로
수집 중 수락/종료 개수 차이를 같은 정책 평가 성공률로 해석하지 않는다.
`training-independent-review.json` SHA
`69f01df22ff19d1c65e03e1b3bdb46ed639d0ad757c856dbf1d714403879824f`.

| 기본 개발2장면×2명령 | 기존 f64 균등1,000 | 최근 replay1,000 |
| --- | ---: | ---: |
| 목표 / 충돌 | 2 / 2 | **0 / 4** |
| tick | 236 | 256 |
| 거절 | 44(18.64%) | 98(38.28%) |
| 제동 | 36(15.25%) | 90(35.16%) |
| 정지 | 12(5.08%) | 36(14.06%) |
| 평균 reward return | −9.981245 | −20.226825 |

고속 KEEP/FOLLOW의 두 성공을 모두 잃었다. CPU 원본 비교 exit0, 초기frame/map/seed동일.
post46/proposal45부터 기존NEW/OK와후보REUSED/PREDICTED_OFFROAD로 분류가 갈렸다.
이후post48제동→53비상제동→71정지→82후방접촉(객체4877),마지막37tick연속거절이다.
최초 물리제안의 최대점차는약1.092cm,post45이미ego위치차8.864cm로 이후 상태도 다르다.
이는 시간순서 설명이며 단독 인과·사고 책임의 증명은 아니다. 나머지 장면은 두 명령 모두
기존과 같은46tick충돌이고 실행 분류 차이는 없었다.
기본 검토 SHA `0495247b34b56395d9caf531af3d3b34974824452af499a211cf747ef2e6245d`,
퇴행 검토 SHA `0a2e4d59893f5ba9b6f647cd8a7c176c4666f5afd0d9613eda6eece397646870`.

정지4건도 모두충돌·328/328tick정지. 거절146→136,제동108→106은 줄었지만 출발 성공은 아니다.
최대변위7.343mm,최대속도0.002458m/s이하이며 기본+정지8건 모두 실패했다.
정지 검토 SHA `2c56eea748ba06dd935852c8562264061304a17c9f608e7c5a6422aafde33471`.

위치 추종 오차도 저장 실행경로의 첫 점과 실제post-step ego위치로 사후 복원했다.
`CapturingEnvironment._advance`의 첫점 시간=(pre_tick+1)×DT,
`tracking_error` 기준 시간=post_tick×DT이며 원본의tick연속성/유효위치를 검사했다.
기본 후보 RMS0.000410917m(176/256tick,EMERGENCY80제외),기존0.000372659m
(216/236tick,EMERGENCY20제외)다. 정지 후보0.0000383354m(250/328tick,78제외),
기존0.0000536834m(254/328tick,74제외)다. 최대 기본오차는둘다약0.002402m이다.
이는 기존 저장점에서 복원한 **위치오차만**이며 새 runtime측정이나 속도/방향RMS가 아니다.
짧은 매step재계획의 오차·제외된비상제동분모를 고려하며 작은RMS를 안전한경로/성공으로보지않는다.
새시뮬레이션/GPU작업/제어변경0, CPU 검토exit0.
추종 검토 SHA `8a881245398125152cc3bfcf9d6b57034f816acba011668a734e084bd714f730`.

**후보 미채택·안정 정책 미확보.** 경험 노출 증가는 달성했지만 주행 개선은 없었다.
Q·최종 정책의 조건부 경험 겹침·실제branchQ·미래보상·terminal평균선호 검토가 남았다.
전체12job검토 전 다음예산/수집방식/다른seed/독립test를 정하지 않는다.


### 2026-11-20 12:46 KST — 최근 replay 검토 종료, 구간별 Q 오차 추적으로 진행

12개 job exit0·실제 PID 종료·36소스와 모든 결과 hash를 확인한 최종검토 exit0.
`scientific-review.json` SHA `ac43885b418e558dc65c7d01ad1346a57e6c0eadddb925a93a3b850fcc042d87`,
미채택/추가예산없음 결정 SHA `e78d3afc9f33407a89b794f83aacc0895d21351bc091f3b27bb0b434fa0178c7`.
기본4/정지4와 실제train미래보상8회 모두충돌이다. 동일2train상태×4seed에서
Q−실제soft-return평균은KEEP13.03894→12.43961, FOLLOW10.47807→10.10064로 조금 감소했으나
여전히크다. 현재 실제평균 KEEP−13.23453(std2.00981),FOLLOW−10.69530(std0.72199).
KEEP soft-return증가0.27646 중0.25325는충돌보상의할인차이이며 비할인reward는0.00841악화했다.
FOLLOW 증가는0.06329,그중할인된종료보상차이0.04656이다. 성공으로해석하지않는다.

고정종료44개 Q MAE5.39711→4.78517,신규종료표본 Q MAE2.61044(82개)→0.97849(54개)다.
후자는같은상태집합이아니므로동일표본오차비교가아니다. 현재평균을실제로실행해같은충돌전이를
확인한Q오차는고정KEEP7.53772/FOLLOW4.14435,branch KEEP2.30574/FOLLOW1.46057이다.
branch FOLLOW의선택tick은기존64/현재68이라정확히같은상태의비교로쓰지않는다.

전체256,000실제batch추출에서정지비중은64.13→52.26%로줄었다. 그러나마지막100update에서는
64.07→69.59%다. 신규마지막100개경험은모두정지,최종정책의조건부95%ellipsoid포함률3%,
행동거리sigma RMS중앙값26.196이다. 전체신규1,000포함률46.2%(기존45.3%)와구분한다.
이는Gaussian support/신뢰구간/성공률이아니다. 전체평균·마지막구간·실제주행분모를섞지않는다.
CPU `review_draw_distribution.py`, `review_draw_timecourse.py` 모두exit0.

현재 수집은 충돌직전 상태만 쓰지 않고 앞선 수락상태도 일부 재실행한다.
CPU `review_probe_exposure.py`로feature/field mask/item mask를모두hash대조했다.
고정KEEP초기관측과정확히같은행은warmup1+online2=3개,실제추출137→354회다.
그상태현재평균Q는batch평가−0.794983인데실제미래보상평균−13.23453이다.
입력상태부재만으로설명되지않으며반복추출을다양한새행동경험으로세지않는다.
FOLLOW검사관측은두실험모두정확일치0이나인근상태경험부재의증명은아니다.
최초CPU분석은numpy int64 JSON직렬화오류로결과저장에실패했고,기존스크립트를보존한뒤
index를Python int로명시해재실행exit0. 학습/평가소스·가중치·replay변경은없었다.

다음은 **저장된 실제 주행의 어느구간에서Q낙관이누적되는지분리**하는읽기전용진단이다.
`diagnose_sac_bellman_trace.py`는같은2train상태·각4개기존주행의raw행동을그대로재실행하고
모든reward/검사/종료/next관측hash를검증한다. 새정책표본·학습·replay삽입은없다.
`sac_bellman_trace.py`는Q0−실제soft-return을할인된1-step잔차합으로정확히분해하고,
각잔차를target Q기준잔차와target/online Q차이로구분한다. terminal다음값과첫평균행동의
entropy항은없다. 같은raw후속표본에조건을둔계산이며기대Bellman오차나정확한Q정답이아니다.

target지연항이존재해도tau변경의성능효과를입증하는것은아니다. 진단뒤다음수집/학습개입을
선택하며단순배율·보상·안전검사·PD·원래Gaussian·SAC손실변경은하지않는다.
관련pytest8개통과(새분해7개+기존target검사1개),compile/CLI/diff검사통과.
이기록시점실제8경로재실행진단은미실행,기준HEAD ac68c73 이후새진단코드는미커밋이다.
안정정책·새seed·최종Rule-based비교·독립test는미완료로목표를계속진행한다.


### 2026-11-20 13:05 KST — 405전이 Q 추적 종료와 다음 수집 비교

실행코드0b72846. Bellman runner2760118/child2760119 exit0·실제종료를확인했다.
기존8경로405전이의raw행동/보상/검사/종료/다음관측을대조했으며새독립episode는아니다.
최종검토13:00:45 exit0, 로컬 `outputs/sac_bellman_trace_review_20261120/scientific-review.json`
SHA `e71187a47ab81b97c3ff61b16ed82af654b33bd0bbf19fb38c29d4d5b691ccdc`. 모델47192c03·replay·core hash보존.
KEEP 초기gap12.43964 중종료4.76747/비종료7.67217, FOLLOW10.10066 중종료1.63498/비종료8.46568.
target기준할인잔차합은각−19.07442/−25.39850, target/online차이항은31.51407/35.49916으로
서로크게상쇄된다. 지연항비율이나tau변경인과효과로해석하지않는다.
첫Q 재계산차이는최대2.873e−5/2.027e−5로사전1e−3허용범위다.

KEEP tick27에는같은scene/provider/tick의신규main1행·branch0행이145회추출됐다.
현재정책의다른실패지점31/34/60은신규대응행0이다. FOLLOW45/52/61도main1·branch0이다.
이는정확관측일치가아니며warmup제외다. 반복추출을새행동경험으로세지않는다.
다음은 **동일초기조건의완료경로수집1,000 비교**: 최근replay quota8후보와같은자기warmup5000,
seed0/f64 raw gain.0001/half_recent256를사용하고기존complete-path옵션(samples1/window6/
moving_predecessors/horizon200, restart없음)으로현재정책의실제종료까지새경험을수집한다.
과거다른모델/uniform의완료경로1,000은기본4건모두충돌해실패했음을보존한다.
이번비교는길이뿐아니라root다양성·종료반복노출도함께바뀌며성공을전제하지않는다.
collector변경을기존continuation에섞지않고fresh동일조건비교로한다. 보상/검사/PD/원래Gaussian/
SAC손실/lr/tau/BCteacher0유지. 추가예산·다른seed·독립test확대는결과검토전미허용.

별도로저장감사호출부가continuation의replay_sampling을전달하지않아최근replay를uniform으로
오인하는문제를CPUfixture로재현한뒤수정했다. 기존fresh1,000은continuation없어영향없다.
`scripts/audit_sac_branch_replay.py`와통합회귀test변경. 같은mode허용/metadata제거거절검증.
13:04 CPU pytest sampling/coverage/continuation/complete_path/reconstruction/rollout_budget
6파일 **46passed(0.50s)**. 실제최근replay continuation GPU검증은미실행이며실행했다고쓰지않는다.
안정정책은미확보이며목표를계속진행한다. 다음실험은이기록시점준비중이다.


### 2026-11-20 13:10 KST — 완료경로 비교 실제 시작·초기 재현 한계

실행코드a41deb74feba17ab9aaa92ac02af93e22d408172(원격확인).
로컬 `outputs/sac_recent_full_path_20261120/` runner2763938/child2763958을13:05:57시작했다.
학습1,000→저장감사→독립root/path검토→기본4→정지4→고정Q/coverage/branchQ→train미래보상8회
→value/종료평균선호검토의13job이며현재학습중이다. 첫완료branch는29실제step에서충돌했다.
39소스동결. 결정SHA c3b4525237262c5b6c996fdcc9d265af96907f4b2ee02cd8a2f96e0318d56ae5.
성공을전제한예산연장/평가확대없음. 성능결과는미완료다.

첫물리행동/실행결과·첫update지표는최근quota와정확히같다. 그러나초기29main step의전체동일성
검사는update2 xy_world에서실패했다. 실패를숨기거나사후허용오차로정확일치통과로바꾸지않는다.
CPU `review_initial_prefix.py`로실제차이를별도측정: XY최대1.126931e−6m,raw_u3.37730e−8,
anchor위치/속도/가속도차이0. 29step모든보상/검사이유/source/종료·batchindex는같다.
update지표최대차이는Q1loss0.00462694,Q1gradnorm0.04488003,actorloss0.00143145다.
첫collector개입이전부터수치차이가있으므로이후변화를수집방식만의정확한인과효과로쓰지않는다.
수치차이의원인은확정하지않았다. replay원본/가중치/소스는실행중수정하지않는다.

완료후기존Bellman피크의정확관측hash(features/fieldmask/itemmask)일치와scene/provider/tick
일치를구분해비교하는CPU검토기를준비했다. 같은tick의경험을동일상태경험으로세지않는다.
추가GPU작업은시작하지않았으며독립test/다른seed/최종Rule-based비교는미실행이다.
노션02/03/04/Home에는완료Bellman·감사오류수정46tests·현재실행을갱신하고다시읽어확인했다.
이초기수치차이후속기록도다음갱신에포함한다.


### 2026-11-20 13:29 KST — 완료경로1,000 수집·기본4주행 검토

로컬 `outputs/sac_recent_full_path_20261120/`, 실행코드a41deb7·기록기준66412dd.
학습1,000/저장감사/독립root선택검토/기본4평가/원시frame검토 exit0, 실제해당PID종료확인.
13:28:27현재runner2763938/정지평가child2772209생존, 전체13job은미완료다.
모델 SHA `2a912dedb2857d440ef274c320f6074e70c988e9c09c6810d57570d57e2053bc`,
replay6000 SHA `c0181207b4948d9a392f915edb9b8031c1da26a6de624f8f6640f49e9c8c0f62`.
자기warmup5000의17배열byte동일·teacher0·새1actual step=1update 확인.

신규main615/branch385,9고유train장면이다. 과거main결과만으로root선택을독립재구성했고
branch7개는29/31/47/64/86/55/73step 모두실제충돌종료했다. 가짜terminal없음.
기존최근quota는main712/branch288,9장면. 새경로수락368(기존477),사전정지482(443),
실제종료15(56),그중reward≤−10종료14(54)다. 학습경로가달라져동일정책성능비교는아니다.

CPU `review_training.py`는1,000개batch indexhash·최종replay RNG를재현하고모든update유한성확인.
256,000draw중신규123,110/초기132,890는동일하다. 실제충돌종료학습노출은전체8,846→3,265,
신규7,595→2,014로줄었다. 정지draw는133,789→140,956(52.26→55.06%)로증가했다.
두실험모두마지막신규100개는모두정지·총2,604draw·2개미추출이다. 긴경로수집이편중을
자동해결하거나종료학습을늘린것으로쓰지않는다. 검토SHA
`c7f9ba24f0d93b257388b63f6ef3ec517f0dc9a0d09543252e7d8d9e82f521b1`.
root검토SHA `b1e7d567e123ea89a121a6fa777a4fa89298e8ba3dd32ced5bfcbb25c309e357`.

기본2장면×KEEP/FOLLOW4평가에서기존최근quota0목표/4충돌→후보2목표/2충돌이다.
256→236tick,거절98→44(38.28→18.64%),제동90→36(35.16→15.25%),
정지36→12(14.06→5.08%),평균return−20.2268249→−9.9813130.
원시frame/map/seed의초기동일성4쌍확인. 성공2건은고속8b4c의KEEP/FOLLOW로,
post72목표종료당시둘다BRAKE/PREDICTED_COLLISION이며마지막5step도BRAKE다.
실패8a4f두건은46tick충돌,마지막EMERGENCY·정지중후방접촉이며귀책판정은아니다.
평가검토SHA `160028bbdaa3cd6cd90b547aa47aeabc9347bca3c25b446257a25f0af5ff5615`.

같은초기관측첫제안경로차이는기존최근quota대비최대약9.35mm다. 고속장면실행범주첫차이는
post46/proposal45에서기존OFFROAD거절/REUSED→후보OK/NEW다. 직전ego위치도이미약7.67cm
다르므로그제안하나의인과효과로단정하지않는다. 나머지장면실행범주이력은같다.
`regression-review.json` SHA `51b527b26f9d446c49160e3291f86f2d39dfd7ba12574c76976e7f1451378594`.

더이전f64 uniform1,000과별도원시frame대조도했다.4건모두성공/충돌·tick·거절/제동/정지수와
매tick실행범주가같다. 실제위치최대차이는4.70~5.51cm이며return은각4.64e−5~9.38e−5작다.
즉직전후보의퇴행은회복했지만이전최고기본4결과를넘은것은아니다. 반복개발평가이며새독립
증거로세지않는다. `historical-uniform-comparison.json` SHA
`624eccc8f9d781d5b802c6e7ea5feb62a27819fe8aa1a7b18464c03730981fb4`.

정지4/Q/coverage/실제branch평균/같은train미래보상8회/최종검토는진행중또는미실행이다.
기존Bellman피크에대한정확관측hash와scene/provider/tick경험대조,실제batch시간분포,
위치추종사후복원검토를준비했다. 현재실행소스/보상/검사/PD/순수SAC변경없음.
안정정책미확보·기본채택없음·추가예산/다른seed/독립test확대는전체검토뒤결정한다.


### 2026-11-20 13:39 KST — 완료경로 정지4·Q/경험 대조 완료, 미래보상 검토중

실행소스a41deb7 유지, 기록기준0fa0afd. 정지4평가/독립frame검토/Q/coverage/branch-Q
exit0·해당PID종료확인.13:38:33 runner2763938/soft-return child2778184생존.
전체13job과최종검토는아직미완료다. 원본가중치/replay/동결소스를수정하지않았다.

정지4는모두충돌·328/328tick정지다. 최근quota대비거절136→100(41.46→30.49%),
제동106→90(32.32→27.44%),평균return−20.5700693→−20.4798630이지만출발성공0이다.
최대변위0.0347206m·최대속도0.00929512m/s,초기상태/map/seed4쌍동일.
검토SHA `0a27c9a7e9f130b3b0b0a5d67716847cae56700a6c74fd51df0aed0404428300`.

CPU저장경로위치추종복원: 기본RMS0.000394632m(216/236tick,EMERGENCY20제외),
정지RMS0.0000632056m(254/328tick,74제외). 최근quota는각0.000410917m
(176/256,80제외)/0.0000383354m(250/328,78제외). 위치만의사후복원이며속도/방향오차나
새runtime측정이아니다. 제외분모·매step재계획을명시하고작은오차를안전성증명으로쓰지않는다.
SHA `5869d1840750d6795d807db68ad7ba9b1fb83222428a74ab5ca9a1be249dc5ef`.

추가CPU `review_stopped_proposals.py`는수락된NEW제안228개만원시좌표로대조했다.
pre-step ego기준첫0.1초점거리최대0.000891367m,3초마지막점거리최대0.0268366m다.
같은수락tick의실제가속도는최대0.00261189m/s²이고episode별중앙값약0.00124다.
거절100개는이기하통계에서제외했다. 점사이거리를순간목표속도/가속도로해석하지않으며,
이자료만으로PD의모든문제를배제하지않는다. 수락경로도매우작은이동을제안한관측이다.
SHA `780c820763fc6d0fe049ecba2ee8f8df92a7588e1d1cb6bd2e03ae81abb2df13`.

고정warmup충돌패널44개저장행동Q MAE는최근quota4.78517→7.19601로악화했다.
실제현재평균을재실행한고정종료Q오차는KEEP9.35310/FOLLOW8.60459로,
기존7.53772/4.14435보다크다. stored/current mean은각각같은실제종료보상·next관측을확인했다.
신규14개severe표본Q MAE5.01643(기존54개0.97849)은서로다른집합임을명시한다.
현재branch평균의실제종료Q오차는KEEPtick28에서10.70574,FOLLOWtick63에서7.88647이다.
기존quota의FOLLOWtick68과정확상태paired비교가아니다. stored와현재평균의보상/종료/next
관측이같아도현재평균을더좋게예측하는오류가남았다. 고정Q SHA
`1f6bdb06ea7299ac5d44e43727d8b413a7b207ad88777e7870800e638b36daa2`, branchQ SHA
`1d0edafc4c473a360054e329a61686c27ba3683c951b106309ce6949244bbd4b`.

coverage는batch256,000의모든indexhash/최종RNG를확인했다. 신규1,000의현재95%조건부
ellipsoid포함률44.6%(기존46.2%),sigma RMS거리중앙값1.51306(1.28893).
마지막신규100개는모두정지,포함률2%(3%),거리중앙값7.47523(26.1960)이다.
이는Gaussian support/신뢰도/주행성공이아니다. 마지막100update의정지draw67.2461%
(기존69.5938%),severe draw0.71094%(1.53906%). 전체구간과마지막구간을구분한다.

기존Bellman피크를사전고정하고완료replay와CPU대조했다. KEEP27의같은scene/provider/tick
신규행은1→2(main1+branch1),실제draw145→284. FOLLOW45/52/61도1→2,각286/257/268draw.
그러나features+fieldmask+itemmask정확관측일치는두실험모두0이다. KEEP34는coarse일치도0.
다른종료피크31/60도새일치가없다. 정확일치0을인근상태경험부재나유일원인의증명으로쓰지않는다.
긴수집이동일시각경험을늘렸지만현재실패행동을충분히배웠다는증거는아니다.
`failure-exposure-review.json` SHA
`b4dd95b665188868961887991da57587ee2c8db5efce2d408b05e3add12b060b`.

기본4의2목표회복과정지4의전부실패·Q악화를함께유지한다. 전체미래보상8회·value/종료선호·
최종hash검토후다음개입을결정한다. 지금시점새예산/새seed/독립test/기본채택없음,
안정정책미확보·전체목표계속진행이다.


### 2026-11-20 13:48 KST — 완료경로13job 최종 미달·고정정책 Q 진단 준비

전체13job exit0·실제runner/child종료·39소스/원본증거/결과hash확인. 최종검토의첫실행은
보고서dictionary key를tuple로잘못넣은KeyError로출력작성전실패했다. 실패한검토기를
`finalize_review-before-key-fix.py`로보존하고개별key를명시해수정한뒤재실행exit0.
학습/평가원본·동결소스·모델·replay는변경하지않았다. 최종검토SHA
`6424aa2a40778af87b478a1f306d9fc9f7e8bfeb091c61acdc843f421a304c0d`.
과거uniform/현재의8개원시frame에서도보고서횟수·reward합을독립재계산해확인했다.

같은2train초기상태×4seed의실제미래보상8회는모두충돌이다.
KEEP Q−실제soft-return평균12.43961→13.14753(현재Q−0.4295455,실제평균−13.57708,
표본std1.73639), FOLLOW10.10064→10.39273(현재Q−0.2958649,실제평균−10.68860,
std0.331456). KEEP실제return평균차이−0.342549,FOLLOW+0.006704.
작은표본의조건부정책return추정이며정확한Q정답이나새독립평가가아니다.
같은실제terminal전이를만드는평균을저장행동보다좋게평가하는오류도고정/branch4상태모두남았다.
기본4는과거수준2목표회복,정지4는전부실패,Q도악화해 **미채택·추가학습예산없음**이다.

다음은정책/alpha를고정한 **가치함수원인분리진단**이다. 현재모델의복사본에서
같은2개사전선정train root를번갈아현재원래Gaussian으로실제주행하고,새1,000전이마다
기존SAC수식의Q/target만1회보정한다. 자기replay6000/RNG를복사하고half_recent256유지.
actor/alpha갱신은보존하지않아진단중정책과비교할soft-return목표를고정한다.
보상/검사/PD/원래Gaussian/critic optimizer/lr/tau/BCteacher0 유지,가짜terminal없음.
원본모델/replay를수정하거나이복사본을정책후보/production checkpoint로채택하지않는다.

이진단은 **규약에맞는동시SAC학습후보가아니며학습규칙개정채택도아니다**.
정책이안바뀌어도Q가실제비용을배울수있는지점검한다. 개선만으로정책변동이유일원인이라고
결론내리지않고,1,000회내미개선도학습불가능의증명으로쓰지않는다. 새seed비교/전체개발/
독립test확대없음. 이기록시점구현/실행전이며결정은로컬 `policy-decision.json`에보존했다.
안정정책미확보·전체목표계속진행이다.


### 2026-11-20 13:52 KST — 고정정책 critic 진단 구현·CPU 검증

`scripts/sac_fixed_policy_critic.py`는원래canonical SAC update를복사learner에계산한뒤
actor/log_alpha/alpha optimizer를원래값으로보존한다. Q/target/critic optimizer/RNG는
canonical과같다. actor/temperature에대한지표는버린가상update값임을명시한다.
BC state/teacher표본은거절한다. 핵심회귀3tests CPU통과(14.20s): 두연속update의
Q/target/RNG byte동일·actor/alpha/그optimizer byte불변·원본불변·BC/teacher거절.

`scripts/diagnose_sac_fixed_policy_critic.py`는완료모델/자기replay6000과RNG를복사하고,
이전에선정된2train root를원래main raw행동prefix로복원해관측전체hash를대조한다.
root를번갈아원래Gaussian으로새실제전이를수집한다(첫행동도확률표본).
원본정책/alpha를고정하고새전이당Q1회보정,100회마다고정된MC첫평균행동의Q/targetQ를측정.
MC평균4표본은정확정답이아니며첫평균재계산수치차이상한1e−9raw와첫Q1e−3허용을사전명시한다.

원본17배열prefix·모델/replay/source hash·actor/alpha byte불변,1step/실제종료bootstrap,
train root·branch부모불변을검증한다. actor보정0/temperature보정0을별도기록하며
산출물은 `critic-evaluation-state.msgpack`/`diagnostic-replay.npz`로명명해정책checkpoint와구분한다.
canonical update가계산한actor/temperature변경은폐기되므로동시SAC학습규약준수후보라고
분류하지않는다. 기본SAC코드·보상·검사·PD·분포·기존후보가중치는변경하지않았다.
compile/CLI/diff검사통과. 이시점실제새진단은미실행이며준비중이다.


### 2026-11-20 14:00 KST — 고정정책 진단 실제 시작·초기 Q 경과

실행코드1934f824dab377d02c06117b05f81e35580855e2.
로컬 `outputs/sac_fixed_policy_critic_20261120/` runner2782595/child2782596,
13:53:34시작·14:00:09실제생존/214전이·Q보정.43소스동결.
결정SHA `11f7ae7b9590252e0e4383211fa4554e913c1b1841665a768c7654df09c04b9a`.
두root관측hash복원과초기평균재계산/Q허용차검사가통과해실제수집이진행됐다.

고정MC평균대비초기Q gap은KEEP13.1475322/FOLLOW10.3927302,
100회13.0211678/10.3240451,200회13.1222454/10.3918679다.
200회의Q는−0.4548324/−0.2967272,targetQ−0.3868259/−0.3006059.
아직보정예산1,000중일부이며개선/실패원인을확정하지않는다. 정책을바꾸지않는진단이므로
주행정책개선이나안정주행을달성했다고쓰지않는다.

CPU에서원본state의actor/alpha와optimizer를Flax `to_bytes`로직렬화해
진단에기록한초기hash와각각일치함을독립확인했다. 원시msgpack재포장hash는Flax상태변환을
거치지않아다르며그차이를가중치변조로오해하지않는다.

`scripts/audit_sac_fixed_policy_critic.py`를준비했다. 완료후읽기전용으로원본/산출물
hash·actor/alpha정확보존·17배열prefix·새raw행동/reward/종료/관측연결·각Gaussian롤아웃
시작관측/연속성·모든batchindexhash/최종RNG·예산종료bootstrap·Q경과산술을검토한다.
Q parameter가안바뀐경우도무조건오류로분류하지않고실제변경여부를표시한다.
이검토는새sim재실행이나Q forward측정은아니다. compile/CLI/diff검사통과.
실제완료산출물검토와손상입력거절검증은아직미실행이며실행완료후수행한다.
실행중43소스·원본모델/replay·learner규약변경없음. 노션Home/02/03/04에완료경로최종미달과
고정정책진단을갱신·재독확인했고,이번초기경과도다음갱신에포함한다.

### 2026-11-20 14:13 KST — 실제 확률 행동의 가치 검토 준비

기준 HEAD는 `02d1ea9dcd14b403a4b791acf8efc643af3103e3`이며 이 절과 새 분석기는
작성 시점 미커밋이다. 기존 진단은 14:12:40에 777/1,000 실제 전이·Q 보정으로 실행 중이다.
700회에서 고정 평균 행동 Q−기존 MC 평균은 KEEP 13.0144232/FOLLOW 10.1910149다.
고정 정책의 평균 행동에 대한 오차만으로 실제 표본 행동의 학습 오차까지 단정하지 않는다.

`scripts/diagnose_sac_fixed_policy_returns.py`를 추가했다. 완료 산출물의 독립 감사를 먼저
통과한 뒤, 저장된 관측과 rollout seed에서 원래 GPU Gaussian 표본 및 ell을 다시 계산한다.
raw 행동 재현의 최대 허용차는 실행 전 1e−9로 정했다. 작은 covariance의 역산으로 epsilon을
추정하지 않는다. 실제 실행한 동일 행동에 대해 원본 critic과 Q 보정 후 critic/target을 측정하고,
실제 종료까지의 soft-return과 대조한다. Q 조건부 행동 자신의 entropy는 제외한다.
예산 중단 경로는 return 분석에서 제외하며 임의의 종료/추정 bootstrap을 넣지 않는다.
경로별 첫 확률 행동·종료·비종료 오차를 분리하고, 같은 경로의 suffix들을 독립 표본으로 세지 않는다.
학습 자료 적합 진단이며 독립 평가나 정확한 기대 Q, 주행 개선을 뜻하지 않는다.

compile/CLI/diff 검사와 기존 경로 분해 수학 회귀 `test_sac_bellman_trace.py` 7개를
CPU에서 실행해 통과했다(0.06s). 새 분석기의 실제 GPU 실행은 기존 진단 완료 후 수행한다.
원본 진단의 43개 동결 소스와 생산 학습 코드는 변경하지 않았다.

로컬 후속 검토기 `outputs/sac_fixed_policy_critic_20261120/review_trial.py`는 원래 child가
종료하고 완료 상태·소스 hash를 확인한 후 독립 감사, 복사본 손상 거절 검사 2건, 실제 행동
가치 분석을 순서대로 실행한다. 첫 대기 프로세스2789961은 결과/예외 기록 없이 사라진 것을
실제 PID 검사로 확인했다. 원인은 확정하지 않았다. `review-interrupted-status.json`과
`review-interrupted.log`에 보존하고, 14:10:06에 별도 session으로2790105를 재실행했다.
원래 수집/학습 프로세스2782595/2782596은 유지했고 재시작하지 않았다.

### 2026-11-20 14:22 KST — 고정 정책 Q 진단 완료·실제 표본 행동의 오차

1,000 실제 전이와 Q 보정이 14:17:47에 완료됐고 원래 runner/child는 실제 종료했다.
후속 검토도 14:18:47에 exit0으로 완료됐으며 reviewer2790105/분석2791849가 종료한 것을
확인했다. 43개 동결 소스·입력 hash, 원본 replay 17배열의 byte 승계, actor/alpha 및
그 optimizer의 불변, 전체 batch index hash·최종 RNG, 실제 전이 연결/종료를 검증했다.
복사본에서 raw 행동을 훼손한 경우와 파일 hash를 갱신하면서 actor를 훼손한 경우 모두
감사가 거절했다. 원본 데이터·정책·보상·검사·PD·BC/teacher0은 유지했다.

두 train root에서 22개 경로를 수집했다. 실제 종료한 21개(KEEP11/FOLLOW10)는 모두 충돌이며
947전이이다. 마지막 FOLLOW 53전이는 예산 중단으로 MC return에서 제외했다.
이는 같은 고정 정책의 새로운 확률 경험이며 새 정책의 주행 평가가 아니다.
256,000 batch 추출 중 승계 경험134,555/새 경험121,445, 새 종료21행의 추출2,702회,
새 비종료118,743회, 시작점22행의 추출2,723회다. 모든 새 종료행이 한 번 이상 추출됐다.
추출 수를 독립 경험 수 또는 충분한 학습의 증명으로 해석하지 않는다.

새 분석기는 원래 seed/관측에서 1,000개 raw Gaussian 행동을 GPU 재현했다.
최대 차이6.2088172e−11로 사전 허용1e−9 이내이며 완전 byte 일치라고 부르지 않는다.
원본/보정후 critic을 동일 실제 행동에서 측정했다. 다음 표의 MAE는 정확한 기대 Q가 아니라
각 경로에서 실제로 받은 soft-return과의 차이다. 같은 경로의 suffix는 서로 상관돼 있다.

| 실제 수집 경험의 표본 return 대비 Q MAE | 분모 | 보정 전 | Q 1,000회 후 |
| --- | --- | --- | --- |
| 완료 경로 전체 | 947전이 | 11.80247 | 8.92874 |
| 비종료 | 926전이 | 11.82950 | 9.06906 |
| 실제 충돌 종료 | 21전이 | 10.61039 | 2.74119 |
| 첫 확률 행동 | 21경로 | 12.34737 | 12.07088 |
| KEEP 첫 확률 행동 | 같은 root의11경로 | 13.94431 | 13.67464 |
| FOLLOW 첫 확률 행동 | 같은 root의10경로 | 10.59072 | 10.30675 |

모든 대상 전이에서 오차 부호는 과대평가였다. 첫 확률 행동의 실제 return 평균은
KEEP−14.410525(std1.474375)/FOLLOW−10.897767(std0.646759)이다. 첫 행동이 평균인 기존
각4표본 MC와는 조건/표본이 다르므로 하나의 paired cohort로 합치지 않는다.
별도로 고정 평균 행동 Q−기존 MC평균은 KEEP13.14753→12.86446,
FOLLOW10.39273→10.13155였다.

표본 경로의 망원합 분해에서 보정후 초기 gap의 종료/비종료 항 평균은
KEEP1.59511/12.07952, FOLLOW1.80142/8.50533이다. target residual/target lag의 합은
KEEP−2.29181+15.96645, FOLLOW−8.32044+18.62719로 같은 초기 gap과 일치했다.
부호가 상쇄되는 분해이므로 큰 lag항 하나로 tau가 원인이라고 확정하거나 tau를 바꾸지 않는다.
별도 CPU 검토는 947개 suffix return을 직접 할인합으로 다시 계산해 recurrence와 대조하고,
모든 집계·sampling RNG·소스 hash를 검증했다. 최초 결론의 정책 변동 원인 배제 표현은
짧은 진단의 범위를 넘으므로 원본 검토/코드를 `*-before-wording.*`에 보존한 뒤 정정했다.

판단: **현재 정책의 실제 실패 경험에서 종료 비용 학습은 개선됐으나, 시작점의 큰 과대평가는
남았다.** 평균 행동만의 측정 문제는 아니다. 고정 정책1,000회에서도 남았다는 사실만으로
이전 정책 변동을 주요 원인에서 배제하거나 학습 불가능이라고 결론내리지 않는다.
정책이 고정된 진단이므로 안정 정책 확보/성능 개선/학습 규칙 채택은 모두 아니다.

다음은 같은 실제 후속 관측에서 원래 Gaussian을 여러 번 뽑아 online/target Q 기대값을
추정하는 읽기 전용 검토다. 한 경로의 다음 행동 표본 잡음과 Bellman 적합/가치 전달 오차를
구분한 뒤 제한된 다음 개입을 선택한다. 추가 학습 예산이나 계수·보상·안전검사 변경은
아직 선택하지 않았다. 전체 개발/새 seed/공정 Rule-based/독립 test는 여전히 미완료다.

로컬 산출물 (`outputs/sac_fixed_policy_critic_20261120/`):

- 최종 검토 `scientific-review.json` SHA `f9a35aab24a7ca7629636316b4e9421193fa74145bae0a80970769c498c509ee`.
- 독립 감사 `independent-review.json` SHA `c6e32a0b798cb26125cad2f086f07059a7aa4d756b0261d7981df6824d20f415`.
- 실제 행동 가치 `actual-action-values.json` SHA `20dc04b8f91175b65389642fac25459de107e9f34c80068cff2447eeb8dcbda4`.
- 진단 state SHA `bf458c7e4240c20252ab417eb42f48677c0f5641c4738d24161b8034fbf107af`;
  replay7000 SHA `7350dfb447ca4adae14a6dcfe6f3692c50301933a9493a4a375ae6225a1f07e2`.

### 2026-11-20 14:28 KST — 기대 Bellman 검사 구현·배치 수치 검사 실패 보존·단일 추론 재실행

앞선 완료 진단·실제 행동 분석을 코드/기록 `0b6e5677953d265f96441628b4cee7111d7f7c9e`로
같은 브랜치에 push하고 원격 SHA를 확인했다. 노션 Home/02/03/04를 갱신·재독해 반영과
과거 결과 보존을 확인했다. 이 절과 새 분석기는 작성 시점 미커밋이다.

`scripts/diagnose_sac_expected_bellman.py`는 같은 실제 후속 관측에 원래 정책의 Gaussian
64표본을 적용해 online/target Q 기대값과 표본 표준오차를 측정하는 읽기 전용 진단이다.
전후 critic에 같은 표본을 쓰며 seed20261120과 replay 행 index로 key를 정한다.
완료 경로947전이 중 비종료926개만 다음 actor/Q를 평가하고, 종료21개는 보상만 사용한다.
실제 저장 전이/보상/관측/정책/alpha·optimizer·학습 상태는 변경하지 않는다.
표준오차는 Gaussian 수치 적분의 표본 변동이며 critic 정확도/독립 test 신뢰구간이 아니다.

최초64배치 추론은 기존 scalar 기록과의 사전 대조 허용1e−3을 위반해 exit1로 중단했다.
tail index7/보정후Q의 차이는0.0015034675598144531이다. 결과 JSON은 생성되지 않았다.
원본 코드를 로컬 `expected-bellman64-batch-failed.py`, 실패/허용차를
`expected-bellman64-batch-failure.json`, 로그를 `expected-bellman64.log`에 보존했다.
이 차이만으로 수치 오차의 내부 원인이나 큰 미래 비용 오차의 원인을 확정하지 않는다.

허용치를 늘리지 않고 매 표본을 기존과 같은 batch1 critic 추론으로 계산하도록 수정했다.
원래 Gaussian64표본과 비교 대상/seed/허용1e−3은 유지한다. compile/CLI/diff 통과.
14:25경 새 GPU 분석을 시작했고 14:28:06 child2795177 실제 생존·200/947 완료를 확인했다.
첫200전이에서 scalar 참조 대조와 종료/기대잔차 산술 검사를 통과했지만 전체 결과는 미완료다.
로그 `expected-bellman64-scalar.log`, 예정 결과 `expected-bellman64-scalar.json`이며
기존 진단/분석 코드와 데이터는 그대로 유지한다. 추가 학습이나 정책 성능 검증이 아니다.

다음은 완료 후 잔차·MC 표준오차·경로 내 위치별 오차와 실제 미래 보상을 대조하고,
기대 target이 큰 실패 비용을 얼마나 반영하는지 검토하는 것이다. 이때도 tau/lr나 보상/검사를
근거 없이 바꾸지 않는다. 안정 정책 미확보·전체 목표 계속 진행이다.

### 2026-11-20 14:36 KST — 기대 Bellman 검토 완료·동일 정책의 제한 연속 진단 결정

실행 코드 `f5452da420c7b974f1cd90154d2b732942bd29e3`를 같은 브랜치에 push하고 원격 SHA를
확인했다. scalar 분석은 exit0으로 947전이를 완료했다. 비종료926상태×64=59,264개 Gaussian
표본을 계산했으며 이것은 Q 추론 표본 수이지 새로운 실제 경험 수가 아니다.
기존 scalar 기록과의 최대 차이3.8146973e−6으로 사전 허용1e−3을 통과했다.
최초 batch64 실패를 보존했고 허용치를 완화하지 않았다. 원래 정책/학습/데이터 변경0.

별도 `review_expected_bellman.py`를 실제 실행해 입력/소스 hash, 이전 실제 행동·보상·종료,
기대잔차 산술·SE 계산·terminal 다음값0, 기존 path 잔차 보존과 집계를 검산했다.
최종947전이 중 최초21행은 같은 두 root의 첫 확률 행동들이다. Q 보정1,000회 후:

- 첫 행동에서 Q 대 추정 Bellman target MAE는0.0479527이다(전0.1355026).
- 그 target 대 실제 경로 soft-return의 평균 차이는+12.0543666이다(전+12.4351840).
  KEEP+13.6576727/FOLLOW+10.2907300, 첫 행동 target 추정의 MC 표준오차 평균0.00903363이다.
- 비종료926행에서는 target 잔차 MAE0.500296, target 대 실제 return 평균 차이+9.2815473이다.
  603행은 Q−target이 −max(2×MC SE,0.001)보다 작았다. 이는 개별 값의 제곱오차 방향을
  설명할 뿐 공유 신경망의 전체 parameter update나 actor 개선 방향을 결정하지 않는다.
- 첫 행동 target lag 평균0.0696566이며 현재 online Q로 한 단계 계산해도 실제 return과의
  차이는+11.9847101이다. target/online 둘 다 낙관적이며 한 계수 변경으로 해결된다고 단정하지 않는다.

| 실제 충돌까지 남은 step | 전이 수 | Q−실제 return 평균: 전 | 후 |
| --- | --- | --- | --- |
| 종료0 | 21 | 10.61039 | 2.74119 |
| 1–5 | 105 | 11.89632 | 3.87270 |
| 6–15 | 210 | 11.32913 | 7.66116 |
| 16–30 | 297 | 12.38257 | 10.00821 |
| 31이상 | 314 | 11.61868 | 10.85997 |

**이 자료에서는 작은 한 단계 TD 오차와 큰 장기 가치 오차가 동시에 존재한다.** 종료 부근의
적합 개선이 먼 구간에서는 훨씬 작았다. 한 경로의 다음 행동 표본 잡음만으로 초기12점의
차이를 설명할 수 있다고 보지 않는다. 다만64표본은 정확한 적분이 아니고, 같은 두 train root와
상관된 suffix로 보편적 수렴 속도/유일 원인/일반화 성능을 확정하지 않는다.

다음 제한 진단을 선택했다: **같은 actor/alpha를 유지한 Q 진단을 누적5,000회까지**
이어가며 실제 자기 경험을 추가4,000개 수집한다. 종료 가까운 곳부터 개선되는 경과가 있어,
짧은1,000회만으로 가치 전달의 장기 실패를 판정하지 않기 위한 선택이다.5,000회면 충분하다는
보장이나 안정 정책 후보 채택이 아니다. 보상/안전검사/PD/원래 Gaussian/critic optimizer와
lr/tau/half_recent256/새 실제전이당Q1회/BCteacher0은 유지한다.

이는 동시 SAC 학습 규약의 후보가 아닌 **고정 정책 가치 원인 분리 진단**의 연속이다.
원본 모델·정책은 그대로 두고, 진단 state의 Q/target/critic optimizer/전체 RNG와 자기 replay7000을
이어받아 최종11000행을 만든다. 새로운 경로는 같은 두 root에서 시작하므로 simulator의 정확한
중간 resume로 부르지 않는다. 이전 예산 중단53전이의 bootstrap을 바꾸지 않고 승계 경계를
명시해야 한다. 같은 첫947전이의 실제 return과 고정 평균 행동 probe를 재사용해 학습 경과를
대조하되 이를 독립 평가라고 하지 않는다. 추가4,000 이후 예산은 자동 연장하지 않는다.

이 시점은 **후속 설계 선택**이며 continuation 구현·검증·실제 실행은 아직 안 했다.
기존 완료1000 자료의 불변, 총/신규 카운터·RNG·prefix·예산 경계·actor/alpha 불변을 검증하는
continuation 도구를 준비한 뒤 실행한다. stable 정책·전체 개발·새 seed·Rule-based·독립 test는 미완료다.

로컬 `outputs/sac_fixed_policy_critic_20261120/expected-bellman64-scalar.json` SHA
`8a1f4b5aad2c7fae8af4abf556051350efa8b6655d4a50aa7f854995c259342d`;
검토 `expected-bellman64-review.json` SHA
`1823cd275973128fa6cad892002b49c6625905df9ec0f7a473fe39d3a015c4e8`.

### 2026-11-20 14:46 KST — 고정 정책 continuation 구현·승계 경계 검증

기준 HEAD `cfe64a87b4e8ffd5069d3bc38c3a46e96bfb41d5`. 이 절과 다음 신규 파일은 작성 시점
미커밋이다. 기존 완료 진단·코어·동결 소스를 변경하지 않고 별도 진단 도구를 추가했다.

- `scripts/sac_fixed_policy_continuation.py`: 완료1000 진단만 허용하고 총1100/2000/5000,
  실제/보정 카운터·정책 고정 증거·replay 버전/RNG/cursor·no-wrap을 검증한다.
  마지막 실제 terminal과 예산 중단을 구분하며 예산 중단 bootstrap을 보존한다.
- `scripts/continue_sac_fixed_policy_critic.py`: 기존 state/optimizer/learner RNG와 replay7000/RNG를
  복원해 새 실제 전이당 canonical Q1회를 수행한다. actor/alpha는 그대로 유지한다.
  원래 tick/update1000개를 byte prefix로 보존하고 새 rollout22/seed20261142부터 수집한다.
  100회마다 고정 평균 Q와 정책 불변을 검사하고1000회마다 진단 state를 저장한다.
  simulator 중간 resume나 동시 SAC 학습 후보로 분류하지 않는다.
- `scripts/audit_sac_fixed_policy_continuation.py`: 완료 부모 감사와 함께 원본17배열/prefix,
  Q/actor/alpha/optimizer 카운터·snapshot·새 raw/reward/관측/종료·모든 새 batch/RNG와
  중간1000 경계의 bootstrap 보존을 읽기 전용으로 검토한다. 실제 sim 재실행은 아니다.

`tests/training/test_sac_fixed_policy_continuation.py`의17 CPU회귀 통과(0.02s): 경계보존·진짜
terminal 구분·seed 중복 방지·메타데이터 불변·예산/RNG/cursor/정책변경/가짜종료 거절.
compile/CLI/diff 통과. 실제 부모 데이터를 CPU에서 다시 감사하고 두 예산의 계획을 확인했다.
저장 state의 전체/Q update2000·actor update1000·BC0도 확인했다. 실제 GPU 복원은 아직 미실행.
로컬 `outputs/sac_fixed_policy_critic_continuation_20261120/preflight.json`에 보존했다.

긴 실행 전에 총1100(새100 실제 전이)의 별도 통합 시험을 한다. 이것은 저장 경계/복원/수집/감사
검증이며 정책 개선 평가가 아니다. 통과한 뒤 본 진단 총5000은 원래 완료1000 state/replay에서
시작해 새4000을 수집한다. 시험100개를 본 실행의 독립 경험으로 중복 합산하거나 그 결과를
원래1000에 덧붙여 본 실행을 시작했다고 기록하지 않는다. 이 시점 실제 시험/본 실행은 미실행이다.

### 2026-11-20 14:54 KST — 실제100전이 continuation 시험·동일947전이 가치 재현 완료

실행부 `1aa06bbb4eb86609dc3693540b7f847feb9c9746` push/원격 SHA 확인.
로컬 `outputs/sac_fixed_policy_critic_continuation_20261120/`의 pilot은14:47:00에
runner2800418/child2800419로 시작했고 수집100개/감사/후속 가치 분석·손상거절까지 완료했다.
후속검토2803939/분석2803940은14:52:30 완료했으며 관련 모든 PID가 종료한 것을 확인했다.
49개 동결 소스와 모든 입력 hash를 재확인했다. 결정 SHA
`91c7b98cb477603db5f9a44306825bf20a87657ceee1ef89cb263132bb724d28`.

GPU에서 원래 완료1000 진단의 learner 전체를 byte 동일하게 복원했고, 초기 KEEP/FOLLOW의
Q·target Q도 이전 마지막 probe와 정확히 일치했다. actor/alpha·그 optimizer와 원본 모델/replay는
바뀌지 않았다. 기존7000행의17배열 및 tick/update1000줄 prefix를 byte 동일하게 승계했다.
새100개 batch index hash/최종 RNG를 검산했다.25,600draw 중 승계22,974/새2,626이다.
예산 경계1000과1100의 bootstrap을 모두 보존했고, 건강한 임시 복사본의 감사도 통과했다.
이전 경계 bootstrap을 바꾸거나 신규 rollout에 이전 seed를 재사용한 복사본은 각각 거절됐다.

새 경로는 KEEP64step 실제충돌(수락20/return−20.5534115), FOLLOW36step 예산중단
(수락32/return+0.0459135)이다. 중간에 끊긴 FOLLOW를 성공이나 실제 완료 경로로 세지 않는다.
고정 평균 행동의 기존 MC 대비 gap은1100에서 KEEP12.8306785/FOLLOW10.0957099로 여전히 크다.

`scripts/evaluate_sac_fixed_critic_snapshots.py`를 추가했다. 진단의 저장 감사 후 actor/alpha 불변을
확인하고 **같은 기존 완료21경로947전이**의 관측/raw 행동/실제 return으로 모든 snapshot을
비교한다. 기존 예산중단53전이는 계속 제외한다. scalar critic을 사용해 처음0회/1000회 Q를
재현했고 최대차1.9073486e−6/3.8146973e−6≤사전1e−3이었다. 원래 분포의 재표본화나 새 실제
주행을 수행한 것이 아니다. compile/CLI/diff 및 실제GPU3snapshot 평가·CPU 집계 검산 통과.

| 같은947전이의 실제 return 대비 Q MAE | 1000 | pilot1100 |
| --- | --- | --- |
| 전체947 | 8.9287355 | 8.5220849 |
| 비종료926 | 9.0690577 | 8.6782902 |
| 종료21 | 2.7411927 | 1.6341768 |
| 첫 확률 행동21 | 12.0708817 | 12.0188560 |

종료 부근 적합은 더 개선됐지만 초기 오차는 크다. 수집/복원/감사 통과와 주행 정책 확보를
구분한다. 이 시험은 actor가 고정된 가치 진단이며 안정 정책이나 동시 SAC 후보의 채택이 아니다.
기존 보상/안전검사/PD/원래 Gaussian/BCteacher0·원본 learner는 유지했다.

최종 pilot 검토 SHA `0645b23aba8f25e3c38d1cea5c9aa792126bd9525c4ab4fe4e7774eee07041bf`;
저장감사 SHA `975a3b33d1908d62ebe150212d5db31a1ada7be654d7d32b66440939b97a947e`;
동일cohort 가치 SHA `1f1151bb0bf11dd6db00a3ef523238aec073f0fcbc1d837ca1bd442c41104057`.
새 evaluator는 첫 실행 당시 미커밋 source SHA
`d536868527b75003c9b8cb2679756df228889cb05dfd5b0ec2fb29f47e98d150`였으며 이번 기록과 함께 커밋한다.

본 진단은 원래 완료1000을 부모로 새4000을 수집해 총5000으로 진행한다. pilot1100을 부모로
사용하거나 두 실행의 겹치는 시험 경험을 독립 자료로 합산하지 않는다. 본 실행은 이 기록 시점
준비 중이며 실제 PID·시각·진행은 시작 후 별도 기록한다. 동일947전이의0/1000/2000/3000/4000/5000
snapshot 가치와 고정 평균 probe를 검토하도록 연결한다. 성능평가 확대/정책 채택/추가예산 자동 연장은 없다.

### 2026-11-20 14:55 KST — 본 고정 정책 진단 누적5000 실제 시작

실행 코드 `c5863d0a8431b4aab3c03b3c49d7c7ca75045684` push/원격 SHA 확인.
14:54:42에 main runner2805374/critic child2805375, 후속검토2805376을 시작했다.
14:55:32에 세 PID의 실제 생존을 확인했다.51개 소스 동결·원본 완료1000 parent 사용,
예산은 새4000 실제 전이/Q 보정으로 누적5000(최종replay11000)이다. pilot1100을 이어받지 않는다.
결정 SHA `6cca338cd1630b7c08c3d906160d3cf61ac934b2d32fbc349468ea1ae50bd30b`.

이 시점 GPU learner 전체 byte 복원·두 root의 관측 복원은 통과했고 신규 전이0/4000으로
초기 update 준비 중이다. 초기 Q/target 재계산의 이전 마지막 probe 대비 최대 차이는
1.6212463e−5로 사전 허용1e−3 이내다. pilot의 초기 Q는 정확히 같았지만 이번 main을
그와 같은 수치 동일 실행으로 표현하지 않는다. 파라미터 직렬화 불변과 추론 수치 차이를 구분한다.

본 수집→저장 감사→동일947전이의0/1000/2000/3000/4000/5000 가치 비교→손상 입력 거절과
집계 검산을 연결했다. actor/alpha 고정이므로 주행 성능 개선 평가가 아니다. 보상/검사/PD/
원래Gaussian/critic lr/tau/half_recent256/BCteacher0 유지·production 정책 checkpoint 없음.
안정 정책/전체 개발/다른 seed/공정 Rule-based/독립 test 목표는 미완료이며 계속 진행한다.

상태 확인(로컬):

```bash
python -m json.tool outputs/sac_fixed_policy_critic_continuation_20261120/main-status.json
python -m json.tool outputs/sac_fixed_policy_critic_continuation_20261120/critic5000/report.json
python -m json.tool outputs/sac_fixed_policy_critic_continuation_20261120/main-post-status.json
```

### 2026-11-20 14:59 KST — pilot/main의 첫100전이 수치 차이 공개

14:59:05 본 실행은누적1117/5000으로 진행 중이었다. 완전히 저장된 첫100개 신규 전이와
update만 읽어 pilot과 대조했다. 수집ID/rollout/seed/step·검사결과/실행출처/종료 분류는
모두 같고100개 batch index hash도 같다. 하지만 tick/update byte prefix는 동일하지 않다.
최대 raw 행동 차이9.0028156e−10, 보상 차이9.5166930e−15였다. 이 비교에는 사후 PASS
허용치를 새로 만들지 않았으며 원본 실행이나 수집 규칙을 수정하지 않았다.

처음 update1001의 Q1 loss는pilot0.8415557/main0.8406311, Q1 grad norm82.7929271/82.3923747이다.
첫 batch에는 신규행이1회 포함돼 입력 전체가 byte 동일하다고 단정할 수 없다.100회 동안
Q1 loss 차이 최대1.4314133, Q1/Q2 grad norm 차이 최대189.12005/218.25561로 커졌다.
1100의 고정 평균 Q는pilot KEEP−0.7463992/FOLLOW−0.5928853,
main−0.7308869/−0.5787887이다. 입력·추론·학습 수치 차이의 내부 원인은 분리하지 않았다.

원래1000 state/replay의 byte 복원·정책/alpha 고정은 유지하지만 **정확한 결정적 학습 재실행**으로
주장하지 않는다. pilot1100의 수치나 경험을 main의 학습곡선/독립 데이터에 섞지 않고,
본 실행 자체의 snapshot과 실제 기록으로 누적5000 결과를 검토한다. 수집/학습 코드는 동결 유지.
로컬 비교 SHA `0d14a4124c6bccd1ec03402243df9a5c556f29e5c73187b906cb95614d2970c3`
(`main-first100-pilot-comparison.json`), 첫 추출 검토 SHA
`a3536d7fbe83cc7035f0586e36e6a04df364addc4bce9aac0f0c857c43a01291`
(`first-update-input-exposure.json`)이며 위 continuation 출력 폴더에 있다.

### 2026-11-20 15:06 KST — 정확한 한 단계 적합에서도 생기는 종료 비용 전달 지연의 보조 계산

실제 main은15:05:42 누적1398/5000(새398/4000)으로 실행 중이며 critic2805375 실제 생존을
확인했다. 본 실행의51개 동결 소스·규약·예산은 변경하지 않았다. 기다리는 동안 GPU를 추가로
사용하지 않고, 현재 learner의 gamma0.99/Polyak tau0.005를 코드 문자열·SHA와 대조한
**별도의 수학 모형**을 CPU에서 계산했다. 실제 Waymax 학습량 예측이나 계수 변경 실험이 아니다.

가정: 결정적인 단일 경로, 각 상태의 값을 독립적으로 표현, online/target 모두0 초기화,
종료 보상−20만 존재, 중간 보상/entropy 없음. 매 iteration마다 모든 상태를 직전 target의
한 단계 Bellman 값에 정확하게 적합한 뒤 원래 Polyak 식을 적용한다. 실제 replay 추출 부족,
유한 optimizer 보정, 신경망의 특징 공유, 확률 주행, 이전 학습 이력은 없는 모형이다.

종료까지 남은 step을 d, 보정 횟수를 k, terminal 보상을 r이라고 하면:

```text
Q_k(0) = r
Q_k(d) = gamma * T_(k-1)(d-1)       (d >= 1)
T_k    = (1 - tau) * T_(k-1) + tau * Q_k
Q_k(d) / (gamma**d * r) = Pr[Binomial(k-1, tau) >= d]    (k >= 1)
```

정규화한 target은 `Pr[Binomial(k,tau) >= d+1]`이며 이항분포 재귀와 같은 관계를 만족한다.
직접 배열 recurrence와 독립적인 binomial 확률 합을63개(k,d)에서 대조해 통과했다
(상대2e−10/절대1e−300). 최초 terminal·1/2step 전달과 tau0/1의 경계 사례도 별도 검산했다.
이 모형에서의 값 전달 비율과50% 도달 iteration은 다음과 같다(실제 학습 횟수가 아님).

| 종료까지 d | 1000 iteration의 전달 비율 | 5000의 전달 비율 | 50% 도달 |
| --- | --- | --- | --- |
| 5 | 0.559067 | 0.999999744 | 935 |
| 15 | 0.000213544 | 0.987731 | 2935 |
| 28 | 7.49247e−13 | 0.299104 | 5535 |
| 51 | 7.05869e−34 | 3.11510e−6 | 10135 |

의미: 당시 한 단계 target을 잘 맞춘다는 사실만으로 장기 비용까지 정확하게 학습했다고
볼 수 없는 구체적인 예다. **실제 신경망의 수렴 하한/상한, 필요예산, tau가 유일 원인이라는
증거가 아니다.** 현재 모델은 이미 학습된 비영점 critic·공유 특징·mixed replay·확률 주행·
중간 보상/entropy를 사용하므로 이 모형보다 빠르거나 느릴 수 있다. 같은 두 train root에서
실제로 관찰한 종료 인접 구간 개선과 먼 구간 오차를 해석하는 보조 자료로만 사용한다.
실행 중5000 예산을 늘리거나 다른 학습 규칙을 채택하는 근거로 쓰지 않는다.

로컬 `outputs/sac_polyak_chain_20261120/`에 재현 코드 `analyze.py`, 결과 `report.json`,
과학 그림 `propagation.png`/`propagation.svg`를 저장하고 그림을 실제 열어 축·범례·가정
표기를 확인했다. 보고서 SHA `004d35001c27879f989dd9c09a3f074bebe54109e9b21c337497d1b039df99d1`,
산출물 hash 목록 SHA `26ecff5c6fcb00d71ee67ee130fa578a2bc8cf2adb5089ce4a452eb94040b938`.
계산 iteration30000을 실제 수집/학습 step에 합산하지 않는다. 안정 정책 목표는 미완료다.

### 2026-11-20 15:21 KST — 누적2000 중간 저장·실제 경험 노출 CPU 검토

15:20:49 CPU 검토 완료, 당시 report는누적2019로 실행 중이었다. 원래1000 parent에서
추가1000 실제전이/Q 보정을 마친 **2000 snapshot**을 대상으로 검토했으며 본5000 실행은
중단하지 않았다. 실행 기준c5863d0·문서 기준HEAD91cd6b3, 보상/검사/PD/Gaussian/순수SAC
데이터 조건과51개 동결소스를 유지했다. critic-only 진단이며 actor 학습/정책 후보가 아니다.

로컬 `outputs/sac_fixed_policy_critic_continuation_20261120/review_milestone.py --count 2000`
를its-merge Python으로 실행해exit0을 확인했다. CPU JAX만 사용했다. parent 대비 actor와
온도 및 각 optimizer 직렬화 불변, BC0, Q step2000→3000/actor step1000 유지,
learner RNG의1000회 split 재구성 일치를 검증했다. 상속 tick/update1000개 byte prefix와
새1000개 batch index hash, observation chain/종료/seed 규칙을 대조했다. snapshot SHA
`efa3bf76c71de1c79b9821c9c5a64955cf44c51f362dd782f97cab2d6a2b0ed3`.
최종 replay 배열 검토는 아직 저장 전이므로 예정된 종료 후 감사에서 수행한다.

추가1000 update의256000 추출 중 상속7000행135809회/새1000행120191회다.
새 terminal18행은2023회 추출됐다. 새행 중1행은 아직 미추출이며 이를 coverage 완료로
표현하지 않는다. 새전이259/1000 경로수락, NEW259/REUSED110/EMERGENCY518/BRAKE113.
검사 거절은 도로이탈238/충돌예측356/운동한계143/sweep미해결4이다.
step 전 anchor속도<0.1m/s인행381개·그행추출43398회이며, 이는 평가 규약의 **step 후
정지율이 아니다**. 위합계와 terminal outcome은 저장된1000개 원본에서 다시 집계했다.

같은 두 TRAIN root의 신규수집 중 완료18회는 KEEP9회 모두충돌, FOLLOW9회는2목표·7충돌이다.
추가 KEEP23tick은 snapshot시점 진행 중이며 종료/예산절단으로 바꾸지 않았다. 두 목표는
rollout23(seed20261143)114tick/103수락/return0.1898425/끝BRAKE,
rollout35(seed20261155)117tick/67수락/return−0.2405907/끝REUSED다.
**고정 actor의 다른 확률 표본에서 나온 목표 도달이며 학습에 따른 정책 개선이 아니다.**
독립 장면18개나 validation 성공률로 확대하지 않는다. 초기947cohort가 모두충돌이었다는
관찰도 이 정책의 모든 확률 주행이 반드시충돌한다는 의미가 아니다. 완료 후 기존947의
동일행동 Q 비교와 전체 신규수집의 성공/실패 구성을 함께 검토해야 한다.

2000 고정 root 평균행동 Q는KEEP−2.3796086/FOLLOW−0.9721057,
기존 root당4개 mean-first 실제return 평균과의차이는11.1974691/9.7164894다.
이전1000의12.8644621/10.1315477보다줄었지만 여전히 크다. 작은 과거 확률 표본 평균은
정확한 기대Q가 아니며, root당 서로 다른 확률 첫행동의 신규 성공과 직접 짝지어진 값도 아니다.
동일947개 실제행동의2000 Q/return 추론은 아직미실행이며 본실행종료후 예정된GPU검토로
확인한다. 정책 안정성/전체개발/다른seed/공정Rule-based/독립test 목표는 미완료다.

원본은로컬 `milestone2000-evidence/`, 검토는 `milestone2000-review.json`이며 위출력폴더내에
있다. 검토SHA `8d2efaa72eed969d2f545d5ef6ead9fe9c8efe9ec9b88d530a5ba79fb67251c6`,
검토코드SHA `dbdb1074631c378e8c1c5c2c2283126cdce5804423e83d113fd4f0f22de03a1a`.
실제2000 snapshot을읽어완료한 검증이며 소스구문검사만통과한 결과가아니다.
본5000 main runner2805374/critic2805375/후속2805376을계속사용하며 재실행하지않는다.

### 2026-11-20 15:26 KST — 신규 목표 도달까지 포함하는 전체 경험 가치 검토 연결

직전2000 검토에서 고정 actor의 새로운 확률 표본 중 FOLLOW 목표2건이 관찰됐다.
기존947개 완료전이 비교는 동일행동의 학습 경과를 보는 데 유지하되, 그 cohort가 모두충돌한
결과만 전체수집을 대표하게 하지 않도록 **전체 완료경로의 읽기 전용 가치 검토**를 추가했다.
보상/검사/PD/정책/수집/Q 갱신과 기존51개 동결소스는 바꾸지 않았다.

`scripts/diagnose_sac_continuation_returns.py`는 기존6000행 이후의실제5000전이를 읽고,
완료경로 전부에 대해 원래 Q와 최종 Q를 같은 실제 행동/return과 대조한다. 원래1000과
신규4000, KEEP/FOLLOW, terminal outcome, 종료까지거리별로 분리한다. 기존53개 budget
절단과 최종미완료경로는 실제미래return에서제외하며 가짜종료를만들지않는다. actor/alpha 및
optimizer 불변·전체 state byte복원·BC0을검증하고 원래Gaussian을기록된seed로재현한다.
이미사용한 행동1e−9/Q1e−3/동일return1e−10 허용치를 유지해 이전1000/947 자료와 대조한다.
초기947의 결과나 새 성공에 맞춰 허용치를 넓히지 않는다.

구문검사와 `--help` 실행은통과했다. CPU환경에서 아직RUNNING인 본자료를입력한실행은
`audit_sac_fixed_policy_continuation.py:28`의 완료상태검사에서예상대로exit1, 결과파일없음을
확인했다. 이음성검사를 GPU값검증 통과라고하지않는다. 실제5000완료자료의추론·return
검산은 **미실행/대기**다. guard기록SHA
`858da0143e742d88cf257a3df67a8ef5223304de92f08af2085e830b098d873f`.

15:26:17 로컬후속runner `all_experience_review.py` PID2815256을기동했다.
15:26:23 본runner2805374/critic2805375/기존후속2805376/새후속2815256의실제생존확인,
본진단2274/5000·새후속 `WAITING_FOR_MAIN_POST_REVIEW`, 아직GPU child없음이다.
기존후속의성공완료와모든기존child/runner종료를확인한뒤에만새GPU추론을시작한다.
이후CPU에서모든완료suffix의조건부할인return을직접합산해검산하고,
원본tick/보상/종료/예산절단/cohort별집계를대조한다. 오류나수치허용치미달은기록후중단하며
자동재시작·추가학습·정책채택은없다. 이것은새경험5000개를추가수집하는작업이아니다.

새후속은기존51소스+새evaluator/runner총53소스를동결한다. evaluator SHA
`2fc1ad23e7d0c9fed160c963b646a8aa742e229cf70c3d474a652e33709db7f4`,
runner SHA `e7f4bcd4086b06bc8be0840d8ccc4f8e817d7353a8d952213fbef59d16169b2e`.
실행시기준HEAD1a1902c에서새evaluator는미커밋이었으며아래기록과함께동일내용을커밋한다.
로컬결과확인은동일continuation출력폴더의 `all-experience-status.json`,
`all-experience-values.log`, 완료후 `all-experience-values.json`/`all-experience-review.json`.
현재안정정책목표는미완료이며본5000진단과기존947비교를끝낸후전체경험검토까지함께해석한다.

### 2026-11-20 15:44 KST — 누적3000 저장본 검증과 명령별 Q 경과

15:28~15:43에는 본critic2805375와기존후속2805376/추가후속2815256의실제생존 및
진행카운터 증가를 확인하며 같은 실행을 기다렸다. 관찰시간 경과로 재시작하지 않았다.
15:43:42 `review_milestone.py --count 3000` CPU검토 exit0, 당시 본report3003/5000이다.
2000 검토와같은 소스SHA `dbdb1074631c378e8c1c5c2c2283126cdce5804423e83d113fd4f0f22de03a1a`
를사용했다. Q step은parent2000→4000, actor step1000 고정, BC0이다.
정책/온도/각optimizer byte불변·learner RNG2000회 split·상속1000로그 byteprefix·
신규2000개 batchhash/관측chain/종료/seed·본실행51소스불변을검증했다.
3000 snapshot SHA `523e55d05bf04754560bc0f151b31f84513919bea038699b512bd7923995a35c`.

부모1000 이후새2000전이/Q 갱신의512000배치추출 중 상속행241312/새행270688이다.
새 terminal39행은5102회추출, 새행1개는아직미추출이다. 경로수락435/2000,
NEW435/REUSED220/EMERGENCY1107/BRAKE238. 검사거절은도로이탈492/충돌예측759/
운동한계306/sweep8이다. step 전 anchor속도<0.1행811/그행추출109895이며
평가규약의step 후정지율로바꾸어쓰지않는다. 원본2000행에서수락·종료집계를다시확인했다.

신규완료39회는KEEP20충돌, FOLLOW3목표·16충돌이다. FOLLOW22tick은snapshot시점진행중으로
분리했다. 새세번째목표는rollout57/seed20261177,115tick/88수락/return0.0200786,
마지막실행REUSED이다. 앞선2목표의BRAKE/REUSED 종료도같다. 같은두TRAIN root의고정정책
확률표본이며 독립39장면·학습개선·validation성공률로주장하지않는다.

아래는본실행의고정root 평균행동Q와과거root당4개mean-first실제return 평균의차이다.
1000은부모완료시점값이고2000/3000은본실행값이다. 정확한기대Q나동일947행동의평균오차가아니다.

| frozen-Q 횟수 | KEEP Q | KEEP 기존표본과차이 | FOLLOW Q | FOLLOW 기존표본과차이 |
| --- | --- | --- | --- | --- |
| 1000 | −0.712616 | 12.864462 | −0.557047 | 10.131548 |
| 2000 | −2.379609 | 11.197469 | −0.972106 | 9.716489 |
| 3000 | −5.351509 | 8.225569 | −1.518630 | 9.169965 |

KEEP의초기값은더많이변했지만FOLLOW는차이가크게남으며매100회수치도단조개선하지않았다.
정책은고정돼있으므로Q하락을주행성능개선으로표현하지않는다. 동일947전이의구간별가치와
전체완료경로의성공/실패별가치는현재5000수집종료후GPU검토에서확인한다.
실행/검토구현기준318c9f4이며 보상/안전검사/PD/Gaussian/학습규칙변경없다.

로컬continuation출력폴더의 `milestone3000-evidence/`에고정prefix와report를보존했고,
`milestone3000-review.json` SHA는
`c70828701a49e8cc8b5018d254672845ece010246a38c886a476ed653b42adff`다.
본5000진단과두후속검토는계속실행/대기중이다. 안정정책/전체개발/다른seed/공정Rule-based/
독립test 목표는미완료이며누적3000을성공판정이나예산변경의근거로쓰지않는다.

### 2026-11-20 16:08 KST — 누적4000 저장본 검증, 초기 Q 개선과 주행 미개선 구분

16:07:14 동일 `review_milestone.py --count 4000` CPU검토 exit0,
당시report4019/5000. Q step은부모2000→5000, actor step1000 고정·BC0이다.
정책/온도/optimizer byte불변·learner RNG3000회 split·상속로그prefix·신규3000 batchhash/
관측chain/종료/seed·기존51소스불변 검증통과. snapshot SHA
`bc9109804697364a09311cd7b556061efa60d1d725dd7d291b70a9ffe877be11`.

새3000전이/Q갱신의768000추출 중 상속335961/새행432039, 새terminal60행8433추출이다.
미추출새행2개가남는다. 수락542/3000,NEW542/REUSED340/EMERGENCY1753/BRAKE365.
거절은도로이탈761/충돌예측1187/운동한계500/sweep10이다. step 전속도<0.1행1270,
그행추출180182이며 평가의step 후정지율과구분한다. 수락/종료합계는원본3000행에서재검산했다.

신규완료60회는KEEP30충돌, FOLLOW3목표·27충돌이다. KEEP19tick은snapshot시점진행중이다.
3000 점검이후새로완료된21회는모두충돌했고 기존목표3건은변하지않았다. 정책이동결돼있으므로
이차이를학습퇴행이나개선으로단정하지않으며 동일두TRAIN root의추가확률표본으로취급한다.
4000 root 평균행동Q는KEEP−10.0334597/FOLLOW−2.7008457, 기존root당4개 mean-first
실제return 평균과차이는3.5436181/7.9877494다. 앞선3000의8.2255691/9.1699648보다줄었으나
정확한기대Q나전체경로오차가아니다. 특히FOLLOW의큰차이가남아있다. 같은947전이및전체완료
경로의성공/실패별값은본5000종료후검토하며 Q하락을정책주행개선으로바꾸어쓰지않는다.

기준서v1.2 10절과현재 `rule-based-vs-pure-sac.md`·`learner.py`도읽기전용으로대조했다.
기본규약은매유효전이 Q→actor→alpha→target 갱신이다. 현재진단의actor/alpha 고정은별도
원인분리목적이며, 진단state를그대로정상SAC checkpoint로분류해이어쓰면규약차이를숨기게된다.
이범위구분은유지하며 이번에는학습규칙/기본채택을변경하지않았다. 처음검색한
`docs/architecture/sac-learning.md`는없어실제존재하는위설계문서로대조했다.

로컬continuation출력의 `milestone4000-evidence/`·`milestone4000-review.json`에보존했다.
검토SHA `9435cd744505cf9e548f5c4f4f5cad89c045f032332e551cd1701f522832ba9f`,
검토코드SHA는기존dbdb107과동일하다.16:07:37 본4029/5000·네PID 실제생존·전체53소스불변확인.
실행/검토코드318c9f4, 직전기록73535c1. 본수집→기존947비교→전체경험검토를계속진행한다.
안정정책/전체개발/다른seed/공정Rule-based/독립test는여전히미완료다.

### 2026-11-20 16:46 KST — 고정정책5000 완료·수치 재현 실패 보존·별도 정밀도 분석

본수집5000/신규4000과저장감사5000은각exit0으로완료했다. replay11000,
상속17배열·로그prefix·모든신규batchhash/RNG·actor/alpha/optimizer불변·예산경계1000/5000
bootstrap 보존·4개snapshot 검증통과. 전체105rollout은100충돌/3목표/2예산중단이며,
그중신규83rollout은79충돌/3목표/1예산중단이다. 완료경로는103개4920전이, 원래53전이와
마지막27전이는미완료로분리한다. 동일두TRAIN root의고정정책표본이며정책개선이아니다.
최종state SHA `d78cf59d476a69af90db2256e85b85fcb9eb745f3c7fb8abff04a78c58c5235a`,
replay SHA `b29b1aceea34b09b42f572d282f5c6b9b8bc7c87d935cdfbceb1a1d354a22bd8`,
감사SHA `ee31bead6344bbc0a51ab90ad92f5661ca524687231a2162ae36173649cde2db`.

그러나기존후속검토는원래0회모델의행index28에서기존Q/target과최대차0.0022287369가
사전허용0.001을넘어exit1로실패했다. `main-post-status.json`의FAILED와로그를보존했고,
전체경험후속도선행실패에따라추론전중단됐다. 원래runner/critic/audit/후속PID가모두없음을
확인했다. main을재수집/재학습하지않았으며 허용치를늘리지않았다. 실패기록
`main-reference-failure.json` SHA `9ad64fa087c82a5c9a048f703cdc5750fd58336445f1cb3af4f2ee0f226f0dd6`.

같은원본모델/입력을읽기전용으로10개→전체1000개대조했다. 명시적GPU matmul `default`는
기존기록대비최대1.90735e−6/허용초과0개였다. `highest`의GPU/CPU는서로최대1.1444092e−5
이내로일치했으나기존GPU기록대비최대0.0183172/0.0183086,각352행에서0.001초과였다.
이결과는계산설정에따른차이를보여주지만, 최초실패의개별kernel/컴파일원인은분리하지않았다.
새default 실행이맞았다는이유로원래실패를없애거나결정적재현이라고하지않는다.
`precision-probe1000.json` SHA `8a73851fcd7e13bdbada288a8ec1f0c4ffaaf1b299bb30575473a746f17c94e7`.

공식 [JAX precision 문서](https://docs.jax.dev/en/latest/201/precision.html)와
[config 문서](https://docs.jax.dev/en/latest/config_options.html)는32bit dot 연산의정밀도설정과
저장dtype를구분한다. [OpenXLA 결정성 문서](https://openxla.org/xla/determinism)는GPU
컴파일/autotuning의결정성을다룬다. 문서의일반설명을이번오류의유일원인증명으로쓰지않는다.
설치된JAX0.6.2에서실제실행해확인했으며학습의전역정밀도/가중치/정책은변경하지않았다.

#### 같은947경험의별도최고정밀도분석: 실제완료

`scripts/evaluate_sac_critic_precision.py`는critic추론에만명시적`highest`를사용하고
모든947행×6snapshot의Q/target을CPU와대조한다. 기존default참조와의차이는실패행수로
기록하며원래참조검사를통과했다고재분류하지않는다. 실제GPU/CPU분석exit0,
각snapshot 최대차5.72e−6~1.91e−5≤새분석에서정한CPU/GPU허용1e−3이다.
같은가중치·실제행동·이미확인한원래return을비교한수치재분석이며학습후보가아니다.

| frozen-Q 횟수 | 전체947 MAE | 첫확률행동21 MAE | 종료21 MAE |
| --- | --- | --- | --- |
| 0 | 11.803229 | 12.347266 | 10.618846 |
| 1000 | 8.926828 | 12.070882 | 2.734428 |
| 2000 | 6.127329 | 11.022301 | 0.240844 |
| 3000 | 4.783391 | 9.173580 | 0.223514 |
| 4000 | 3.265025 | 6.420791 | 0.340255 |
| 5000 | 2.516540 | 3.738918 | 0.395318 |

5000의첫행동MAE는KEEP11개0.691719/FOLLOW10개7.090837이다. FOLLOW종료까지31+tick
275행은평균+5.309597점과대평가가남는다. 반면KEEP362행평균부호오차는−1.475581로
과소평가도있다. 종료적합후먼상태값의개선은관찰됐으나정확한기대Q·독립평가·정책개선의
증명은아니다. 정책고정/수집분포/누적학습의효과를단일원인으로분리했다고하지않는다.
로컬 `precise-fixed-cohort-values.json` SHA
`96e5d7bb170fcd0283e0121e2d0ddcdee72c84259bc911cda6ebe60f5c47af2c`.

#### 전체경험의정확재현실패와구간분석준비

`scripts/diagnose_sac_precise_continuation_returns.py`의전체경험검토는actor/sampler의
원래정밀도를유지하고critic만최고정밀도로대조하도록구현했다. 하지만행index9의raw재현차이
1.3705964e−8>기존1e−9로exit1,결과파일없음이다. 이실패도보존하고허용치를늘리지않았다.
`precise-all-reference-failure.json` SHA
`dddca2ef044db77144972b5b248041b34190e59a2d202ad133ab808d070a50e5`.
따라서추가4000개의정확한ell/soft return을재구성했다고주장하지않는다.

대신기록된실제raw/reward/종료와rollout RNG,규약의sigma범위[0.005,0.03]로실제경로의
soft return **구간**을계산하는별도분석을준비했다. 평균mu/rho의재현이나작은공분산의역산을
사용하지않는다. 원래식 `ell=30−0.5*sum(epsilon^2)−30*sum(log(sigma/0.02))`에따라
`base−60*log(1.5) <= ell <= base−60*log(0.25)`이며계산여유1e−10을명시한다.
Q조건부현재행동의entropy는제외한다. 저장seed의epsilon제곱합GPU/CPU차이≤1e−12를
요구하며,원래1000ell과947완료return이구간에포함되는지도검사한다. **기대Q 신뢰구간이나
정확return이아니며,오차하한0이정확도검증을뜻하지않는다.** 원래재현gate의통과대체도아니다.

`scripts/sac_soft_return_bounds.py`, `scripts/evaluate_sac_bounded_returns.py`,
`tests/training/test_sac_soft_return_bounds.py` 추가. 실제sampler의sigma극값/비대칭축·현재entropy
제외·종료·직접할인합/구간폭·alpha0·잘못된alpha/경계/예산중단거절을CPU9테스트로검증했다
(0.98s). compile/CLI/diff통과.16:46:01 실제PID2836419가전체5000중1000추론까지진행중이다.
전체구간산출물과종합CPU검토는아직미완료이며실행중source는동결한다.
원래보상/안전검사/PD/학습/순수SAC조건과기존53소스는유지했다. 안정정책목표는미완료다.

### 2026-11-20 17:02 KST — 전체 실현 return 범위 검산 완료·정책 방향 진단 준비

앞 절의 별도 범위 분석은16:49까지5000행을 완료했고,16:51:50 별도 CPU 검산을 마쳤다.
`bounded-all-experience-values.json`의 완료103경로/4920전이는 같은TRAIN2roots의
100충돌·3목표다. 기존53+최종27 budget 행은bootstrap을유지해제외했다.

- 전체4920개 실현return 범위에 대한 Q 오차하한 평균10.9979→2.0638.
  종료103개는entropy항이없어정확한오차10.2963→0.4063이다.
- 첫KEEP52개 오차하한평균14.0023→0.4704, 첫FOLLOW51개9.7853→6.5807.
  최종FOLLOW 첫행동48/51은범위위이며남은초기낙관오차가크다.
- 목표에 도달한3경로346전이 중최종327개는범위아래다. 성공경로의오차하한평균은
  0.04494→1.87639로악화됐다. 실패과대평가감소만으로좋은정책을보장하지않는다.
- critic CPU/GPU최대차1.4305115e−5,같은highest947참조차0,
  epsilon제곱합CPU/GPU최대차4.26326e−14. 과거default 재현게이트와별개다.
- 범위는기록된실현경로의entropy불확실성을감싼것이지기대Q의신뢰구간이아니다.
  오차하한0이Q정확도를증명하지않으며동일학습자료검산을일반화평가로부르지않는다.

로컬 `outputs/sac_fixed_policy_critic_continuation_20261120/review_bounded_results.py`는
4920개모든suffix에대해재귀구현을사용하지않고직접할인합으로상·하한을검산했다.
6snapshot×947의Q/실현return집계와전수CPU대조,전체범위집계/행정체성,
건전한임시복사본continuation감사,상속budget bootstrap변조·신규seed재사용거절2건,
원래53소스불변을확인했다. 이전default검토와정확actor재구성실패는FAILED로보존했다.

| 로컬 산출물(위 continuation 디렉터리) | SHA256 |
|---|---|
| `bounded-all-experience-values.json` | `6ccd735fa320a4919f394458e09f6c8e4d394dfc23bed4edfe55cc539e5a6855` |
| `bounded-scientific-review.json` | `72625c8b0c83e104ee87b29f4f4d002011a009cf88e88e435850d241cf52904d` |
| `review_bounded_results.py` | `8af7ee785606c53e34bd73f689c4270318c27dd6027e2de93a88b0ef0867b24a` |

코드71ead6ce0fb2e88093824314021db2e0eef98c25 push·원격동일확인.
노션Home/02/03(67절)/04에완료·실패·범위한계·다음진단을반영하고4페이지재조회로확인했다.

다음은 `scripts/diagnose_sac_calibrated_actor_direction.py`의별도읽기전용진단이다.
아직실행결과/성능채택은없다. 보정전/5000보정후critic bundle(online/target/optimizer)을
동일actor/온도/optimizer·동일원본update RNG·동일최종replay batch256에대조한다.
한번의canonical update는메모리복사본에서만계산하고모델/replay파일을저장하지않는다.
이는새전이없는가상계산이며전이당1회인정규SAC학습으로재분류하지않는다.
critic-only warmup의기본학습편입/생산체크포인트채택은아니다.

같은TRAIN2roots에서원래평균·보정전Q의가상업데이트평균·보정후Q의가상업데이트평균을
비교한다. 각각첫행동만변경후원래고정확률정책으로종료까지이어가는4공통난수반복
(2×3×4=24주행),별도로각actor전체의결정적주행(2×3=6주행)을수행하도록구현했다.
전자는조건부Q의실현표본이며현재행동entropy를제외한다. 후자는다른정책의주행으로
동일Q의정답으로쓰지않는다. 시작상태지문·원본불변·batchhash/RNG·Q의CPU대조·
실제종료/보상/거절/제동/step후정지·원시경로를저장한다. source/보상/검사/PD변경없음.

py_compile/CLI/help·diff check와관련기존5pytest(13.97s)는통과했다.
새실제30주행은다음실행대상으로아직미실행이며,단일배치·동일2TRAINroots·4표본의
진단한계를유지한다. 안정정책→전체개발→다른seed→Rule-based·별도test목표는미완료다.

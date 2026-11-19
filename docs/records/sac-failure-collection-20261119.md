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

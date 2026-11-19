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

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

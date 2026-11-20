# 자기 성공 종료 경험을 유지하는 SAC 비교

2026-11-20 별도 비교 제안·구현. 기본 설정 채택이나 안정 정책 확보가 아니다.

## 관찰과 가설

고정 정책의 자기 실패 경험/Q 보정5000을 마쳤다. 같은 TRAIN2roots의 완료103경로는
100충돌·3목표였다. 충돌 종료의 적합은 개선됐지만 성공 경로의 가치까지 낮아졌다.
초기 자기 warmup5000 중 완료62경로/4998전이에 실제 목표 도달6경로/359전이가 있다.
원래 checkpoint와 현재 online1000의17개 replay 배열 prefix가 바이트 동일하고 teacher0다.
소스 runtime이 다른 재사용 warmup이라는 기존 비교의 한계는 그대로 남는다.

현재 online1000의 실제 batch256×1000 추출을 모든 index hash/RNG와 대조했다.
자기 성공 경로359전이는8470회(3.3086%), 성공 종료6행은155회(0.06055%) 사용됐다.
이들이 전혀 학습되지 않았다는 뜻은 아니다. 해당6개 실제 저장 종료 행동에서
CPU highest Q MAE는 보정 전0.487650→고정 정책5000보정 후1.890138로 악화됐다.
종료 target은 실제 reward라서 entropy/미래 가치 추정이 필요 없다.
단일사례에서는같은초기관측/두종류mask가바이트동일한30a56b458fc5b63d KEEP를
초기자기수집은목표도달,이후online1000수집은충돌로끝냈다. 난수·정책학습이달라
그한번의재방문만으로망각의유일원인이나현재정책성능을확정하지않는다.

로컬 근거는 `outputs/sac_calibrated_actor_direction_20261120/`의
`own-success-coverage.json`, `own-success-terminal-q.json`이며, 입력·검토기SHA를포함한다.
Rule-based teacher 자료를 사용한 결과가 아니다.

가설: 실제 실패 경로 수집을 계속하면서 자기 성공 종료 target도 배치에 일정량 유지하면
실패 비용 학습과 성공 가치 보존을 함께 개선할 수 있다. 현재는 미검증 가설이다.

## 규약 대조와 정확한 변경

- 기존 비교: 전체 replay에서128개 + 최근256행에서128개, 모두복원추출.
- 후보 `own_goal_terminal8`: 전체128 + 최근256행120 + 자기 목표 종료8개, 모두복원추출.
  목표 종료가 없으면 마지막8개는 전체 replay에서 추출한다. 총 batch256은 같다.
- 목표 종료 pool은 실제 저장 전이 중 bootstrap=false, training reward가[0.05,0.12]인행이다.
  기존 보상식의 목표 event+10,step clip[-5,2],합/100에서나온구간이다.
  다른 종료 event0/−5/−2000은이구간이될수없다. 감사기는원래warmup episode의
  ROUTE_GOAL과신규tick의실제terminal_reason을다시대조해수치분류를검증한다.
- reward·loss식/계수·안전검사·공통PD·Gaussian·mean gain/B·학습률/tau·batch256을유지한다.
  최초동일자기warmup5000을재사용한freshlearner seed0,이후새실제전이마다
  Q→actor→alpha→target을각1회수행한다. BC/teacher replay0,validation삽입0이다.
- 기준서v1.2 10절의batch256/replay50k/전이당update1을유지한다. 균등추출은명시되어있지
  않지만기존추출확률과actor의상태분포가달라지므로**별도비교실험**으로기록한다.
  importance weight를추가하지않고이전목적함수/분포와동일하다고주장하지않는다.
- 고정 정책 Q 진단5000의보정가중치를학습초기값으로쓰지않는다. 그가중치는정규SAC의
  동시업데이트스케줄을따르지않았으므로채택하지않는다. 새후보는원래fresh초기화다.

## 구현과 검증 상태

`sac_goal_retention_sampling.py`는명시적pool·fallback·범위제한·teacher거절을구현한다.
`diagnose_sac_goal_retention.py`는기존online driver를격리복사해sampler/비교scope만바꿨다.
기존실험의sourcehash와실행중인진단을보존하기위함이며생산학습기를변경하지않는다.
continuation은명시적으로거절한다. 신규runner의불필요한기존복구분기는실행되지않는다.
`audit_sac_goal_retention.py`는기존branch/실제전이감사에원래17배열prefix·목표종료정체성·
모든batch index hash/RNG·actor/Q/BC카운터검사를더한다.

CPU12pytest 통과(0.09s): 실제reward계약4종류·budget bootstrap제외·goal구간경계·
미래capacity행제외·최근구간·empty pool fallback·손상입력·teacher거절.
py_compile/CLI/help/diff검사통과. 실제학습·저장감사·주행비교는아직미실행이다.

## 제한 실험과 판정

1. 진행중인30회가상actor방향진단과독립검산을먼저마친다. GPU작업을겹치지않는다.
2. 기존half_recent256 대조군과새후보를각fresh100 실제전이로실행한다.
   동일초기warmup/learner seed0·mean/B/f64·physicalQ·complete-path수집을쓴다.
3. 각저장/추출검사후고정개발2장면×KEEP/FOLLOW4회와정지2장면×2행동4회를평가한다.
   primary IDs 8b4c1e7e09f5f4c3/8a4fba3463c6c62d,
   stopped IDs 8dd5939855c15b3f/a62f5a7d2cd9b3d9. 개발반복평가로독립test가아니다.
4. 자기성공종료6개와실패종료의Q오차·새경험통계,성공/충돌/거절/제동/정지/보상을
   함께검토한다. 목표종료6개를반복추출해도독립경험수가늘었다고쓰지않는다.
5. 새로운오류/회귀/성공망각을점검한뒤fresh1000 비교여부를결정한다.
   기본채택·확대평가·추가예산을자동승인하는gate가아니다.

안정정책→전체개발validation→다른seed→같은조건Rule-based·별도test는여전히미완료다.

## 17:25 실행 시작·17:31 중간 확인

구현 코드 `1c86e9f5eb1587cc86bfbd29dc2a7214766e1d47` push/원격 SHA 동일 확인.
17:25:37 KST `outputs/sac_own_goal_retention_20261120/run_pilot.py`가 실제 실행을 시작했다.
runner2847542, 첫control100 child2847555였다. 17:31 대조군100전이와저장감사(exit0)를
완료했고 후보child2851200은29/100전이 실행중이었다. 이것은중간관측이며최종성능결과가아니다.

`decision.json` SHA `243a29b46a8ab874c92dba7c593a046e3b93453644536daa46e3cd0bb024fcb4`에
동일조건·정확한8job명령과56개동결소스/입력해시를기록했다. 과거source53개와신규3개
sampler/driver/auditor를보존했다. `sources/`는로컬실행소스사본이다.
진행은`status.json`의jobs와각job별log, 각arm의`report.json`에서확인한다.

17:30:35 후속CPU검토runner2853566도시작했다. 두저장감사완료후
`review_training.py`가warmup17배열/모든추출hash/RNG와성공6·실패44·기타12의
동일실제종료행 Q를대조한다. 전체8job/실제runner종료후`review_pilot.py`가16주행의
원시frame에서지표를재계산하고두모델의초기frame/map/장면쌍을대조한다.
두검토기는compile만통과했고실제완료산출물검토는아직미실행이다.
상태는`reviews-status.json`이며실패하면기록하고추가예산을자동실행하지않는다.

노션Home/02/03의68절/04에이전30주행실패·자기성공가치악화·새구현/실행시작을반영했고
4페이지를재조회해핵심상태/수치/커밋링크와기존67절보존을확인했다.

## 17:49 pilot 완료: 성공 보존 신호와 남은 실패

두 fresh100 학습·저장 감사, 각8개 동일 개발쌍의 총16주행을 완료했다.
17:49:15 별도 검토기가 모든 원시 frame의 보상·거절·제동·step후 정지·종료를
재계산했고 초기 frame/map, 모델/manifest, 56개 동결 소스를 확인했다.
`pilot-review.json` SHA `855d21387748751c5f2f4914d225e17b507efa2da0ac9ec3f0e22b3849fe4bd5`.
학습/평가 runner2847542와 검토 runner2853566은 종료했다. 안정 정책 확보는 아니다.

| 평가 | 기존 recent | 자기 목표 종료 유지 |
| --- | --- | --- |
| primary 4회 | 목표0·충돌4 | 목표2·충돌2 |
| primary ticks | 224 | 218 |
| 거절/제동/정지 tick | 110/98/44 | 58/46/22 |
| 거절/제동/정지 비율 | 49.11/43.75/19.64% | 26.61/21.10/10.09% |
| 평균 training return | −20.265201 | −10.016100 |
| 정지 출발4회 | 충돌4,328/328 정지 | 충돌4,328/328 정지 |
| 정지 출발 거절/제동 | 80/76 | 80/76 |
| 정지 출발 평균 return | −20.429955 | −20.429955 |

고속 KEEP/FOLLOW의 성공2건은 초기 mean0도 성공했던 쌍이다. 새로운 주행 능력이 아니라
이 예산에서의 기존 성공 보존 신호다. 성공 종료 frame도 BRAKE/PREDICTED_COLLISION,
속도 약11.02m/s라서 무개입 안정 주행으로 부르지 않는다. 반복 개발4장면/8쌍·seed0이며
독립 test나 통계적 일반화 근거가 아니다. 정지 출발 문제는 전혀 개선되지 않았다.

고속 KEEP tick42: 실제 위치 차이0.015832m, 제안 경로 최대 차이0.073698m에서
기존은 PREDICTED_OFFROAD, 후보는 OK였다. 작은 차이가 검사 경계를 넘고 이후
제동/충돌 결과를 바꿨다. 단일 trace 비교로 유일 인과나 견고함을 주장하지 않는다.
`paired-divergence.json` SHA `809c6107af71d67e30dfc3df40f1b002aaec6be0663a9ff6dcfb54888fed0d53`.

같은 저장 종료62행의 CPU highest Q MAE는 기존/후보 순서로 전체13.083282/13.023559,
목표6행0.274441/0.240822, 실패44행18.309669/18.223490, 기타12행0.324288/0.348515다.
목표/실패의 소폭 개선과 기타 종료 악화를 함께 기록하며 실패 과대평가는 여전히 크다.
후보는 실제 자기 목표 종료6행을 총824회 추출했고 기존은24회다. 독립 경험수가
824개라는 뜻이 아니다. 동일 warmup17배열과 모든 batch index hash/RNG를 검산했다.
`training-review.json` SHA `81e1fcb27db545285905e312be0107968c1b25f9ea525b5806ce20b439eda359`.
첫 실제 신규 전이의 관측·행동·보상·next 관측·종료/mask도 바이트 동일함을 별도 확인했다.

### 실제 주행 GIF

위 pilot 로컬 폴더의 `visual/high-speed-keep.gif`와 `.png`는 동일 고속 KEEP의
기존 SAC/자기 목표 유지 SAC를 나란히 렌더링했다. Rule-based 비교 영상이 아니다.
실제 저장 trace만 사용했고 재시뮬레이션하지 않았다. 1440×900,39frame,
주행7.6초/재생9.1초(마지막1.5초 유지)다. 최종 PNG를 시각적으로 확인했다.
GIF SHA `8f56bb07fd9feddba6ee5d5d4e134f784bf62d406106956f3ae66f10f7381db8`.

원래 wrapper는 GIF/PNG 저장 성공 후 PIL.Image에 savefig를 호출해 exit1이었다.
실패 로그/원본을 보존하고 `finalize_visual.py`가 기존 결과 전체 frame·크기·duration·입력
hash를 검증했다. `visual/render.json`은 COMPLETE_VERIFIED_EXISTING_RENDER이며
원래 wrapper의 FAILED_AFTER_SUCCESSFUL_RENDER도 함께 남겼다. 실패를 삭제하거나
원래 실행을 성공으로 바꾸지 않았다.

## 17:50 fresh1,000 동예산 비교 시작

pilot를 검토한 뒤 두 방식 모두 동일 초기화에서 fresh1,000을 실행하기로 결정했다.
목표는 기존 성공 보존이 더 긴 학습에서도 유지되는지와 실패 가치 오차/정지 문제를 함께
확인하는 것이다. 기본 설정 채택, 안정성 승인, 추가 예산 자동 연장이 아니다.
과거1,000 결과만 재사용하지 않고 대조군도 재실행한다. pilot100을 이어받지 않으며
고정Q5000 가중치도 사용하지 않는다. 앞 절의 모든 조건과 reward/checker/PD를 유지한다.

로컬 `outputs/sac_own_goal_retention_1000_20261120/`, main runner2861692는
17:50:04 시작, review runner2863133은17:50:39 시작했다. 실제 학습 코드1c86e9f,
launch HEAD는문서커밋eda68fb다. 각학습/저장감사→개발4회/정지4회와 동일62행Q검토,
원시16trace 검토를 연결했다. 실행 중 source56개와 입력을 동결했다.
`decision.json` SHA `9124d279d282d1969ad7402e19fcb9eea6a1ad522a792112b79374b788dd285b`.
준비 중 절대경로 소스 복사에서 SameFileError가 났으나 launch 전이었으며, 원본은
바뀌지 않았다. workspace 상대 경로로 사본을 만들고 원본/사본56개 hash를 재확인했다.
현재 상태/자식PID/정확한 명령은 `status.json`, 검토 상태는 `reviews-status.json`이다.
1,000 결과·기본 채택·다른seed·독립test는 아직 미완료다.

## 17:54–17:56 정지 실행과 출발 경험의 추가 분리

pilot의 두모델×정지4쌍에서 수락된 NEW 경로496개를 읽기전용으로 재구성했다.
모두 pre-step 속도/직전 두 관측의 유한차분 가속도가 정확히0이었고,
3초 제안의 최대 변위는 기존0.242769~0.242956mm,후보0.399054~0.399196mm였다.
원래 `Trajectory`를 동일 anchor/raw XY에 적용하면1mm이내 terminal suffix 정렬이
30점을모두anchor로맞췄다. 재구성 실행XY는저장된실행XY와모든값이정확히같았으며
기록된실제가속도/step후속도도전부0이다. 거절160frame은이재구성에서제외했다.
검사/PD나정렬규칙을바꾸지않았고새시뮬레이션도하지않았다. 이는이496개수락경로의
정지 원인으로, 모든정책의출발불가능성이나Gaussian표본의성공확률은입증하지않는다.

로컬 pilot의 `review_stopped_alignment.py` → `stopped-alignment-review.json`,
SHA `343ae60414edbb68ac1697134615e5132d012f93b366f9791d87127dfff63050`.
저장관측/mask 기반출발경험점검 `review_launch_coverage.py`는같은own warmup5000중
속도0.1m/s미만3347전이에서다음속도0.1m/s이상인경우0개를확인했다.
해당다음속도최대0.001413m/s,속도증가최대0.0001878m/s다. 신규100의55저속전이도
두방식모두임계통과0개이며,유효next mask를모두확인했다.0.1은기존평가정지기준이지
새로운안전기준이아니다. 자기성공6경로359전이는전부9.4557~14.6712m/s구간으로
정지출발성공경험이아니다. 성공종료추출유지가출발경험부족까지해결한다고주장하지않는다.
`launch-coverage-review.json` SHA
`8f5a3c3003fcab0c3e5e72369707a5da298e232a33ae74ab400e13b380932a2d`.

후속fresh1,000은설정변경없이계속실행한다. 완료후같은정지/실제신규경험을대조해
제안이이정지구간을벗어나는지와안전검사/실제주행을함께본다. 현재학습결과는미완료다.
17:57 확인대조군307/1000,main2861692/child2861699,review2863133생존.
노션Home/03/04에는100회결과와1,000진행을반영하고3페이지재조회로확인했다.

## 18:00 추가 경험 분포 검토 연결

같은 완료pilot100의 실제 저장 전이와25600회 batch 추출을 새 읽기전용 검토기로
전수재현했다. 각index hash와최종RNG가일치했다. 기존/후보순 신규100전이의실제추출은
2628/2462회,그중이동전이추출1168/1087회였다. 전체저속draw는18087/17490회다.
성공종료강조가신규경험노출을일부줄이는비용도기록한다. 실제신규전이는양쪽100개로
같으며다중추출횟수를독립경험수로세지않는다. 동일관측속도/mask·main/branch·
실제제어와종료label을분리했다. `outputs/sac_own_goal_retention_20261120/experience-review.json`
SHA `9e06ef6105ba90306a9758ca8a71c1c1b08033320ae202364e60e016e08c9786`.

검토기 `outputs/sac_own_goal_retention_1000_20261120/review_experience.py`는
실제pilot100에서exit0이며SHA `a2907d4944e2175592928d624ffb76fab9ea36c2eb19ffebfddec1a35cea23b4`다.
이파일은로컬비추적검토산출물이며원래56개동결학습소스나기존runner를변경하지않았다.
18:00:49 별도runner2867314를실행했다. 기존두저장감사/종료Q검토가완료되면1,000의
실제속도별경험/출발임계통과/제어/추출분포를검토하고`experience-review.json`에저장한다.
진행상태는`experience-status.json`,실패하면FAILED로남기고재학습/예산연장을하지않는다.
1,000자료에서의검토는아직미실행이다. 이절은현재HEAD942c372 기준미커밋기록이다.

## 18:07 대조군1,000 완료·18:09 동일 조건 반복 차이

대조군1,000실제전이/전체업데이트와저장감사exit0,후보child2868850실행을확인했다.
대조군report SHA `2c49e9e3a8c8b441eeb2719b45aa7a1e8c26e2ab77f72047129b9198603de4a9`,
learner SHA `94d39067395e298b55c47d24cbd1eb32f1488bd2a5118b547f24165c8dbbd5f8`.
후보학습·양쪽주행·종료Q/경험분포비교는아직미완료다.

과거동일설정 `outputs/sac_recent_full_path_20261120/online1000`과새대조군을
읽기전용으로대조했다. 버전/seed/초기화/모델설정/수집설정/샘플러가같고, 초기자기warmup
17배열/첫실제신규전이가바이트동일,1,000개batch index hash도전부같았다.
그러나첫update의gradient norm/loss부터작은수치차이가있었다(Q2norm차이2.50e−8).
두번째신규raw행동최대차이6.38e−11,신규offset30에서관측차이2.29e−7이처음나타났다.
이후수집main/branch수는과거615/385→현재627/373으로달라졌다.
양쪽main결과는충돌7·RECORD_END1·예산종료1이지만일부장면의충돌tick도달랐다.
따라서같은설정/seed의실행을가중치·궤적까지정확히재현한것으로부르지않는다.

후반replay같은index는서로다른상태/시간일수있어동일상태대체행동으로해석하지않는다.
하드웨어/컴파일러의유일원인이나highest정밀도변경의해결효과는이번에입증하지않았다.
작은단일seed차이를후보효과로과장하지않고현재동시기대조군과반복주행근거가필요하다.
`control-repeat-review.json` SHA
`7748d1eb2652435f7a9fc8cb455c4864f867bb2de6f290ea7531d7719b324488`.
검토기는로컬출력폴더에보존했고source/hash를보고서에기록했다. 이절도HEAD942c372기준
미커밋이며진행중학습설정/소스는바꾸지않았다.

## 18:14 pilot 추종 위치 오차·행동별 결과·충돌 접근 방향

평가 운영 기준의 추종 위치 RMS/KEEP·FOLLOW/충돌 이력을 추가 확인했다.
새 로컬 `review_driving_details.py`를완료pilot100에실행해16개실제trace의
post-step위치와그시점의executed_xy[0]를대조했다. EMERGENCY의경로없음은
0오차로넣지않고제외했다. 속도/방향RMS나새Waymax재실행은아니다.

| 같은4개개발장면×2행동 | 기존100 | 후보100 |
| --- | --- | --- |
| 목표/충돌 | 0/8 | 2/6 |
| KEEP 목표/충돌(각4회) | 0/4 | 1/3 |
| FOLLOW 목표/충돌(각4회) | 0/4 | 1/3 |
| 거절/제동/정지비중 | 34.42/31.52/67.39% | 25.27/22.34/64.10% |
| 평균training return | −20.347578 | −15.223028 |
| 위치추종RMS(m) | 0.000271878 | 0.000268613 |
| 추종분모/경로없는EMERGENCY 제외 | 388/164 | 440/106 |

작은위치오차는수락/재사용/제동경로추종의사후기록이며경로안전성이나정지문제해결이아니다.
14충돌은모두실제종료속도0,충돌직전연속거절16~35tick·연속제동15~33tick이었다.
종료OBB겹침을원시객체/ego로재계산해기록된colliding_ids와모두일치했다.
해당상대차량은종료이전최대3초의센터상대위치분류에서모두REAR로접근했다.
사고책임/제동의인과/회피불가능성을입증하는분류는아니다. 동일장면KEEP/FOLLOW와
두모델반복을독립14장면으로세지않는다.

로컬pilot `driving-details-review.json` SHA
`71bad38fe74c02d78dc8831ef3cab19c34d9c16d750dbd7371a8e506cee7974d`.
검토기는 `outputs/sac_own_goal_retention_1000_20261120/review_driving_details.py`,
완료pilot100에서exit0이다.1,000평가의최종pilot-review완료후같은파일을인자없이실행한다.
그검토는아직미실행이며후보학습/기존56개소스는유지한다. 이절은HEAD942c372기준미커밋.

## 18:26 두 방식의 SAC 학습·저장·경험 검토 완료, 주행 평가 중

현재 비교는 **SAC를 학습해 안정적인 정책을 만드는 과정**이다. 먼저 안정 정책을 만든 뒤
그것으로 SAC를 다시 가르치는 절차가 아니다. 두 모델의 actor/Q를 실제 자기 경험으로
각1,000회 업데이트했다. 안정 정책이 확보되면 가중치를 고정하고 Rule-based와 비교한다.
이번fresh1,000은pilot100을이어받지않은별도초기화이며,여러진단의업데이트를한모델의
누적학습량으로합산하지않는다. BC/teacher replay0·보상/검사/PD 조건은 유지했다.

두 학습과 저장 감사, 동일62개 저장 종료 행동의 Q 검토, 실제6,000행/256,000회
batch 추출 검토가 완료됐다. 마지막 후보 목표 pool은 초기 자기 성공 종료6행 그대로이고,
총8,155회(지정8,000+기타155) 사용했다. 대조군은155회다. 반복 추출을 신규 성공이나
독립 경험수 증가로 세지 않는다. 후보report SHA
`5b04deee4c1e20ee7c73183b075fb9c7efa81777513070f7848e3e974ebb6eae`, learner SHA
`67cc5e7d7ce05d83337c0e45b03327c4c50a5b6d35b0075c66e1b33249b68386`.

| 같은 저장 종료 행동의 CPU highest Q MAE | 대조군1,000 | 후보1,000 |
| --- | ---: | ---: |
| 전체62개 | 6.597882 | 5.875929 |
| 목표6개 | 0.501221 | 0.109321 |
| 큰 실패44개 | 9.102374 | 8.124218 |
| 기타 종료12개 | 0.463075 | 0.515507 |

목표/실패 오차는 줄었지만 기타 종료 오차는 늘었다. 실패 오차8.12는 여전히 크다.
종료 target은 실제 저장 reward라서 미래 return/entropy 추정 불확실성은 없다.
이 수치 개선 자체가 실제 주행 개선이나 새로운 상태의 정확도를 입증하지 않는다.
`training-review.json` SHA `8c4a019c0fbc23809e88bfba6444b5a78085b590ba3c56de6a619eb2f1f1fafc`.

신규1,000전이 중 저속은 대조군490/후보525, 모두 next speed0.1m/s 이상으로 넘어간
사례가0개다. 해당 next speed 최대는0.005737/0.002740m/s다. 새 목표 종료도양쪽0개다.
새 경험의 batch 노출은123,110/116,118회, 그중이동경험은70,910/62,619회였다.
실제main/branch 전이는627/373 대668/332로달라졌으며, main/branch 종료도분리했다.
후보main 충돌8·RECORD_END1·예산종료1,대조군7·1·1은학습중서로다른길이의수집결과다.
이를고정모델의동일장면성공률비교로사용하지않는다.
`experience-review.json` SHA `ba6a06bf3851024c572afa8bd9710e8f6a4fd5698e410c40f9e440fa56c19d4b`.

18:26:59 실제control-primary child2876172생존,고속KEEP/FOLLOW2건은충돌로완료했다.
나머지primary/후보primary/양쪽정지평가와최종16trace검산은진행중또는미실행이다.
이상태에서기본채택/추가예산/평가확대를결정하지않는다. 완료후추가driving-details검토로
행동별결과·추종분모·충돌접근이력까지검토한다. 안정정책은여전히미확보다.

## 18:40 평가 완료 및 후속 원본 검토: 안정 정책 미확보

두 fresh1,000 모델의 학습·저장 감사·동일 초기 frame/map 주행 평가가 완료됐다.
main은18:40:30, review는18:40:33 종료했고 실제 PID2861692/2863133/2880825가
종료했음을 확인했다. `COMPLETE_PENDING_REVIEW`는 보존된 main 실행 상태이며,
후속 `pilot-review.json`은 COMPLETE/qualified=false다. 추가 학습은 실행하지 않았다.
기준 HEAD07e0145, 학습 코드1c86e9f, seed0, 고유 개발4장면×KEEP/FOLLOW 8쌍/모델이다.

| 전체8주행 지표 | 대조군 | 자기 목표 종료 유지 후보 |
| --- | ---: | ---: |
| 목표 / 충돌 | 0 / 8 | 0 / 8 |
| 실제 tick | 486 | 514 |
| 경로 거절률 | 55.97% | 36.19% |
| 제동률 | 45.68% | 31.91% |
| 정지 비중 | 74.49% | 69.26% |
| 평균 training reward 합 | -20.468682 | -20.355843 |

KEEP/FOLLOW 각각4회 모두 충돌했다. 정지 출발4회는 양쪽328/328tick 정지다.
후보의 일부 지표 개선으로 목표0/8을 상쇄하지 않으며 후보는 미채택이다.
pilot100 후보의 고속2성공도 fresh1,000에서는 유지하지 못했다. 서로 다른 fresh 학습이며
100→1,100 연속학습이나 예산만의 인과 효과로 해석하지 않는다.

실행 명령: `/home/autonav/miniconda3/envs/its-merge/bin/python outputs/sac_own_goal_retention_1000_20261120/review_driving_details.py` (exit0).
기존16개 trace를 검산했으며 새 시뮬레이션은 아니다. 기록된 실제 시점의 선택 경로 대비
위치 RMS는0.000284006/0.000282690m, 분모330/368이고 참조 없는 EMERGENCY156/146은
제외했다. 작은 추종 오차가 안전한 경로 계획을 입증하지 않는다. 충돌16건 모두 최종속도0,
마지막 연속 거절16~29tick·제동14~27tick이었다. 실제 OBB 접촉 ID를 재검산했고
직전3초의 상대 위치 이력에서 모두 뒤쪽 접근이었다. 책임·불가피성·유일 원인 판정은 아니다.

로컬 공통 경로 `outputs/sac_own_goal_retention_1000_20261120/`:

- `pilot-review.json` SHA `d5507713232300e39dfaca0ca18979d2af41d7e52677661c68244e1edc9582f4`.
- `driving-details-review.json` SHA `7c8fc90a74420520f101d22d6aebccb45959f77572ff56009402937a0ac3253d`.
- `terminal-exposure-review.json` SHA `c2c6ec7845d57ac4a7af0eaa3cd3ee56e0ecbc51abeaf20b16bc5b8abf6db847`.
- `visual-budget-comparison/own-goal-100-vs-1000-keep.gif` SHA `43cb6283bce519e801f59a134ad746d20c6f1276c3f219c3dbb1457126e6bf78`.

종료 경험의 추출 재검산에서는 초기 실패44행의 사용이 크게 늘지 않았다.
초기 큰 실패44행의 추출 횟수는 대조군1,251/후보1,232, 행당28.43/28회였다.
같은 목표6행은155/8,155회다. 실패Q MAE8.124가 남았고 성공 경험을 강조한 이번
실험은 주행 성공으로 이어지지 않았다. 추출 빈도만이 원인이라고 단정하지 않는다.

다음은 실제 자기 성공·실패 종료 경험의 학습을 함께 유지하는 방안을 규약과 대조한다.
아직 설계 제안이며 새 sampler 구현·예산·학습 실행·기본 채택은 미완료다.
보상/안전검사/PD/순수 SAC 조건을 유지하고, 실패Q 개선과 실제 주행을 함께 확인해야 한다.
전체개발16쌍·다른seed·새 동일조건 Rule-based 비교·별도test 확대는 미실행이다.

## 18:51 자기 성공·실패 종료 유지 비교 구현 및 실행 시작

직전 단계는 최종16trace 검산과 미채택 판단을 완료해 다음 보완 근거를 얻은 진행이다.
다음 비교는 성공 종료8을 유지하면서 최근 경험8개를 실제 실패 종료8개로 대체한다.
batch256 = 균등128 + 최근256행에서112 + 자기 목표 종료8 + 자기 큰 실패 종료8이다.
빈 pool만 균등 추출로 대체하며 실제 보상과 bootstrap=false를 함께 사용한다.
이 구분을 원본 종료 사유와 대조하고, 전이/보상/종료를 새로 합성하지 않는다.
이는 별도 추출 분포 비교로 critic과 actor 상태 분포 모두 달라지며 importance weight는 없다.
기본 채택·균등 replay와 같은 목적함수·성능 개선을 주장하지 않는다.

실행 코드 `82d5fc7de08499ee4939b5853f229078de70cf14`는 같은 브랜치에 commit/push했고
원격 SHA도 확인했다. 새 PR/병합은 없다. 새 파일은
`scripts/sac_terminal_retention_sampling.py`, `scripts/diagnose_sac_terminal_retention.py`,
`scripts/audit_sac_terminal_retention.py`, 관련 테스트다. driver는 기존goal driver의
sampler/선택 인자/scope만 바꾼 격리 복사본이다. 기존56소스 해시와 모든src는 유지했다.
`docs/architecture/rule-based-vs-pure-sac.md`에 별도 비교 조건을 명시했다.

검증 명령:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest -q \
  tests/training/test_sac_terminal_retention_sampling.py \
  tests/training/test_sac_goal_retention_sampling.py \
  tests/training/test_sac_diagnostic_sampling.py
```

its-merge Python에서40 passed. 실제 보상 함수의4실패/목표/기타종료 분류,
bootstrap·미래capacity 제외·각 빈 pool fallback·배치 원본 action/mask/reward 보존·
저장 RNG 재현·teacher/ring 거절을 확인했다. 실제 자기 warmup5,000의 목표6행·실패44행도
checkpoint의 원본 episode 종료 기록과 일치했다. 전체 회귀/새GPU 성능 검증을 대체하지 않는다.

첫 실행은 seed0·fresh learner·같은 자기 warmup5,000·신규 전이1,000/SAC전체update1,000이다.
기존 B평균/f64/raw gain0.0001/physical_whitened critic/zero-action 초기화,
실패수집 samples1/window6/moving_predecessors/horizon200/complete_path를 유지한다.
보상/안전검사/PD/Gaussian/lr/tau/BCteacher0도 유지한다. 이전 fixed-Q 가중치는 사용하지 않는다.
대조군은 직전 `outputs/sac_own_goal_retention_1000_20261120/candidate1000`와
그 candidate-primary/candidate-stopped 실제8trace를 해시 고정해 재사용한다.
새 비교에서 대조군을 다시 학습/평가했다고 세지 않는다. 별도 실행의 수치 차이와 seed0 한계를 유지한다.

로컬 새 루트 `outputs/sac_own_terminal_retention_1000_20261120/`.
decision SHA `f225fe03e6e38896de9ece34d41928e8dcf5b94c186abfa5da61dfc3fa9ab796`.
66개 소스·22개 입력을 고정했고 원본과 보존 복사본 SHA를 대조했다.
18:51:02 runner2885167(session83060), candidate child2885178 실행 시작.
18:51:10 검토 runner2885246(session59015) 시작, WAITING_FOR_TRAINING.
18:51:35 실제 두 실행 PID와 학습 child 생존 확인; 당시 첫 update는 아직 기록되지 않았다.

연결 순서는 후보1,000학습→저장/종료pool/배치해시/RNG/full-update 감사→같은primary4회·
정지4회 평가다. 별도 CPU 검토는 저장 감사 후 같은62개 종료 Q·실제 경험/배치 노출,
평가 완료 후 초기 frame/map·16개 저장trace·추종 분모/접촉 이력을 대조한다.
후속 검토 파일은 준비·구문 검사를 마쳤으며 새 결과 대상 실행은 아직 대기다.
오류·비유한·hash 불일치는 기록 후 중단한다. 자동 추가 예산/채택/전체validation 확대는 없다.
안정 정책을 확보했다는 뜻이 아니며 최종 목표는 계속 미완료다.

```bash
python -m json.tool outputs/sac_own_terminal_retention_1000_20261120/status.json
python -m json.tool outputs/sac_own_terminal_retention_1000_20261120/reviews-status.json
```

18:53 완료된 초기33update를 읽기 전용으로 추가 검산했다. 모든 배치 index hash,
이전 policy update 번호, finite 기록, 목표/실패 지정 slot264/264회가 일치했다.
첫 제안의 수치 필드도 대조군과 정확히 같았다. 최초 진단은 제안 dictionary 전체 동일성을
요구해 실패했으며, 차이는 독립 실행의 episode UUID와 그로 만든 proposal ID뿐이었다.
실패를 `live-prefix-initial-comparison-failure.json`에 보존했다. 각 실행 내부 ID 일관성과
나머지 모든 필드를 따로 검증한 `review_live_prefix.py`는exit0이다. 학습은 변경/재시작하지 않았다.
`live-prefix-review.json` SHA `6d40c505846fa8a31096a05487b2c380c7884090e82722275b93ea898326c707`.
이는 실행 중 prefix 확인이며 최종 저장 감사나 주행 성능 검증을 대신하지 않는다.

## 18:58 현재 정책 행동의 실제 미래 보상 검토 연결

저장된 과거 종료 행동의 Q 적합만으로 현재 정책의 Q 과대평가가 해결됐다고 볼 수 없다.
현재1,000 비교의 설정/소스/decision을 변경하지 않고, 완료 후 두 모델을 고정해 실제 종료까지
주행시키는 추가 읽기 전용 검토를 연결했다. train 상태2곳에서 평균 첫 행동을 실행한 뒤
원래 Gaussian으로 종료까지 주행한다. 각 상태×4개 동일 seed, 모델당8회이며 학습/replay에
넣지 않는다. 조건부 표본평균은 정확한 기대 Q나 독립 test가 아니다.

상태는 이전부터 고정한 KEEP `77ae312df21e80b9` tick0와 FOLLOW `baac7893b025167a` tick4다.
`outputs/sac_physical_q_20261119/branch1000`의 실제 관측/과거 행동으로 재구성하고
`outputs/sac_physical_q_20261119/predecessor-actions1000.json`의 선택을 유지한다.
새 모델의 Q나 결과를 보고 상태를 다시 고르지 않는다. 기존
`scripts/diagnose_sac_soft_returns.py`를 그대로 실행한다. 새 검토기는 실제 trace의 매 tick
보상·discount·entropy·종료를 재계산하고 두 모델의 시작 관측 동일성을 확인한다.

로컬 새 루트의 `post-decision.json` SHA
`d4718382297d4c09ef1db9737724317e999e828a97680d6832e34ca7a892f0df`.
69소스/29입력을 고정했다. `run_post_review.py`는 main 평가와 검토의 실제 PID 종료 및
성공 상태를 기다린 뒤 모델/replay/report SHA를 고정한다. 순차 GPU 검토는 대조군→후보,
그 뒤 `review_current_actions.py` 검산과 `render_terminal_comparison.py`의 실제 저장
고속KEEP 나란히 GIF를 연결했다. 새 학습·새 Rule-based 비교는 아니다.

검토기 사전 실행은 과거 `sac_recent_full_path_20261120/soft-returns1000`을 양쪽 참조로 둔
계산 검증이며 새 모델 결과가 아니다. exit0, 원래 KEEP/FOLLOW Q−표본평균 차이
13.147532/10.392730을 재현했다. 로컬
`current-action-review-preflight/current-action-review.json` SHA
`b85c6d7769f521bdf24a461c85295dbe78f2701c9259c9b178912a317ab5e802`.
최초 준비 명령의 닫는 괄호 누락은 Python 실행 전 SyntaxError로 종료했고 그 사실을
post-decision에 보존했다. 준비 전 파일 변경/학습 재시작은 없고 수정 후 사전 검증했다.

18:58:23 추가 runner2890300(session19010)을 시작했다. 19:00:18 실제확인에서
main2885167/학습child2885178/reviewer2885246/post2890300 모두 생존했다.
학습407/1,000, 기존 검토는 WAITING_FOR_TRAINING, 추가 검토는 WAITING_FOR_MAIN_REVIEW다.
현재 새 모델의 주행/Q/미래 보상 결과는 아직 미완료이며 기본 채택/추가 예산은 결정하지 않았다.

```bash
python -m json.tool outputs/sac_own_terminal_retention_1000_20261120/post-status.json
```

## 19:09 새1,000 학습·저장/Q/경험 검토 완료

학습은19:09:04, 저장 감사는19:09:07 종료했다. 실제1,000전이/actor·Q 전체update1,000,
초기 자기warmup17배열·실제종료pool·모든배치hash/RNG·BCteacher0 검토를 통과했다.
후보는main739/분기261전이, 목표pool6→7·실패pool44→58이다. 목표/실패 지정slot은각8,000회,
전체목표추출8,265·실패추출10,825회다. 기존목표만유지모델과 같은 종료62행의 CPU highest Q:

| 종료 Q MAE | 기존 목표8 | 새 목표8+실패8 |
| --- | ---: | ---: |
| 전체62개 | 5.875929 | 2.082132 |
| 목표6개 | 0.109321 | 0.109951 |
| 큰실패44개 | 8.124218 | 2.606761 |
| 기타종료12개 | 0.515507 | 1.144582 |

실패 적합은 개선됐고 목표 오차는 비슷하지만 기타종료의 과소평가는 악화됐다.
현재 정책 행동의 실제 미래 보상 검토는 아직 대기이며 이 표만으로 Q 문제가 해결됐다고 보지 않는다.
초기실패44행의추출은1,232→7,965회, 신규실패14행은2,860회다. 신규1,000전이의저속은554개,
next speed0.1m/s 통과0·최대0.007046m/s로 실제 출발 경험 부족은 남았다.
새 목표 종료1건은 train KEEP `30a56b458fc5b63d`의68tick주행이며 초기warmup에서도
성공했던같은쌍이다. 새로운장면성공이나고정정책성능으로세지않는다.

로컬 `outputs/sac_own_terminal_retention_1000_20261120/`의 확인한SHA:

- 후보report `e50299e6c30f0991342bfcdaf2f4dc2ec592708056b192e232bb7eb8a5829f24`.
- learner `67a2a12c5bc16d1ccbcdc901a683fae7e6f0cfcede7782de004c4bf23e331831`.
- replay `bc217661ac46d0ab4ca3d00ed267f9e483f7156107e106d5d5ee12b190863d89`.
- audit `41fc1daa11a600843c619a860fe358e68fa98b12046ad07dd124ffe6769d11c4`.
- training-review `e543d1acf553f0b9a17f95f2f50b2cf519c1b6bd60e09899a364cd436d7ca0ce`.
- experience-review `377f0d886361ec4d203d9739c49267ade18eef1b549e849dcb68de9b2a1446f7`.

19:11:55 primary child2893552와main2885167생존확인. 고속KEEP/FOLLOW는각72tick목표,
저속KEEP는46tick충돌로완료했고나머지FOLLOW·정지4회평가는진행중/대기다.
고속2성공은초기평균0과pilot100에서도있던성공의회복이며새능력확보로과장하지않는다.
추가seed/전체개발/Rule-based/독립test/학습예산확대는아직미결정이다.

## 정지 출발 가능성 분리: 수작업 진단이며 SAC 성과 아님

공통 검사·PD에서 출발 가능한 경로가 있는지 같은정지2개발장면×KEEP/FOLLOW를
별도CPU환경에서 검사했다. SAC/production Rule-based를변경하거나시범을학습에넣지않았다.
실제anchor 위치/v/a/yaw와원래60차원encode/decode·검사·fallback·PD·Waymax를사용했다.
현재heading방향의cubic longitudinal경로와초기측방v/a를보존하며감쇠하는성분을구성했다.
실제초기anchor와기존평가frame의일치,encode/decode roundtrip도검증했다.

첫진단은고정grid jerk0.05/0.1/0.2/0.4/0.8에서초기검사를통과한최대0.2를유지했다.
모든4사례에서0.05/0.1/0.2는통과,0.4/0.8은curvature/curvature_rate로거절됐다.
3초실행에서0.0842~0.0857m움직였지만속도0.05825~0.05855m/s라서0.1m/s출발기준은미달이다.
두번째진단은같은초기경로를사용하되실제속도가기존moving-heading기준0.01m/s이상이되면
같은grid의0.8을요청했다. 모든신규경로는기존검사와fallback을그대로거쳤다.

| 같은 고유 장면의 KEEP/FOLLOW | 0.1m/s 최초 도달 | 3초 속도 | 3초 이동 |
| --- | ---: | ---: | ---: |
| 8dd5939855c15b3f | 둘다1.8초 | 0.196047m/s | 0.243688m |
| a62f5a7d2cd9b3d9 | 둘다1.9초 | 0.195747m/s | 0.242741m |

각30tick에서실제event0·제동0,2번째tick만KINEMATIC_LIMIT로거절후기존경로REUSED였다.
나머지29제안은통과했다. 종료는모두`PROBE_HORIZON`이며bootstrap=true로남겨
ROUTE_GOAL/전체장면안전/안정정책으로세지않는다. 같은장면의두명령은반복측정이다.
공통검사/PD에서출발이전부불가능하다는설명은이4사례에해당하지않는다.
그러나이수작업경로는SAC학습결과가아니며teacher/replay/새baseline으로채택하지않는다.

실행: `python outputs/sac_stopped_launch_feasibility_20261120/probe.py`,
`python outputs/sac_stopped_launch_phased_20261120/probe.py`,
`python outputs/sac_stopped_launch_phased_20261120/review.py`,모두its-merge에서exit0.
첫probe PID2892232(session83540),두번째2892801(session44798)는실제종료확인.
원본report SHA는각 `b7e7ad20ab8b0cbe4c5c1692e5fa5c44dabc597380b65a4d96444054a04e2555`,
`4388e7381bcf94cd1739e8d1722c543c43df8060e457c77b7b9b2dc41e982f14`.
새검토 `review.json` SHA `d549c422d7e60a5f95fcc9f73f57bab56da9e3da52cec1b119383eb210859c74`.

동일초기상태의기존목표8모델평균과첫수작업경로도비교했다. raw절댓값최대는약0.000264 대0.03333,
현재B/gain0.0001을역산한동등head절댓값최대는약78.3 대2000이다. 이는저장경로의수학적좌표이며
실제NN head조사·필요최소출력·학습시간하한·gain이유일원인이라는증명이아니다.
출력크기만확대하면된다고판단하거나진행중모델/gain/보상/검사를변경하지않았다.

## 19:24 확인: 전체 8회 주행 완료, 현재 행동 가치 평가는 진행 중

19:17:57까지 새 후보의 4개 개발 장면×KEEP/FOLLOW 8회 평가와 저장 trace 검토가
완료됐다. 같은 초기 frame/map의 기존 목표8 모델 8회는 보존된 결과를 재사용했다.
독립 장면은 4개이며 두 명령과 두 모델의 반복을 독립 test로 세지 않는다.

| 전체 개발 8회 | 목표8 대조군 | 목표8+실패8 후보 |
| --- | ---: | ---: |
| 목표 도달 / 충돌 | 0 / 8 | 2 / 6 |
| 실제 tick | 514 | 564 |
| 경로 거절 비중 | 36.19% | 30.14% |
| 제동 비중 | 31.91% | 24.82% |
| 정지 비중 | 69.26% | 60.28% |
| 평균 reward 합 | -20.355843 | -15.263125 |

고속 KEEP/FOLLOW의 기존 성공을 회복했지만 저속 2회와 정지 4회는 모두 충돌했다.
정지 4회만 보면 양쪽 모두 328/328tick 정지이고, 후보의 거절은92→126,
제동은82→104로 악화됐다. 전체 평균의 개선으로 이 회귀를 숨기지 않는다.
추종 position RMS는 대조군0.000282690m(368samples)/후보0.000270578m(468samples)이며
참조 경로 없는 EMERGENCY146/96tick은 제외했다. 작은 추종 오차는 안전한 경로 생성의
증거가 아니다. 안정 정책·기본 채택·전체 개발/새seed/독립test 확대는 미완료다.

로컬 `outputs/sac_own_terminal_retention_1000_20261120/`:

- `pilot-review.json` SHA `ebe4fe00337d6ff2ecc67994443940c69b0a302cafd34e3ee9bc8182a9897b18`.
- `driving-details-review.json` SHA `43c762869e2a1d632423a19f0a195956fed4fb801f5d1d4d35064a85e336b3f3`.

19:23:57 KST 실제 post runner2890300/soft-returns-control child2897254 생존을 확인했다.
학습은 종료됐으며 현재는 고정 정책의 Q와 실제 종료까지 미래 보상을 대조하는 평가다.
대조군→후보→검산→같은 고속 KEEP 저장 trace GIF 순서이며 새 학습/replay 쓰기는 없다.
최종 현재 행동 Q 비교와 GIF는 아직 미완료다. 단순히 좋은 수치가 나올 때까지 반복하는
것이 아니라, 실패 행동의 가치 추정과 실제 주행을 함께 확인해 다음 실험을 결정한다.

## 같은 learner 이어학습 준비: CPU 복원 확인, GPU 이어학습 미실행

코드 `0ea2739491780029ffe263db3bcfb104831c8c06`은 같은
`feature/yuntae-sac-training` 브랜치에 commit/push 및 원격 SHA 확인을 마쳤다.
새 파일 `scripts/continue_sac_terminal_retention.py`,
`scripts/sac_terminal_retention_continuation.py`,
`scripts/audit_sac_terminal_continuation.py`,
`scripts/audit_sac_terminal_branch_replay.py`,
`scripts/diagnose_sac_terminal_soft_returns.py`는 기존 동결 실험 소스와 분리했다.
전체 learner/optimizer/target/RNG와 replay/RNG 및 기존 tick/update prefix를 복원한다.
환경과 수집 RNG는 명시적으로 재시작하므로 정확한 시뮬레이터 중간 재개는 아니다.
보상·검사·PD·Gaussian·목표8+실패8 추출·실제 전이당 전체 SAC update1을 유지한다.

앞서 실행한 관련 pytest는69 passed다. 실행 명령:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest -q \
  tests/training/test_sac_terminal_retention_continuation.py \
  tests/training/test_sac_diagnostic_continuation.py \
  tests/training/test_sac_terminal_retention_sampling.py
```

새 auditor의 실제 fresh1,000 저장 감사와 CPU 실제 복원 사전 검토도 완료됐다.
19:19:57 복원 결과는 직렬화 learner 바이트 동일, actor/Q 각1,000·BC0,
replay6,000행의17배열 동일, RNG복원과 다음 메모리상 배치 hash 일치다.
`continuation-preflight-audit.json` SHA
`3e663a92eac7b9cc444267500fd626ccab012de44a14ead622266ccb93f7521f`,
`continuation-restore-preflight.json` SHA
`0eda4994f38bfa02a50c81318357fb05c9c2d6e9347883f0dc7ff11971ae0f88`.
사전 검사에 사용한1001은 허용 범위 검사 값이며 실제 선택/실행한 후속 학습 예산이 아니다.
GPU 이어학습·새 전이·추가 optimizer update는 미실행이다. 현재 행동 가치 평가를 마친 후
추가 예산을 판단하고, 이어학습한다면 실제 GPU 복원/신규 전이/누적 감사를 먼저 확인한다.

## 19:27 연결: 실제 GPU 이어학습 1회 검사 대기

현재 정책의 평가가 끝난 뒤 같은 learner를 실제로 복원하고 신규 자기 경험1개와
전체 SAC update1회를 수행하는 별도 smoke를 연결했다. 후속 성능 학습의 예산을
선택한 것은 아니며, 이 검사 모델을 자동으로 최종 후보나 다음 학습 조상으로 채택하지 않는다.
원래 fresh1,000은 보존하고 별도 `online1001`에 저장한다. 총update1001·이전1000·신규1,
actor/Q/BC0·6,001 replay행·기존tick/update prefix 및 모든 배치/RNG를 검사한다.
보상/검사/PD/Gaussian/추출방식·전이당update1은 유지한다. collection_seed1000으로
환경/수집 RNG를 명시적으로 재시작하므로 정확한 simulator 중간 resume은 아니다.

로컬 `outputs/sac_terminal_continuation_smoke_20261120/`의 `run_smoke.py`를
its-merge Python으로 실행했다. 19:27:22 PID2901620(session83679),
`WAITING_FOR_POST_REVIEW`이며 신규 GPU 학습은 아직 미실행이다.
실제 post PID2890300의 process starttime과 runner hash를 확인한 뒤 대기한다.
기존 평가가 성공 종료하고 실제 PID가 사라진 경우에만 수행하며, 실패/누락이면 중단한다.
기존 GPU 평가와 겹치지 않는다. 75소스/30입력 hash 및75소스 보존 복사본을 검증했다.

- decision SHA `7fbb4f58374f141dd2d74836c1080067eb8bab7bfdd29c0391eabfe72d6dc4ba`.
- runner SHA `68a4ef9b81dd930faff8959cadbaeb0f5e58a686dab38ac7b3731cb27b55dfbd`.
- 실행 코드0ea2739, 실행 당시 문서 HEAD0f57347. 기존 동결 source 변경0.

대조군 현재행동 평가는 완료됐다. 두 train상태×4확률seed는 모두 충돌했고,
Q−실제soft-return평균은 KEEP13.681079/FOLLOW10.212227이다. 후보 KEEP4회도
모두 충돌하며 해당 차이는12.198158이다. 후보 FOLLOW와 최종 검산은 아직 진행 중이다.
이 부분 결과를 전체 완료나 안정 성능으로 쓰지 않는다. 상태 조회의 최초 편의 명령은
존재하지 않는 `states` 키로 KeyError였고, 실제 `cases` 키로 수정했다.
실험 과정/모델/파일 변경이나 재시작은 없었다.

```bash
python -m json.tool outputs/sac_terminal_continuation_smoke_20261120/status.json
python -m json.tool outputs/sac_own_terminal_retention_1000_20261120/post-status.json
```

## 19:35 현재 행동 가치 검토·GIF 완료

기존 목표8과 새 목표8+실패8의 동일 train2상태×각4Gaussian seed 검토가 완료됐다.
두 모델 모두8/8충돌이다. 실제 reward·할인·entropy·최종bootstrap·관측hash·raw action을
원본trace에서 검산했다. 아래는 평균 첫 행동 이후 원래Gaussian으로 실제종료까지의
조건부 표본이며 정확한 기대Q나 독립test가 아니다.

| 항목 | 목표8 KEEP | 목표+실패 KEEP | 목표8 FOLLOW | 목표+실패 FOLLOW |
| --- | ---: | ---: | ---: | ---: |
| 현재 평균 첫 행동 Q | -0.371438 | -1.451190 | -0.241021 | -0.946604 |
| 실제 soft-return 표본평균 | -14.052518 | -13.649348 | -10.453248 | -10.821256 |
| Q−표본평균 | 13.681079 | 12.198158 | 10.212227 | 9.874652 |
| 표본 표준편차 | 1.083250 | 1.711044 | 0.259192 | 0.520338 |
| 평균 실제 tick | 35 | 38 | 62.25 | 59.25 |

과대평가 차이는 줄었지만 여전히 크다. KEEP의soft-return은+0.403170,
FOLLOW는-0.368008이다. FOLLOW는 평균3tick 먼저 충돌했고 할인 reward 합도
-11.256366→-11.593023으로 악화됐다. KEEP은 평균3tick 늦게 충돌했을 뿐
목표도달이 없으므로 안전한 개선으로 판정하지 않는다. 저장종료 Q 개선과
개발고속2성공회복만으로 이 결과를 상쇄하지 않는다.

로컬 `outputs/sac_own_terminal_retention_1000_20261120/current-action-review.json`
SHA `b201feffff99f6434dd0d6c5b4f42fff8c13fe7ad5775421cc79fd446a3b5f4f`.
post runner2890300과 실제 평가/검토 child는 모두종료했다.
같은고속KEEP 두SAC의 저장trace GIF도 생성됐고 최종PNG를 실제 열어 확인했다.
기존목표8은충돌후정지, 새목표+실패8은11.03m/s·BRAKE 상태로ROUTE_GOAL임이 표시된다.
거절/제동/주행 궤적과종료후held표시가 있으며 단일성공예시를 전체성능으로 보지 않는다.

- 로컬 `visual-terminal-comparison/own-goal-vs-goal-failure-keep.gif` (위루트내).
- 1440×900,37frames,시뮬레이션7.2초·마지막hold포함재생8.7초.
- GIF SHA `c407246fd2d371043c44d7c01ca95ba93e2f1f419cfa706a618f105fc1ec6e4b`.
- 같은1,000update의SAC/SAC 비교이며 Rule-based 영상이 아니다.

## 19:36 GPU 이어학습 검사 통과 및 같은 정책 총3,000회 시작

별도smoke는19:36:06 완료했다. 실제 GPU learner/optimizer/target/RNG 복원바이트동일,
새실제자기전이1개·전체update1회, 누적actor/Q각1,001·BC0·replay6,001행을 확인했다.
원본tick/update prefix와17배열·실제train분기/종료·전체배치hash/RNG 감사도 통과했다.
smoke runner2901620/학습2903894/감사2905457의 실제종료를 확인한 뒤 후속을 시작했다.
로컬 `outputs/sac_terminal_continuation_smoke_20261120/review.json` SHA
`1975ed4b733ba800a2543f15d6f8eb9319424e2c756d78baf4f48807702d2507`,
`audit1001.json` SHA `31ba360035a749b8fe17b7ca26e2af7bebd45524bf15e5947ccf18fcdc47e723`.
이는실행검증이며성능학습이나안정성통과가아니다.

다음선택은새초기화가아닌 **완료된목표+실패1,000모델에서총3,000까지이어학습**이다.
종료실패Q적합과고속성공회복은관찰됐지만현재행동미래비용/출발이남아있어서,
같은정책의추가경험과학습이비용전달·성공유지·실제주행에어떤영향을주는지검사한다.
학습량부족이유일원인이거나3,000이면충분하다고가정하지않는다.
smoke1,001가중치는사용하지않고원래1,000의전체상태/replay를복원한다.
collection_seed1000으로환경/장면·정책수집RNG를명시적으로재시작한다.
정확simulator resume이나순수update횟수만의인과비교로주장하지않는다.

보상·안전검사·공통PD·원래Gaussian·actor/Q구조·gain·lr/tau·목표8+실패8배치·
자기전이당전체SAC update1·BCteacher0유지. 추가2,000실제전이/전체update를목표로하며
기존1,000기록은그대로보존한다. 새예산경계에가짜종료나성공을넣지않는다.

로컬 `outputs/sac_terminal_retention_curve3000_20261120/`:

- 19:36:28 runner2905593(session32911),19:36:31 학습child2905600 실제시작/생존확인.
- decision SHA `0b378b9d873ebe31871c99768e5654112f7ebcbc261095f5c01df4fc9bcc0aed`.
- runner SHA `610eeba7d7a6c9cf78d69c4100a100486e4454b1bf1400fede9a22a47cad9c65`.
- 84소스/67입력hash고정및84소스복사본검증. 실행코드0ea2739,실행시문서HEADb3c6af3.
- 총3,000학습→누적저장감사→같은62종료Q/기존·신규경험분리→고속·저속4주행+
  정지4주행→초기frame/map·지표·접촉이력검산→같은고속KEEP GIF→같은2train상태×4seed
  현재행동Q/실제미래보상→검산의11jobs를직렬연결했다.
- 1,000모델의8주행/MC는완료자료를hash고정재사용한다. 중복새실행이나독립장면으로세지않는다.
- 검토script는사전동일1,000snapshot양쪽비교에서종료Q/경험/주행지표의일치를실제로확인했다.
  `preflight/training-review.json` SHA `8d6888b372214091871f3ce7d9b19785bc2ad28db0c184942319e6d9d555745c`.
  사전검토를새3,000학습/주행성능의완료로세지않는다.

현재3,000결과는아직미완료다. 오류/비유한/hash/lineage실패는기록후중단하고,
새예산/채택/전체개발·다른seed·최종Rule-based·별도test는자동연결하지않는다.
개선과회귀를함께검토한뒤다음판단을한다. 안정정책미확보·전체목표진행중이다.

```bash
python -m json.tool outputs/sac_terminal_retention_curve3000_20261120/status.json
python -m json.tool outputs/sac_terminal_retention_curve3000_20261120/online3000/report.json
```

## 19:40 중간 이력 검사와 현재 행동/경험 대조 추가 연결

현재 총3,000 이어학습은 그대로 유지한다. 19:40:53 읽기 전용 중간검사는
누적1,182개(기존1,000+새182)의 완성된tick/update쌍을고정복사해검산했다.
기존기록바이트보존, 새전이당정확한update1·현재정책카운터·원래bootstrap/실제종료·
raw수치유한성·저장replay RNG에서재구성한모든새배치hash가일치했다.
목표/실패지정slot각1,456회, 새main114/branch68·충돌종료4건이다.
저속제안73개는모두거절(운동한계5/예측충돌68),양의가속도0이다.
학습중경험의일부이며최종저장감사/고정정책성능/전체회귀를대신하지않는다.

로컬 `outputs/sac_terminal_retention_curve3000_20261120/live-prefix-001182/review.json`
SHA `1aa2c5cd6c833f905c198bf9620084bdb68a5ff8165c7c6fff4973faa76881b8`.
최초중간검사는저장npz가capacity가아닌유효6,000행만가진다는점을놓쳐
`IndexError: index6000 out of bounds`로실패했다. 원본script/오류를같은루트의
`live-prefix-audit-initial-failure/`에보존하고, 검사메모리에검토길이만큼별도배열을
할당해원본6,000행과새기록을채우도록수정했다. 학습소스/진행/원본replay변경·재시작0이다.

저장행동에대한Q적합과현재평균행동의Q가달라질수있으므로, 현재행동과자기경험의
관계를직접대조하는후속읽기전용검사를준비했다. 새파일:

- `scripts/sac_terminal_retention_exposure.py`: 실제목표/실패추출의전체배치hash/RNG와노출수재구성.
- `scripts/audit_sac_terminal_coverage.py`: 기존분석을종료유지/이어학습감사와연결한별도진입점.
- `scripts/diagnose_sac_terminal_branch_q.py`: 같은실제실패상태의저장행동·현재평균·원래4Gaussian표본재실행.
- `tests/training/test_sac_terminal_retention_exposure.py`: 이력변조/부족·보상분류변경거절,원본배치/RNG복원검증.

기존동결소스/production replay/학습/보상/검사/PD/Gaussian변경0이다.
행동수집을deterministic mean으로바꾸거나진단평균행동을학습에넣지않는다.
원본v1.2의5절분포·평가u=mu 및10절step/실제전이규칙과대조했다.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest -q \
  tests/training/test_sac_terminal_retention_exposure.py \
  tests/training/test_sac_terminal_retention_sampling.py \
  tests/training/test_sac_replay_coverage.py
```

실제28passed. 기존완료1,000의실제6,000replay행/256,000추출을새helper로재구성했고
목표8,265·실패10,825추출과최종RNG가기존감사와일치했다.
Q를보지않는첫실패분기선택은KEEP77ae312df21e80b9/tick28/index57,
FOLLOWbaac7893b025167a/tick59/index358이다. 두행모두현재3,000에상속된prefix다.
사전검사 `terminal-coverage-preflight.json` SHA
`266a368bf58be104f5b9ea396ca90e3b737a2b21eb300aa4118d7cfbc0718f2b` (현재3,000루트내).
GPU forward/Q/반사실적실행의완료검증은아직아니다.

후속은로컬 `outputs/sac_terminal_policy_overlap_20261120/`에연결했다.
19:45:01 runner2910702(session89956), `WAITING_FOR_CURVE`.
현재3,000의11jobs가성공종료하고실제runner2905593이종료한뒤에만실행한다.
두모델의coverage/실패분기실행을순차수행한뒤CSV/실제대안결과를검산한다.
현재학습/주행과GPU작업이겹치지않는다. 90소스/69입력hash및90소스복사본검증.
post decision SHA `99a8172fef91718bf09b972469d18834798d8c6d3e4cbc8d00d3be4410df411c`,
runner SHA `087db7904c98e676bd6e346648b4da65741ebb8213a173bd0bc170b3fdefe5cd`.
원래3,000decision/예산/학습조건/동결소스는변경하지않았다.

같은6,000관측의분포거리와신규2,000행을따로집계하고, 실제terminal대안만Q−reward를계산한다.
Gaussian의95%타원체포함비율은조건부거리진단이지문자그대로support·신뢰구간·안전성·
새행동경험의충분성증명이아니다. 현재평균과저장행동의보상이같다고미리단정하지않는다.
19:45:23 학습runner2905593/child2905600와post2910702실제생존,학습1,569/3,000·추가569확인.
최종학습/주행/현재행동검토는아직미완료이며안정정책미확보다.

## 19:52 같은 learner 중간2,048 검산: 저속 출발 경험은 아직 없음

학습을재시작/변경하지않고실제main2905593/child2905600을확인하며대기했다.
19:52:06 완성된2,048tick/update쌍(기존1,000+추가1,048)의읽기전용검산이통과했다.
기존바이트prefix·현재정책update카운터·전이당update1·원래실제종료/bootstrap·
유한raw/학습지표·저장RNG로재구성한추가배치hash가일치했다.
추가main765/branch283,목표/실패지정slot각8,384회,실제종료는충돌16/기록종료2다.
추가목표종료는없으며기록종료를성공으로세지않는다.

추가구간의저속제안635개는OK279/운동한계176/예측충돌154/UNRESOLVED_SWEEP26이다.
양의가속도입력298개가있지만사례구성이다르므로앞선182전이와의성능개선으로해석하지않는다.
별도읽기전용검토에서main은episode_uid/tick,branch는previous_collection_id로
다음실제정책호출을대응시켰다. 속도를가속도입력으로추정하거나새로시뮬레이션하지않고
다음호출의actual anchor velocity를사용했다.

- 대응가능한저속후속호출619개:0.1m/s이상으로넘어간사례0.
- 관측된최대다음속도0.004567698m/s,최대속도증가0.000726395m/s.
- 종료때문에다음호출없는16개는별도제외(충돌14/기록종료2). 최종replay의next_obs전체검사로대체하지않는다.
- 양의가속도와후속호출을함께확인한296개도포함했다. 입력부호만으로출발성공을주장하지않는다.

로컬 `outputs/sac_terminal_retention_curve3000_20261120/live-prefix-002048/`:
`review.json` SHA `3cec589c365ad2886b8c2684316e1b6334bd556036c00b29bdeca213319bdeec`,
`successor-speed-review.json` SHA `7d5c35bfacf25e64619dbfe331aff52e2b65e5d749b59acf407efd77295eb167`.
이것은학습중정책의일부자기경험이며고정정책평가/최종저장감사/전체성능결과가아니다.
현재3,000학습을포함한11jobs와추가현재행동대조5jobs연결은유지하며결과를기다린다.

## 20:04 같은 정책 총3,000 학습·저장·종료Q·경험 검토 완료

총3,000 actor/Q update와8,000replay행(자기warmup5,000+online3,000)을저장했다.
기존1,000에서새실제자기전이2,000/전체update2,000을추가했으며BC/teacher0이다.
누적감사와CPU검토에서전체학습상태복원·원본6,000행17배열/이력prefix·모델/규약버전·
실제종료분류·모든배치hash/RNG를확인했다. 학습child2905600/감사2915923/
training-review2915970/experience-review2916232는정상완료했고주행평가로넘어갔다.

| 동일 자기warmup 종료행 Q MAE | source1,000 | continued3,000 |
| --- | ---: | ---: |
| 전체62행 | 2.082132 | 0.521524 |
| 목표6행 | 0.109951 | 0.205988 |
| 큰실패44행 | 2.606761 | 0.499291 |
| 기타종료12행 | 1.144582 | 0.760810 |

실패종료적합은개선됐지만목표종료오차는늘었다. 현재평균행동/앞선상태의미래비용이나
주행성능으로이표를대체하지않는다. 추가2,000전이는main1,348/branch652,
실제종료충돌32/기록종료2·새목표0이다. 전체누적main2,087/branch913,
누적목표1건은이전1,000에서이미수집한것이다.
추가저속1,139행의실제next_obs는모두유효였지만0.1m/s이상전환0,
최대다음속도0.004567698m/s다. 중간다음호출대조보다분모가큰최종replay검토에서도
출발경험부족이남았다. 이학습경험통계와고정정책평가는구분한다.

로컬 `outputs/sac_terminal_retention_curve3000_20261120/`의완료해시:

- `online3000/report.json`: `bd60c00d413f371d5fb47d60c57c36f0e8f7971793c25e92a63361711a9a8ea2`.
- learner: `94d62f1dd749a3673c98682990214bb4382d53bc485e76b713fd1a46b358467b`.
- replay: `5ee44e74b745b8bbaf071964b7de63575c72942a4e8765c19aaf57a2bbd495e3`.
- `audit-candidate3000.json`: `3d0cf1c377035be325f9c9bb246439a3d7735176fd27d01e5a3553fa9f0e76c1`.
- `training-review.json`: `e36849ae394b400d67c79860c2cca79fad6da33d066d7ad6d0725e06368cb528`.
- `experience-review.json`: `a9899b1825a197fcecd738e7fa450f8b3b410dea70c6129e4440a7123c6286ce`.

## 20:05 최초 완료 주행에서 고속 KEEP 성공 상실

같은개발장면8b4c1e7e09f5f4c3/KEEP의완료trace를먼저검산했다.
초기frame/map이source1,000과같고,실제tick/reward/거절/제동/종료를원본에서계산했다.

| 같은 고속 KEEP | source1,000 | continued3,000 |
| --- | ---: | ---: |
| 결과 | ROUTE_GOAL | COLLISION |
| tick | 72 | 67 |
| 거절 / 제동 / 정지 | 11 / 8 / 0 | 33 / 30 / 7 |
| reward 합 | 0.325933 | -20.145044 |
| 종료 속도 m/s | 11.029097 | 0 |

기존성공을잃은명확한회귀이며종료Q적합개선으로상쇄하지않는다.
현재모델을채택하거나학습예산을자동연장하지않고,남은동일8주행/실제미래보상/
현재행동과저장경험거리·실패분기검토를예정대로끝내실패원인을더분리한다.
20:05:44까지의한쌍검토이며나머지7후보사례/전체paired검토완료를의미하지않는다.
로컬 `first-high-keep-review.json` SHA
`94c40eb56b8a4cacec8a10de6a0df0a022064da0ce392927baa297ab50fe7552`.
main2905593/primary평가2916264/추가검토대기2910702실제생존확인.
원래학습/평가조건은유지했고새소스변경·추가학습은없다. 안정정책미확보다.

## 20:13 개발 8회 평가 완료: 성공 상실과 거절·제동 시간대

같은 4개 개발 장면의 KEEP/FOLLOW 8쌍을 대조했다. source1,000의 저장
평가를 hash 고정 재사용하고 continued3,000의 8회를 새로 실행했다.
초기 frame/map, 원본 tick·보상·종료·제동·정지 집계를 검산했다.
반복 사용한 개발 장면이며 독립 test나 8개 고유 장면이 아니다.

| 같은 개발 8회 | source1,000 | continued3,000 |
| --- | ---: | ---: |
| 목표 / 충돌 | 2 / 6 | 0 / 8 |
| 전체 tick | 564 | 559 |
| 거절 tick / 비율 | 170 / 30.14% | 175 / 31.31% |
| 제동 tick / 비율 | 140 / 24.82% | 157 / 28.09% |
| 정지 tick / 비율 | 340 / 60.28% | 356 / 63.69% |
| episode 평균 보상 합 | -15.263125 | -20.332767 |

정지 시작 4회는 두 모델 모두 328tick 전 구간 정지 후 충돌했다.
후보의 정지 장면 거절 수는126→84로 감소했지만 출발이나 성공으로 이어지지 않았다.
고속 KEEP/FOLLOW의 기존 두 성공을 잃었으므로 현재 후보는 채택하지 않는다.
참조 경로가 있는 PD 추종 표본은468→415, 위치 RMS는0.000271→0.000289m다.
EMERGENCY로 참조가 없는96→144tick은 이 RMS에서 제외했다.
작은 추종 오차가 경로의 안전성이나 성공을 의미하지 않는다.

고속 장면8b4c1e7e09f5f4c3의 원본 타임라인에서:

- source1,000 KEEP/FOLLOW: 최종 연속 거절은tick63의 PREDICTED_COLLISION부터,
  첫 제동tick65, 정지 없이tick72 ROUTE_GOAL.
- continued3,000 KEEP: tick36 PREDICTED_OFFROAD부터 최종 연속 거절,
  첫 제동38·첫 정지61·충돌67. 접촉 object4877은 뒤쪽에서 접근했다.
- continued3,000 FOLLOW: tick37–39 도로 이탈 예측 거절/제동 후tick40은 수락,
  tick41부터 다시 연속 거절, 첫 정지64·충돌72. 같은 object4877 접촉이다.
  첫 제동39가 최종 연속 거절41보다 앞서므로 단일 단조 과정으로 요약하지 않는다.
- 후보8회 모두 마지막 접촉 물체는 뒤쪽 접근으로 분류됐다. 시간 순서와 기하학적
  분류이며 사고 책임·충돌 불가피성·유일한 학습 원인으로 해석하지 않는다.
  후속 상태는 모델별로 달라 동일 상태 반사실적 비교가 아니다.

실행 산출물은 로컬 `outputs/sac_terminal_retention_curve3000_20261120/` 아래다.
`review_pilot.py`, `review_driving_details.py`, `render_comparison.py`는 연결된
runner에서 exit0 완료했고 별도 읽기 전용 `review_regression_timeline.py`도 완료했다.

- `pilot-review.json`: `20d69639992ad0a49391783ec74d58ad6da33b570b20a5a248a2ebee8b3dfce1`.
- `driving-details-review.json`: `f2a1e6250b9d83daa29e5d99c2067f1808462b1850ca22204d8de69c23943478`.
- `regression-timeline-review.json`: `67d6a1cbafa308daa4bf50dd611e0dc63c2485bcd1c7c9d8cd01505f5d3b7c23`.
- `visual-terminal-comparison/same-policy-1000-vs-3000-keep.gif`:
  `7491ecc2819d3f73ea0fd5faba9b644d70661363d3334480a45559d4b2efd862`.

GIF는1440×900·37frames, 시뮬레이션7.2초/마지막 hold 포함재생8.7초다.
생성 PNG를 직접 열어 두 모델 표시·제동 시간대·성공/정지충돌 표시를 확인했다.
같은 learner의 학습 예산 비교이며 Rule-based 최종 비교 영상이 아니다.

20:18경 실제 main2905593/현재행동 평가2919730/후속 대기2910702 생존 확인.
현재는 추가 학습이 아니라 고정 모델의 실제 종료까지 미래 보상을 평가 중이다.
그 뒤 현재 Gaussian과 저장 행동의 거리 및 같은 실패 상태의 실제 대안을 비교한다.
보상·안전검사·공통 PD·BC/teacher0와 동결 조건을 유지한다.
추가 예산·정책 채택·전체 개발/다른 seed/최종 Rule-based/별도 test는 미완료다.

## 20:21 같은 정책 3,000회 실제 미래 보상 검토 완료

원래 선택한 train 상태 KEEP77ae312df21e80b9/tick0와
FOLLOWbaac7893b025167a/tick4에서 현재 평균으로 첫 행동을 실행하고,
이후 원래 Gaussian의4seed(88019–88022)로 실제 종료까지 실행했다.
두 모델의 초기 관측 hash는 같고 각 모델의 첫 행동은 다르다.
실제 보상·gamma0.99 할인·첫 행동 이후 entropy·종료 bootstrap=false·
trace hash를 원본에서 검산했다. 두 모델 모두8/8충돌이다.

| 지표 | 1,000 KEEP | 3,000 KEEP | 1,000 FOLLOW | 3,000 FOLLOW |
| --- | ---: | ---: | ---: | ---: |
| 첫 평균 행동 Q | -1.451190 | -1.473330 | -0.946604 | -1.326877 |
| 실제 soft-return 표본평균 | -13.649348 | -13.643837 | -10.821256 | -10.651342 |
| Q - 표본평균 | 12.198158 | 12.170507 | 9.874652 | 9.324465 |
| 평균 종료 tick 수 | 38 | 40 | 59.25 | 63.5 |
| 할인된 실제 보상 평균 | -14.188298 | -13.950617 | -11.593023 | -11.098459 |
| 할인된 entropy 기여 평균 | 0.538950 | 0.306780 | 0.771767 | 0.447117 |

저장 실패 종료행 Q 오차 감소가 초기 현재 행동의 미래 비용까지 충분히 전달되지 않았다.
KEEP 차이는 거의 그대로이고 FOLLOW도9점 이상 남았다. 두 상태×4표본의 조건부
추정이며 정확한 기대 Q, 일반화, 독립 장면 수나 개선 유의성을 주장하지 않는다.
개발8주행의 성공 상실과 함께 판단해 총3,000 후보는 미채택이다.
현재 행동 평가/검산까지 원래11jobs는 COMPLETE_PENDING_DECISION으로 종료했고
실제 main2905593/평가2919730/최종검토2922310 종료를 확인했다.
새 모델/replay 쓰기 없이 후속 runner2910702의 coverage/실패 상태 대조가 시작됐다.

로컬 `outputs/sac_terminal_retention_curve3000_20261120/current-action-review.json`
SHA `a347094c4832b7c4436566d283ba7b24bb572b8905626bedc90544a666542508`.
재현 입구 `review_current_actions.py`, 실행 exit0. 앞선1,000 자료는 동결 재사용이다.

## 20:28 현재 행동과 저장 경험 대조 완료, 집계 오류 별도 복구

후속 coverage-control/branch-control/coverage-candidate/branch-candidate는 모두
exit0 완료했다. 마지막 reviewer는 CSV의 float32 차이를 Python float64로 다시
빼면서1e-10 허용오차를 적용해 실패했다. 원본 runner의 FAILED 상태·script·log는
보존했다. 두 CSV6,000/8,000행 모두 원래 float32 뺄셈으로는 정확히 일치했고
최대 float64 재계산 차이는4.76837158203125e-7이었다.

별도 로컬 `outputs/sac_terminal_policy_overlap_20261120/float32-review-recovery/`에
`failure-analysis.json`, 수정한 `review_overlap.py`, `review.log`, `review.json`을
저장했다. 해당 검사만 원래 float32 계산의 정확한 일치로 고쳐 exit0 완료했다.
허용오차 확대·원본 결과 수정·학습/시뮬레이션 재실행은 없었다.
원래90소스/69입력 및 두 모델/report/replay의 동결 hash를 다시 확인했다.
복구 review SHA `7a86479ce4a26ade1b4ae2b89e9ebecb2e30c802ecbb6c2a08a35340346e2d25`.

| 같은 실제 실패 직전 상태 Q−실제 보상 | 1,000 저장 행동 | 1,000 현재 평균 | 3,000 저장 행동 | 3,000 현재 평균 |
| --- | ---: | ---: | ---: | ---: |
| KEEP77ae312df21e80b9/tick28 | 1.501171 | 5.508746 | -0.005691 | 0.797901 |
| FOLLOWbaac7893b025167a/tick59 | 7.745564 | 10.001662 | 2.008344 | 8.556472 |

각 상태/모델에서 저장 행동·현재 평균·원래 Gaussian4표본의 총6개 실제 대안을
실행했다. 모두 충돌했으며 각 상태 안에서 실제 보상·bootstrap·다음 관측 hash가
정확히 같았다. 따라서 이 두 상태에서는 Q−보상이 실제 종료 오차다.
특히3,000 FOLLOW는 같은 실제 결과인데 현재 평균을 저장 행동보다6.548127점
좋게 평가했다. 현재 Gaussian4표본도 오차8.539423–8.564999로 같은 문제가 남았다.
KEEP는 현재 행동 오차도 개선됐다. 모든 실패 상태의 같은 현상이나 유일 원인으로
확대하지 않는다. 고정 종료행 평균 적합만으로 현재 정책 평가를 대체할 수 없다.

같은 과거 관측에서 저장 행동의 현재 Gaussian95% 타원체 포함 비율:

- 공통 자기warmup5,000행:30.6%→30.1%.
- 공통 source online1,000행:37.5%→36.4%.
- 신규2,000행:35.2%. 새 장면18개·저속1,139행이 포함됐다.
- 각 모델 마지막100행:21%→69%. 서로 다른 관측/장면이며 직접 개선 대조가 아니다.

타원체는 조건부 거리 진단이다. Gaussian의 실제 support나 통계적 신뢰구간,
현재 행동 경험의 충분성·안전성·정책 성능을 뜻하지 않는다.
두 모델의 batch/RNG 및 공통6,000행 메타데이터·CSV계산·실제 대안 결과를 검산했다.

### 다음 보완의 규약 대조: 현재 Gaussian으로 실패 직전 상태 재수집

**구현 전 제안**이다. 기존 complete-path 수집은 과거 실패의 앞선 상태에서 한 경로를
끝까지 이어 수집한다. 저장한 종료 행동을 반복 추출해도 업데이트된 현재 행동의
실제 결과를 추가한 것은 아니다. 확인된 FOLLOW 차이를 겨냥해, 실제 train 주행에서
획득한 실패 직전 상태를 보존하고 최신 정책의 원래 Gaussian으로 다시 실행하는
수집 옵션을 다음 구현 대상으로 검토한다.

- 실제 실행한 전이만 원래 보상·종료·bootstrap·next_obs로 저장한다. 대안이 같은
  충돌을 일으킬 것으로 가정해 라벨을 복사하거나 가짜 실패를 만들지 않는다.
- 확률 행동을 평균 행동으로 교체하거나 유리한 표본만 고르지 않는다. 원래
  mu/g/B/sigma/ell과 관측·실행 계약을 유지한다. BC/teacher replay는0이다.
- 기존 main/완료 경로 수집을 보존하면서 재수집 비중을 사전에 제한한다.
  최신 자기 행동 경험의 추가와 replay 재추출은 별도로 계수한다.
- 새 실제 전이당 전체 SAC update1, 같은 batch256·손실·lr·tau·보상·검사·PD 유지.
  v1.2 5절의 Gaussian 행동,10절의 전이당update1,종료/수집 경계 계약에 대조한다.
- 실패 상태 편중과 성공 상실을 더 키울 수 있으므로 실제 종료 재현·이력·배치 검사와
  같은 예산의 기존 수집 대조가 필요하다. 반복 상태를 독립 장면으로 세지 않는다.

이 제안은 기본 설정 채택이나 성능 입증이 아니다. 아직 코드 구현/새 학습 실행0이며
구체적인 비중·비교 예산은 구현 검토 후 고정한다. 최종 주행 개선·전체 개발 validation·
다른 seed·동일 조건 Rule-based·별도 평가는 계속 미완료다.

## 20:37 실패 직전 상태 재수집 구현과 fresh100 실제 검사 시작

앞선 FOLLOW 현재 행동의 실제 종료 Q 오차를 겨냥한 별도 수집 옵션을 구현했다.
`scripts/sac_terminal_refresh.py`는 실제 main/완료 경로의 충돌·이탈 직전 상태를
FIFO8개까지 보존하고, 기존 수집 전이4개마다 최대1개를 순환 선택한다.
현재 actor의 원래 Gaussian으로 실제 실행하며 재수집 실패는 bank에 재투입하지 않는다.
행동을 골라 버리거나 과거 보상을 복사하지 않는다. 현재 정책의 실제 대안이 비종료면
원래 bootstrap과 next_obs를 그대로 보존한다. 최대 재수집 비중은 새 전이의20%다.

`scripts/diagnose_sac_terminal_refresh.py`는 기존 fresh terminal-retention driver와
분리했다. 기존 complete-path와 목표8+실패8 sampler,모델·초기화·배율·원래Gaussian·
보상·검사·PD·전이당전체SAC update1·BCteacher0 유지. 원래 경로의 collection_id를
중간 재수집 ID와 분리해 successor를 실제 이전 경로 전이에 연결한다.
continuation은 거절하고 최초 자기warmup5,000만 재사용한다.

`scripts/audit_sac_refresh_replay.py`와 `scripts/audit_sac_terminal_refresh.py`는
실제 parent/next_obs 배열·실패source와관측fingerprint·FIFO/주기/순환·train분할·
원본warmup17배열·종료pool·전배치hash/RNG·전체actor/Q/BC카운터를 검산한다.
`scripts/verify_sac_terminal_refresh_execution.py`는 명령/source종류별 첫 재수집을
선택해 실제 원본 경로를 재현한 뒤 저장 행동과 재수집 행동을 각각 실행하고
관측·보상·검사·제어·종료·다음관측 및 부모 불변성을 대조한다.

실행한 사전 검사:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest -q \
  tests/training/test_sac_terminal_refresh.py \
  tests/training/test_sac_failure_complete_path.py \
  tests/training/test_sac_terminal_retention_sampling.py
git diff --check
```

33passed. 새 bank7개와 기존 경로/배치 검사를 포함하며 실제 GPU/Waymax·주행 성능은
아직 대체하지 않는다. 새 파일 문법 검사도 통과했다. 과거 동결 소스와 src/*는 수정하지 않았다.

로컬 `outputs/sac_terminal_refresh_pilot100_20261120/`에 seed0·fresh100회씩의
후보/기존 수집 대조를 연결했다. 같은 자기warmup5,000, B-float64/raw mean gain0.0001,
physical_whitened Q/zero-action init, 목표8+실패8배치, samples1/window6/
moving_predecessors/horizon200/complete_path다. 기존1,000/3,000/critic-only 가중치를
가져오지 않고 같은 초기 learner에서 각각 시작한다. 다른 맥락 파일의 해시가 입력
목록에 포함돼 있으나 학습 초기화에는 checkpoint의 자기warmup prefix만 사용한다.

runner2929909(session39768),학습child2929929 실제생존 확인. 최초 새 실제전이3개와
update3개가 기록됐다. 61소스/22입력 및61소스복사본 동결,
decision SHA `7e07134a4ab05404d75b29ec2c5738e4a58ad75ff91567b46dc683f03f3a3594`.
7jobs: 후보100→저장감사→실제재현→기존100→저장감사→기존4주행→후보4주행.
주행은 같은2개 개발장면8b4c1e7e09f5f4c3/8a4fba3463c6c62d×KEEP/FOLLOW이며
독립test가 아니다. 예산/입력/소스 실패 시 중단하고1000회나 채택으로 자동 확대하지 않는다.
현재 실제 재수집·완료 저장·재현·주행 결과는 미완료이고 전체 목표도 미완료다.

## 20:43 후보100 저장·실제 재현 완료, 기존100 대조도 완료

후보100 실제전이/전체SAC update100·replay5,100 저장과 새 감사가 통과했다.
main57/failure_branch29/terminal_refresh14이며 refresh14개 모두 실제COLLISION이다.
실제 원본 source ID는29(main)와65(failure_branch) 두 개이고 둘 다 같은 train
KEEP77ae312df21e80b9/tick28이다. 독립 장면2개나 FOLLOW 검증으로 세지 않는다.
14개 source관측이 replay의 실제 parent관측과 정확히 같고 FIFO/순환/4base주기,
warmup17배열·종료pool·배치hash/RNG·actor/Q100·BC0를 검산했다.

명령/source종류별 첫 재수집을 실제로 재현했다. 각28step의 원래 경로를 따라
source29→refresh33과 source65→refresh73을 각각 실행했다. 원래 행동과 새 행동 모두
실제충돌·보상-20.014였고 관측/mask·검사·제어·종료/다음관측이 기록과 일치했다.
부모 snapshot 불변성도 확인했다. 새 행동의 raw값은 원래 저장 행동과 다르다.
실제 비종료 refresh는 이번100개에서 관찰되지 않았으며 이를 실제 검증 완료로
확대하지 않는다. FOLLOW와 정책 성능도 아직 검증하지 않았다.

로컬 pilot100루트의 완료 산출물:

- `candidate100/report.json`: `ba4ed8b2ddf59c4eba9ed2d2675569c9b245fee7c75064ec17a00a83e8d304cc`.
- `audit-candidate.json`: `e005430e2730330f790882ba00d425505a1560d8390faf61a139479d1837f610`.
- `execution-review.json`: `5019961df2768e418a9d48054dd42ba6a361283f7f4b358ef629d10b9f0444dd`.

후보 학습/감사/재현 child2929929/2934626/2934659는exit0 종료했다.
20:46:30 기존수집control100/감사도exit0 완료했고 같은개발4주행이실행중이다(child2938828).
main2929909는계속생존,다음후보4주행까지원래7jobs를유지한다.

최종 두모델의 동일초기설정·warmup17배열·첫refresh전32전이/배치/관측·실제4+4주행
원본지표/초기frame/map을 검산하는 읽기전용 후속2jobs를 연결했다.
로컬 `post-review/` runner2939148(session6334),WAITING_FOR_PILOT.
원래runner의성공종료와실제PID소멸을함께확인한뒤에만실행하며학습/예산확대는없다.
64소스/23입력과복사본동결,post decision SHA
`79b42b1f2478296c7622be541e406b856f4cddda2dec5a72f6504567ed39d9ab`.
모델/replay/report는완료후추가동결한다. 현재전체성능·안정정책은아직미확보다.

## 20:53 fresh100 주행 완료와 재수집 전 수치 차이

원래7jobs는 COMPLETE_PENDING_DECISION으로 정상 종료했다. 그러나 후속
`review_training.py`의 재수집 전32행 수치동일성 검사는 실패했다.
원본 `post-review/status.json`의FAILED,script/log를 보존했으며 원인을 확인하지 않고
허용오차를 늘려 통과시키지 않았다. 실제 재수집은collection33에서 시작하지만 첫
gradient 지표 차이는update1부터 나타났고 raw행동은collection2부터 달랐다.

첫학습입력배열(원본warmup+첫실제전이,학습에쓰지않는ID제외)은바이트동일하고
32회배치index hash도같았다. 그러나 실제초기learner바이트는기록하지않았으므로
seed/설정동일성을초기상태바이트동일성으로확대하지않는다. 차이의유일원인은미확정이며
모든차이가수치비결정성때문이라고확정하거나재수집효과로설명하지않는다.

- 재수집전32행 raw_u 최대차이1.5943615795947687e-7,서로다른행31개.
- obs 최대차이6.351619958877563e-7,서로다른행2개(collection31/32).
- next_obs 최대차이같음,서로다른행3개(collection30–32).
- 보상최대차이5.169392341269896e-9,서로다른행3개.
- mask·검사판정/이유·source·종료·bootstrap·장면/tick·배치인덱스는일치했다.

별도 `prefix-difference-review/`에서 차이를 그대로 보고하는 검토를 수행했다.
`training-review.json`은COMPLETE_WITH_NONIDENTICAL_PREFIX,
`pre_refresh_equivalence_passed=false`이며원래검사실패를성공으로바꾸지않았다.
두실험의동일설정·warmup17배열과실제실행은검증됐지만엄밀한동일경로반사실적비교는아니다.
별도주행검토는4+4원본trace/초기frame/map/집계를검산한기술적비교다.

| 같은두개발장면×KEEP/FOLLOW | 기존수집100 | 재수집후보100 |
| --- | ---: | ---: |
| 목표 / 충돌 | 2 / 2 | 2 / 2 |
| 전체tick | 218 | 218 |
| 거절 / 제동 / 정지tick | 58 / 46 / 22 | 58 / 46 / 22 |
| 평균보상합 | -10.0160922573 | -10.0160973947 |

두모델모두고속KEEP/FOLLOW72tick목표,저속KEEP/FOLLOW37tick충돌이었다.
주행개선을관찰하지못했고작은보상차이를효과로해석하지않는다. 후보재수집은한train
KEEP장면의14전이뿐이며FOLLOW수집/정지출발/전체개발/다른seed/독립평가검증이아니다.

로컬 `outputs/sac_terminal_refresh_pilot100_20261120/prefix-difference-review/`:

- `difference-analysis.json`: `81feb1a14ecf1914f7b1e616c8c2c36e88a0d6be89010836f0acd89fcaecd718`.
- `training-review.json`: `cf485e22b9b0bdb47649ffb1308f8d42b448d61fd34d7e93cd032d73d1b70f9d`.
- `pilot-review.json`: `3ed4b74e4d39529166ba7c6e6c15814ea701ccb64387db8b39acd254d600ed9e`.

main2929909/post2939148/평가2940353은종료했다. 원래61소스/22입력 및후속64소스/
23입력·동결모델/report/replay hash를재확인했다. 원본후속FAILED와별도비동일성기술적
검토완료를구분한다. 이차이분석에학습/시뮬레이션재실행이나모델/replay수정은없었다.

다음은새refresh이력에도현재행동Q/거리·실제미래보상검증도구가올바르게작동하도록
연결하고,모델초기화/수치동일성검증의범위를명확히한뒤FOLLOW를포함하는같은예산의
더긴비교를준비하는것이다. 아직다음학습은시작하지않았고이100모델은채택하지않는다.
보상·검사·PD·Gaussian·순수SAC조건과원래전체목표는유지한다.

## 21:05 — 재수집 이력의 현재 정책 Q·실제 return 진단 연결

`feature/yuntae-sac-training`, 기준 HEAD `4fa07e754c2a9e076c836a7a06ee92102ab20db6`.
새 진단 4파일과 테스트 1파일을 추가했다. 기존 학습/환경/동결 스크립트는 변경하지 않았다.

- `scripts/sac_collection_diagnostic_audit.py`: 수집 scope에 맞는 실제 lineage 검사를
  선택한다. refresh 이력을 기존 수집으로 잘못 표기하거나 알 수 없는 scope이면 거절한다.
- `scripts/diagnose_sac_refresh_branch_q.py`: 평가할 모델과 실패 상태의 원본 이력을
  별도 인자로 받는다. 원본의 seed/실제 행동으로 상태를 재현하고, 현재 모델의 저장행동 Q,
  평균/원래 Gaussian 4표본의 Q와 실제 종료 보상을 비교한다. 다른 모델의 상태를 모델
  자신의 replay로 오인하지 않도록 report/replay SHA를 각각 남긴다.
- `scripts/diagnose_sac_refresh_soft_returns.py`: 같은 감사 어댑터를 사용해 기존 고정
  선행 상태에서 평균 첫 행동과 확률적 후속 행동의 실제 종료까지 할인 보상·entropy를 계산한다.
- `scripts/audit_sac_refresh_coverage.py`: 새 lineage와 실제 추출 횟수를 검산하고
  refresh/main/완료경로를 분리한다. Gaussian 타원체 거리는 안전성이나 경험 충분성 증명이 아니다.

CPU 관련 pytest **36개 통과**, 새 4파일 py_compile 통과. 실행 명령:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest -q \
  tests/training/test_sac_collection_diagnostic_audit.py \
  tests/training/test_sac_terminal_refresh.py \
  tests/training/test_sac_terminal_retention_sampling.py
python outputs/sac_terminal_refresh_diagnostics100_20261120/run_diagnostics.py
```

새 로컬 루트 `outputs/sac_terminal_refresh_diagnostics100_20261120/`는 완료한
fresh100 두 모델을 읽기 전용으로 검토한다. 21:05:23 runner PID2944744/session79319 시작,
68소스/43입력 및 소스 복사본 동결. decision SHA
`158e1dfd7ec47e8e714da70ec8a27b990cb9813baac46dc5ec6e92d8aa312d0c`.
7 jobs는 저장감사 → 양쪽 coverage → 양쪽 실제 branch Q → 양쪽 실제 soft return이다.
두 모델/원본 branch1000/원본 predecessor1000의 실제 저장감사는 exit0 완료했다.
GPU 진단은 실행 중이며 최종 결과는 아직 없다.

실패 상태는 이전부터 고정한 train KEEP77ae312df21e80b9/tick28와
FOLLOWbaac7893b025167a/tick59, 각 저장행동·현재평균·확률4개다.
미래 return은 별도로 고정한 같은 두 train 장면의 KEEP tick0/FOLLOW tick4에서
각 4개 후속 RNG seed88019–88022로 종료까지 실행한다. 모델마다 8회이며
독립 장면 8개나 validation/test로 세지 않는다.

추가 학습/가중치·replay 변경은 없다. BC=0/teacher replay=0·보상·검사·PD·원래
Gaussian 유지. 이전 재수집 전 엄격 수치 동일성 실패는 그대로 보존하며,
이번 결과도 동일 설정의 기술적 비교이지 완전히 동일한 학습 경로의 인과적 효과 증명이 아니다.
현재100 모델 미채택·안정 정책 미확보. 이 결과를 검토한 뒤 FOLLOW까지 실제 재수집되는
더 긴 동일 예산 비교를 결정한다. 전체 개발 validation/다른 seed/최종 Rule-based/별도 평가 미완료.

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

## 21:11 — 다음 대조를 위한 실제 초기 learner 기록 준비

이전 fresh100에서는 실제 초기 learner 바이트를 저장하지 않아 초기화 차이를
배제할 수 없었다. 다음 실행에 사용할 별도 진입점
`scripts/run_sac_initial_capture.py`와 기록 함수 `scripts/sac_initial_state_capture.py`를
추가했다. 기존 두 fresh driver/source는 변경하지 않는다.

wrapper는 첫 실제 전이가 수집된 후 **첫 full SAC update에 전달되는 state**를
직렬화해 `initial-learner.msgpack`과 `initial-state-audit.json`으로 기록한다.
원래 update 함수에 같은 state/batch/인자를 전달하고 반환값도 그대로 돌려준다.
옵션 `--expect-initial`이 있으면 원본 바이트를 직접 비교하고 불일치 시 첫 update 전에
실패한다. BC 또는 이미 업데이트된 learner는 거절한다. 모델 복원이나 초기화 변경은 없다.
이 기록은 이후 GPU 연산/학습 경로의 동일성 보장이 아니며 과거 prefix 실패를 해소하지 않는다.

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest -q \
  tests/training/test_sac_initial_state_capture.py
python scripts/run_sac_initial_capture.py --driver retention -- --checkpoint ... --output ...
python scripts/run_sac_initial_capture.py --driver refresh \
  --expect-initial CONTROL/initial-learner.msgpack -- --checkpoint ... --output ...
```

새 테스트4개 통과: 입력/출력 정체성, 최초 1회만 기록, 바이트 불일치 시 update 차단과
실패 원본 보존, nonfresh/BC 거절. py_compile/CLI help 통과. 위 실행 예시는 필수 driver
인자를 생략한 사용 형식이며 실제 학습 실행 명령이 아니다.
**GPU 실제 초기 상태 기록 및 두 실행 바이트 비교는 아직 미실행**이다.
현재 read-only 진단 runner2944744/candidate-branch-Q2948776은21:11:11 실제 생존했다.
다음 학습 예산/실행은 아직 결정하지 않았으며 원래 전체 목표는 미완료다.

## 21:13 기록 — fresh100 실제 실패 행동 Q·수집 노출 검산 완료

양쪽 실제 branch Q와 coverage가 exit0 완료됐다. 같은 train 두 실패 직전 상태에서
모델마다 저장/평균/Gaussian4표본을 실행한 **총24행동이 모두 충돌**했다.
각 상태/모델 안에서 다음 관측 hash도 같았으며 Q−실제 종료 보상을 직접 검산했다.

| 현재 행동의 Q−실제 종료 보상 | 기존100 | 재수집100 |
| --- | ---: | ---: |
| KEEP 현재 평균 | 16.2476106300 | 16.0122690811 |
| FOLLOW 현재 평균 | 16.1541376514 | 16.0623581810 |
| KEEP 저장 행동 | 16.2361374512 | 16.0089984550 |
| FOLLOW 저장 행동 | 16.1488423748 | 16.0495407982 |

작은 오차 감소는 보이나 두 모델 모두 약16점 과대평가가 남았다. 현재100 예산의 실패
근거이며 수집 방식의 수렴 가능성/장기 성능을 확정하지 않는다. 기존 prefix 비동일성도 유지한다.

새 refresh14행의 학습 batch 추출은 총303회/전체25,600 draw다(복원 추출 포함).
최종 정책의 같은 관측에서 95% Gaussian 타원체 안에 드는 것은1/14행이고,
그 마지막 행(replay5097)은 실제 학습 batch에0회 등장했다. 나머지13행은 이미 달라진
최종 정책과의 거리가 컸다. 이것은 **실제 경험 추가와 현재 행동에 대한 학습 노출을
구분해야 한다는 근거**다. Gaussian support 부재·안전성·유일 실패 원인으로 확대하지 않는다.
같은 warmup5,000행의 타원체 포함은 양쪽32.4%; 각자의 online100은49%/47%로
서로 다른 상태 집합임을 유지한다. CSV각5,100행의 q_gap은 원래float32 뺄셈과 정확히
일치했고, 추출횟수25,600과68소스/43입력 hash도 재검산했다.

로컬 `outputs/sac_terminal_refresh_diagnostics100_20261120/branch-coverage-review.json`
SHA `7ffc7f3bd79c269d4f2ddddcfc702e9133abeabaa6a1925150c5040fc7b2ebd3`.
범위는 `PARTIAL_BRANCH_AND_COVERAGE_REVIEW_ONLY`이며 미래 return 전체 검산은 미완료다.
runner2944744는 유지되고 후속 control-soft-returns2950139가 실행 중이다.
다음은 양쪽 실제 종료 return 완료/검산 후 FOLLOW 재수집까지 포함한 다음 예산을 결정한다.
초기 상태 기록 코드는 `f2ef2a1198d718dc397fb9487523861049bbf259`에 커밋·push/원격SHA확인했으며
실제 GPU 초기 기록 비교는 아직 미실행이다. 새 PR/병합/모델 채택은 없다.

21:15 후속 검산 연결: 로컬 같은 루트의 `post-review/run_review.py`를
PID2951779/session87680으로 시작했다. 실제 생존 중인 main2944744의 성공 완료와
PID 소멸을 확인한 후 `review_results.py`만 실행한다. 후속은 초기
`WAITING_FOR_DIAGNOSTICS`이며 학습이나 추가 예산을 자동 실행하지 않는다.
reviewer SHA `1a4d01e7b981cb46f4b7734da680e0ad72cf7f6b890869f8f03119f0575aba72`.
원래68소스/43입력 불변, 두 모델의 동일 상태 출처, 실제24행동의 종료 오차,
각 CSV5,100행의 원래float32 q_gap/추출횟수, 총16확률주행의 실제 종료·원본
trace hash·할인 보상·entropy 산술을 검산한다. 검증 실패는 원본 보존 후 중단한다.
예정 결과는 `diagnostic-review.json`; 아직 완료 결과가 아니다.

## 21:28 — fresh100 전체 진단 완료, FOLLOW 포함 fresh1,000 대조 시작

읽기 전용7jobs와후속산술검산이모두exit0 완료했다. 원래main2944744는21:27:21,
post-review2951779는21:28:00 완료됐고실제PID소멸을확인했다. 두train선행상태×4seed,
모델당8주행은 **기존/재수집모두8/8 COLLISION** 이다. 완료원본을검산한결과:

| 미래 보상 진단 | 기존100 | 재수집100 |
| --- | ---: | ---: |
| KEEP 평균 실제 soft return | -13.9566173152 | -13.9561767645 |
| KEEP 첫평균행동 Q−평균실제return | 13.4954274193 | 13.5074215002 |
| FOLLOW 평균 실제 soft return | -10.7077526953 | -10.7071619887 |
| FOLLOW 첫평균행동 Q−평균실제return | 10.2250729110 | 10.1365617175 |

작은차이를주행개선으로보지않는다. 현재100 모델은미채택이며,앞선24행동실제충돌과
개발4주행양쪽2목표2충돌결과도유지한다. 초기평균행동이후원래Gaussian4seed의실현
return표본이며정확한기대Q나독립test가아니다. 기존prefix수치동일성실패도보존한다.

최종 로컬 `outputs/sac_terminal_refresh_diagnostics100_20261120/diagnostic-review.json`
SHA `cc4478b25b4b4b11bb1496919304e478ba3ef19446c8c658d18418f73113209b`.
68소스/43입력불변·같은상태출처·CSV각5,100행/25,600draw·24행동종료오차·총16주행
trace hash/실제종료/할인보상/entropy합을검산했다. control return report SHA
`f9198f4c282c02cb748aebb1d54de7340ed364dd29697f01fdd317c85891ffde`,candidate
`275d8e2e33439b75dbe4baed34e3cd3321350fe8f6aa6d96cee89c4185eab33b`.

이결과와KEEP에만국한된14refresh/303학습draw를검토하고,수집방식이FOLLOW까지
실제로작동하고더학습했을때가치/주행에영향을주는지 **각fresh1,000회** 고정대조를
선택했다. 100회실패를수렴불가능으로단정하거나100회모델을채택하는결정이아니다.
배치/재수집비중/보상/검사/PD/모델을추가변경하지않는다.

- 로컬새루트 `outputs/sac_terminal_refresh_paired1000_20261120/`.
  실행은 `python outputs/sac_terminal_refresh_paired1000_20261120/run_trial.py`.
- seed0·동일자기warmup5,000·각fresh learner·각새전이1,000/fullSAC update1,000.
  이전100/1,000/3,000가중치를이어받거나학습량을합산하지않는다.
- B-f64/raw mean gain0.0001/physical_whitened Q/zero-action critic init·원래Gaussian·
  complete-path samples1/window6/moving_predecessors/horizon200·목표8+실패8·BCteacher0 유지.
  후보만FIFO8/base4당refresh1/재귀0의기존옵션을사용한다. 기본채택아님.
- 첫fullSAC update입력의전체learner를각각저장한다. 후보는새control의초기바이트와
  일치해야첫update를진행한다. 이후GPU/수집prefix수치차이는측정해서보고하며
  과거엄격동일성실패를바꾸거나후속경로동일성을가정하지않는다.
- 11jobs:control1,000→감사→candidate1,000→감사→실제source재현→학습검토→
  양쪽primary/정지출발각4주행→원본주행검산. 총4개발장면×K/F×2모델=16주행.
  명령별실제refresh와실제재현이모두있어야FOLLOW수집검증으로기록한다.
- 기존개발장면8b4c1e7e09f5f4c3/8a4fba3463c6c62d,정지출발8dd5939855c15b3f/
  a62f5a7d2cd9b3d9 유지. 전체개발validation/새seed/heldout가아니다.
- 성공상실·충돌·거절·제동·정지·보상을함께판단한다. 초기상태/입력hash/실제lineage/
  비유한값등실패는원본보존후중단한다. 1,000이후자동예산확대/기본채택없음.

74소스/50입력/소스복사본동결,decision SHA
`907873d784959c6de19f27239fde3c5f82e80d5ce095d839a9c17d6f9d5241d0`.
21:28:55 runner2955496/session72760와control2955503 실제생존확인.
기존진단소스/현재src변경0,실행기준HEAD45f8d447f0e6601a704d55464a9ad468bfb272b4,
초기기록wrapper구현f2ef2a1198d718dc397fb9487523861049bbf259.
후속실제control GPU초기기록은COMPLETE,learner SHA
`7363d29a0c9500d58a259ba8953ff21428ede1660ee9e66be9b9f571ef014c2c`.
이것은단일초기기록확인이고두실행바이트대조/1,000회완료/주행성능검증은아직미완료다.
안정정책·전체개발·다른seed·최종Rule-based·별도평가의원래목표는계속미완료다.

## 21:40 — paired1,000 이후 현재 행동 진단 연결

현재 main2955496/control2955503은실행중이다.21:40:26 보고서570새전이/fullSAC update570,
실제trace KEEP326/FOLLOW244를확인했다. 후보1,000은기존수집완료후실행되며아직미시작이다.
기존11jobs와모델/수집/학습소스는변경하지않았다.

읽기전용후속을로컬
`outputs/sac_terminal_refresh_paired1000_20261120/policy-diagnostics/`에연결했다.
runner2959705/session65181은`WAITING_FOR_PAIRED1000`,현재실제생존이다.
원래main PID2955496과process start ticks73667462를함께확인하며,main의학습/감사/
실제재현/16개발주행검산성공과원래process종료후에만시작한다. PID재사용을원래작업의
생존으로오인하지않는다. 부모실패/조기종료는기록후중단하며자동재시작/추가학습없음.

76소스/51기존입력과소스복사본을동결했다. decision SHA
`58acb1fa492ec827021f7a7b28bd333f72c8b71dca8c2c48d27b5e2229e1b08a`.
완료후새모델·replay·초기state·실제ticks/update·부모감사결과를추가로hash동결하고,
첫진단전과각job사이에대조한다. 아직미완료인모델을확정hash로표시하지않았다.

7jobs는양쪽coverage→양쪽같은실패상태Q→양쪽같은선행상태soft return→산술검산이다.
첫100회에서실제검증한진단소스를그대로사용한다. 후속집계만각6,000 replay행과
256,000 draw에맞췄으며원래float32차이계산/원본trace검산/4seed를유지했다.
같은train실패KEEP tick28/FOLLOW tick59의저장/평균/확률4행동,
선행KEEP tick0/FOLLOW tick4의seed88019–88022각4주행이다. 신규독립test가아니다.
후속runtime/집계py_compile와실제대기시작은확인했지만GPU진단결과는아직없다.
예정결과는후속루트`diagnostic-review.json`,실행상태는`status.json`이다.

## 21:40 — 별도 평가 입력 재검산과 외부 조회 제약

기존7,040프로젝트hash-test후보의전수검사보고서·manifest·32개도로후보cache SHA를
새로대조했다. 결과는29고유장면/56명령쌍(KEEP29/FOLLOW27),목표32에3개부족,
기존데이터오류17건으로변함없다. 새policy실행/원본전체재스캔/environment reset은0이다.
로컬 `outputs/sac_heldout_availability_20261120/reset-inventory-review.json`은이전결과와
바이트동일한SHA `a5105287234e6a5564e6e227b7326b24346176d7482f000b67b474f27e4733c0`다.
기존1,000raw파일/7,040후보전체를이미검사했으므로같은후보풀을반복검색해서
새로운적격3개가확보된다고가정하지않는다.

추가원본목록을조회하려했으나`gcloud storage ls`가인증갱신실패로종료했다.
자격증명없는공개GCS JSON목록조회도HTTP401을반환했다. 따라서현재버킷의추가
20초prefix존재/부재를확인한것이아니다. 사용자에게`gcloud auth login`재인증을
비동기로요청했으며,현재로컬SAC학습은계속진행한다. 응답/인증복구없이조회성공으로
기록하거나다른계정/자격증명을시도하지않는다.

[공식Motion설명](https://www.waymo.jp/intl/jp/open/data/motion/)은일반training/validation을
10Hz의91sample window로설명하고,시간은장면내0부터/좌표원점은장면마다달라진다고명시한다.
따라서일반validation을기존20초cohort에그대로혼합하거나이정보로drive독립성을선언하지않는다.
[공식Scenario proto](https://github.com/waymo-research/waymo-open-dataset/blob/master/src/waymo_open_dataset/protos/scenario.proto)
GitHub blob SHA `22b1c3c55cad11fe6601261eea603dc7bcdf0ec2`와실제로사용하는vendored descriptor의
노출필드에는별도drive/log식별자가없었다. 이는외부매핑정보가없다는증명이아니다.
[공식저장소이슈720](https://github.com/waymo-research/waymo-open-dataset/issues/720)의열람내용도
후속training_20s/validation중복질문을해결하지않았다.
최초외부패키지이름의직접import는ModuleNotFoundError였으며,
실제`src.scenarios.vendor.waymo_open_dataset.protos`를사용해descriptor를확인했다.
의존성설치/환경변경은없었다.

로컬 `availability-review.json` SHA
`b9aed0dc267ff01f75df511c9525c83f3e031fa0a49abfdf430d15906dc37c98`에현재재검산범위,
공식근거/조회실패/남은확인을분리했다. 추가원본또는검증가능한매핑이확보되기전까지
29/32·17오류·drive/시간독립성미확인상태를유지한다. 목표개수를채우려고train/dev를
test로바꾸거나오류배열을자르거나ODD/reset/안전검사를완화하지않는다.

## 21:52 — 기존1,000 완료, 두 실제 초기 learner 동일성 확인

기존수집 control1,000과 저장 감사가 exit0 완료됐다. 새 전이1,000개에 actor/Q 각각
1,000 update, BC0, 자기 warmup17배열 보존, 실제 lineage·종료pool·전체256,000 batch draw와
RNG를 검산했다. main739/완료경로261,실제 신규목표종료1/실패종료14다.
수집 중 목표1건은 평가 성공률이 아니며, 개발 주행 평가는 아직 실행 전이다.

로컬 `outputs/sac_terminal_refresh_paired1000_20261120/`:

- control1000/report SHA `83d7617463a4dfc5233f510eccb4f1cf062be8eda0da8d4624e7c94e5fc10bcf`.
- audit-control SHA `62d4acd195f41d5fe6de85a046a454d928c523fcc3de7ca3110849adcb0245a0`.
- 완료 learner SHA `8f2684693274526393030eb529b3879fa94df682ac220d4419f660d0a61b233d`.
- 완료 replay SHA `c4c9a585c637e7e0409e3ef549148b489014973e56b69a340105d496edd81fef`.

후보2962566이 실행됐고, 두 모델의 **첫 full SAC update 직전 전체 learner가 실제 바이트
단위로 같음**을 확인했다. initial SHA는양쪽
`7363d29a0c9500d58a259ba8953ff21428ede1660ee9e66be9b9f571ef014c2c`다.
후보 initial-state-audit SHA `d855f97b456d4c418f39b920fd461f7d2a0d97f36045a7a4d91046d616d15aab`,
reference_bytes_equal=true. 21:52:38 후보181 update/refresh30이며,현재관찰한refresh는
모두KEEP/실제충돌이다. FOLLOW수집·최종저장감사·주행개선으로확대하지않는다.

실제 초기 파일을 읽어 mu head kernel/bias0, rho kernel0와 float32(log1.5) bias,
actor/Q/전체update/BC카운터0을 따로 검산했다. 저장 rho에서 계산한 sigma는
0.019999999928762665,log_alpha는−6.907755374908447이다. 기준서5절의평균head0
초기화를유지하며임의가속bias를추가하지않았다. 로컬 `initial-control-review.json`
SHA `c2554adfdb696121705cfd55c20a664dabf84b024d2f5c38b6f24d6fd7255b1f`.
이는실제초기파일검사이며새모델재구성/정책성능검사가아니다.

### 초기 동일성과 후속 수치 차이는 구분

최초refresh는collection33이다. 실행중인후보의변하지않는첫32개완료tick/update줄만
`prefix-observation/`에복사해대조했다. 이후계속늘어나는전체파일을동결했다고표시하지않았다.

- 초기전체learner바이트일치,첫32개batch index hash일치.
- 장면/provider/tick·판정/이유·source·종료/bootstrap의차이0.
- 첫update의기록된scalar지표는같고,update2부터지표차이가나타났다.
- raw_u의최대차이9.447475999039764e−8,서로다른31행.
- 보상최대차이1.0146021175994058e−11.

따라서초기화차이를배제한이번두실행도후속학습경로의완전수치동일성을주장하지않는다.
모든숨은중간learner상태를기록한것은아니며,obs/next_obs전체prefix는후보replay저장후
원래예정된검토에서대조한다. 재수집전의차이를재수집효과로설명하거나유일원인을
확정하지않는다. 기존fresh100의엄격동일성실패도그대로보존한다.
로컬prefix review SHA `3239177a968cb67cd3e2f968036e9431c37fbccd74248fa1a34e05f01b8084db`.

기존 main2955496·후보2962566·후속2959705는유지한다. 실행조건/소스/예산변경0,
후보완료→저장/실제source재현→16개발주행→후속현재행동진단의기존순서를따른다.
안정정책과원래전체목표는아직미완료다. 추가평가원본인증에대한사용자응답은대기중이다.

## 21:59 — 첫400개 실제 수집에서 정지·명령 편중 확인

진행 중인 paired1000 실험의 두 tick 로그에서 이미 완성된 첫400줄만 별도 복사했다.
로컬 `outputs/sac_terminal_refresh_paired1000_20261120/collection-observation/`의
`review_prefix.py`를 실행해 원본 collection ID 연속성, 재수집 원본 ID의 존재,
명령·장면·시각 일치와 복사본 SHA를 확인했다. 학습 모델·수집기·실행 조건 변경은 없다.
`review.json` SHA는 `2ade6887a6626b5965432984e87bb6d542785c0028d56ea0d21cc3dcf6e2b88a`다.

| 첫400개 수집 구성 | 기존 | 후보 |
| --- | ---: | ---: |
| main / failure_branch / terminal_refresh | 233 / 167 / 0 | 181 / 145 / 74 |
| KEEP / FOLLOW 전이 | 234 / 166 | 325 / 75 |
| 고유 장면 | 5 | 4 |
| 실행 전 속도 <0.1m/s | 184 | 226 |

후보 재수집74개는 KEEP73/FOLLOW1이며, **모두 실행 전 정지·PREDICTED_COLLISION 거절·
EMERGENCY 제어·실제 COLLISION 종료**다. FOLLOW 첫 재수집은 collection388에서 발생했다.
원본 수집 사건은7개, 관측 fingerprint는6개, 고유 장면은4개다. 첫 KEEP 장면의 main/branch
두 원본이 재수집45/74개를 차지한다. 같은 장면의 서로 다른 원본 사건을 독립 장면으로
세지 않는다. 원본과 재수집 사이 간격은4–364 update이며 이 간격을 최신 정책 행동에 대한
replay 노출 횟수로 해석하지 않는다.

따라서 이 구간에서 현재 정책의 실제 실패 행동을 확보한 것은 확인되지만, 대부분 정지 후
충돌 직전의 상태다. 이동 중 회피·출발 경험이 충분해졌다는 증거는 아니다. 후보의 전체
정지 전이 증가에는 의도적으로 추가한 정지 terminal 재수집이 포함되므로 곧바로 정책의
주행 악화라고 해석하지 않는다. 재수집을 제외하면 후보 정지152/326, 기존184/400이다.
서로 방문한 상태·학습 경로가 달라 수락 수나 충돌 수를 동일 장면 평가 성공률로 비교하지 않는다.
정지 상태라는 사실만으로 충돌 불가피성을 판정하지도 않는다.

이는 첫400전이의 부분 관찰이며 최종 learner 성능/행동 coverage/저장 감사가 아니다.
예정한1,000회와 전체 저장·실행 재현, 양쪽 총16개 개발주행, 현재 행동 Q·실제 미래보상
진단을 마친 뒤 정지 편중과 FOLLOW 경험 부족이 최종에도 남는지 판단한다. 진행 중에
수집 cadence·배율·보상·안전검사·PD·순수SAC 조건을 바꾸거나 추가 예산을 시작하지 않았다.
직전 Notion 갱신은 첫 화면/02/03/04를 다시 읽어 a0707f2 링크와03의89절 단일 삽입,
과거 기록 보존을 확인했다.

## 22:11 — 양쪽1,000회 학습·저장 감사 완료, 전체 수집 분포 검산

paired1000의 후보 학습과 저장 감사가 exit0 완료됐다. 실제 새 전이1,000개에 actor/Q 각각
1,000 update, BC0, 기존 자기 warmup17배열 보존을 확인했다. 후보의 main375·완료경로431·
terminal_refresh194가 실제 replay6,000행에 저장됐고, 재수집194개 모두 원본 관측/실제
후속 배열·종료값·FIFO/cadence 검사를 통과했다. 전체256,000 batch draw와 RNG도 검산했다.

로컬 `outputs/sac_terminal_refresh_paired1000_20261120/`:

- candidate1000/report SHA `05851f10506cff1a46119429c96d7fdf1fe6e9cb7e33f27c3496acfd595f0ddb`.
- audit-candidate SHA `0f77a18e395e1a27db0aa9d010c62e6d9080cb702c2e8759c1b7ade33d07e998`.
- 후보 learner SHA `c58d9b706e0b44f85705d40d50a2d48c19f626d29e51af004b7765294b4b1da5`.
- 후보 replay SHA `ef939b05cc8862b497d7d0a861c30c4fe64c7ddee6f1772aab8c6355553f9e06`.

완료 보고서/감사를 요구하는 읽기 전용 `collection-final/review.py`를 실행했다.
두 전체 로그의 ID 연속성, 원본 참조·명령/장면/시각, 복사본과 완료 입력 SHA를 확인했다.
`collection-final/review.json` SHA
`c96922e249b2e2eb9da27ffb19ed992f69e70ab67f63ce996fa059aee26e7f74`.

| 전체 신규1,000개 경험 | 기존 | 후보 |
| --- | ---: | ---: |
| main / failure_branch / terminal_refresh | 739 / 261 / 0 | 375 / 431 / 194 |
| KEEP / FOLLOW 전이 | 603 / 397 | 676 / 324 |
| 고유 수집 장면 | 10 | 8 |
| 실행 전 속도 <0.1m/s | 557 | 497 |
| 이동 상태 전이 | 443 | 503 |
| 재수집 제외 정지 전이 | 557 / 1,000 | 303 / 806 |

재수집194개는 KEEP142/FOLLOW52, 원본 사건14개/관측 fingerprint13개/고유7장면이다.
명령·원본종류별로 KEEP main75/branch67, FOLLOW main29/branch23이다. **전부 정지 상태,
거절, EMERGENCY, 실제 COLLISION**이며 거절 이유는 PREDICTED_COLLISION187/
KINEMATIC_LIMIT7이다. FOLLOW가 전혀 재수집되지 않았다는 초반 우려는 해소됐지만,
재수집 자체가 정지 충돌 직전에 한정된다는 관찰은 전체에서도 유지된다.

전체 수집의 정지 비중은 후보49.7%/기존55.7%로, 첫400의 후보56.5%/기존46.0%와
방향이 다르다. 따라서 부분 관찰을 전체 후보의 정지 증가나 주행 악화로 확대하지 않는다.
후보 이동 경험503개 역시 서로 다른 상태에서 모은 학습 경험이며 평가 성능 향상의 증거가 아니다.
기존 신규 목표종료1·실패14, 후보 신규 목표0·실패208(이 중재수집194)도 수집 과정의
종료 건수이지 동일 장면 정책 성공률이 아니다. 고정 배치 goal8/failure8은 유지했고
후보 실패 pool44→252·목표pool6 유지, 전체 실패draw32,020/목표draw8,155였다.
반복 재수집으로 terminal pool 구성도 달라지므로 건수 증가만으로 유효 경험이 충분하다고
결론내리지 않고 예정된 현재 정책 행동 coverage/Q 검증과 함께 판단한다.

22:11 이후 main2955496과 실제 상태 재현2975399가 살아 있고 후속2959705는 대기 중이다.
명령/source별 상태를 실제로 재실행하는 검사, 최종 초기/prefix 검토, 양쪽16개 개발주행,
현재 행동 Q/실제 미래보상 진단은 아직 남았다. 새 학습/예산/채택/규약 변경은 없다.
안정 KEEP/FOLLOW·전체개발/seed/Rule-based/별도평가 목표는 여전히 미완료다.

## 22:17 — 네 종류 실제 재현과 전체 배치 노출 검산 완료

22:14에 `verify-execution`과 `review-training`이 exit0 완료됐다. KEEP/FOLLOW 각각의
main/failure_branch에서 첫 재수집을 선택하는 사전 지정 규칙으로4건을 재현했다.
매번 reset에서 실제 저장 경로를 실행해 원본 직전 상태를 복원하고 원본 행동과 재수집
행동을 별도 fork에서 실행했다. 검사·실제 제어·보상·종료/bootstrap·다음 관측을 대조하고
부모 환경이 변하지 않았음을 확인했다. 범위는4대표 원본이며194개 전부를 다시 실행한
검사는 아니다.194개 전체 저장 lineage/배열 검사는 앞선 감사 결과다.

| 명령 / 원본 | 원본 collection → refresh | 재현 prefix step | 원본/재수집 보상 |
| --- | --- | ---: | ---: |
| KEEP / main | 29 → 33 | 28 | -20.014 / -20.014 |
| KEEP / branch | 65 → 73 | 28 | -20.014 / -20.014 |
| FOLLOW / main | 375 → 388 | 53 | -20.017 / -20.017 |
| FOLLOW / branch | 455 → 463 | 63 | -20.017 / -20.017 |

4건 모두 실제 COLLISION/bootstrap false였다. 로컬 paired1000 `execution-review.json`
SHA `579837adbe3db134c61667d9784902370ac5c86b538647d8a42e0758318ca42a`.
`training-review.json` SHA
`5baf1ab7ed935a71a79b2f56abdfabdcd252742bc67179bbe34a51252ee2a9e8`.
최종 replay에서도 최초 재수집 전32개 batch hash와 이산 플래그는 같고, 실제 초기 learner와
warmup17배열 바이트 일치를 확인했다. 다만 raw_u 최대차9.447475999039764e-8(31행),
obs/next_obs 최대차1.8067657947540283e-7(각2/3행), 보상 최대차1.0146021175994058e-11(3행)이
남았다. 초기 동일성을 이후 계산 전체의 동일성으로 확대하지 않으며, 이전100회의
엄격 수치 동일성 실패를 바꾸지 않는다. 관측 벡터 차이를 물리 거리로 해석하지 않는다.

### 실제 배치 사용량과 정책 행동 coverage는 구분

학습 종료 뒤 `collection-final/exposure_review.py`를 실행했다. 완료 replay의 reward/
bootstrap와 seed로 각1,000 update의256,000 draw를 재구성하고 **모든 기록 batch index
hash를 대조**했다. 원본/복사 입력 SHA를 종료 시 다시 확인했고 학습/GPU 실행은 추가하지 않았다.
`collection-final/exposure-review.json` SHA
`d3ca1509edd5a559d99312f6270f8259ba5156aaffb45c1d0dc3cfafd4d1959a`.

| 실제 배치 추출 수 | 기존 | 후보 |
| --- | ---: | ---: |
| 기존 자기 warmup | 145,119 | 141,963 |
| 신규 main | 77,361 | 39,273 |
| 신규 failure_branch | 33,520 | 49,058 |
| 신규 terminal_refresh | 0 | 25,706 |
| 신규 경험 중 이동 상태 | 54,065 | 56,043 |
| 신규 경험 중 정지 상태 | 56,816 | 57,994 |

후보 재수집194개는 모두 최소 한 번 추출됐고 KEEP20,362/FOLLOW5,344 draw다.
총25,706회 반복 추출은25,706개의 독립 경험이 아니다. 전체 warmup의 속도 분류는
이번 분석에서 하지 않았으므로 위 이동/정지 분모는 신규 경험의 draw로 한정한다.
마지막100개 신규 경험은 기존2,317/후보2,333 draw이며 두 실행 모두 미추출2개가 있다.
미추출은 모두 main, 재수집 미추출은0개다.

따라서 이번1,000회에서 재수집 경험이 학습 배치에 들어간 것은 확인됐지만, **마지막 정책이
생성하는 행동과의 거리 및 그 행동의 Q 적합**은 아직 후속 검사 대상이다. 반복 노출만으로
현재 정책의 행동을 충분히 학습했다고 보지 않는다. 현재 기존 모델 primary 주행 평가가
진행 중이며 후보·정지 출발·최종16주행 검산은 미완료다. 정책 채택/추가 예산은 없다.

## 22:27 — 기존8회 실패 검산, 이동 출발4회 대조와 GIF

기존 모델의4고유 개발장면×KEEP/FOLLOW,8회 평가를 완료했다.
`driving-details/review.py --arm control`로 완료 모델/보고서 해시와 원본 프레임의
연속 tick·sample_valid·종료·거절/제동/정지 수·보상 합을 검산했다.
로컬 paired1000 `driving-details/control-review.json` SHA
`ad8665c369a76423564a2755d06dabde889ce7a2c4dab30905dc3f3995118278`.

기존은 **목표0/8·충돌8/8**,614tick 중거절314·제동240·정지376,
평균 return−20.494750736216602였다. 정지 출발4회는328/328tick 정지 상태였다.
고속 장면은 KEEP/FOLLOW 둘다55tick부터 연속 거절,57tick 제동,80tick 정지,
97tick 충돌이었다. 중속 장면은37tick부터 연속 거절/제동,41tick 정지,46tick 충돌이었다.
8회 모두 마지막 기록 충돌 상대의 중심이 ego 종방향 뒤쪽에 있었지만, 이는 접촉 시점의
위치·시간 순서이며 책임·불가피성·유일 원인을 판정한 결과가 아니다.

후보의 이동 출발 primary4회도 완료했다. `driving-details/review_primary.py`로
두 모델의 같은2장면×KEEP/FOLLOW4쌍에 대해 초기 프레임/지도 일치, 실제 저장 프레임,
집계/보상 합과 입력 SHA를 검산했다. 이는 전체 개발8장면 평가가 아니다.
`primary-paired-review.json` SHA
`808da946692298aafababdeed42ec8677f3aa65ac17d892477797365119a0496`.

| 같은 primary4회 | 기존 | 후보 |
| --- | ---: | ---: |
| ROUTE_GOAL / COLLISION | 0 / 4 | 2 / 2 |
| 실제 tick | 286 | 236 |
| 거절 | 110 (38.46%) | 44 (18.64%) |
| 제동 | 102 (35.66%) | 36 (15.25%) |
| 정지 | 48 (16.78%) | 12 (5.08%) |
| 평균 return | -20.249332356470862 | -9.981296819481493 |

후보의 성공2건은 고속 `8b4c1e7e09f5f4c3`의KEEP/FOLLOW로 모두72tick ROUTE_GOAL이다.
다만63tick부터 연속 거절,65tick부터 제동하며 목표에 도달했고 각거절11/제동8tick이다.
규약의 목표 성공으로 계산하되, 목표 이후의 주행이나 안정적인 전구간 경로를 입증한 것으로
확대하지 않는다. 중속 `8a4fba3463c6c62d`의두명령은 후보도46tick 충돌·거절11·제동10·
정지6으로 기존과 같았다. 현재 대조에서 고속 성공 유지와 지표 차이는 확인됐지만,
안정 정책 채택·전체개발/다른seed/독립평가 성능 확인은 아니다.

### 같은 고속 KEEP 기록의 나란한 GIF

첫 사전 지정 primary장면/KEEP의 이미 완료된 두 trace를 복사하고 표시 label만
`Pure SAC | existing collection` / `Pure SAC | terminal refresh`로 구분했다.
**양쪽 모두 순수SAC 수집 방식 비교이며 Rule-based 비교가 아니다.** 원본 정책이나
시뮬레이션을 다시 실행하지 않고 초기 프레임/지도가 같은 실제 기록만 사용했다.

기존 `scripts/visualize_sac_waymax.py render`를 stride2로 실행했다. 로컬 산출물은
`outputs/sac_terminal_refresh_paired1000_20261120/visualization/high-speed-keep.gif`.
PIL로50frame·1440×900·총11,200ms(실제9.7초+마지막1.5초hold)를 확인했고,
마지막PNG를 열어 두label·실제 궤적·거절/제동·충돌과종료 후hold표시를 시각 확인했다.

- GIF SHA `773ffc51710028cb7858ec5c1a94cfb6e3cb56b89f9792422d519c0c1b07847a`.
- evidence JSON SHA `07bd3a52995d8d5942ae8c9e72db12d5a9153a82a9a72b27bba25651473c100a`.
- 마지막 PNG SHA `8b5075861353492c6c85d090f0be7c631f9da81546974c9dfb1094753e4a1c10`.

후보 정지 출발 평가2981404가 이어 실행 중이다. 그 결과와 양쪽 총16주행 원본 검산,
대기 중인 현재 정책 행동 coverage/Q/실제 미래보상 진단을 마친 뒤 다음 보완을 결정한다.
실행 중 조건 변경이나 추가 학습 예산은 없고 전체 목표는 계속 미완료다.

## 22:42 — 16개 비교 주행 완료, 현재 행동 terminal Q 개선과 남은 정지 실패

paired1000의11jobs가 모두 exit0 종료됐고 main2955496은 실제 종료됐다.
`driving-review.json`은 모델당4고유 개발장면×KEEP/FOLLOW8회, 총16개의 원본 프레임과
초기 상태/지도 일치 및 집계를 검산했다. SHA
`460252b6dd0d5213832e1f07b113e9c0d23bee831f7e044781d35f2dd36ac017`.
후보의 별도 `driving-details/candidate-review.json` SHA는
`72b20e207a9ded90c88d88e53d97a266f5ddc25320d23be65967f58ad89d7fb1`이다.

| 같은8회 개발 주행 | 기존 | 후보 |
| --- | ---: | ---: |
| 목표 / 충돌 | 0 / 8 | 2 / 6 |
| 실제 tick | 614 | 564 |
| 거절 | 314 (51.14%) | 128 (22.70%) |
| 제동 | 240 (39.09%) | 116 (20.57%) |
| 정지 | 376 (61.24%) | 340 (60.28%) |
| 평균 return | -20.494750736216602 | -15.21061642036133 |

후보도 정지 출발4회는 모두 충돌·328/328tick 정지다. 같은 정지 장면의 거절204→84,
제동138→80, 평균return−20.740169115962345→−20.439936021241166 개선을 출발 성공으로
보지 않는다. 수락 tick은124→244개로 늘었지만, 수락된 모든 tick도 속도0.1m/s 미만이다.

`driving-details/review_stopped_motion.py`로 실제 위치/속도와 제안 경로 범위를 검산했다.
후보의 장면별 최대 속도는약0.00256–0.00443m/s, 초기위치 대비 최대변위는약6.14–17.47mm다.
따라서 정지는 평가 임계값0.1m/s에 따른 표현이며 위치가 수치적으로 완전히 고정됐다는
뜻은 아니다. 수락 제안 경로의 최대 범위도13.77–17.27mm다. 이는 저장된 경로 기하이며
실행 가능 속도나 유일 실패 원인을 의미하지 않는다. `stopped-motion-review.json` SHA
`7e5fd05eb5d807e421136e5a1b844e7b0ec3298ba890b646e7ad6e4b57e77485`.

### 최종 정책 행동과 실제 terminal Q

후속2959705가 완료 모델을 추가 동결한 뒤 양쪽coverage와 실제 terminal 행동검사를
완료했다. `review_coverage_only.py`에서각6,000 CSV행의 ID/거리영역 판정/Q gap 산술과
256,000draw 및 앞선 독립 lineage별 추출 집계를 대조했다. 같은 warmup의고정 충돌44개
저장 행동Q MAE는기존2.7537842184217647→후보0.8621346657938189로 감소했다.
후보 재수집194개 중 최종 조건부Gaussian95%영역 안에는1개(index5997,KEEP,4draw)만
포함됐다. 전체 재수집은25,706draw, 저장 행동Q MAE0.1823610791896771이다.
이95%영역은 조건부 행동 거리 지표이며 Gaussian의 실제 support·안전성·성공률·통계적
신뢰구간이 아니다. 서로 다른online 상태의37.4%/45.2%를 같은 상태 모집단의 개선으로
직접 비교하지 않는다. `coverage-only-review.json` SHA
`84425d50bfd115304c4d8ad7cfbc6e8faa4035c166d08ce0de9a85d1b87927e8`.

고정된 두 train실패상태(KEEP77ae312df21e80b9 tick28, FOLLOWbaac7893b025167a tick59)에서
각모델의 저장 행동/현재 평균/원래Gaussian4표본, 총24개의 행동을 실제 실행했다.
`review_branch_only.py`는 모델/상태 원본 hash와지정indices일치, 실제 COLLISION 여부,
bootstrap false, Q−보상 산술, 각상태의 모든행동/두모델에서 동일한 다음관측hash/보상을 검산했다.

| 실제 terminal Q−보상 | 기존 저장행동 | 기존 현재평균 | 후보 저장행동 | 후보 현재평균 |
| --- | ---: | ---: | ---: | ---: |
| KEEP | 1.67456061 | 6.19118315 | 0.08577162 | 0.29832083 |
| FOLLOW | 9.44808654 | 11.44048480 | 0.30214481 | 0.39308147 |

24행동 모두 실제 충돌이었다. 후보의 확률4표본 오차도 KEEP0.29796–0.30019,
FOLLOW0.39200–0.39435로 줄었다. **현재 행동의 충돌 직전Q 적합 개선은 확인했지만,
그보다 앞선 상태의 미래 가치와 실제 회피/출발 학습까지 개선됐다는 증거는 아니다.**
`branch-only-review.json` SHA
`dbe2e3e1c04b8db7581dcfe49f8c5f32aa83d4f58cb7c6119e7dd9c39854f2fa`.
이 두 partial review의미래return 미완료 표시는 작성 시점 범위이며 최종 진단을 대신하지 않는다.

### 남은 전체 개발 장면으로 비교 범위 확장 연결

위 주행 대조와 양쪽 명령의 실제 terminal Q 개선을 근거로, 추가 학습 대신 남은 개발
장면을 평가하기로 결정했다. 일부 고속 장면의 두 성공만 반복 확인하는 것으로 안정성을
판단하지 않기 위한 읽기 전용 진단 확장이다. 현재 정지/중속 실패는 그대로 보존하고
정책 채택이나 안정성 수락으로 취급하지 않는다.

로컬 `outputs/sac_terminal_refresh_full_validation1000_20261120/`에 준비·실행했다.
새 runner2990076/session86804는현재진단2959705의실제PID/startticks73701579와최종
성공 상태를 확인한뒤에만 실행한다. 현재상태는WAITING_FOR_POLICY_DIAGNOSTICS다.
관찰timeout을종료로취급하거나기존진단을재시작하지않는다.

- 남은4장면:57eb0840351f6cdc,22a504c8e79dc7ba,83c142d159860844,fd56befbc8f7e7db.
- 각장면KEEP/FOLLOW, 모델당새8주행. 완료된8주행을합쳐모델당총16쌍/8고유개발장면이다.
- 3jobs:기존remaining→후보remaining→전체원본검산. 양쪽추가합계16주행이며새독립장면16개가아니다.
- 동일완료1,000모델/평균평가/seed0/manifest/검사/PD. 보상/학습/replay/BC/teacher 변경0.
- source78개와입력42개를hash검사하고source복사본을보존했다. 전체validation16쌍과
  기존8쌍/남은8쌍의차집합·원본8장면SHA·스크립트AST를사전검사했다.
- decision SHA `f1538d261da7fd6089694196d992a6d047058788e53607069d09016b0f194d57`.
- source-copies SHA `574c9e90f247463f43d2aafa92713a41b6151a889a77c929435eb4acd793bedd`.

22:42:29 후속2959705/기존soft-return2988470/전체개발대기2990076의실제생존을확인했다.
새전체개발주행은아직0회이며소스/입력이바뀌거나앞선진단이실패하면기록후멈춘다.
전체개발도반복사용한development이며독립test가아니다. 다른seed·동일조건Rule-based·
별도평가·안정정책은미완료이며새예산/추가seed/기본채택은자동실행하지않는다.

## 22:55 — 실제 미래 보상 대조 완료, 초기 상태 노출과 남은 과대평가

후속 진단2959705/session65181의7jobs가 모두 exit0 완료됐으며 세션의 실제 종료를
확인했다. 완료 시점은22:54:51 KST다. `policy-diagnostics/review_results.py`는 원본
16개 rollout의 실제 종료/bootstrap, gamma0.99 보상 합, 첫 행동 이후에만 포함되는
정규화 entropy 보너스, Q−soft-return 산술 및 동결 소스/입력 SHA를 검산했다.
`diagnostic-review.json` SHA는
`3a3d8dc2b98d722f86854797b2575be4dadf790c0df0ffb861244d9b51ccb506`이다.

같은 두 train 상태(KEEP77ae312df21e80b9 tick0, FOLLOWbaac7893b025167a tick4)에서
모델별 평균 첫 행동 뒤 원래 Gaussian 정책을 실제 종료까지 실행했다. 각 상태의
난수 seed88019–88022는 양쪽에 동일하게 적용했다. 각 모델8회 모두 실제 충돌이다.
이는 두 상태의 작은 표본 Monte Carlo 진단이며 독립8장면이나 정확한 기대 Q가 아니다.

| 상태 | 기존 Q | 기존 평균 soft-return | 기존 차이 | 후보 Q | 후보 평균 soft-return | 후보 차이 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| KEEP | -1.31101716 | -13.27854000 | 11.96752284 | -2.05209088 | -13.64969671 | 11.59760583 |
| FOLLOW | -0.97444117 | -10.69658035 | 9.72213918 | -1.51817107 | -10.63552183 | 9.11735076 |

후보의 충돌 직전 현재 행동 Q 오차가 약0.30/0.39로 줄었어도 앞선 상태의 미래 비용
차이는 약11.60/9.12로 크게 남았다. 작은 차이 감소에 통계적 유의성이나 주행 안정성을
부여하지 않는다. 후보 KEEP의 실제 평균 soft-return은 오히려 약0.37 낮아졌다.
**terminal 재수집만으로 초기 미래 비용과 실패 회피까지 해결됐다는 결론은 기각한다.**

### 같은 초기 관측의 학습 노출 확인

읽기 전용 로컬 `early-state-exposure/review.py`를 실행해 고정 MC 원본 replay 관측과
양쪽6,000행의 관측/두 mask를 대조하고 기존 완료 coverage CSV의 실제 추출 수를
연결했다. 원본/현재 replay SHA와 입력 불변을 검산했다. 비교는 scalar 최대차1e-6
이하와 mask 완전 일치를 요구하고, 완전 동일 행은 따로 집계했다.

- KEEP은 양쪽 모두 완전 동일한 packed 관측3행(index0/5000/5029)이 있다.
  warmup/main/branch의 추출은각20/147/141, 합계308draw다. 세 저장 행동 모두 각
  최종 정책 조건부95% 거리영역 안에 있다. 중복 관측 행과 반복draw는 독립 경험 수가 아니다.
- FOLLOW의 고정 tick4 원본 관측과 위 기준으로 일치하는 행은 양쪽 모두0이다.
  이는 유사 상태 부재나 일반화 불가능을 입증하지 않으며 실제 시뮬레이터 상태 전체의
  동등성을 검사한 것도 아니다.
- CSV의 Q는 packed replay 관측에서 계산됐고 실제 재현 경계에서 계산한 위 MC 첫 Q와
  작은 수치 차이가 있다. 둘을 동일한 계산으로 취급하지 않는다.

`early-state-exposure/review.json` SHA
`72a08f6096b3f551a21ddd67fcf441845655b71906f119ea07ffd7425701d11a`.
KEEP의 미래 손실 과소평가는 단순 초기 상태 미방문만으로 설명되지 않는다. 현재 행동
주변 저장 경험, 그 후 실제 경로의 상태/행동 분포 및 가치 전달을 함께 살펴야 한다.
이 검토에서 새 모델 업데이트나 replay 변경은 없었다.

### 전체 개발 평가 실제 시작

대기2990076/session86804가 앞선 진단 성공과 실제 종료를 확인한 뒤 RUNNING으로
전환됐고 control-remaining2993680의 실제 실행을 확인했다. 앞 절에서 동결한 남은
4장면×KEEP/FOLLOW 평가이며 모델당 기존8회와 합쳐 전체16회가 되는 동일 범위다.
평가 확대는 진행 중이고 결과는 아직 미확정이다. 이후 후보remaining과 원본 전체검산을
이어간다. 보상/검사/PD/원래Gaussian/BC·teacher0 및 완료 모델은 바꾸지 않았다.

현재 재수집 bank는 실제 COLLISION/OFFROAD 바로 전 상태만 저장한다. 기존 complete-path는
과거 이동 중 수락 predecessor를 우선 골라 원래 정책으로 실제 종료까지 실행하지만,
그 root를 여러 후속 정책 시점에서 재방문하는 bank는 없다. 더 이른 이동 실패 상태의
현재 정책 경험을 확보하는 방안은 검토 대상이며 이번에 구현·선택·실행한 기능이 아니다.
전체 개발 주행과 위 비용 차이를 함께 검토한 후 다음 한 가지 수집 변경을 결정한다.
안정 정책·다른 seed·동일 조건 Rule-based·별도 성능 평가는 계속 미완료다.

## 23:01 — 이동 경험의 현재 행동 거리 확인과 다음 가치 오차 진단 연결

전체 개발 평가2990076/session86804는 기존 모델의 남은8회를 완료하고 후보
remaining2995653을 실행 중이다. 기존 완료 보고서 세 cohort를 합치면 전체8고유장면×
KEEP/FOLLOW16회에서 목표2/충돌14, 1,032tick·거절506·제동426·정지442,
평균return−17.857806718919633이다. 두 목표는83c142d159860844의KEEP/FOLLOW다.
이는 완료된 기존 보고서의 부분 합산이며 양쪽 전체 원본 검산은 예약된 최종 job에서
진행한다. 후보의 전체16회 성능은 아직 미확정이다.

새 읽기 전용 `outputs/sac_terminal_refresh_bellman1000_20261120/review_exposure.py`로
기존 완료 coverage6,000행과 실제 online1,000행의 ID/장면/수집 종류를 대조했다.
이동 여부는 실제 제안 anchor의 실행 전 속도0.1m/s를 기준으로 분류했다.

| 집계 | 기존 건수 / 최종 조건부95% 영역 내 | 후보 건수 / 영역 내 | 기존 / 후보 실제 draw |
| --- | ---: | ---: | ---: |
| 이동 상태 | 443 / 370 | 503 / 450 | 54,065 / 56,043 |
| 이동 중 수락 | 170 / 132 | 204 / 178 | 21,350 / 23,075 |
| complete-path 시작 지점 | 7 / 5 | 7 / 7 | 781 / 881 |

후보 이동 경험의 대부분은 최종 정책의 조건부 행동 거리영역 안에 있다. 따라서 더
이른 이동 상태 재수집을 바로 구현하기 전에, 현재 경험에 대한 가치 전달 오차도
분리해야 한다. 이 영역은 확률적 거리 지표이며 안전/충분한 coverage/Q 정확성의
보장은 아니다. 두 모델이 방문한 상태가 달라 위 비율을 같은 상태의 개선으로 보지
않고, 반복 추출이나 분기 root를 독립 장면으로 세지 않는다.
`moving-exposure.json` SHA
`759993be8357805f104dad5996d735ea2dd5cc166d82605c2a5514cc97d8cbc9`.

### 다음 실행: 저장된 실제 경로의 Bellman 오차 분해

이미 구현된 `scripts/diagnose_sac_bellman_trace.py`와 `scripts/sac_bellman_trace.py`를
읽고 현재 완료 MC 결과/모델과의 입력 호환성을 확인했다. 현재 MC의 두 train 상태×
4표본을 각 모델에서 원본 행동 그대로 재현하며 실제 보상/종료/다음관측 SHA를 대조하고,
각 tick의 online Q와 target Q로 초기 Q−실제 soft-return 차이를 분해한다.
새 행동 생성·replay 쓰기·학습 업데이트가 없고 새로운 독립 주행 평가도 아니다.

목적은 충돌 직전 오차 감소 후에도 남은 차이가 비종료 구간의 sampled residual과
online/target 차이에 어떻게 분포하는지 확인하는 것이다. 단일 경로의 다음 행동은
기대 Bellman target의 정확한 적분이 아니며 target-lag 항이 커도 tau 변경이 주행을
개선한다는 인과 근거는 아니다. 보상/검사/PD/Gaussian/loss/lr/tau/BC·teacher0 유지.

로컬 새 runner2996225/session43612를 실행했다. 전체 개발 부모2990076의 실제
startticks74129488과 성공 완료/실제 종료를 기다리는 WAITING_FOR_FULL_VALIDATION이다.
기존 경로 진단→후보 경로 진단→합계/원본 검산의3jobs이며 현재 새 진단0회다.
source83개/input64개 SHA와 소스 복사본을 보존하고 Python AST를 검사했다.

- decision SHA `655f3f339a325f42064ab229f96488e8a17dc0339c6638a9ab6bfb4bd3fd7434`.
- runner SHA `73b92d8cf314ac5a6e6239161fa82cca8ac6252ae89db9f9bee13ad78c6165a2`.
- 실행/대기 상태: `outputs/sac_terminal_refresh_bellman1000_20261120/status.json`.
- 향후 결과: 각 `control/report.json`, `candidate/report.json`, `bellman-review.json`.

23:00경 전체 개발2990076/후보2995653/후속대기2996225의 실제 생존을 확인했다.
기존 경로/모델은 변경하지 않았고 추가 학습 예산·새 seed·기본 채택은 연결하지 않았다.
다음 수집 보완은 현재 실제 성능과 이 오차 분해를 함께 보고 선택한다. 전체 목표는
계속 미완료이며 이동 상태 재수집은 아직 검토 대상이다.

## 23:05 — 전체 개발16회 대조 완료: 성공 증가 없이 성공 장면 교환

전체 개발 runner2990076/session86804의3jobs가 모두 exit0로23:04:47에 완료됐고
실제 종료를 확인했다. 남은4장면의각8회와 기존4장면의각8회를 합친 모델당16회,
8고유 개발장면×KEEP/FOLLOW의 전체 범위를 검사했다. 두 모델의 초기 프레임/지도
동일성, 모델/실행 버전, 원본 tick/유효성/종료/보상 합과 지표를 검산했다.
`outputs/sac_terminal_refresh_full_validation1000_20261120/full-validation-review.json`
SHA `54e66e64896e65d72d1ddcd1c9f386eba210b111673709ebae6e87537d525165`.

| 전체 개발16회 / 모델 | 기존 수집 | terminal refresh 후보 |
| --- | ---: | ---: |
| 목표 / 충돌 | 2 / 14 | 2 / 14 |
| 성공률 / 충돌률 | 12.5% / 87.5% | 12.5% / 87.5% |
| 실제 tick | 1,032 | 962 |
| 거절 | 506 (49.03%) | 348 (36.17%) |
| 제동 | 426 (41.28%) | 330 (34.30%) |
| 정지 | 442 (42.83%) | 430 (44.70%) |
| 평균 return | -17.857806718919633 | -17.757717198794897 |

KEEP와FOLLOW는각각8회 중 양쪽 모두1목표/7충돌이다. 초기 속도별로 고속(10m/s이상)
6회 목표0→2, 중속(2이상10미만)4회 목표2→0이며 저속(0.1이상2미만)2회와
정지출발4회는 양쪽 모두 충돌이다. 정지출발328/328tick 정지도 해결되지 않았다.
정지 tick 절대 수가12 줄어도 종료까지의 전체tick이70 줄어 비율은 오히려 늘었다.
평균 보상과 거절 감소만으로 주행 개선을 판정하지 않는다.

**후보는 현재 안정 정책으로 미채택이다.** 앞선 부분8회에서 보인0→2목표 증가는
전체16회에서 성공 증가로 이어지지 않았다. 고속8b4c1e7e09f5f4c3의KEEP/FOLLOW를
회복한 대신 중속83c142d159860844의KEEP/FOLLOW 성공을 잃었다. 두고유장면의 네명령
결과가 바뀐 것이며 독립 네장면이나 새로운4성공이 아니다. 반복development/seed0의
결과이고 다른seed·독립평가 안정성은 여전히 미검증이다.

### 성공을 잃은 중속 장면의 실제 실패 과정

새 읽기 전용 `review_tradeoffs.py`는 전체검산 결과와 outcome이 달라진 모든4쌍의
원본 프레임/초기상태·지도/종료·거절/정지 집계 및 상태 변화를 대조했다.
`tradeoff-review.json` SHA
`b51033c25c258057a8df5dea5f404295f8c5d1d2ca9c0d81ab6d3b3442cc7f8c`.

83c142d159860844의두명령은초기속도7.318985160574789m/s이며:

- 기존은94tick 첫PREDICTED_COLLISION 거절/제동,95tick 일시수락,96tick부터 연속거절,
  108tick EMERGENCY를거쳐110tick ROUTE_GOAL이다. 각거절16/제동16/정지0tick.
  목표에 도달했지만 마지막15tick 거절과제동이있어 안정적경로나목표이후안전을보장하지않는다.
- 후보는66tick PREDICTED_OFFROAD와REUSED,68tick BRAKE,75tick EMERGENCY,
  76tick KINEMATIC_LIMIT,86tick 속도0.1m/s미만,89tick PREDICTED_COLLISION,
  99tick 실제COLLISION이다. 각거절34/제동32/정지14tick,66tick이후수락회복없음.
- 최종 접촉차량2724의중심은ego종방향약−4.468m/횡방향약−0.336m다. 시간순서와
  상대위치의관찰이며 책임·불가피성·유일원인을판정하지않는다.

### 실패 비교 GIF

같은장면KEEP의완료실제trace를복사해표시label만기존수집/terminal refresh로구분하고,
기존renderer로생성했다. **양쪽모두순수SAC이며Rule-based비교가아니다.** 성능회귀를
설명하기위해선택한사후시각화이며모델/시뮬레이터재실행이나새독립평가는없다.
56frame·1440×900·시뮬레이션11.0초+최종1.5초hold,마지막PNG를열어도로/주변차량/
제안·실행·실제궤적/거절·제동·충돌및종료hold표시를확인했다.

- 로컬GIF: `outputs/sac_terminal_refresh_full_validation1000_20261120/visualization/medium-speed-keep.gif`.
- GIF SHA `1fa14dd693f3d771b5f94f86ad77ae1e52270a78aa3bb1b91aa5b16ac475f186`.
- evidence JSON SHA `9ff20fc0bfc5e1eeaa1f70227eeac7b66e49dfa6f7948542d607087a1d0aea4e`.
- PNG SHA `435595e1d1660ae9b27e719272c4d6479ac97924e571d8b350dbfdc7c27d3b08`.

후속2996225/session43612의동결된경로별Q진단은그대로이어간다. 현재후보의채택이나
추가학습은없으며terminalQ개선/전체성공동일/중속회귀/초기미래비용오차를함께설명할
근거를검토한다. 보상·안전검사·PD·순수SAC조건은유지하고전체목표는계속미완료다.

23:06경 후속2996225가RUNNING으로전환되고기존경로진단2997617의실제생존을확인했다.
전체개발완료status/review/frozen-inputs를추가동결한후실행하며새source/model변경은없다.

## 23:23 — 경로별 가치 오차 완료와 비종료 거절 재수집 구현

Bellman runner2996225/session43612의3jobs가23:21:27에 모두exit0로 끝났고 실제 종료를
확인했다. 기존 MC의각8경로를 원본 행동 그대로 Waymax에서 재현해 보상/종료/다음관측
SHA를 검증했다. 별도 `review_components.py --arm control|candidate`는 기존404tick,
후보395tick의 **모든 suffix를 직접 할인합**으로 다시 계산하고 각 잔차와 초기차이 합,
실제 제어 source별 집계를 대조했다. 새 독립 평가나 기대 Q의 정확한 적분은 아니다.

| 평균 경로 오차 분해 | 기존 KEEP | 후보 KEEP | 기존 FOLLOW | 후보 FOLLOW |
| --- | ---: | ---: | ---: | ---: |
| 초기 Q−실제 soft-return | 11.967498 | 11.597699 | 9.722138 | 9.117232 |
| 마지막 종료 기여 | 7.435951 | 1.853520 | 6.237524 | 0.340082 |
| 비종료 구간 기여 | 4.531547 | 9.744179 | 3.484614 | 8.777151 |
| sampled target 잔차 합 | -2.494604 | -6.748242 | 15.228529 | -23.914554 |
| target−online 차이 항 합 | 14.462102 | 18.345941 | -5.506391 | 33.031786 |

두 마지막 행의 합과 종료/비종료 두 행의 합은 각각 초기차이와 같다. 부호 있는 항의
합산이며 오차 설명 비율이나 독립 인과 기여가 아니다. 기존 FOLLOW의 target 차이는
반대 부호여서 단순히 target 갱신이 느리다는 하나의 설명으로 묶지 않는다.
후보 비종료 EMERGENCY 구간의 평균 잔차 기여는 KEEP9.709227/FOLLOW7.369012이다.
비종료 NEW 구간은−2.002022/−3.050221로 음수이며 source별 항 사이에 상쇄가 있다.
서로 다른 실제 경로이므로 같은 중간 상태의 정책 개선 대조로 확대하지 않는다.

충돌 직전 재수집이 종료 오차를 줄였어도 앞선 거절/제동 구간의 미래 비용 차이는
해결하지 못했다. 이를 근거로 다음 수집 옵션은 **실제로 거절된 비종료 상태까지
재수집 source를 확장**하는 것으로 정했다. 보상/검사/PD/Gaussian/loss/lr/tau를
바꾸거나 제동 상태를 가짜 종료로 바꾸지 않는다. 효과는 아직 미검증 가설이다.

로컬 `outputs/sac_terminal_refresh_bellman1000_20261120/` 완료 근거:

- `bellman-review.json` SHA `8b800e18dc606a91b4ccfaf3f3c14498b347cde462ce3d1339b9c8788d19b69b`.
- `component-review-control.json` SHA `653991d77572de38981e38f77bda2e80ed429b96cf2bdc5a0ce36c8bc6a86e81`.
- `component-review-candidate.json` SHA `07a564590c013a2ff8bb5988afd49b5c6b54188bf71b3a0e2c4da6be3c899efe`.

### 구현·규약 대조·단위 검증

`scripts/sac_rejection_refresh.py`는 원래 TerminalRefresh의 실제 전이/관측/bootstrap
검사와 FIFO8/순환/base4당1회/재귀 금지를 재사용한다. 기존 COLLISION/OFFROAD 직전
상태에 `bootstrap=true && check_valid=false`인 main/failure_branch를 추가한다.
수락된 비종료 및 ROUTE_GOAL/기타 종료를 새 자격으로 넣지 않는다. 최대20% 재수집은
같지만 FIFO 구성/첫 실행시점/종료 상태 노출량은 바뀔 수 있음을 명시했다.

별도 `diagnose_sac_rejection_refresh.py`, `audit_sac_rejection_replay.py`,
`audit_sac_rejection_refresh.py`, `verify_sac_rejection_refresh_execution.py`를 준비했다.
driver는 기존 수집/학습 루프를 보존하고 bank/lineage scope와 초기 learner 기록만
분리했다. fresh1..1000만 지원하며 첫 실제 update에 전달되는 learner를 기록하고
지정한 reference와 바이트 동일성을 검사한다. 기존83개 진단 소스는 변경하지 않았다.

감사기는 source의실제check/bootstrap·원본/후속배열·FIFO/주기·모든batchhash/RNG·
actor/Q update수·BC0·자기warmup17배열·초기model을 대조한다. 실제 재현기는 각 관측된
provider/source종류/비종료·종료 조합의 첫 사례를 원래 collector로 실행한다.
존재하지 않은 조합은 미검증이며 원본이 거절됐다고 새 결과까지 거절로 정하지 않는다.

규약5절의 원래 Gaussian,10절의 전이당 전체update1·batch256과 기존sampler를 유지하는
상태 수집 분포 실험이다. 별도 B평균 허용 범위 안에서 같은 모델 설정을 쓴다.
학습 보상/안전검사/PD/BC·teacher0을 그대로 유지하며 기본 설정 채택은 아니다.

CPU30pytest 통과(0.21s): 신규8개, 원래bank7개와 complete-path/초기기록/실제branch
복원 관련15개다. 새driver/감사/실제재현 CLI와Python AST, diff 검사도 통과했다.
신규 테스트는 비종료 bootstrap 보존, 통과한 실제충돌도 포함, goal/기타종료 제외,
FIFO/주기/최대비중/메모리격리/재귀금지/손상된종료거절을 확인한다.

다음은 새옵션100전이의 실제 GPU 수집→저장감사→관측된 source조합의Waymax 재현이다.
학습방식의 사전 실행 검증이며100회로 안정정책이나 성능 우위를 판정하지 않는다.
동일 seed0·자기warmup5,000·기존 실제초기learner를 reference로 쓰고 추가 비교 예산은
그 결과를 검토한 뒤 정한다. 현재 새옵션의 실제 실행/성능은 아직 미검증이다.

## 23:25 — 새100회 실제 사전검증 시작·초기 learner 일치 확인

구현 `da419dd009f277108652eec6c8fac8b2769e36a0`을 같은 브랜치로 push하고 원격 SHA를
확인했다. 로컬 `outputs/sac_rejection_refresh_pilot100_20261120/`의 runner3003336/
session21062, 실제 수집 child3003341이23:25:05에 시작했고 실제 생존을 확인했다.
이전 Bellman 작업의 성공 완료와 실제 종료를 확인한 뒤 실행했으며 GPU 작업을 겹치지 않았다.

- source91개/input69개 SHA·복사본·AST 검사를 완료하고 조건을 동결했다.
- decision SHA `52d180749f64d69e8dc7c9669a695fbe187dce0b35811afd5e14a85ea204faf0`.
- 4jobs: `train100` → `audit100` → `verify100` → `review100`.
- 동일seed0/자기warmup5,000/B-f64/raw gain0.0001/physical Q/complete-path/
  goal8+failure8/원래Gaussian과 실제전이당전체update1이다. source자격만 새옵션이다.
- 첫 실제 전이 뒤 첫 update에 전달되는 초기 learner가 기존reference와 바이트 동일했다.
  `initial-state-audit.json` COMPLETE, reference_bytes_equal=true,
  SHA `7363d29a0c9500d58a259ba8953ff21428ede1660ee9e66be9b9f571ef014c2c`.
  초기동일성이후의수치동일성이나주행성능을뜻하지않는다.

23:25:40 기준 수집 프로세스 실행 중이며100회 완료/최종 저장감사/실제source재현은
미완료다. 새정책의 성능 평가는 아직 없다. 실행 결과를 검토한 후 같은1,000회 예산
대조 여부를 결정하며 추가 학습/새seed/정책채택은 자동으로 연결하지 않았다.

## 23:34 — 비종료 재수집100 완료·동일 예산1,000과 전체 개발 비교 시작

이전 대화 응답은 강화학습/모방학습 구분 설명이며 새 실험 증거를 만들지 않았다.
이번에는 실제 프로세스와 산출물을 다시 확인했다. pilot runner3003336은 종료됐고
train/audit/verify/review4jobs가23:29:00에 모두exit0로 완료됐다.
`outputs/sac_rejection_refresh_pilot100_20261120/pilot-review.json` SHA
`de18fb865f7cc71901cdf150cc9d120340e8e33ef4e425c495eb900fcb97fd21`.

- 실제100전이 = main51 + complete-path29 + refresh20; actor/Q 각100회, BC0.
- 자기warmup17배열 보존, 배치25,600추출/RNG와 모든 source/후속 관측을 검산했다.
  최초 learner는 기존 reference와 바이트 동일했다. teacher replay0 유지.
- refresh20개 모두 실제 비종료, bootstrap=true. source는main9/branch11이며
  첫 source 각1개를 원본 경로와 함께 Waymax에서 재실행해 실제 보상/검사/제어/
  next_obs와 부모 상태 보존을 확인했다. KEEP만 관찰됐으며 FOLLOW/종료 source는 미검증.
- pre-state anchor 속도 기준 이동6/정지14, 최대3.738856m/s. 새 경로는20/20거절,
  PREDICTED_COLLISION14/OFFROAD3/KINEMATIC_LIMIT3, REUSED2/EMERGENCY18이었다.
  `exposure-review.json`은 이 기술통계이며 새 수집이 유효 경로를 확보했다는 증거가 아니다.
- `audit100.json` SHA `d857971c660e5ca82374a0706e3265e0c19c2b4924e0b83c5bff5b0e08d5958e`,
  `execution100.json` SHA `b7d268f48e0b9ced67cfc55df0e8d67c3122f915edba91677613eb0c322736de`.

관측된 실제 비종료 경험의 저장/업데이트가 규약대로 수행됐으므로 같은fresh1,000회
예산의 성능 비교를 선택했다. 100회 가중치를 이어받지 않는다. 이전 완료된
`sac_terminal_refresh_paired1000_20261120/candidate1000`(terminal-only refresh)를
이번 대조군으로 재사용한다. 이 모델의 전체 개발 성적은2/16목표·14/16충돌이다.
대조군을 Rule-based나 안정 정책이라고 부르지 않는다.

새 로컬 `outputs/sac_rejection_refresh_comparison1000_20261120/`:

- 23:33:40 KST runner3008213/session49022, train3008214 시작·실제 생존 확인.
- source94/input83 SHA와 source복사본을 동결했다. decision SHA
  `3ba7df2b85530a8278a78bd5ad406a96d093106d6f8bb86a01ad7bc3885cc6b5`.
- 6jobs: train1000 → audit1000 → verify1000 → review-training → validation-full
  → review-full. 오류는 보존하고 실패 시 이후 단계는 실행하지 않는다.
- 같은 seed0/초기learner/자기warmup5,000/config/override/sampler/complete-path 설정을
  원본 report와 대조했다. 새 실행 첫update에서도 초기 바이트 검사를 다시 수행한다.
  새 최종 learner와 실제 행동 분포의 동일성을 가정하지 않는다.
- 변경 요인은 재수집 source 자격이다. 보상·검사·PD·원래Gaussian·loss/lr/tau·
  전이당전체update1·BC/teacher0 유지. 별도허용 B-f64/rawgain0.0001은 기본채택아님.
- 새 모델의8고유개발장면×KEEP/FOLLOW16회 전체를 평가한다. 대조군 완료16회는
  원본 trace/모델 hash를 동결해 재사용하고 두 모델 초기frame/지도·모든tick의
  보상/종료·거절/제동/정지·명령/속도별 집계를 대조한다. 새로운 독립 장면은 아니다.

새 성능 결과는 아직 없으며 안정정책·다른seed·동일조건Rule-based 최종비교·별도평가는
미완료다. 1,000회 뒤 지표와 실패 경로를 검토해 다음 행동을 정하며 자동 추가예산이나
채택은 없다. 현재 동결된 학습/평가 source는 실행 중 수정하지 않는다.

```bash
python -m json.tool outputs/sac_rejection_refresh_comparison1000_20261120/status.json
python -m json.tool outputs/sac_rejection_refresh_comparison1000_20261120/candidate1000/report.json
```

## 23:41 — 새 이력의 Q/미래보상 진단 연결·첫200전이 분포 검토

직전 goal turn은 pilot 완료 검토·새1,000 비교 실행·노션 반영으로 실질 진전이었다.
이번에는 runner3008213와 train3008214의 실제 생존을 다시 확인했고 조건을 유지했다.
실행 중 동결한94source/83input SHA를 재검사했으며 모두 그대로였다.

기존 정책 진단은 rejection scope를 지원하지 않으므로 별도
`sac_rejection_diagnostic_audit.py`와 `diagnose_sac_rejection_branch_q.py`,
`diagnose_sac_rejection_soft_returns.py`, `audit_sac_rejection_coverage.py`를 연결했다.
새 scope는 전용 실제 lineage 감사, 기존 scope는 원래 감사기를 통과해야 한다.
계약 누락/혼합/위장·미완료 감사는 거절한다. Q와 return 진입점은 원래 파일에서
감사 import/hash만 바꿨고, coverage는 lineage종류/감사/결과 contract만 바꿨다.
학습·보상·Gaussian·물리검사·PD·할인/entropy 계산은 바꾸지 않았다.
기존 파일이 진행 중 비교에 동결돼 있어 별도 진입점으로 두었으며 src 변경0이다.

검증: 관련CPU pytest30개 통과(0.38초, 신규adapter8+기존adapter8+bank8+복원6).
세 새CLI --help, AST, diff검사 통과. 새adapter를 실제 pilot100·이전terminalrefresh1000·
이전control1000 저장 이력에 CPU로 실행해 모두 COMPLETE를 확인했다. 새 GPU 정책
진단 자체는 아직 미실행이며 테스트/저장감사를 주행 성능으로 확대하지 않는다.
코드 `a99f463a514b52aa7515e870a60aa41d50a79d41` push/원격SHA 확인.

### 후속 진단 실행 연결

로컬 `sac_rejection_refresh_comparison1000_20261120/policy-diagnostics/`에
runner3012859/session69685를23:39:52에 시작했고 실제 생존 확인했다.
현재 WAITING_FOR_COMPARISON이며 부모 PID3008213/start_ticks74453793과 성공 종료를
확인한 뒤에만 실행한다. 시간 만료로 재시작하지 않으며 부모 실패 시 그 상태를 남긴다.

- source101/input149 SHA와 source복사본 동결. decision SHA
  `4f48731e5fa0b99665164dfce8f6bc0baac50a0ae573f6d742c0be8668d1cc5e`.
- 완료된 terminal-only 대조군의 coverage/branch-Q/soft-return 산출물을 바이트 그대로
  복사해 재사용한다. 독립 평가 횟수나 장면 수를 늘리지 않는다.
- 부모의 전체 개발평가/저장감사가 성공하면 새모델/replay/로그/초기learner와 완료
  보고서 hash를 추가 동결하고 candidate-coverage → candidate-branch-q →
  candidate-soft-returns → review를 순서대로 실행한다. 학습GPU와 겹치지 않는다.
- coverage는6,000행·256,000배치추출, branch-Q는 같은 사전선택train 종료상태
  KEEP77ae tick28/FOLLOWbaac tick59에서 기록/평균/원래Gaussian4표본을 실제 실행한다.
  새행동이 실제 종료된 경우만 Q−reward를 종료오차로 해석한다.
- soft-return은 같은 선행train상태KEEP tick0/FOLLOW tick4에서 평균첫행동 이후
  원래Gaussian을4seed로 실제 종료까지 실행한다. 원래discount/entropy합과 모델/
  replay/상태원본hash, 동일한 상태/명령/seed를 검산한다. 소표본이고 독립test가 아니다.
- 이 진단은 가중치/경험을 수정하지 않으며 별도 수집·학습 예산을 추가하지 않는다.

### 첫200 실제전이 기술통계: 이동 경험 증가로 단정할 수 없음

로컬 `review_prefix.py`는 두 모델의 첫200개 완전한 JSONL행을 고정해 순서/업데이트/
source연결을 확인하고 `collection-prefix200.json`을 생성했다. 이는 부분 수집 분포이며
최종 replay 감사나 주행평가를 대체하지 않는다. 중간 상태 분포가 다르다.
결과 SHA `087adeff928f784094e2f0e60d39b17908b8a1489bed24adb7f36b5abfced25e`.

| 첫200전이 | terminal-only 대조군 | 비종료거절 후보 |
| --- | ---: | ---: |
| 이동 / 정지 | 86 / 114 | 81 / 119 |
| 수락 경로 | 58 | 38 |
| main / 완료경로 / refresh | 88 / 78 / 34 | 93 / 67 / 40 |
| refresh 이동 / 정지 | 0 / 34 | 9 / 31 |
| refresh 실제 비종료 / 충돌 | 0 / 34 | 38 / 2 |
| 고유 refresh source 전이ID | 3 | 40 |

양쪽 모두 KEEP3고유장면 범위다. 새 refresh40개의 source는 비종료거절38/실제종료2,
새경로수락0, REUSED3/EMERGENCY37이다. 재수집의 이동 경험은 생겼지만 전체 이동 수는
늘지 않았고, 실제 종료 실패 경험은 크게 줄었다. source를 넓히면서 terminal 노출을
줄이는 교환이 생겼으며 이후 Q/미래보상과 전체 주행으로 검토해야 한다.
refresh 충돌34→2는 안전성 향상이 아니라 재방문한 상태 분포의 변화일 수 있다.
이 부분 결과를 전체1,000회나 안정성으로 외삽하지 않고 실행 조건을 유지한다.

## 23:45 — 첫400회 실제 배치 노출 검산: 새 실패와 초기 실패의 교환

직전 goal turn은 새 이력 진단 구현·실제 감사·후속 실행 연결로 진전이었다.
이번에는 학습3008214/부모3008213/대기3012859의 실제 생존과 상태를 확인하고
읽기 전용 배치 분석을 추가했다. 23:44:56 학습442/1,000·refresh88로 실행 중이다.
전체 평가와 새 모델 Q/return 진단은 아직 미완료이며 학습 조건은 그대로다.

로컬 `review_exposure.py --updates 400`는 두 모델의 첫400개 tick/update JSONL
prefix와 공통 자기warmup5,000의 실제 bootstrap/reward를 읽었다. 원래 sampler로
400개 batch의 index hash를 하나씩 재현해 각102,400추출을 검산하고, 분석 후에도
입력 prefix가 동일함을 확인했다. 출력은 `exposure-400.json`, SHA
`b4ba859fab2e6047929d6995d799ffa8dcb18fa0ffff26d5c609bc7b470b8a94`.
script SHA `e127085b62855a10a403bd56a8ce40ca0181e88be4f5b207c2825627a3fad652`.
실행 명령은 프로젝트 환경의 Python에 `JAX_PLATFORMS=cpu`를 지정했다.
최종 replay 자체의 감사·저장 RNG 대조와는 구분한다.

| 첫400회 배치에 실제 추출된 횟수 | terminal-only 대조군 | 비종료거절 후보 |
| --- | ---: | ---: |
| 전체 / 자기warmup | 102,400 / 68,634 | 102,400 / 69,630 |
| 새400전이 | 33,766 | 32,770 |
| 공통 초기 실패44행 | 2,530 | 3,526 |
| 공통 초기 목표6행 | 3,267 | 3,267 |
| 신규 실제 충돌행 | 7,602 (81행) | 1,094 (9행) |
| 전체 실패pool | 10,132 (125행) | 4,620 (53행) |
| 새 이동 경험 | 13,898 (174행) | 14,479 (191행) |
| 새 정지 경험 | 19,868 (226행) | 18,291 (209행) |
| refresh의 비종료 source | 0 | 6,290 (78행) |
| refresh의 실제 종료 source | 6,862 (74행) | 307 (2행) |

그룹은 겹치며 횟수는 batch 표본이지 독립 경험/장면 수가 아니다. 초기44행은 양쪽이
같은 저장 경험이므로 직접 노출 비교가 가능하지만 새 전이의 상태/경로는 서로 다르다.
종료pool의8개 고정추출과uniform/recent의 추가추출이 함께 포함된 수치다.
새후보의 실패 경험 전체가 사라진 것은 아니며, 신규 충돌 노출을 줄이고 초기 실패
노출을 늘리는 교환이 확인됐다. 이것이 Q 적합을 개선할지 악화시킬지는 아직 미확정이다.

400전이는 기존KEEP325/FOLLOW75, 후보KEEP305/FOLLOW95다. 첫200회에서는 후보의
이동 수가 더 적었지만 첫400회 누적에서는174→191로 더 많다. 부분200기록은 당시
범위로 보존하며 전체 수집/주행의 결론으로 쓰지 않는다. 후보의 비종료source78행 중
1행(전이390)은400회까지 배치에 추출되지 않았다. 원래 insert후batch256추출이며
모든행의즉시추출을보장하지않는다. 미추출 index를 별도 검산한 결과 전체 온라인의
미추출2행은 refresh390/완료경로397이었다. 최신400행 자체가 미추출된 것은 아니다.

최종1,000회가 완료되면 같은 명령의 `--updates 1000`으로 전체 노출을 다시 계산하고,
전체 개발 주행 및 연결된 Q/return/coverage와 대조한다. 자동 새학습/보상/검사 변경이나
채택은 없다. 안정정책·다른seed·동일조건Rule-based·별도평가 목표는 계속 미완료다.

## 23:57 — 1,000회 학습·저장감사 완료, 최종 수집/배치 노출 확인

직전 goal turn은 실제400batch 검산으로 진전이었다. 이번에는 확인된 session49022/
학습PID3008214를45초 이하 간격으로 관찰했고, 학습은23:55:54 exit0로 완료됐다.
audit1000 child3016820도23:55:57 exit0·COMPLETE다. 실제source재현3016849가 실행
중이며 전체개발/정책Q/return은 아직 미실행이다. timeout에 따른 재시작은 없었다.

- 새전이1,000 = main433 + 완료경로367 + refresh200, actor/Q 각1,000업데이트·BC0.
- 자기warmup17배열·전배치index/RNG·원본/후속관측·초기learner reference바이트동일
  검산 완료. 원래Gaussian/보상/검사/PD/loss/lr/tau/teacher0 조건을 유지했다.
- 새 learner SHA `937e284a255806fa476b0bb6f1e56b971759385729f3f144a795b20877bc6d3c`.
- audit SHA `218c51b7d1cc6546ae83e68e9a2c3672f8a4ab50e8ca2033f86f454543d8cee3`.

완료 로그로 `review_collection.py`와 `review_exposure.py --updates 1000`을 실행했다.
각1,000batch/256,000추출의index hash를 원래sampler로 재현하고 실제 전이 분포를
집계했다. `collection-final.json` SHA
`04e101d71b154a8321b5b012b0ca455dfb0edae3c048009801dca524f469fe8c`,
`exposure-1000.json` SHA `adaef25c3b1a00914fa775ce48a60844354567e2186e0957de53a2568a5a14d0`.

| 완료된1,000수집 | terminal-only 대조군 | 비종료거절 후보 |
| --- | ---: | ---: |
| 이동 / 정지 전이 | 503 / 497 | 597 / 403 |
| 경로 수락 | 242 | 239 |
| KEEP / FOLLOW 전이 | 676 / 324 | 609 / 391 |
| refresh 전이 | 194 | 200 |
| refresh 이동 / 정지 | 0 / 194 | 93 / 107 |
| refresh 수락 | 0 | 8 |
| refresh 실제 비종료 / 충돌 | 0 / 194 | 195 / 5 |
| refresh 고유 source 전이ID | 14 | 188 |
| 공통 초기 실패44행 batch추출 | 4,293 | 7,858 |
| 신규 실제 충돌행 batch추출 | 27,727 (208행) | 3,332 (17행) |
| 비종료 source refresh batch추출 | 0 | 21,176 (195행) |
| 종료 source refresh batch추출 | 25,706 (194행) | 912 (5행) |

양쪽은train8고유장면을 방문했지만 동일한 중간 상태 분포가 아니다. 새 refresh200개는
KEEP118/FOLLOW82, 실제 source는비종료거절195/종료5다. 재수집이 이동/비종료 경험을
확보한 것은 확인됐지만 전체 수락 수는 비슷하고 대부분의 새 경로는 여전히 거절됐다.
수집 중 충돌208→17을 평가 충돌률 개선으로 해석하지 않는다.

후보에는 실제 목표 도달 전이1개가 추가돼349번 batch에 추출됐다. 이는 main의
train `30a56b458fc5b63d` KEEP, 전이772/tick67에서 학습 도중 관찰한 것이다.
해당68tick은 수락58/거절10, NEW58/REUSED4/BRAKE2/EMERGENCY4이고 마지막4tick은
연속거절·마지막source EMERGENCY였다. 같은train장면의 이전 수집은55tick충돌이었으나
방문 시점/중간 정책이 다르므로 최종모델 성능 대조가 아니다. 목표 도달을 연속적인
유효 경로나 안정 주행으로 확대하지 않는다. 전체개발16회 결과는 아직 없다.

## 23:57 — 현재 버전 Rule-based 재평가·전체 주행 기록 저장 대기

최종 Rule-based 비교 준비에서 과거
`outputs/sac_stability_20261117/revalidated-final/baseline.json`을 현재 규약 함수로
검사했다. 기존16장면/명령쌍과 집계 재계산은 일치했지만 `validate_rule_baseline`은
현재버전불일치로 거절했다. 다른 version 항목은 같고 code SHA는 과거c217dc…/
현재96fa1d…로 다르다. 과거 호환revision ee211b9와 현재tracked core의 차이는
learner/networks/runtime3파일이다. 이 차이 목록을 새로운 측정이나 무조건적인
실행 동등성 근거로 쓰지 않고 원래 보고서를 보존한 채 다시 측정하기로 했다.

로컬 `outputs/sac_rule_baseline_current_20261120/historical-preflight.json` SHA
`82eb34ca4b294be22eca6e7f5c7a530374844cf4de5c72c696e184cdb00cbcb0`.
새 runner3015652/session33463은23:51:15 시작·실제생존확인,
WAITING_FOR_POLICY_DIAGNOSTICS다. 부모3012859/start_ticks74492863과 성공 종료를
확인한 뒤에만 진행한다. 학습/성능 측정과 동시 실행하지 않는다.

- source210/input150 SHA·source복사본 동결. decision SHA
  `75cfee3ec8c255b39057e4c0bc616809fe9ece1da9ed5dbd4621c8601fcbe161`.
- 기존 `evaluate_sac_models.py evaluate --policy teacher --seed 0`로 같은manifest의
  전체16쌍을 평가하고, 원래 `scripted_proposer('teacher')`와 실제 `capture`로
  동일16쌍의 도로/차량/제안·실행경로/거절/제동/종료 프레임을 다시 저장한다.
- 독립검산은 현재버전 Rule-based validator, 원본 모든tick의 수락·source·event·
  보상·정지/종료 집계, 별도 formal평가와의 일치, SAC 평가의 초기frame/지도 일치를
  확인한다. 반복capture를32개 독립장면으로 세지 않는다.
- 두번의주행은 formal 지표/영상근거 확보이며 학습 또는 replay 삽입0이다.
  GIF 자체 생성/최종3seed비교·별도test는 아직 미실행이다.

새 로컬 runner/capture/reviewer는 AST 및 경로/동결hash 검사를 통과했다. 실제 평가와
capture는 아직 대기 중이며 과거 baseline의 버전만 최신으로 바꾸지 않았다.
안정정책 확보·다른seed·동일조건 최종비교·별도평가 목표는 계속 미완료다.

## 23:59 — 실제 source7조합 재현·학습검토 완료, 전체개발 평가 시작

verify1000은23:58:50, review-training은23:58:51 exit0로 완료됐다.
실제 원본경로/저장행동과 새행동을 관측된 provider/source종류/stage별 첫1개씩
총7조합에서 Waymax로 재현했다. KEEP main/branch의 비종료거절과실제종료4조합,
FOLLOW main/branch의 비종료거절과branch실제종료3조합이다. FOLLOW main의종료
source는 관찰되지 않아 미검증이다. 비종료4조합은bootstrap=true, 실제충돌3조합은
false였다. 원래 보상/검사/제어/next_obs와 부모 상태 보존을 확인했다.

- `execution1000.json` SHA `da40085503ac1f2a032ac0b4f1b1ce14840970c9b450787b693d936d65123ee4`.
- `training-review.json` SHA `d4eb10bb8cb285204953e5888a07fc7bafcdd03c103fb9f1a17722a9caa0232d`.

23:58:51 `validation-full` child3017699가 시작했고 실제생존을 확인했다.
8고유개발장면×KEEP/FOLLOW16회 전체 평가 후 원본 대조를 수행한다.
부모3008213/session49022·후속진단3012859/session69685·baseline대기3015652/session33463
유지. 현재 학습·저장·관측된 실제source재현 검증 완료와 주행 성능 검증 미완료를 구분한다.

## 2026-11-21 00:10 — 고속 장면 회귀 과정 검토와 GIF 생성

직전 goal turn은1,000학습/감사/실제재현 완료와 현재baseline 재평가 연결로 진전이었다.
이번에는 실제 평가PID3017699를 확인하면서 완료된고속 `8b4c1e7e09f5f4c3`의KEEP/FOLLOW
두쌍을 원본trace로 대조했다. 같은초기속도11.119323m/s·프레임/지도·모델/버전 일치를
확인하고 모든tick의 보상/수락/제동/정지/종료를 재계산했다.

| 각 명령의 고속 주행 | terminal-only 대조군 | 비종료거절 후보 |
| --- | ---: | ---: |
| 종료 / tick | ROUTE_GOAL / 72 | COLLISION / 73 |
| 거절 / 제동 / 정지 | 11 / 8 / 0 | 35 / 32 / 9 |
| 마지막 수락 tick | 62 | 39 |
| 첫 제동 / 긴급제동 tick | 65 / 없음 | 42 / 47 |
| 첫 정지 tick | 없음 | 65 |
| 종료 직전 연속 거절 | 10tick | 34tick |

새후보는40tick부터PREDICTED_OFFROAD/REUSED,42BRAKE,47EMERGENCY로 진행했고
48tick에는 tangential_acceleration/lateral_acceleration/curvature/tangential_jerk/
curvature_rate 위반이 함께 기록됐다. 최종속도0에서차량4877과 접촉했다.
접촉 차량 중심은ego기준 종방향약−5.011m·횡방향+0.099m다. 시간/위치 관찰이며
충돌 책임·불가피성·유일 원인 판정이 아니다. 이전목표도마지막10tick거절이있어
연속유효경로나목표이후안전을보장하지않는다.

학습중수집의이동수가늘었어도이장면의최종결정적정책은회귀했다.
이는1고유개발장면/2명령의사후분석이며전체개발/새seed/독립평가결론과구분한다.
제동후기본가속도연장과경로거절의연결은앞선
[재계획 진단](sac-replanning-20261118.md)의별도관측과함께읽되이번위반의유일원인으로
확정하지않는다. 보상/검사/PD/학습조건은변경하지않았다.

로컬 `outputs/sac_rejection_refresh_comparison1000_20261120/`:

- `high-regression-review.json` SHA `3fd8ba8bc8eca9a459222780ad43fd297280649b5b9bc569af2c6a2e5145d2be`.
- `visualization/high-speed-keep-evidence.json` SHA `e878fbb0d4fbda23b4f7af79c43b17bb48d8c47671f7dcebc87354cf502374c3`.
- `visualization/high-speed-keep.gif` SHA `ee2e0ac90429ba4b9eef52e8b75564197afdf981c162507f2bc249d874145010`.
- `visualization/high-speed-keep.png` SHA `e818b35bc557830ee4e2d8dfa19c4aea25d47d76e8af9719c7afc31cc04ff18d`.

기존render명령을원본trace의label만바꾼복사본에실행했고새시뮬레이션/정책호출0이다.
GIF38frame·1440×900·7.3초실제시간+1.5초종료hold를PIL로검산했다. 마지막PNG를열어
도로/주변차량/궤적·거절/제동·왼쪽목표종료hold/오른쪽충돌표시를확인했다.
두패널은**순수SAC 수집방식 비교이며 Rule-based 비교가 아니다**.
Rule-based의현재버전실제평가/capture는별도대기중이며로컬GIF를공개링크로표시하지않는다.

## 2026-11-21 00:12 — 전체 개발16회 완료: 새 후보 미채택

부모3008213/session49022의6jobs가00:11:10 모두exit0로완료했고실제종료확인.
새후보8고유개발장면×KEEP/FOLLOW16회와완료된대조군16회의원본trace/모델·버전·
초기frame/지도·모든tick보상/검사/종료와집계를대조했다. 반복개발·seed0 평가다.
`full-validation-review.json` SHA `25e6af45ec7bf4e7da01aff19bb51823260f960d5a54bcb6352288835366d1e6`.

| 전체 개발16회 / 모델 | terminal-only 대조군 | 비종료거절 후보 |
| --- | ---: | ---: |
| 목표 / 충돌 | 2 / 14 | 0 / 16 |
| 성공률 / 충돌률 | 12.5% / 87.5% | 0% / 100% |
| 실제tick | 962 | 962 |
| 경로거절 | 348 (36.17%) | 634 (65.90%) |
| 제동 | 330 (34.30%) | 565 (58.73%) |
| 정지 | 430 (44.70%) | 456 (47.40%) |
| 평균return | -17.757717198794897 | -20.467394395738786 |

KEEP/FOLLOW각8회에서대조군1목표7충돌→후보0목표8충돌이다. 새후보는모든장면에서
실패했으며이전고속장면의두성공을잃었다. **이번1,000회 비종료거절 후보는미채택이다.**
수집에서 이동503→597·비종료실패경험 증가/새train목표1개가 생겼어도 최종개발주행
개선으로이어지지않았다. 1,000회실패를모든후속예산/알고리즘의불가능성으로확대하지않는다.

정지출발2고유장면×2명령4회는양쪽모두충돌·328/328tick정지였다. 후보거절84→302,
제동80→255로악화됐다. 고속6회는대조군2목표4충돌→후보6충돌·정지44→62,
저속2회는양쪽충돌·정지18→26이다. 중속4회는양쪽충돌·정지40동일이고,후보294tick/
기존290tick으로정지비율만조금낮아진것을출발/주행개선으로해석하지않는다.

진단runner3012859/session69685는부모의완료/실제종료후RUNNING으로전환했고
candidate-coverage child3021296은 상태 파일에서 실행을 관찰한 뒤 00:12:08 exit0로
완료했다. 최초 기록의 이 PID에 대한 “실제생존확인”은 당시 별도 ps 근거가 없어
정정한다. 이후 branch-Q child3022588의 실제 생존을 ps로 확인했고 00:14:29에 완료했다.
선행상태4seed실제누적보상진단이이어진다. source/입력을동결했으며가중치/replay쓰기0.
현재baseline runner3015652/session33463은그진단의완료를대기중이다.
다음은새종료경험노출감소·초기실패노출증가·최종정책행동거리와실제return을대조해
다음수집변경의근거를확보하는것이다. 새예산/seed/채택을자동추가하지않는다.
안정정책·다른seed·동일조건Rule-based최종비교·별도평가목표는미완료다.

## 2026-11-21 00:22 — 실패 상태 소실 확인·분리 보존 옵션 구현

직전 질문 응답은 개념 설명으로, 목표 작업의 진척으로 세지 않는다. 이번에는 현재
작업 트리와 실제 PID3012859/3023936/3015652를 다시 확인하고 완료된 coverage와
branch-Q 원본을 검토했다. 기준 HEAD `510098bf63c63e94cbdcced69a78c3979d95c1f9`,
브랜치 `feature/yuntae-sac-training`. 아래 새 옵션은 이 시점 미커밋이며 성능 미검증이다.

두 모델의 coverage 각6,000행을 실제 replay와 대조하고 공통 warmup17배열, Q 차이,
조건부95% 경계 포함 여부, 유한한 값 및 총256,000 배치 추출을 독립 검산했다.
로컬 `policy-diagnostics/coverage-review.json` SHA
`0416663621aeb478ecc9dd104cf55200ec6a3b334e55ddec22b60c1ca2cc6f08`.

| 동일 초기 실패44행 / 고정 train 상태 | terminal-only 대조군 | 비종료 거절 후보 |
| --- | ---: | ---: |
| 초기 실패44행의 배치 추출 | 4,293 | 7,858 |
| 동일 저장 행동의 Q MAE | 0.862135 | 2.651968 |
| KEEP tick28 현재 평균 행동의 Q−실제 종료 보상 | 0.298321 | 7.611898 |
| FOLLOW tick59 현재 평균 행동의 Q−실제 종료 보상 | 0.393081 | 11.607156 |

두 고정 상태에서 저장 행동·현재 평균·원래 Gaussian4표본은 모델별12회 모두 실제
COLLISION/bootstrap=false였다. 그러므로 이 표의 현재 평균 Q−reward는 실제 종료
오차다. 과거 대조군 결과는 재사용이며 새 독립 표본으로 세지 않는다. 새후보의
refresh200행 중81개가 최종 정책의 조건부95% 경계 안에 들어왔지만, 종료 source5개는
0/5였다. 고정 상태의 행동 거리이지 새 정책 전체 상태 방문이나 안전성의 증거가 아니다.

`review_source_eviction.py`로 실제1,000전이의 FIFO와 모든200회 선택을 원본 lineage와
대조했다. source 자격은 비종료556개/종료12개였고, 종료12개 모두 비종료 source 삽입
때 퇴출됐다. 종료 source가 하나 이상 있는 선택 시점은67회였지만 실제 종료 선택은
5회뿐이었고, 종료 source8개는 한 번도 재방문되지 않았다. FOLLOW의 첫 두 종료
source388/468도 재방문0회였다. 종료 상태 소실을 확인했으나 이를 주행 회귀의 유일
원인으로 단정하지 않는다.

같은 과거 도착 순서에만 각각4개 보존·번갈아 선택을 적용한 선택 계산은 종료97회/
비종료103회였다. **실제 새 정책 실행이나 성능 예측이 아니다.** 새 수집은 이후 상태와
학습 자체를 바꾸므로 실제 결과를 다시 검증해야 한다.
로컬 `source-eviction-review.json` SHA
`f5205d705d61a6bf90cf890bbaa9c51499c7511982fb7f6c8d819bb8a2bc4560`.

### 구현한 후속 비교 옵션과 검증 범위

- `scripts/sac_stratified_refresh.py`: 종료 실패/비종료 거절을 각각 FIFO4에 보존하고
  둘 다 있으면 번갈아 선택한다. 한 종류만 있으면 그 종류를 선택한다. 총보존8개,
  기존 전이4개당 최대1회, 재귀 금지와 원래 상태 검증을 유지한다.
- `scripts/diagnose_sac_stratified_refresh.py`: fresh 학습 진입점이다. 기존 동결된
  드라이버에서 bank·scope 식별자만 바꾼 별도 복사본이며 실제 수집/보상/업데이트
  계산은 동일함을 대조했다. 보상·안전검사·PD·Gaussian·loss/lr/tau·배치 추출·
  전이당 전체SAC1·BC/teacher0을 바꾸지 않는다. B 평균은 기존 별도 허용 비교다.
- `audit_sac_stratified_replay.py`, `audit_sac_stratified_refresh.py`,
  `verify_sac_stratified_refresh_execution.py`: 종류별 FIFO/순환/교대 선택, 실제 source와
  후속 관측, 실제 종료/부트스트랩, warmup·batch/RNG·전체 업데이트 및 Waymax 재현을
  검사하도록 연결했다. source가 종료였다는 이유로 새 결과를 종료로 복사하지 않는다.
- 관련 CPU36pytest가 0.42초에 통과했다. 신규7개와 기존 bank/adapter/branch 재현
  검사를 포함한다. AST·CLI help·드라이버 식별자 외 동등성 확인을 완료했다.
  CPU 테스트는 실제 주행 성능을 대신하지 않는다.

규약5절의 확률 경로와10절의 실제 전이당 업데이트를 유지하는 수집 분포 비교다.
기본 채택이 아니며 안정 정책 미확보다. 다음 실제100전이 pilot은 현재 Q/return 진단과
Rule-based 재평가가 끝난 뒤 실행하도록 준비한다. 아직 새 학습 실행 확인은 없다.
100회는 구현/저장/관측된 source 조합의 실제 재현을 확인하는 예산이며, 같은1,000회
성능 비교·다른 seed·최종 Rule-based 비교·별도 평가로 자동 승격하지 않는다.

## 2026-11-21 00:25 — 실제 미래 보상 진단 완료·후속 pilot 대기

진단 runner3012859/session69685는 00:23:18에 coverage/branch-Q/soft-return/검산
4jobs 모두 exit0로 완료했고 실제 PID 종료를 확인했다. 모델/입력/source 해시,
같은 train 상태의 식별자, 각 trace의 실제 종료와 보상·entropy 할인합을 검산했다.
`policy-diagnostics/diagnostic-review.json` SHA
`031ff39d96f493d2920b0326341c7dacdc4b610e4c53a191bdcce699f318aff3`.

| 고정 train 선행상태 × 원래 Gaussian4seed | terminal-only 대조군 | 비종료 거절 후보 |
| --- | ---: | ---: |
| KEEP tick0 실제 soft-return 평균 | -13.649697 | -12.748185 |
| KEEP Q−실제 평균 | 11.597606 | 11.454919 |
| FOLLOW tick4 실제 soft-return 평균 | -10.635522 | -10.475469 |
| FOLLOW Q−실제 평균 | 9.117351 | 9.383596 |
| 실제 종료 결과 / 모델 | 충돌8/8 | 충돌8/8 |

같은2개 train 상태의 소표본 진단이며 대조군 완료 결과는 재사용했다. 새 후보의
KEEP return이 조금 높아도 전부 충돌했고, FOLLOW의 Q 차이는 커졌다. 이 결과는
개발16회 실패와 함께 미채택 판단을 유지한다. 보상 증가를 안전한 주행으로 해석하지
않는다. 학습/가중치/replay 수정0인 진단이다.

현재 버전 Rule-based 평가 runner3015652/session33463은 00:23:34에 formal 평가
child3026743을 시작했다. 두 PID의 실제 생존을 확인했다. 같은16쌍의 평가 후 실제
capture와 원본 검산이 이어진다. 이 시점 새 Rule-based 결과는 미완료다.

새 구현 `c1e1f03470875ee9d1f9b5d2f12665dac06b7c5d`를 같은 브랜치에 push하고
원격 SHA 일치를 확인했다. 추가 PR/병합은 하지 않았다. 로컬
`outputs/sac_stratified_refresh_pilot100_20261121/`에 fresh100 검증을 준비하고
00:24:06 runner3027013/session33870의 실제 생존과 WAITING_FOR_RULE_BASELINE을
확인했다. 부모3015652/start_ticks74564168의 성공 완료·실제 종료를 확인한 뒤
train100 → audit100 → verify100 → review100을 실행한다. 부모 실패 시 근거를
남기고 pilot을 시작하지 않는다.

source97/input73의 SHA·복사본을 동결했다. decision SHA
`3b29ec1b5abc91f9e71e49d157e65551c0cfb129f259fdf954c7757e2d03a161`.
seed0·같은 자기warmup5,000·최초 learner 바이트 대조·별도허용B-f64/rawgain0.0001/
physicalQ/complete-path/goal8+failure8 조건을 유지한다. 바뀌는 것은 상태의 종류별
보존과 선택이다. 검사/보상/PD/원래Gaussian/전체SAC1·BC/teacher0은 동일하다.
실제100전이와 source 두 종류의 관측·재현을 모두 요구하는 검토를 연결했지만,
현재 새 학습은 아직 시작하지 않았다. 이후 예산은 결과 검토 뒤 정한다.

재개 명령:

```bash
python -m json.tool outputs/sac_rule_baseline_current_20261120/status.json
python -m json.tool outputs/sac_stratified_refresh_pilot100_20261121/status.json
```

안정 정책·다른 seed·동일 조건 최종 Rule-based 비교·별도 평가는 아직 미완료다.
별도 평가 후보29/32와 원본 데이터 오류17, GCS 재인증 대기 및 drive/시간 독립성
미확인도 그대로 남아 있다. 현재 새 수집을 기본 설정으로 채택하지 않았다.

## 2026-11-21 00:37 — 현재 버전 Rule-based 정식 평가16회 완료

직전 목표 작업은 구현·테스트·완료 진단·Notion 반영까지 진행한 progress였다.
이번에는 HEAD `226e4c4d1b6a714657932f7d38544bbaf0892f17`과 실제 실행을 재확인했다.
Rule-based formal child3026743이 00:35:35 exit0로 종료한 뒤 capture3029682가
실행 중이다. 부모3015652/session33463의 생존도 확인했다. 아래는 완료된 formal16회이며
capture/전체 원본 대조는 아직 미완료다.

같은 manifest·seed0·현재 FrenetTeacher v4·기존 보상/검사/PD를 사용했다.
8고유 개발 장면×KEEP/FOLLOW16회다. 새 학습이나 teacher replay 추가는 없다.

| 현재 Rule-based formal 평가 | KEEP8회 | FOLLOW8회 | 전체16회 |
| --- | ---: | ---: | ---: |
| 목표 | 4 | 3 | 7 (43.75%) |
| 충돌 | 4 | 4 | 8 (50%) |
| TASK_LOST | 0 | 1 | 1 (6.25%) |
| 실제 tick | 534 | 543 | 1,077 |
| 경로 거절 | 153 | 86 | 239 (22.19%) |
| 제동/비상제동 | 145 | 78 | 223 (20.71%) |
| 정지 | 194 | 194 | 388 (36.03%) |
| 평균 return | -10.046467 | -10.091431 | -10.068949 |

로컬 `outputs/sac_rule_baseline_current_20261120/review_formal_result.py`를 실행해
완료 상태/16쌍·episode tick/행동/이유 합계·수락/정지 범위·유한 보상·집계/추종 RMS를
검산했다. 같은 장면의 모든 episode 필드를 과거
`outputs/sac_stability_20261117/revalidated-final/baseline.json`과 대조한 결과 차이0개다.
version에서 달라진 항목은 code_sha256 하나이며 과거 보고서를 수정하지 않았다.
실제16회 재실행으로 같은 수치를 확인한 것이며 과거 수치를 현재로 재표기하지 않았다.

- 현재 `baseline.json` SHA
  `fb5cfb3f96ad4fc0c26f14143bc0a81fa7f1d907c88e90016f47f7b0ca2d7ae2`.
- `formal-baseline-review.json` SHA
  `3071e627d501d93b49f2fbc9018590566dcbc1a517575c44015751c19ccab797`.
- 현재 core SHA `96fa1d625b0d8525b1de6de05a12cc132721a03ee47eb51dff3d6e25f02e30ef`.

실패한 비종료 수집 SAC seed0는 전체 개발 목표0/16·충돌16/16·거절65.90%·
제동58.73%·정지47.40%·평균return-20.467394로 현재 Rule-based보다 미달이다.
Rule-based도8회 충돌했으므로 안전한 상한이나 실패 장면의 회피 불가능성을 입증한
결과는 아니다. 실패 진단이며 새3seed·최종평가·독립test 완료로 세지 않는다.

### 실제 궤적 대조와 비교 GIF 준비

capture는 같은16쌍을 실제 제어하며 궤적을 저장한다. formal 결과와 tick별 보상/수락/
제어출처/종료/정지를 대조하고 SAC와 최초frame·지도를 비교한다. 독립32장면으로 세지
않는다. 이 검산이 끝나야 부모가 완료된다.

읽기 전용 `render_reviewed_comparison.py`는 이미 분석한 고속 장면
8b4c1e7e09f5f4c3/KEEP 한 쌍을 선택했다. baseline 전체 검산·실제 종료 뒤 원본 trace의
label만 바꾼 복사본으로 Rule-based/SAC GIF를 만든다. 추가 정책 호출/episode는 없고
대표 장면 선택이나 전체 성능 판정으로 주장하지 않는다. frame/해상도/물리 재생시간
검사와 PNG 직접 확인을 연결하며, 이 시점 새 GIF는 아직 생성되지 않았다.

로컬 `outputs/sac_rule_baseline_current_20261120/visualization-current/`의
runner3029047/session57026은 00:32:49 시작·실제 생존 확인,
WAITING_FOR_REVIEWED_BASELINE이다. decision SHA
`8bbbc31a18c4bfd39f53bde4d0dd358d748dd9e2414819456079c47b43cd252a`.
새 수집 pilot3027013/session33870도 실제 생존·부모 성공 종료 대기를 확인했다.
00:37 현재 새 학습은 시작 전이며 기존 조건/동결 파일을 유지했다.
다음은 capture/원본 검산 → 새100회 학습·감사·실제source 재현 검토다.
안정 정책과 전체 목표는 미완료다.

## 2026-11-21 00:42 — 종류별 수집의 정책 진단 연결

Rule-based capture3029682와 부모3015652, 후속 pilot3027013, GIF 대기3029047의
실제 생존을 재확인했다. formal16회는 완료, capture는 진행 중이며 새 학습은 아직
시작 전이다. 동결된 baseline source210/사전input150 및 pilot source97/input73의
SHA를 다시 검산했고 기존 파일은 변경하지 않았다.

새 `scripts/sac_stratified_diagnostic_audit.py`는 종류별 수집 scope에서 전용 실제
lineage 감사를 필수로 실행한다. 과거 scope는 기존 rejection/terminal/일반 감사기를
거치며 새 계약의 위장·혼합·누락과 미완료 검증을 거절한다. report/replay/adapter의
해시를 결과에 남긴다.

`diagnose_sac_stratified_branch_q.py`와 `diagnose_sac_stratified_soft_returns.py`는
기존 rejection 진입점에서 감사 import/해시만 바꾼 별도 복사본이다.
`audit_sac_stratified_coverage.py`는 실제 수집 종류·감사·출력 contract를 연결했다.
원래 정책 행동/실제 보상/Q/할인·entropy 계산은 바꾸지 않는다. 진행 중인 동결 파일을
유지하려고 별도 진입점을 사용했으며 이 진단은 learner/replay를 수정하지 않는다.

관련 CPU32pytest(신규 adapter11 포함)가0.42초에 통과했다. 계약 누락/혼합/위장,
과거 감사 우회, 미완료 검증을 거절하는 테스트와 기존 adapter/종류별 bank/분기 재현
테스트를 포함한다. AST·세 CLI help도 통과했다. 새 adapter를 실제 완료된 비종료
수집1,000·terminal-refresh1,000·그 이전 control1,000의 저장 이력에 실행해 모두
COMPLETE를 확인했다. 각 실제 원본의 기존 검사를 그대로 수행했다.

새 종류별 모델은 아직 학습 전이므로 그 모델의 실제 Q/coverage/return 진단은 미실행이다.
pilot의 실제 실행·저장·source 재현 결과를 먼저 검토하고, 같은 예산 성능 비교를
선택할 때 이 도구를 연결한다. CPU 통과를 성능 통과로 해석하지 않는다.

## 2026-11-21 00:56 — Rule-based 원본 검산·GIF·종류별100회 완료

Rule-based runner3015652/session33463은00:47:51에 formal/capture/review 모두
exit0로 완료했고 실제 종료를 확인했다. 8장면×KEEP/FOLLOW16쌍의 formal 지표와
별도 실제 capture의 tick별 보상/수락/종료/정지/제어출처를 대조했고, SAC와 초기
frame·지도도 모두 일치했다. 재실행을 독립32장면으로 세지 않는다.
로컬 `outputs/sac_rule_baseline_current_20261120/baseline-review.json` SHA
`c07391861f47c5b4e23e159ad8e05cc37b0b5caba8d5b85749ff7cb2da9bb8da`.

### 실제 Rule-based/SAC 비교 GIF 저장·시각 확인

`visualization-current/high-speed-keep.gif`를 원본 trace로 생성했다. 8b4c1e7e09f5f4c3
KEEP 한 장면의 사후 예시이며 추가 정책 호출/episode0이다. Rule-based는74tick 목표,
실패한 비종료 수집 SAC는73tick 충돌이다. 각각 거절5/35·제동5/32·정지0/9tick이다.
Rule-based도 마지막5tick은 거절/제동이므로 연속 유효 경로나 이후 안전을 보장하지 않는다.

GIF38frame·1440×900·실제7.4초+종료hold1.5초=8,900ms를PIL로 검산했다.
중간4.4초 frame과 마지막PNG를 열어 도로/차량/제안·선택 경로/궤적·속도/제동/거절
표시를 확인했다. 오른쪽 충돌 후 hold, 왼쪽 목표 종료와 서로 다른 정책 label도 확인했다.
영상 범위로 전체 성능이나 안정 정책을 판정하지 않는다. 원본 render 결과를 보존하고
별도 시각 확인 파일을 저장했다. 로컬 위 baseline 루트의 `visualization-current/`:

- GIF SHA `ac5b10b74339db2d71ab1d2e08e592adc38e958218ab479c4fc2816e9ef7771b`.
- PNG SHA `b66a501c489ad3e01a4ac8afe30e2cdb16f3f1ea916130982a728a3add0b5ab6`.
- evidence SHA `e960c7895b3e5d990f54559ea875b29a0a07e8ba9b174d121192ada2042b642d`.
- `visual-inspection.json` SHA `3ec7a6f69e176ac6bd082879bc458813a045df621d4ad704053637d09afb08c6`.

### 종류별 수집100회 실행 검증 완료

pilot3027013/session33870은00:53:20에4jobs 모두exit0 완료·실제 종료를 확인했다.
실제100전이=main51+완료경로29+refresh20이며 actor/Q100·BC0다. 초기 learner는
기존 reference와 바이트 동일하고 warmup17배열·배치/RNG·source/다음 관측·실제 종료
플래그를 감사했다. 새 진단 adapter로도 완료된 실제 이력 검사를 통과했다.

재수집의 source와 실제 결과는 각각 비종료13/충돌7이었다. source의 종료 여부를
새 전이에 복사한 것이 아니라 원래 collector/Waymax에서 실제 결과를 확인했다.
KEEP main/branch×종료/비종료4조합의 첫 사례를 원본 prefix와 함께 실제 재현했다.
비종료 bootstrap=true, 충돌 false 및 보상/검사/제어/후속 관측·부모 보존을 대조했다.
FOLLOW는 관찰되지 않아 미검증이다.

refresh20개는 수락0·이동4/정지16·REUSED1/EMERGENCY19이며 거절 이유는
PREDICTED_OFFROAD2/KINEMATIC_LIMIT2/PREDICTED_COLLISION16이다. 고유 source15개다.
이동/정지 수집 통계는 pre-state anchor 속도0.1m/s 경계로 구분했다.
전체100전이의 이동44·수락23도 수집 중 통계이며 최종 정책 주행 성능이 아니다.
**구현/실행 검증은 통과했지만 새 안정 정책이나 유효한 재수집 경로 확보는 아니다.**

로컬 `outputs/sac_stratified_refresh_pilot100_20261121/`:

- pilot-review SHA `7fc1cb103fefc28439960ad6928efc054c88a95b51e0dac365ec11009965ceaa`.
- audit100 SHA `57636d14b084a03c50c910b8fff636ec6ea10c603f40442fb5f4417077690630`.
- execution100 SHA `75d89275285ced0b9775d5b2e607d2b69dc84d8764742b983140075bab7cad37`.
- collection-review SHA `1d7e6c2a1956db056618684c3429cfdd8cf905fc7918039ab7c05fce91f851fa`.

### 같은1,000회 비교와 후속 정책 진단 시작/대기

동일 예산 비교를 선택했다. 100회 가중치를 이어받지 않고 같은 seed0/자기warmup5,000/
초기 learner로 fresh1,000을 실행한다. 주 대조군은 직전의 보존 구분 없는 비종료
수집1,000(0/16목표), 보조 참조는 terminal-only1,000(2/16)과 현재 Rule-based(7/16)다.
모델 구성/override/수집 설정/배치 추출/초기바이트를 대조했다. 바뀌는 것은 종류별
보존과 선택이며 자격·총8개·base4/최대20%·재귀 금지는 유지한다. 비종료만 있을 때
보존량도 기존8→4가 되므로 종료 보존 하나만의 인과 실험으로 주장하지 않는다.
보상/검사/PD/Gaussian/loss/lr/tau/전체SAC1·BC/teacher0은 같다.

로컬 `outputs/sac_stratified_refresh_comparison1000_20261121/`의
runner3037500/session91162·train3037519가00:54:27 시작했고 실제 생존 확인.
source214/input138 SHA·복사본 동결, decision SHA
`a851698bb83b3e7a7e4943d583e45603b72a1dd248ae5f4776c5c1930927040b`.
첫 update 입력 learner도 SHA7363d29a0c9500d58a259ba8953ff21428ede1660ee9e66be9b9f571ef014c2c로
reference와 실제 바이트 동일했고 이후22update를 확인했다. 이후 모든 계산의 동일성을
주장하지 않는다. 학습→감사→실제 재현→학습 검토→전체 개발16회→원본 검산6jobs다.

비교 검산기는 이전의 완료된 자료만으로 사전 실행했다. 이때 control/candidate 모두
같은 과거 비종료 모델을 가리켰고 추가 주행0·새모델 결과0임을 별도 scope로 명시했다.
새 실행 결과로 오인하지 않는다. `reviewer-preflight/full-validation-review.json` SHA
`c441b1c27e8d81a118544ba6b6caa3f9e17d5591695320c3de6695c4f7558237`.

같은 폴더 `policy-diagnostics/`의 runner3039851/session65591이00:55:59 시작·생존 확인,
WAITING_FOR_COMPARISON이다. 부모3037500/start_ticks74961685의 성공 완료와 실제 종료
후 coverage→같은train 종료 상태Q→같은 선행 상태4seed 실제 미래보상→검산을 수행한다.
source216/사전input217 동결, decision SHA
`78c2f5eb16cf5ee14d35612d92e1fffdd9ea92801e52bc64a799ff8314ad4e80`.
대조군 진단은 완료된 비종료 수집 결과의 바이트 복사본이며 새 독립 표본으로 세지 않는다.
모델/replay 쓰기0, 새 진단 성능은 아직 없고 추가예산/seed/채택 자동연결은 없다.
안정 정책·다른seed·최종 동일조건 Rule-based 비교·별도 평가는 계속 미완료다.

## 2026-11-21 01:05 — 종류별 수집의 첫300전이 보존·학습 노출 검산

fresh1,000회 실행을 유지하며 첫300전이의 실제 bank 선택과 학습 배치를 검산했다.
주 대조군은 완료된 비종료 수집1,000의 첫300개이며, terminal-only 대조군과 혼동하지
않는다. 양쪽 모두 main127/branch113/refresh60, 실제300전이·300update다.
초기 learner는 같지만 이후 상태/행동이 달라질 수 있어 일치 상태의 인과실험이 아니다.

| 첫300전이 내 항목 | 기존 비종료 수집 | 종류별 보존 수집 |
| --- | ---: | ---: |
| 적격 source: 비종료 / 종료 | 197 / 5 | 197 / 5 |
| 재수집 source: 비종료 / 종료 | 58 / 2 | 33 / 27 |
| 비종료 유입 때문에 퇴출된 종료 source | 5 | 0 |
| 아직 한 번도 선택되지 않은 종료 source | 3 | 0 |
| 재수집 경로 수락 | 0 / 60 | 0 / 60 |
| 재수집 이동 상태 | 19 / 60 | 10 / 60 |
| 실제 신규 충돌 전이 | 7 | 32 |
| 신규 충돌 전이의 배치 추출 횟수 | 681 | 2,500 |
| 종료 source 재수집의 배치 추출 횟수 | 191 | 2,036 |
| 비종료 source 재수집의 배치 추출 횟수 | 3,908 | 2,454 |
| 전체 신규 이동 전이 / 배치 추출 | 124 / 8,560 | 115 / 8,064 |

종류별 FIFO4와 선택 교대, 기존 FIFO8의 실제 모든 선택을 독립적으로 재구성했다.
종료 source5개는 새 방식에서 모두 재방문됐고, 그중 최초 source36은 이후 종료
source242가 들어오며 퇴출됐다. 비종료 source가 종료 source를 밀어내는 문제는 이
prefix에서 관찰되지 않았다. 재수집 실제 결과는 기존 비종료58/충돌2, 새 방식
비종료33/충돌27이며 source 라벨로 새 결과를 합성하지 않는다.

양쪽 각각300개 배치의 실제 indices SHA를 RNG로 재구성했다(각76,800추출).
공통 warmup5,000의 실패44개 추출은2,743→2,378, 신규 경험 추출은20,604→20,969다.
새 실패 경험의 노출이 늘었지만 이동 경험 노출은 감소했다. 전이/배치 추출 수는
독립 episode 수가 아니며 중첩 그룹을 합산하지 않는다. 이동은 사전 anchor속도
0.1m/s 기준이고 validation의 사후 정지 비중과 구분한다.

**충돌 직전 상태의 보존/학습 노출이라는 작동 의도는 확인했으나, 안정적인 주행이나
Q 오차 개선은 아직 입증하지 못했다.** 첫300개는 KEEP만 포함한다. 재수집 경로는
모두 거절됐으므로 실패 상태를 더 방문한 사실을 안전 경로 확보로 해석하지 않는다.
후속 전체 replay/실제 Waymax source 재현/전체 개발16회/Q·실제 미래보상 검증을
완료한 뒤 판단하며, 중간 통계만으로 수집 비율이나 학습 예산을 바꾸지 않았다.

로컬 `outputs/sac_stratified_refresh_comparison1000_20261121/`에서 실행:

```bash
JAX_PLATFORMS=cpu python outputs/sac_stratified_refresh_comparison1000_20261121/review_exposure.py --updates 300
python outputs/sac_stratified_refresh_comparison1000_20261121/review_bank_prefix.py
```

- `bank-prefix300.json` SHA `9df1e84c54bf99890c7419d43fa9e2708ddbbaa41b2af720c528d2346a29bd45`.
- `exposure-300.json` SHA `7c1ebdb7e7a873ceb7f68ae15597664bb78bf24a0f399192a0e7939c054325b7`.
- `prefix300-sources/`에 양쪽 tick/update 원본 prefix 복사본을 보존했고 전후 SHA를 대조했다.
- `prefix300-verification.json`: 학습/후속 진단의 동결 source/input 전부 일치.
  01:05:31 KST runner3037500·train3037519·후속3039851의 `/proc` 실제 command/stat을
  확인했고388/1,000update 진행 중이었다. 후속 진단은 같은 부모 성공 종료 대기 중이다.
- 이전 Notion4페이지 갱신은 재조회하여 실제 반영을 확인했다. 03의104절은1회 존재하고,
  과거103절·04의 과거 기록·첫 화면의4개 하위 링크가 유지됐다. `notion-readback.json` 참조.

이 작업은 읽기 전용 분석이며 실행 중 학습 소스/설정/보상/검사/PD·BC/teacher0을
변경하지 않았다. 새로운 학습/정책 평가/pytest를 추가로 실행한 결과가 아니다.

## 2026-11-21 01:12 — 완료된 대조군의 실제 학습 장면·명령·초기속도 노출

진행 중인 종류별1,000회가 끝나기 전에 이전 두 완료 대조군의 입력 분포를 검산했다.
새로운 정책 평가가 아니라 저장 replay와 실제 배치 추출 이력에 대한 읽기 전용 분석이다.
현재 후보의 최종 범위나 성능을 이전 결과로 대체하지 않는다.

`review_dataset_exposure.py`에서 실제 replay 해시·warmup provenance/6개 메타배열·
episode/scene/tick 연결·관측의 명령 one-hot와 mask·배치 indices SHA/RNG를 대조했다.
manifest의 train59장면/110쌍에만 속하는지 확인하고 실제 main episode 순서가 seed0
permutation과 일치하는지 확인했다. 관측 명령 검사는 warmup과 신규를 합친6,000행이다.

두 대조군 모두 자기warmup5,000은63개의 episode에서 온47고유장면/63쌍이며,
신규1,000은8고유장면/8쌍이다. 신규8쌍은 이미 warmup에 포함되어 전체 replay도
47장면/63쌍이다. 이는 저장/추출 범위이지 현재 정책의 성공 경험 또는 행동 coverage가 아니다.

| 실제 배치 사용 범위 | 비종료 수집1,000(주 대조군) | terminal-only1,000(보조 참조) |
| --- | ---: | ---: |
| 전체 배치 추출 | 256,000 | 256,000 |
| 과거 자기warmup5,000 추출 | 145,285(56.752%) | 141,963(55.454%) |
| 신규1,000 추출 | 110,715 | 114,037 |
| 신규 KEEP/FOLLOW 전이 | 609 / 391 | 676 / 324 |
| 신규 KEEP/FOLLOW 배치 추출 | 62,240 / 48,475 | 75,137 / 38,900 |
| 정지 출발 장면의 신규 전이 / 추출 | 0 / 0 | 0 / 0 |
| 정지 출발 장면의 warmup 전이 / 추출 | 687 / 22,189 | 687 / 21,578 |

여기서 정지 출발은 manifest reset 초기속도<0.1m/s다. 이동 장면에서 제동 후 정지한
전이를 이 그룹에 포함하지 않으므로 기존 전체 정지 전이3,347개라는 수치와 다르다.
warmup의 정지 출발 경험은5고유장면/7쌍을 포함한다. **정지 출발 학습 데이터 전체가
없다는 뜻이 아니라, 이 두 짧은 비교에서는 현재 정책의 신규 경험이 없다는 뜻이다.**

train에는 KEEP/FOLLOW 각각 정지 출발5쌍·저속4쌍, 중속27/23쌍·고속23/19쌍이 있다.
seed0 순서의 첫 정지 출발은9번째1bedf33cd20c1e7b/FOLLOW이며 두 완료 대조군은
8번째 장면 진행 중1,000전이 예산을 소진했다. 분기/재수집도 실제 학습 예산을 쓴다.
짧은 fresh 실험을 같은 seed에서 반복하는 것만으로 전체 train 경험이 늘지는 않는다.

이 분포는 초기속도별 평가를 해석할 때 고려해야 할 한계다. 분포 편중을 주행 실패의
유일 원인으로 단정하거나, 장면 순서 변경만으로 정지 출발 문제가 해결된다고 주장하지
않는다. 현재 후보 완료 후 동일 검산과 정지 출발4개 개발 평가·Q 결과를 함께 검토한다.
이전부터 확인된 정지 상태의 매우 작은 행동 탐색/평균 변화 문제도 별도 원인으로 유지한다.
이 분석 때문에 현재 실행의 장면 순서·전이 예산·배치·보상·안전검사·PD를 바꾸지 않았다.

로컬 `outputs/sac_stratified_refresh_comparison1000_20261121/`:

- `JAX_PLATFORMS=cpu python …/review_dataset_exposure.py --arm control`,
  `--arm terminal_reference` 실행 완료. `--arm candidate`는 학습 미완료라 아직 미실행이다.
- `dataset-exposure-control.json` SHA `3950ffa1663b6651462ee2ac45dcb3a2f338890cab4954a1de8df2e02614e547`.
- `dataset-exposure-terminal_reference.json` SHA `6369ca6bf33fc8c767b00fec380746517360c96ce146f2dcb1c0a564b8dc1dfb`.
- `dataset-order-review.json` SHA `20916b481f686a617e31e9b94d1bb376c4152a59b57dc8fef58e9b5434b816e5`.
- 분석 소스 SHA `982ad7f7491e02370aec1fc47d7cbcc1f7d00711faa6f1d82514630ac7b54e74`와
  사용한 source7개 복사본은 `dataset-review-sources/manifest.json`에 보존했다.

분석기 최초 실행은 command_id를 행동 종류 번호로 간주한 잘못된 검증에서 실패했다.
규약의 command_id는 episode 내 명령 식별자이므로 해당 가정을 제거하고 실제 관측의
maneuver one-hot/mask로 바꾼 후 두 완료 replay의6,000행씩을 검증했다. 학습/환경 오류나
명령 규약 변경이 아니다. 보고서는 수정 후 통과한 실행에서만 생성했다.

01:12:03 KST 실제 `/proc` command로 runner3037500·train3037519·후속3039851을
재확인했고663/1,000update RUNNING이었다. 후속 전체개발/Q 검증은 계속 대기 중이다.
안정 정책·다른seed·최종 동일조건 Rule-based 비교·별도 성능 평가는 아직 미완료다.

## 2026-11-21 01:25 — 종류별1,000회 학습·실제8조합 검증 완료, 전체개발 평가 시작

runner3037500/session91162의 train3037519는01:19:13, 감사3048479는01:19:17에
exit0 완료했다. 실제 source 재현3048512는01:24:13, 학습 검토3049746은01:24:14에
완료했다. 학습/검증 자식의 실제 종료를 확인했고 다음 평가3049753이01:24:15 시작했다.

실제1,000전이=main384+branch416+refresh200, actor/Q 각각1,000update·BC0이다.
warmup17배열·teacher0·전체 배치 indices/RNG·source/다음 관측·실제 종료 플래그·
같은 초기 learner 바이트를 검산했다. 기록된 과거 source 종료값을 새 결과로 합성하지 않는다.

### 실제 수집과 배치 노출 비교

| 완료된1,000회 수집 | 기존 비종료 수집 | 종류별 수집 |
| --- | ---: | ---: |
| 적격 비종료 / 종료 source | 556 / 12 | 557 / 13 |
| 재수집 비종료 / 종료 source | 195 / 5 | 103 / 97 |
| 종료 source의 비종료 유입에 의한 퇴출 | 12 | 0 |
| 재수집 경로 수락 | 8 / 200 | 5 / 200 |
| 재수집 이동 상태 | 93 / 200 | 56 / 200 |
| 전체 신규 이동 상태 | 597 / 1,000 | 569 / 1,000 |
| 전체 신규 경로 수락 | 239 / 1,000 | 234 / 1,000 |
| 신규 실제 충돌 전이 / 배치 추출 | 17 / 3,332 | 110 / 15,769 |
| 종료 source 재수집 배치 추출 | 912 | 13,843 |
| 비종료 source 재수집 배치 추출 | 21,176 | 11,298 |
| 공통 warmup 실패44개 배치 추출 | 7,858 | 5,297 |

종류별 모든 실제 bank 선택을 재구성했고 비종료 유입이 종료 source를 퇴출시키는 현상은
없었다. 종료13개 중 미선택1개는 마지막 branch source999로, 예산 종료 때 bank에
남아 있었다. 이를 퇴출로 오인하지 않는다. 재수집 실제 결과는 비종료103/충돌97이며,
재방문과 배치 추출은 독립 episode 수가 아니다. 이동은 사전 anchor 속도 기준이다.

종류별 신규 경험은 KEEP601/FOLLOW399,7고유장면/7쌍이다. 신규 배치 추출113,195,
warmup142,805(55.783%)이며 전체 replay는47장면/63쌍이다. 기존 두 대조군은 신규
8장면/8쌍이었다. 종류별 후보도 정지 출발 신규 경험0이며 해당 warmup687전이의
추출은21,721회다. 전체 명령 one-hot/mask6,000행과 manifest train membership,
seed0 main 순서·최종 RNG를 검산했다. 이전 장면 분포 검산기를 같은 코드로 실행했다.

**보존/노출 문제는 줄었으나 이동/수락 수집 지표가 좋아진 것은 아니다.** 새 데이터가
더 많아졌다는 이유로 안정적인 정책이나 주행 성능 개선을 주장하지 않는다.

학습 중 main7회는 모두 충돌했다. 별도 branch에서는30a56b458fc5b63d/KEEP의
collection852에서 ROUTE_GOAL1회가 있었다. root tick0에서68step을 진행했고55개
제안이 수락됐으나 마지막4step은 거절·fallback이었다. 마지막은 KINEMATIC_LIMIT/
EMERGENCY다. 매 step 정책이 갱신되는 수집 결과이며 고정된 최종 정책의 성공률이 아니다.

### 8조합 실제 재현과 평가 진입

KEEP/FOLLOW×main/failure_branch×비종료 거절/종료 실패8조합 각각의 첫 source를
원본 prefix와 함께 실제 재현했다. 원본 행동과 재수집 행동, 보상·검사·제어·다음 관측·
부모 환경 보존을 대조했다. 네 비종료 사례의 bootstrap=true, 네 충돌 사례=false다.
FOLLOW main 비종료의 실제 원래 보상−0.007874470434608202와 새 보상
0.0022282608987103385도 재현됐다. source 값을 그대로 복사한 결과가 아니다.
재현은총8reset/227step이며 성능 평가8episode로 합산하지 않는다.

평가3049753은 `evaluate_sac_short_updates.py --variant online --labels updated`로
전체 개발8장면×KEEP/FOLLOW16회를 실행 중이다. 주 대조군은 기존 비종료 수집
0목표/16충돌, 보조 참조는 terminal-only2/14 및 현재 Rule-based7목표/8충돌/TASK_LOST1.
참조 평가는 재사용하고 새 독립 표본으로 세지 않는다. 주행 원본 검산 완료 후 대기 중인
진단3039851/session65591이 coverage→같은 train 종료 상태Q→실제 미래보상을 수행한다.
01:25에 runner3037500·평가3049753·진단3039851 실제 `/proc` command와 모든 동결
source/input SHA 유지도 확인했다. 새로운 예산/seed/채택은 연결하지 않았다.

로컬 `outputs/sac_stratified_refresh_comparison1000_20261121/`:

- 모델 SHA `f020d96c2fb4642556da5f1decf9a2126884a934fcd3c46c6f599cd685a217da`.
- replay SHA `43b33177c3c9be3c72c229f698abab204956b4e7d788c0aad30e3ba8d54f1cca`.
- `audit1000.json` SHA `1aad666465dfed249e02080daabbc6dc5d46db0ec608bcc8b4ad4d72516226ac`.
- `execution1000.json` SHA `2eb230f2ebd528586ba917df380f7c0c6cebd1ecfdc567c4dafbfb1659212431`.
- `training-review.json` SHA `1fbb73189c350dcb4b445073e1e183a9e89c19da54ea8ae7e06011e08dac8f9b`.
- `bank-prefix1000.json` SHA `63608155541a3e420ed8fc287e3c1cbe9e2c724ffefd09c5600de34471073d6a`.
- `exposure-1000.json` SHA `ef2886d24de51c5f045f575b74845e499941ac3f54efd59bd87ef29834940e99`.
- `dataset-exposure-candidate.json` SHA `4ae321b05d2e882313fcc40086ff480290afac3f32ee8c48ab30014dcc215d93`.
- `collection-completion-review.json` SHA `4295fad40c2beeb3465b10fbc3d9e7c3bef0f3e0129ad6161c9c2763ccd61128`.

CPU에서 `review_exposure.py --updates 1000`, `review_dataset_exposure.py --arm candidate`,
`review_bank_full.py`를 실행했다. 위 저장 감사·실제 재현과 별도의 집계 검증이며,
새 학습·정책 평가·pytest를 추가 실행한 결과가 아니다.
안정 정책·다른seed·최종 동일조건 비교·별도 성능 평가는 미완료다.

## 2026-11-21 — 종류별1,000 전체개발 실패와 완료된 종료Q 검증

전체개발 평가3049753은01:35:00, 원본 검산3052940은01:35:02에exit0 완료했다.
runner3037500/session91162는01:35:03 COMPLETE_PENDING_DECISION 후 실제 종료했다.
같은8고유개발장면×KEEP/FOLLOW16쌍의 원본tick·모델/버전·초기frame/지도·종료·
보상합과 모든 지표를 검산했다. 기존 세 참조의 평가는 재사용이며 새 독립 주행이 아니다.

**종류별 수집 후보는 목표0/16·충돌16/16으로 미채택이다.** KEEP와FOLLOW 각각0/8·8/8.
보조 참조 terminal-only는2목표/14충돌, 현재 Rule-based는7목표/8충돌/TASK_LOST1이다.

| 전체16회 | 기존 비종료 수집 | 종류별 수집 |
| --- | ---: | ---: |
| 목표 / 충돌 | 0 / 16 | 0 / 16 |
| 실제 tick | 962 | 862 |
| 거절 | 634(65.90%) | 411(47.68%) |
| 제동 | 565(58.73%) | 387(44.90%) |
| 정지 | 456(47.40%) | 460(53.36%) |
| 평균 return | -20.467394395738786 | -20.336391644312826 |

### 전체 거절률·보상 개선으로 오인하지 않는 코호트 검산

같은 초기 상태를 기준으로 이동 출발12회와 정지 출발4회를 나눠 실제 trace를 다시
검산했다. 정지 출발4회는 두 모델 모두328/328tick 정지 후 충돌했다. 이 그룹의
거절이302→78로 줄어 전체 거절률을 낮췄지만 출발/회피에 성공한 것은 아니다.

이동 출발12회는 tick634→534, 거절332→333(52.37%→62.36%),
제동310→311(48.90%→58.24%), 정지128→132(20.19%→24.72%)로 악화됐다.
중속4회의 충돌까지tick294→202가 전체 분모 감소의 대부분이다. 빨리 충돌한 주행은
거절 기회도 줄어드므로 절대 개수와 비율을 함께 판단한다.

평균 return 증가+0.131002751426은 정지 출발4회의 기여+0.140014467013과
이동 출발12회의 기여−0.009011715587로 분해된다. 가중합이 실제 전체 변화와 일치한다.
보상 증가를 목표 도달이나 안전 개선으로 취급하지 않는다.

두 모델 각각16회 모두 마지막3초에 제동이 관찰됐고, 최종 자차속도<0.1m/s 및 뒤쪽
차량 접촉이 함께 기록됐다. 원본의 실제 colliding_ids와 마지막3초 상대 중심/속도를
대조했으며 접촉 객체 누락0이다. 이 중 이동 후 첫 정지 이전/동시에 제동이 관찰된 경우는 각각12회이고, 처음부터
끝까지 정지한 경우는 각각4회다. 후자를 제동 때문에 출발하지 못한 것으로 분류하지
않는다. 최초 보고서의 동시관찰 라벨은 `failure-analysis-initial-chronology-label/`에
원본과 분석 소스를 보존하고 수정했다. 이는 관찰된 순서/기하이며 제동의 인과 효과, 과실,
회피 불가능성의 증거가 아니다. 추가 Waymax 주행 없이 완료된 trace에서 분석했다.

로컬 `failure-analysis/report.md`와 `report.json`, `cohort-metrics.png/svg`를 만들었다.
PNG를 열어3지표×3코호트의 막대/수치/범례·episode 및tick 분모 표시가 읽힘을 확인했다.
순서 라벨 정정 후 PNG는 앞서 열어 본 파일과 SHA가 동일함을 검산했다.
그래프는 같은 개발 평가 자료의 표현이며 새로운 평가 표본이 아니다.

### 완료된 coverage·실제 종료상태Q 검증

후속 runner3039851/session65591이 부모 성공 종료를 확인한 후 모델/입력을 추가
동결했다. coverage3053035와 branch-Q3054352는 완료·실제 종료 확인. 같은 train
77ae312df21e80b9/KEEP tick28, baac7893b025167a/FOLLOW tick59에서 저장행동·현재평균·
원래Gaussian4표본을 실행했고 모델당12행동 모두 실제COLLISION/bootstrap=false다.

| 현재 평균 행동의 실제 종료Q 오차(Q−실제 보상) | 기존 비종료 | 종류별 |
| --- | ---: | ---: |
| KEEP tick28 | 7.61189752960205 | 0.19342428588867122 |
| FOLLOW tick59 | 11.607155601501464 | 0.8950326843261713 |

종류별 현재평균 Q는−19.820575714111328/−19.121967315673828, 실제 종료보상은
−20.014/−20.017이다. 종료에서는 bootstrap0이므로 이 비교의 target은 실제 보상이다.
이 두 상태에서의 오차 감소와 전체 주행 개선을 혼동하지 않는다.

공통 warmup 실패44행의 저장행동Q MAE도2.651967976030352→1.1554850835604256으로
줄었다. 같은44행의 배치 추출은7,858→5,297이었다. 이를 재추출 증가만으로 설명하지 않는다.
양쪽 CSV6,000행/256,000추출의 산술·현재분포95%거리 판정·공통5000행 식별자·
모델/원본 해시·고정 상태/저장 행동 일치와 실제 종료Q 오차를 별도 검산했다.
Gaussian 거리 영역은 문자 그대로의 support/안전/충분한 탐색의 보장이 아니다.

미래보상 child3055692는01:46:38, 최종 검산3058193은01:46:45 완료했다.
runner3039851/session65591도01:46:48 COMPLETE_PENDING_DECISION 후 실제 종료했다.
같은 두 선행train상태에서 평균 첫 행동 후 원래Gaussian으로 실제 종료까지4seed씩
실행했다. 새8회 모두충돌이며 기존 대조군8회는 완료자료 재사용이다.

| Q−실제 soft-return 표본평균 | 기존 비종료 | 종류별 |
| --- | ---: | ---: |
| KEEP tick0 | 11.454919395485373 | 12.317480893367206 |
| FOLLOW tick4 | 9.383595679324893 | 9.421274820925062 |

종류별 KEEP의 초기Q−1.7710416316986084 대비 실제 평균−14.088522525065814,
FOLLOW는 초기Q−1.253022313117981 대비 실제 평균−10.674297134043043이다.
종료 직전Q 적합은 좋아졌지만 더 이른 상태의 미래 비용 차이는 개선되지 않았다.
소수 확률표본의 평균이므로 정확한 기대Q 또는 통계적 유의성으로 주장하지 않는다.
최종 검산은 실제8trace의 할인보상·entropy·종료/부트스트랩·동일상태/모델·동결SHA를
대조했다. 기본 채택/추가 예산/다른seed로 자동 확장하지 않았다.

새 학습 전 기존 `diagnose_sac_bellman_trace.py`로 저장된8경로의 실제 행동을
재생하여 온라인Q·targetQ·보상/entropy를 구간별 대조했다. 프로세스3059295/
session27308은01:59 exit0 및 실제 종료 확인. 로컬 `bellman-trace/`에 완료 결과와
`bellman-trace.log`를 남겼다. 이는 추가 학습이나 독립 평가8회가 아니다.
소스·보상·검사·PD·BC/teacher0·loss/lr/tau는 변경하지 않았다. 경로의 오차 분해만으로
수집 방식이나 tau 변경의 인과 효과를 확정하지 않으며, 남은 비종료 가치 오차의 위치와
기존 후속경험 수집 방식의 차이를 검토한 뒤 보완을 정한다.

로컬 `outputs/sac_stratified_refresh_comparison1000_20261121/`:

- `full-validation-review.json` SHA `86d3ac63e8a4b989803c34c73682c81cf12888791aaa9df4b83299461d87a98e`.
- `failure-analysis/report.json` SHA `2dac8b5919a75d67ac72374d6f7684b2d0d9d9592af62ccf73138f3467f99ae8`.
- `failure-analysis/visual-inspection.json` SHA `fa0b0407ff01361db56b34e6dc7f96cf89f5e0531908b794856af840925ec990`.
- `policy-diagnostics/completed-terminal-q-review.json` SHA `cd15e2519fc4d13a20bbf3d60b39f0c1a6a06651b5eb9d4a3666cdd46eebfebf`.
- `policy-diagnostics/diagnostic-review.json` SHA `a735bbe242dedb0c3d051076b56d0b18a66c0149073d2db35763325492ae4c4b`.
- `bellman-decision.json` SHA `53ea7fd2c7e33b277d7ce94e7a928d3de9a6f24418a0c1fc9adb01a20f5e1006`.
- 추가 집계 명령은 `review_validation_failures.py`와
  `policy-diagnostics/review_completed_terminal_q.py`다. CPU에서 실행했고 원본은 수정하지 않았다.

안정 정책·다른seed·최종 동일조건 Rule-based 비교·별도 성능 평가는 미완료다.


### 실제 후속 관측·행동의 replay 포함 범위와 다음 수집 가설

읽기 전용 `review_successor_coverage.py`로 최종 replay6,000행의 모든 관측 feature/
field/item mask를 byte fingerprint로 대조했다. 현재 모델의 coverage CSV와 연결하여,
각 비종료 전이의 다음 관측에서 실제 행동 경험이 저장·추출됐는지도 확인했다.

| 비종료 전이의 다음 관측 | 기존 비종료 재수집 | 종류별 재수집 |
| --- | ---: | ---: |
| 비종료 전이 | 195 | 103 |
| 같은 관측을 입력으로 한 저장 행동 있음 | 187 | 98 |
| 해당 전이 뒤에 수집된 같은 관측 행동 있음 | 58 | 24 |
| 현재 분포95% 거리 영역 안의 저장 행동 있음 | 72 | 43 |
| 위 행동이 실제 배치에 한 번 이상 추출됨 | 72 | 43 |

종류별 main377/branch409의 비종료 다음 관측은 모두 나중 전이로 이어졌다. 따라서
**후속 상태가 전혀 없다는 설명은 기각**한다. 다만 재수집 다음 관측103개 중43개만
현재 정책 분포의95% 거리 영역 안에 저장 행동이 존재했다. 나머지60개를 탐색 불가,
안전 위반 또는 데이터 support 밖으로 단정하지 않는다. 완전히 일치하는 관측만 세므로
작은 수치·시간·피드백 차이도 미일치가 될 수 있고, 근처 경험/일반화는 별도 문제다.

완료된 train 확률 경로의 prestate와 replay를 대조하면 KEEP는 각 경로 시작1개만,
FOLLOW는0개가 완전히 같았다. 기존/종류별 모두 같은 양상이다. 이는 현재 경로와
과거 경험의 exact identity 범위이며 새로운 독립 주행이나 일반화 실패의 단독 증명이 아니다.
처음 집계 source도 원본 SHA와 동일한 `review_successor_coverage_initial.py`로 보존했다.

**설계 가설, 아직 구현/실험/채택 전:** 기존 비종료 재수집은 한 step 뒤 환경을 버린다.
그 결과 상태에서 현재 정책으로 실제 후속 행동을 계속 수집하는 옵션을 검토한다.
기존 main 실패 후 한 번 실행하는 complete-path와 달리, 재수집 슬롯에서 이어가는
현재 정책 경로다. 종료 source4/비종료 source4 보존, 총 재수집 최대20%, 원래Gaussian,
보상/안전검사/공통PD, 실제 새 전이당fullSAC1, BC/teacher0은 유지한다.
한 active 비종료 환경을 다음 비종료 슬롯에 이어 쓰고 종료 슬롯과 교대하는 구상이며,
새 후속 상태는 bank 적격 source로 재귀 삽입하지 않는다. 실제 terminal 또는 예산 소진까지
이어가되 예산 소진을 가짜 terminal로 기록하지 않는다. 추가 active 환경 메모리와
source 퇴출 후 이어지는 root lineage, parent 보존·실제 후속 재현을 별도로 검증해야 한다.

과거 quota/complete-path도 주행을 개선하지 못했으므로 길이 증가만으로 성공을 예상하지
않는다. 현재 terminal 보호를 유지한 채 최신 정책의 후속 행동을 수집하는 차이를 검증할
가설이다. 기존8경로 Bellman 검산도 아래와 같이 마쳤다. 다음은 이 수집 옵션의 제한된 구현과
실제 후속 상태 재현/저장 감사 pilot이며, 주행 성능과 채택은 별도로 판단한다.


추가 로컬 근거:

- `policy-diagnostics/completion-summary-review.json` SHA `2951cf2895d0faec296fc4918e16e56357ce90fe0e78834222797c102fa04930`:
  두 모델의 soft-return 평균·표준편차·gap 재계산, 모든 동결 source/input 일치 재확인.
- `successor-action-coverage-review.json` SHA `75a8d5ba34b7721b10cef98febbadfaf68342e25e9676ea150baa5e1159f2072`.
- 최초 관측만 집계 `successor-coverage-review.json` SHA `09487eb192e7eaed6e343ce756d2644fc0cb7435eb8dfb3dfb6256704d534396`.

완료한 읽기 전용 재생 명령(같은 출력 폴더가 있으면 다시 시작하지 않는다):

```bash
/home/autonav/miniconda3/envs/its-merge/bin/python scripts/diagnose_sac_bellman_trace.py \
  --diagnostic outputs/sac_stratified_refresh_comparison1000_20261121/candidate1000 \
  --states-from outputs/sac_physical_q_20261119/branch1000 \
  --returns outputs/sac_stratified_refresh_comparison1000_20261121/policy-diagnostics/candidate-soft-returns \
  --manifest outputs/sac_readiness_20261116/inventory/eligible_manifest_reset_v2.json \
  --output outputs/sac_stratified_refresh_comparison1000_20261121/bellman-trace
```

`python outputs/sac_stratified_refresh_comparison1000_20261121/review_bellman_components.py`
로 모든 suffix 직접 할인합·구간 잔차·초기차이 합·제어 source별 항을 검산 완료했다.


### 01:59 — 저장된8경로381step 재현·구간별 가치 오차 검산 완료

KEEP139/FOLLOW242step의 원래 행동·보상·검사·제어·종료·다음 관측 SHA와 부모 보존,
모델/replay/동결 source를 확인했다. 새 정책 행동이나 독립 평가 표본을 추가한 것이 아니다.
모든 suffix를 직접 할인합하여 재귀 계산과 대조했고8개 전체 합도 검산했다.

| 평균 경로의 부호 있는 오차 항 | KEEP | FOLLOW |
| --- | ---: | ---: |
| 초기 Q−실제 soft-return | 12.317420 | 9.421223 |
| 마지막 종료 기여 | 2.992297 | 1.093048 |
| 비종료 구간 기여 | 9.325123 | 8.328176 |
| sampled target 잔차 합 | 2.826457 | -4.695507 |
| target−online 차이 항 합 | 9.490964 | 14.116730 |

종료/비종료 합과 마지막 두 항의 합은 각각 초기 gap과 같다. 상쇄가 있는 부호 합이며
독립 인과 기여/설명률이 아니다. target 차이가 크다는 이유로 tau를 바꾸지 않는다.
이번 batched critic forward와 원래 forward의 f32 차이는 KEEP−0.0000605583/
FOLLOW−0.0000516176이며 기존1e−3 허용 내다. 앞선 MC표의 초기Q를 바꿔 쓰지 않았다.

비종료 EMERGENCY의 평균 잔차 기여는 KEEP9.503471/FOLLOW6.518008이고,
FOLLOW BRAKE는2.470880이다. 그러나 일부 새 종료 상태의 오차도 크다. KEEP rep3는
실제 충돌tick38에서Q−8.6191339493 대비 reward−20.014, 오차11.3948660507이었다.
고정된 tick28의0.1934만 보고 현재 정책이 만드는 모든 충돌상태를 학습했다고 할 수 없다.
다음 수집은 종료 source 보호를 유지하면서 재수집의 현재 정책 후속 행동·실제 새 종료를
얻는지 확인한다. 효과가 확정된 것은 아니며 종료/비종료 양쪽 진단을 함께 유지한다.

- 로컬 `bellman-trace/report.json` SHA `c87f8bda99b13b7504beac092cf9906ee933a772d743551fd1a2bb9f48619093`.

- 로컬 `component-review-candidate.json` SHA `cbd03fc60a8b2e49527bab64743cb24a7033baa7c0f8933120da818777888617`.


## 2026-11-21 — 현재 정책의 비종료 후속 경험 수집 옵션 구현

종류별1,000의 실패 분석을 근거로 별도 successor-refresh 비교 옵션을 구현했다.
기존 소스와 actor/critic/loss/reward/checker/PD는 수정하지 않았다. 상태 수집 분포만
바꾸며 BC/teacher0, original Gaussian, 실제 새 전이당fullSAC1을 유지한다.

비종료 재수집 슬롯은 한 active환경을 실제 다음 관측부터 이어 실행한다. 종료 source4/
비종료 source4·base4당최대1회·종류 교대와 종료 source 단일 실제 행동은 유지한다.
root 퇴출 후에도 active경로를 보존하고, 실제 종료 뒤에 새 root를 선택한다. 재수집에서
다시 source를 만들지 않으며 수집 예산 경계는 비종료 bootstrap을 유지한다. 정책은 각
슬롯의 최신 가중치이므로 고정 최종 정책의 연속 평가 경로가 아니다.

지속 보관 대상은 bank8+active1+퇴출된 원본root최대1로10환경까지다. 실행 중인
main/branch 환경과 일시적인 실행fork는 별도다. 원본 root도 끝까지
검증하기 위해 보관하므로 단순히 기존8에1만 추가한다고 설명하지 않는다. 실제 결과의
보상/종료를 source에서 복사하지 않는다. terminal-source의 새 결과가 비종료여도
bootstrap=true를 유지하되 이 경로는 이어 수집하지 않는 구분이다.

새 파일:

- `scripts/sac_successor_refresh.py`: pending 실제 결과/부모·대기환경 보존과 bounded bank.
- `scripts/diagnose_sac_successor_refresh.py`: 기존 학습계산에 별도 bank와 실제결과 feedback 연결.
- `scripts/audit_sac_successor_replay.py`, `scripts/audit_sac_successor_refresh.py`:
  root/후속전이/종류선택/FIFO/관측/flags·배치/RNG/초기learner/teacher0 검증.
- `scripts/sac_successor_reconstruction.py`, `scripts/verify_sac_successor_refresh_execution.py`:
  interleaved main/terminal 경로를 섞지 않고 root의 실제 이력+successor prefix를 재현.

검증: 관련CPU41pytest(신규13 포함)통과,3CLI help·컴파일·diff 검사를 완료했다.
신규 테스트는 실제 환경 이어짐/원본root 퇴출/종료 슬롯 교대/종료 후 새 root/비종료
bootstrap/미기록 결과/잘못된 flags/환경변조/재귀source/끊긴 이력·다른장면을 확인한다.
드라이버 diff는 bank/scope/식별자와 `after_refresh` 결과 연결 외 학습계산 변경이 없다.

다음은 같은seed0·자기warmup5000·초기learner 바이트·B-f64/rawgain0.0001/
physical-whitened zero-action critic·complete-path1/window6/horizon200·같은 sampler의
fresh100전이 pilot이다. 학습→저장 감사→관찰된 root/후속경로 실제 Waymax 재현→집계
검산을 연결한다. 소스/입력 동결 후 실행하며 에러는 보존하고 자동 재시작하지 않는다.
아직 실제 수집·FOLLOW/종료된후속경로 조합·주행 성능·기본 채택은 미검증이다.


### 예비100전이 실행 시작 확인

구현 `80ba5e1daf6900b418d222760684eb5e34b22576` push/원격SHA 일치를 확인했다.
로컬 `outputs/sac_successor_refresh_pilot100_20261121/`의 runner3063654/session62388은
02:07:59, train3063661은02:08:00 시작했다. 2026-11-21T02:10:08.704879+09:00에 두 실제
command/starttime과 source274/input66 SHA 일치를 다시 확인했고 실제 저장 전이
23/100개를 확인했다. 첫 update 입력 learner는 reference와 바이트 동일,
SHA7363d29a0c9500d58a259ba8953ff21428ede1660ee9e66be9b9f571ef014c2c다.

`decision.json` SHA960c62d5bc746ad063130701ff020958317017ca8a58fadb385bf872397a5907,
`preflight-review.json`은 bank/scope/결과callback 치환 후 driver AST 동일과 core diff0을
확인했다. `launch-review.json`은 실제 생존/중간 저장 확인이다. 수집 전체 완료·저장 감사·
후속경로 실제 재현·FOLLOW/종료된후속경로 조합·성능 검증은 아직 미완료다.

확인 명령:

```bash
python -m json.tool outputs/sac_successor_refresh_pilot100_20261121/status.json
python -m json.tool outputs/sac_successor_refresh_pilot100_20261121/candidate100/report.json
tail -n 5 outputs/sac_successor_refresh_pilot100_20261121/train100.log
```

실행은train100→audit100→verify100→review100만 연결했다. 관찰된 root·비종료후속·
종료후속의 첫 사례와 마지막후속을 실제 원본 경로로 재현한다. 미관측 조합은 미검증으로
남긴다. 실패는보존·정지, 시간경과만으로 재시작하지 않는다. 새1,000학습/새seed/평가/
기본채택은 자동 연결하지 않았으며100회 검산 뒤 다음 실험을 정한다.


### 후속 경로 수집 결과의 읽기 전용 진단 연결

새 `sac_successor_diagnostic_audit.py`는 새 scope에 actual successor lineage 검사를
필수로 적용한다. 기존 scope는 `sac_stratified_diagnostic_audit.py`를 통해 기존 검사를
그대로 실행한다. missing/mixed/mislabeled contract와 미완료 감사는 거절한다.
`diagnose_sac_successor_branch_q.py`, `diagnose_sac_successor_soft_returns.py`,
`audit_sac_successor_coverage.py`를 별도 추가했으며 기존 동결 파일은 수정하지 않았다.
이름/adapter/출력계약 치환 후 AST 동일성을 확인해 Q/coverage/return 계산은 유지했다.

관련CPU33pytest(신규14 포함) 통과. 최초 명령은 없는 terminal-diagnostic 테스트 경로를
지정해0tests/exit4였고, `rg --files`로 실제 경로를 확인한 뒤33개를 실행했다. 결과를
누락하지 않는다. 기존 종류별·비종료·terminal-only 각각1,000회 실제 이력도 새 adapter로
검사해 모두 COMPLETE였다. 새100회도 실제 이력 감사12successor/20refresh를 확인했다.
새 CLI3개 help 확인. 새 GPU 정책진단·주행 성능은 아직 미실행이며 학습 파이프라인과
별도다. 보상·안전검사·PD·Gaussian·loss/lr/tau·BC/teacher0 조건은 유지한다.


## 2026-11-21 02:14 — 후속경로100전이 실제 검증 완료·동일예산 비교 준비

runner3063654/session62388의 train/audit/verify/review4jobs 모두 exit0,
02:14:18 COMPLETE_PENDING_DECISION 및 실제 종료를 확인했다. 학습은02:13:18,
감사는02:13:22 완료했다. 실제100전이=main51+branch29+refresh20, fullSAC100·BC0,
초기learner reference 바이트·자기warmup17배열·배치SHA/RNG/teacher0·실제flags를 검산했다.

재수집 source 비종료13/종료7, 실제결과 비종료13/충돌7이다. 비종료는 main source2,
tick1에서시작한 한 실제경로를tick13까지 이어13step을 수집했다. 그 중후속12개이고
마지막collection95도bootstrap=true다. 예산끝 active는(root2,path1,nextstep13,last95)이며
가짜종료를 만들지 않았다. 종료 source는 main36과branch72의tick28을 재방문했다.

재수집20경로 모두거절, 이동8/정지12다. 이전종류별pilot은 이동4/정지16·수락0이었다.
이동은prestate anchor기준이며 학습 중 수집통계다. 정책이 매전이 바뀌므로 고정모델
주행평가나 성공률로 쓰지 않는다. 현재관찰된 provider는 KEEP뿐이다.

실제재현5사례는 main 비종료root/첫후속/마지막후속 및 main·branch 종료root다.
원본source 행동과 저장된각후속행동의 보상/검사/제어/다음관측·부모보존을 대조했다.
총5reset/82step(원본prefix59+원본source5+재수집18)이며 독립평가5회가 아니다.
FOLLOW, branch의비종료root, 비종료후속경로의실제terminal은 아직 미관측/미검증이다.

로컬 `outputs/sac_successor_refresh_pilot100_20261121/`:

- `pilot-review.json` SHA `5017af848786850080dc458836c1c8da2e6ea4768c80a71e6a2bfebe971aab24`.
- `audit100.json` SHA `b76199c6dcc742ebf0e3fda1cd1c423fcb4a5cfabd7182494eefe5cce6816d23`.
- `execution100.json` SHA `0b4f615bad453d6ec9c1db5e20474b2aa9380fa9f7cc3555bc1287f7e5eef4f8`.

검산 통과는 실행 근거이며 안정적인 정책이나 주행개선은 아니다. 같은 초기모델·자기
warmup5000으로fresh1,000 비교를 선택했다.100회가중치를 이어받지 않는다. 주대조군은
종류별1,000(0/16목표), 보조참조terminal-only2/16·현재Rule-based7목표/8충돌/TASK_LOST1.
기존평가를 재사용하고 새독립표본으로 세지 않는다. 기존B평균 별도비교조건도 유지한다.

로컬 `outputs/sac_successor_refresh_comparison1000_20261121/`에 학습→저장감사→
실제관찰조합재현→학습검토→전체개발16회→원본검산을 준비했다. 기존 source선택과
달리 active경로가후속슬롯을 사용해root재방문빈도·지속보관메모리도 달라진다. 한 요인만의
인과효과로 단정하지 않는다. 보상/검사/PD/Gaussian/학습계산/전이당fullSAC1/BCteacher0 유지.

새정책의coverage·고정종료상태Q·동일선행train상태2개×4seed 실제미래보상도 후속연결한다.
성능/기본채택/추가예산/다른seed를 자동확정하지 않으며, 종료source보호와 실제현재정책
후속행동·새종료경험의확보 및다중주행지표를 함께판단한다. 안정정책 목표는미완료다.


### 동일 예산 fresh1,000 및 후속 진단 시작

구현33800602edc8a505b71adbd49a1644dd55d59151 push/원격SHA 확인. 새 비교는
runner3069123/session56549·train3069130, 시작2026-11-21T02:16:35.674870+09:00다.
소스280/input142 동결, decision SHA170ee6e1dabdc727d9b3ff5d78ce7841caf3b78cbdc0663f952b990ebc59a6b3.
첫 update 입력learner도 reference와 SHA7363d29a0c9500d58a259ba8953ff21428ede1660ee9e66be9b9f571ef014c2c로
바이트 동일했다. 2026-11-21T02:18:20.749269+09:00 실제 저장10/1,000전이를 확인했다.

같은 폴더 `policy-diagnostics/`의 runner3070752/session67143은
2026-11-21T02:17:33.848164+09:00 시작, 실제부모3069123/start_ticks75478893의
성공완료·실제종료를 기다린다. source282/input314 동결, decision
SHAa49e01055472351196d8f5f785e9d8766deda997aaacdbead3442fa0fd63751a.
주대조군 진단은 이전종류별1,000의 완료candidate 자료를control로 바이트 복사한 것으로,
이전 비종료 수집 대조군이나 새 독립 표본과 혼동하지 않는다.

학습/저장/실제후속재현/전체개발 검산이 끝난 뒤coverage→같은train충돌직전Q→
같은선행train상태2개×4seed 실제미래보상→원본검산 순서다. 현재는학습실행/후속대기이며
새 성능결과는 없다. `launch-review.json`에 모든 동결SHA와 실제PID/command 및 초기
모델 일치 확인을 저장했다. 현재 소스/설정 변경·자동재시작·추가예산/seed/채택 없음.

```bash
python -m json.tool outputs/sac_successor_refresh_comparison1000_20261121/status.json
python -m json.tool outputs/sac_successor_refresh_comparison1000_20261121/candidate1000/report.json
python -m json.tool outputs/sac_successor_refresh_comparison1000_20261121/policy-diagnostics/status.json
```


### 2026-11-21 — 첫 실제 후속 종료 재현 및 첫100전이 비교

새1,000학습을 그대로 유지하며 collection245에 기록된 첫 실제후속terminal을
읽기 전용으로 재현했다. source2의원래step은비종료였고, tick1부터28개 새행동을
이어간 결과tick28에서실제COLLISION/bootstrap=false였다. 중간의종료source재방문과
main/branch행동을 섞지 않고 원래실제chain만 재생했다. 원본prefix1+source1+후속28,
총1reset/30step에서 보상/검사/제어/매후속pre관측SHA/종료와부모보존을 대조했다.
새학습·replay쓰기·독립성능평가가 아니다. PID3076489/session14733 exit0 확인.

로컬 `first-successor-terminal/ticks.jsonl`은첫245개의동결prefix이며
`first-successor-terminal/replay.py`와 `report.json`에재현근거를보존했다.
report SHA `91ef49cd13a31e3bd81941bdb61a9d7687633e13ec786b996d94c14ac0cd33f8`.
다음비종료슬롯collection255은source251/path51/step0,
previous_refresh_collection_id=null로새root를선택했다. 이는현재저장기록이며 전체1,000
감사/관찰조합재현은끝난뒤별도로수행한다. FOLLOW/branch비종료/다른종료경로까지
검증한것으로확대하지않는다.

첫100개tick/updateprefix도완료pilot과대조했다. 최초learner바이트와100개batch-index
SHA는같고, 장면/명령/tick/검사유효·이유·위반/제어출처/bootstrap/종료판정은모두같았다.
그러나후속수치가완전히같지는않았다. raw행동차이는update3부터나타나최대
5.0286278771e−7, 보상차이는최대2.5218067239e−8, 제어가속도2.6897323551e−6m/s²,
곡률6.1328522861e−5m⁻¹였다. 첫update의일부학습metric에도작은차이가있었다.
실행별episode UUID차이와수치차이를구분했고, 원인을검증하지않고단정하지않았다.
최초모델동일성을이후완전수치동일성으로해석하지않는다. 허용오차/설정/정책은바꾸지않았다.

`review_prefix100.py`, 양쪽원본prefix복사본 `prefix100/`,
`prefix100-review.json` SHA `187efe7ea13288336038eb9405bdaf39fcf360027c9a758eb450a5e3798f09cc`에저장했다.
이수치비교가주행성능평가나개선증거는아니다.

2026-11-21T02:30:03.916837+09:00 확인: 실제324/1,000전이 저장,
학습runner3069123/train3069130와후속대기3070752를유지하며동결source/inputSHA를
모두재확인했다. 새실험/예산/seed/채택으로전환하지않았다.


### 2026-11-21 — 동일 예산400전이에서 후속길이와 시작상태 다양성 점검

실행 중인 학습은 그대로 두고 양쪽 최초400개의 완결된 tick행을 별도 복사해
`review_chain_prefix.py --count 400`으로 집계했다. 로컬 `chain-prefix400/report.json`
SHA `b7ca4400fef8b40bc96087ed4b7c8e6131304f7e5b353e14c8251237221eb71a`. 별도 집계로 source 수·수락·사전속도·후속41개와
실제종료 collection245, 원본 prefix SHA를 대조했다. 학습 소스/입력 동결도 유지됐다.

| 최초400전이의 재수집 | 이전 종류별 수집 | 실제 후속 수집 |
|---|---:|---:|
| 재수집 비종료/종료 source | 43/37 | 43/37 |
| 비종료 고유 source ID | 43 | 2 |
| 비종료 경로 수락 | 1/43 | 0/43 |
| 비종료 사전 이동 상태 | 15/43 | 23/43 |
| 비종료 KEEP/FOLLOW | 35/8 | 43/0 |
| 비종료 source에서 실제 새 충돌 | 0 | 1 |

새 수집의 두 비종료 경로는 main source2의28step(실제충돌)과 branch source251의
15step(이prefix말 bootstrap=true)다. 첫 경로 pre_tick1→마지막post_tick29이며
충돌 행동의 pre_tick은28이다. branch251은 ed30dabc039abc7a/KEEP의 pre_tick6→21,
collection255→395다. branch 경로는 현재 저장 연결 검산만 했고 실제Waymax 재현은
전체1,000 뒤의 관찰조합 검증에 남겨둔다. 경로간 정책 업데이트 간격은5 또는10회라
최종 고정정책의 연속 rollout/MC가 아니다. 각 실제행동은 당시 현재정책으로 수집된다.

종료 슬롯37개는 양쪽 모두 고유6source·실제37충돌·수락0·사전이동0이다. 새 전체400은
main188/branch132/refresh80, 이전은197/123/80이다. 표의 이동은 사전 anchor속도≥0.1m/s이며
평가의 사후 정지 비중과 섞지 않는다. 서로 다른학습궤적이므로 한 요인의 인과효과로
단정하지 않는다.

후속길이 증가는 확인했지만 root 다양성/명령 노출이 줄었고 수락 개선은 없다.
현재 정책의 실패후속 확보 가설을 시험하는 중이며, 실제 안정 주행/기본채택 근거가 아니다.
최종1,000에서 FOLLOW·새종료·root범위와 전체개발16회 및 Q/실제미래보상을 함께 검토한다.
2026-11-21T02:35:06.263951+09:00 실제프로세스3069123/3069130/3070752와 446/1,000저장을
확인했다. 새학습시작·규약변경·추가예산·seed확대는 하지 않았다.


### 2026-11-21 — 성공/실패 혼합 결과를 처리하는 후속 분석 준비

이전 종류별 실패 분석 스크립트는당시16/16충돌을 assert하고 Markdown/그림에도고정
서술했다. 현재 후보가성공하더라도처리할수있도록 로컬 새분석파일
`outputs/sac_successor_refresh_comparison1000_20261121/review_validation_failures.py`에
성공/기타종료처리, 명령별집계, 실제분모/결과를사용하는표·제목을추가했다. 학습소스는
변경하지않았다. 파일SHA `c234ae5d6a7215f069c6ce97b015d4b4de41e789728da7649fcd27932eb15645`.

과거종류별32trace의 모든case/집계/접촉기하/보상분해를정확히재현했다. 별도도구검증입력은
기존완료terminal-only의2목표/14충돌trace를candidate로재사용해 원본full-review의명령별
수치와대조했다. 목표도달을접촉실패에세지않았다. 이는새모델의성과나새독립평가가아니다.
초기검증그림의candidate범례가현재후속수집을가리켜혼동가능성이있어Control/Candidate로
고쳤고Markdown에실제모델경로를명시했다. 최초그림은보존하며수정그림도직접확인했다.

`failure-review-preflight/`, `failure-review-mixed-input/`, `failure-review-mixed-output-v2/`와
`failure-analysis-job/preflight-review.json`에근거를보존했다. 새pytest/학습/정책실행은0이다.

후속runner3079803/session45025는2026-11-21T02:39:31.476573+09:00 시작해 실제부모3069123/
start_ticks75478893의성공완료·종료를기다린다. decision SHA
`b161366545740cc6b1d1f247ddd3bdf315dad649b1bd865b15f0a6a50446b858`.
완료된full-validation-review와원본trace SHA를동결한뒤실패분석→원본수치대조를수행한다.
모델/replay쓰기와추가학습은없다. 최종그림직접검토는자동완료로표시하지않으며
`COMPLETE_PENDING_VISUAL_REVIEW`에서확인한다.

2026-11-21T02:40:15.019991+09:00 현재583/1,000전이 저장과
학습3069123/3069130, Q진단대기3070752, 분석대기3079803 생존을확인했다. 새정책의
전체개발성능·Q검사와안정정책목표는미완료다.


### 2026-11-21 — 608회 실행 중단 확인과 동일 조건 대체 실행

02:40경 기존exec session56549가exit143을반환했다. 실제3069123/3069130도사라졌고,
최종tick/update각608행·report608에서멈췄다. 원래status/report의RUNNING은과거값으로
남았으며현재생존근거가아니다. 학습로그에Python예외는없고 종료를일으킨주체/원인은
확인하지못했다. 단순관찰timeout으로간주한재시작이아니라실제프로세스소멸을확인했다.

후속Q3070752와분석3079803은부모소멸을감지해FAILED로종료했다. 원본status/log/
608행과앞선분석자료는보존하고 `interruption-review.json`에SHA와소멸근거를기록했다.
학습후모델/replay checkpoint가없고initial learner만있어608지점의정확한복원은불가능하다.
부분prefix검산은유효한관찰이지만최종성능이나전체감사통과로취급하지않는다.

대체폴더는 로컬 `outputs/sac_successor_refresh_comparison1000_20261121_retry1/`이다.
동일학습코드/설정/seed0/초기자기warmup5000으로fresh1,000을다시실행했다. 기존608을
합산한1,608회모델이아니고추가학습조건의성능비교도아니다. 과거실행디렉터리를덮어쓰지않았다.
원래jobs/models/cohorts/seed/budget은출력경로치환후동일,모든동결source/inputSHA도유지했다.
초기learner SHA `7363d29a0c9500d58a259ba8953ff21428ede1660ee9e66be9b9f571ef014c2c`가reference와바이트동일했다.

runner3080418(start_ticks75640081)은02:42:10에별도session/process group으로시작했고
학습child3080425는02:42:13 시작했다. stdin DEVNULL, stdout/stderr파일·start_new_session=True로
호출세션과분리했다. 이변경이알수없는최초종료원인을해결했다고단정하지않는다.
Q진단3080445와분석3080446도별도session이며부모3080418/start_ticks75640081에연결했다.
decision SHA `d15bf4a2ce086628c46b56b8f67032209d95fdd0fe22859bb652018bf40c5a6f`.

2026-11-21T02:43:59.171066+09:00 실제4프로세스와새저장11/1,000전이를확인했다.
`recovery-launch.json`·`recovery-review.json`에실제핸들/명령과비교근거를저장했다. 새실행은
학습→감사→실제후속재현→전체개발16회→원본검산후Q/실제미래보상·실패분석으로이어진다.
현재전체평가/새성능/기본채택/안정정책은미완료다. 보상/검사/PD/BCteacher0변경없음.


### 2026-11-21 — 현재후보와 Rule-based의4개 GIF 생성 연결

대체실행폴더의 `render_reviewed_pairs.py`와 `visualization/decision.json`에
고속8b4c1e7e09f5f4c3·정지출발8dd5939855c15b3f 각각 KEEP/FOLLOW,2고유장면·4쌍을
현재최종결과가나오기전에지정했다. 과거회귀/정지출발을근거로고른설명용사례이며
무작위/대표표본이아니다. 전체개발8장면16회결과를대체하지않는다.

실제부모3080418/start_ticks75640081의성공완료·종료뒤원본full-review를검산하고
동일seed/초기frame/map·최종model/report SHA가일치하는 Rule/SAC trace만읽는다.
새정책실행/episode/학습은0,기존renderer를사용해공통축·물리시간1x·마지막1.5초hold로
4GIF/최종PNG/evidence를만든다. 출력은 `visualization/high-speed-keep.gif`,
`high-speed-follow.gif`, `stopped-start-keep.gif`, `stopped-start-follow.gif`다. 현재미생성이다.

과거완료종류별1,000의4쌍으로evidence조립만검증했고새그림/주행을생성하지않았다.
최초preflight는Rule 보상합의엄격float동일성검사에서5.551115123125783e−17 차이로
실패했다. 기존full-review와같은return절대1e−10허용으로맞추고이산개수는정확히대조해
4쌍통과했다. 물리검사/학습조건의허용오차는변경하지않았다.

renderer3085826은02:49:27 별도session으로시작·현재실제생존확인. decision SHA
969dd4e3d6e57a0fb62ffbccde0a1eea8a864f9fc6f887ef979cec979821b648, preflight SHA
9240100deb0c9de45c86d94d22dde86f2109f59344518404fcc52fa1bf81b5f7.
완료상태는 `COMPLETE_PENDING_VISUAL_INSPECTION`이며프레임수/크기/총시간/원본SHA
검산뒤직접그림검토는남긴다. 2026-11-21T02:50:03.601193+09:00 학습은
168/1,000 진행중이고모든동결학습/진단source/inputSHA는유지했다.


### 2026-11-21 — 대체 후속1,000 학습/저장감사 완료와 FOLLOW 수집 편중

대체실행 `outputs/sac_successor_refresh_comparison1000_20261121_retry1/`의
train3080425는03:20:51, audit3094677은03:20:54 exit0완료·실제종료했다.
실제신규1,000=main415+branch385+refresh200, actor/Q 각각1,000·BC0이다.
자기warmup17배열·초기learner·teacher0·source/후속관측·실제flags·배치SHA/RNG를
감사했다. 6,000replay는warmup5000+신규1000이고 중단된이전608을합산하지않았다.

- 최종model SHA `0b679d97b0ec4a7bf83129cc3c55de930d33fd96929fb83591d31287f2982af1`.
- replay SHA `07f742ab6ec9bf730221836e999ca3bb80b99bf2d35fda3cb24ad2da870f782f`.
- `audit1000.json` SHA `cc04b755e3c1a8578a5157913d2de86cef21e1368cdffc6d1e1127c6fbac35be`.

| 동일1,000의 재수집 | 이전종류별 | 실제후속 |
|---|---:|---:|
| 비종료/종료source 전이 | 103/97 | 103/97 |
| 비종료 고유source | 102 | 4 |
| 비종료 KEEP/FOLLOW | 63/40 | 103/0 |
| 비종료 경로수락 | 5/103 | 13/103 |
| 비종료 사전이동상태 | 56/103 | 58/103 |
| 비종료source에서 실제충돌 | 0 | 3 |

실제후속99개를확보했다. 비종료경로는4개모두KEEP이고길이는28/41/20/14step이다.
앞3경로는실제충돌, 마지막source821/path173은예산말bootstrap=true로아직진행중이다.
수락13개는모두마지막경로에만있다. 중간정책이5또는10update씩바뀌는수집경로라
최종고정정책의성공률/MC로사용하지않는다. 종료슬롯97개는실제97충돌·수락0이며
KEEP60/FOLLOW37, 고유13source다.

FOLLOW 전체신규경험은289/1000, KEEP711이다. 신규8고유train장면/8쌍,
정지출발신규경험0이다. 전체replay는47장면/63쌍, warmup추출142688/256000
(55.7375%)이며정지출발warmup687행은21729회추출됐다. 전체FOLLOW부재나
정지출발데이터전체부재로표현하지않는다. 실제새학습은아직9번째장면에진입하지않았다.
main7실제충돌+마지막1예산경계, branch는6충돌/1목표(collection769)다.
이역시고정정책의주행성과가아니다.

FOLLOW비종료거절 eligible212개가있었다. FOLLOW가비종료bank에있던35슬롯
(collection315~655)에서도활성KEEP경로를계속수집했다. 새root선택4시점
(collection5/255/665/865)에는FOLLOW가bank에없었다. 저장된stage bank ID와
모든관측/선택을확인한완료감사에기반한집계다. 활성경로지속과FIFO가FOLLOW후속
경험의노출을막는패턴은관찰했지만, Q/주행실패의인과원인으로단정하지않는다.
현재수집규칙을바꾸지않고전체개발평가와Q진단을기다린다.

`review_chain_prefix.py --count 1000`, `review_dataset_exposure.py --arm candidate`로
원본전체행/장면순서/command one-hot·mask/배치추출을검산했다. 전자는원본400분석,
후자는이전종류별분석스크립트의바이트동일복사다. `chain-prefix1000/report.json`
SHA `35ccfd010c2bbf1069b8db8c2588752b4929c86dedcb3aba65c00c8ec1eeeac5`,
`dataset-exposure-candidate.json` SHA `c2c2dc3e7135fe85a7e8bb27acb821177bcc552a7e91c552c41639d48b960e55`,
`follow-bank-availability-review.json` SHA `b7d9060e0ad8653e7fcf04c5458b9c480059c646355d186b1af2792fe1251f99`.
별도집계대조와최종model/replay SHA는 `collection-review.json`에저장했다.

실제Waymax재현verify3094709는03:20:56부터실행중이다. 완료전결과파일조회는
FileNotFoundError였고검증실패로해석하지않았다. 실제PID생존을확인했으며
새실제재현/전체개발16회/후속Q·미래보상·GIF는미완료다. 안정정책/채택미확보.
보상/검사/PD/원래Gaussian/학습계산/BCteacher0유지.


### 실제12사례 재현 완료와 전체개발 평가 진입

verify3094709는03:26:00, review-training3096023은03:26:02 exit0완료·실제종료했다.
관찰된provider/source종류/stage별root·비종료후속·종료후속·마지막후속12사례를
12reset/359실제step(원래prefix234+원본source12+새후속113)으로재현했다.
저장보상/검사/제어/관측·마스크/실제종료·bootstrap와원본관측보존을대조했다.
KEEP의main/branch비종료후속과실제충돌종료, 예산말branch후속14step의bootstrap=true,
FOLLOW의main/branch종료source를포함한다. FOLLOW비종료후속은수집되지않아미검증이다.
12사례를독립성능평가12회나모든경로전수재현으로세지않는다.

`execution1000.json` SHA9162f2c72e498e36423f0942d817530f500c5522bbd225fde53e90140a6bc2d4,
`training-review.json` SHAd9599e533ef577d94ca796d7d002ac1d83b08798ed755589b9e554b0c893b574.
학습검토의모든input SHA를재확인했다.

전체개발8장면×KEEP/FOLLOW16회 평가3096030은03:26:03 시작·실제생존확인.
현재최종정책평가실행중이며결과검토는미완료다. 이후full-review를마친뒤대기중인
Q진단/실패분석/GIF가이어진다. 안정정책/채택/다른seed/별도성능평가는미완료다.

### 2026-11-21 03:42 KST: 전체 개발 평가·실패 분석·GIF 검토 완료

대체 실행 `outputs/sac_successor_refresh_comparison1000_20261121_retry1/`의
전체 평가가 03:39:00, 원본 검산이 03:39:03에 exit0으로 완료됐다.
main3080418과 평가3096030의 실제 종료도 확인했다. seed0 최종 평균 정책,
8고유 개발 장면×KEEP/FOLLOW=16회이며 새 독립 test가 아니다.
최종 모델은 앞 절의 `0b679d97…82af1`이고 보상·검사·PD·BC/teacher0는 그대로다.

| 같은 개발16회 | 목표/충돌/기타 | tick | 거절 | 제동 | 정지 | 평균 return |
|---|---|---:|---:|---:|---:|---:|
| 직전 종류별1,000 | 0/16/0 | 862 | 411 (47.68%) | 387 (44.90%) | 460 (53.36%) | -20.336392 |
| 새 실제후속1,000 | 2/14/0 | 982 | 346 (35.23%) | 328 (33.40%) | 432 (43.99%) | -17.754974 |
| 이전 terminal-only1,000 | 2/14/0 | 962 | 348 (36.17%) | 330 (34.30%) | 430 (44.70%) | -17.757717 |
| Rule-based 참조 | 7/8/1 TASK_LOST | 1077 | 239 (22.19%) | 223 (20.71%) | 388 (36.03%) | -10.068949 |

새 후보만16회 추가 실행했고 나머지는 완료 원본을 재사용했다. 초기 frame/map,
모델/report SHA·규약·seed·매 tick의 실제 지표를 대조했다. KEEP/FOLLOW 각각
목표1/8·충돌7/8이다. 성공 쌍은 고속 `8b4c1e7e09f5f4c3`의 두 명령으로,
terminal-only 참조와 동일하다. **직전 후보의 회귀를 회복했지만 새 성공은 없고,
안정 정책과 기본 채택은 미확보다.**

이동 출발12회는 직전 대비 거절333/534→268/654, 제동311/534→252/654,
정지132/534→104/654다. 정지 출발4회는 둘 다328/328 정지·4충돌,
거절78·제동76으로 동일하다. 새14충돌 모두 후반 제동·최종 정지·뒤쪽 접촉이
함께 관찰됐다. 그중10회는 이동 후 제동·정지,4회는 처음부터 정지였다.
이를 제동의 인과효과나 회피 불가능성으로 해석하지 않는다.

terminal-only 참조와 비교하면 이동 출발의 거절264→268, 제동250→252,
정지102→104이고 1장면×2명령의 충돌이10tick씩 늦어졌다. 전체 return의 작은
증가 +0.002743569는 정지 출발 기여 +0.003741850과 이동 출발 -0.000998281의
합이다. **이 보상 증가를 새로운 주행 개선으로 판단하지 않는다.**

`review_rejection_details.py`를 로컬 분석 산출물로 추가 실행했다. 네 모델의
기록된 최초 거절 원인과 운동 위반을 원본 SHA·기존 합계에 대조했다. 새 후보는
예측충돌178·운동한계136·예측도로이탈32회다. 운동한계136회 중 곡률변화율134,
곡률132, 접선가속114, 접선jerk90회가 포함된다. 위반 항목은 중복될 수 있으며
검사에서 먼저 반환된 결과이므로 모든 물리 위반을 전수 검사한 숫자가 아니다.
고속57eb의 KEEP/FOLLOW 첫 거절은 속도 제한이고 이후 다른 운동 위반도 발생했다.
기준을 완화하거나 소스·학습 조건을 바꾸지 않았다.

실패 보고서 `failure-analysis/report.md`와 `cohort-metrics.png`를 직접 검토했다.
그림의 세 지표·분모·0/16 대2/14 표기와 원본 합계가 일치한다. 생성기의
`COMPLETE_PENDING_VISUAL_REVIEW` 이력은 보존하고 별도 `full-result-review.json`에
시각 검토를 기록했다. 추가 pytest/새 주행은 없으며 실제 원본 검산 범위다.

사전 지정한 고속/정지 출발2장면×KEEP/FOLLOW의 Rule-based 대비 GIF4개도 생성했다.
`visualization/{high-speed,stopped-start}-{keep,follow}.gif`는 각각38/35frame,
8.9/8.2초(물리1x+마지막1.5초 hold),1440×900이다. 네 최종 PNG와 실제 GIF 중간
프레임을 직접 확인했고 프레임 수·시간·SHA를 별도로 대조했다. 모든 프레임을
눈으로 검사한 것은 아니다. 고속은 양쪽 목표 도달, 정지 출발은 양쪽 충돌이다.
먼저 종료한 정책은 held로 표시한다. 설명용 선정 사례이며 대표성/새 평가 표본이 아니다.

주요 로컬 근거 SHA:

- `full-validation-review.json`: `7e4de40d0e9718bfd859442e2753a7ac143d4744b3a1d2e7948172b23c97f15b`
- `failure-analysis/report.json`: `a979e3207129085c41d43f417db521937c82d3c6b88fb3be8caffb7c96018de9`
- `rejection-details-review.json`: `8b17bbed27b107d876a2c2a6edd4dab8e57a332d55dc57a9d0b29438897c9436`
- `reference-cohort-review.json`: `7ef3904413fd88628f6556b0b1b9b20853d9afe449ac62931316c5aefd79dea1`
- `full-result-review.json`: `879a4f2308048297650c8de26fe83082513fdf25ffd77bedf7b0e73ce211a424`
- `visualization/visual-inspection.json`: `e92ec47d95e850077cfa346998e6ab7a788c9abd2b937cbdb38f647024f51cf4`

후속 Q진단3080445가 실제 실행 중이다. coverage3099782는03:40:13 완료했고
고정 충돌 Q검사3101049가 이어졌다. 같은2train상태×4seed의 실제 미래 보상과
대조한 뒤 다음 보완을 결정한다. FOLLOW 비종료 후속 누락은 확인된 수집 한계지만
현재 실패의 유일 원인으로 단정하지 않는다. 추가 학습 조건·기본 채택·다른 seed
확대는 아직 결정하지 않았다. 최종 비교·별도 성능 평가·안정 정책은 미완료다.

### 후속1,000 Q 진단 완료와 명령별 후보 보존 옵션

같은 대체 실행의 `policy-diagnostics/`가 모든 실행·검산을 완료했고
03:51 KST 이후 runner3080445/마지막MC3102413의 실제 종료를 확인했다.
source282/input329개 SHA도 유지됐다. 고정 충돌 직전 평균 행동의 Q−실제보상은
KEEP tick28에서 +0.193424→−0.020161, FOLLOW tick59에서 +0.895033→+0.655474다.
자기 warmup의 고정 실패44행 Q MAE도1.155485→0.878041로 줄었다.

그러나 같은2train상태×4확률seed는 후보도8회 모두 충돌했다. 원본 jsonl의
할인 보상·entropy 항·실제 종료·평균·표본표준편차를 별도 표준라이브러리로 검산했다.

| 고정 시작 상태 | 직전 Q−실제평균 | 후보 Q | 후보 실제 soft-return 평균 | 후보 표본표준편차 | 후보 차이 |
|---|---:|---:|---:|---:|---:|
| KEEP 77ae tick0 | 12.317481 | -1.844561 | -13.391503 | 2.060681 | 11.546942 |
| FOLLOW baac tick4 | 9.421275 | -1.270782 | -10.575180 | 0.489050 | 9.304398 |

후보 실제 길이는 KEEP[29,29,60,42], FOLLOW[62,63,64,56]tick이다.
첫 행동은 평균, 이후는 원래 Gaussian이며 실제 종료 뒤 critic bootstrap은 없다.
소수 표본이고 재사용 train상태라 정확한 기대Q/독립test/통계적 개선 확정이 아니다.
충돌 직전 적합 개선에도 앞선 상태의 미래 비용 과대평가가 남아 안정 정책은 미확보다.

새 후보의 신규1,000 행동 중 최종 정책95% 타원 안의 비율은48.9%(직전51.4%),
재수집200행에서는27%(직전24%)다. 조건부 행동 거리이며 Gaussian 지지집합 포함 여부나
새 행동의 충분한 학습 경험을 보장하는 수치가 아니다.

#### 확인된 수집 한계를 보완하는 별도 구현

원본1,000행의 후보 도착 순서와 실제 네 root 선택 시점만 재생했다.
비종료 eligible은KEEP334/FOLLOW212개다. 기존FIFO는 네 선택 시점 모두FOLLOW0이었다.
같은4칸에서 한 명령이2칸을 넘을 때 그 명령의 가장 오래된 후보를 제거하면,
실제collection665/865 시점에FOLLOW613/614 두 후보가 남는다.
이는 역사적 입력에서의 보존 가능성이다. 정책/실제경로/선택시점이 달라지는 새 실행의
주행결과를 예측하거나 FOLLOW 선택을 보장하는 반사실적 성능 실험은 아니다.

`scripts/sac_provider_protected_refresh.py`의 별도 옵션으로 이 교체 규칙을 구현했다.
한 명령만 관찰됐을 때4칸을 모두 쓰고, 두 명령의 후보가 들어오면 다른 명령의
마지막 후보까지 밀어내지 않는다. 종료bank4칸·비종료bank4칸·활성경로1개·최대
보존환경10개·기존round-robin·실제경로완료·base4 간격은 유지한다. 선택 상태에서
당시 정책의 원래 Gaussian을 실행하며 실제 새 전이당 full SAC1회다.
이전 collector와 학습·검사·보상·PD 파일은 바꾸지 않았다.

별도 `diagnose_sac_provider_protected_refresh.py`, 저장/배치/lineage 감사,
`verify_sac_provider_protected_execution.py`, coverage·고정충돌Q·실제미래보상 진단도
새 계약으로 연결했다. 기존 감사 우회나 과거 결과의 새 계약 재분류를 금지한다.
학습 본문 AST는 collector import와 provenance 외에 동일함을 검사했다.
기존 소스282개 SHA가 그대로임을 확인했다.

관련55pytest 통과(새 보존/진단 테스트 포함),9새 Python 파일 AST와6CLI `--help`
통과, diff check 완료. 희소 명령 후보 보존·한 명령에서 기존 선택과 동일·종료후보
격리·활성KEEP 완료 후 저장FOLLOW 선택·잘못된provider의 mutation 전 거절·혼합계약
거절·진단 계산식 보존을 검증했다. 실제 새 학습/저장/Waymax재현은 아직 미실행이다.
다음은 같은 자기warmup/초기learner의100전이 실행·저장감사·실제재현이다.
초반에FOLLOW가 없으면 명령 보존의 실제 주행효과는 그pilot으로 입증하지 않는다.
그 뒤 같은fresh1,000 예산으로 기존후속 모델과 비교할 예정이며 기본채택은 아니다.

원본 기준서10절의 실제전이/next_obs·유효5k 이후 update1회,17절의 환경/상대정책
고정과 대조했다. 원문 BC/teacher 절차는 사용자가 확정한 순수SAC 실험에서 제외되고,
승인된 B평균 별도 비교 및 현재 replay 혼합조건을 유지한다. 데이터 선택의 변경을
원문에 명시된 기본 알고리즘인 것처럼 소급 기술하지 않는다.

로컬 `policy-diagnostics/diagnostic-review.json` SHA
`0a76436d65c7dd31b61cadd83d62aab0e72b38f740373836f3a176810cdaa279`,
`provider-retention-replay.json` SHA
`c37dd76046aad6ea059f7fb088d23fae6ed816e7a25805a2d475edf32a34c53c`.
전체 목표·다른seed·최종Rule-based비교·별도성능평가는 계속 미완료다.

### 2026-11-21 04:04 KST: 명령 보존 pilot 완료와 동일1,000 비교 시작

`outputs/sac_provider_protected_pilot100_20261121/`의 학습은04:02:03,
저장감사는04:02:06, 실제재현은04:02:59, 최종검토는04:03:00에 exit0으로 완료됐다.
runner3106328과 모든 child의 실제 종료를 확인했다. main51/branch29/refresh20,
actor/Q각100·BC0·teacher0·warmup17배열·배치SHA/RNG·실제후속관측/flags를 검산했다.
refresh는 비종료13/충돌7, 실제 후속12개다. 예산말13step 경로의 bootstrap=true를
유지했다. KEEP만 관측됐으며 FOLLOW 실제보존효과와 성능은 미검증이다.

관찰5사례를5reset/82실제step으로 재현해 보상·검사·제어·관측/마스크·원본root보존을
대조했다. 이전KEEP-only pilot과 수락·거절이유·실제제어선택·종료/flags는 같았다.
run별UUID와 이를 포함하는 lineage는 다르며 raw_u 최대차이7.2099422e−7,
reward 최대차이3.2629136e−8이다. 초기learner 바이트동일이 이후 전체연산 동일성을
보장한다고 주장하지 않는다. 최초 읽기전용 비교가 raw_u를1차원으로 가정해TypeError였고
실제30×2 형식으로 펼쳐 재계산했다. 원본실험/허용오차는 변경하지 않았다.

- `pilot-review.json`: `99072bef2209160024076a06dad12063a6aa9bafd3e603f0222e46a9760e729e`
- `execution100.json`: `13386eaf338bebc64a33a71c7de6ad8b6dd2e232ba0e4c113c69b3d76023111b`
- `pilot-integration-review.json`: `e84693cd5b00525796a9726ecfdcc4d0765402ebd6950caa931f9d094b9b7f1c`

동일 예산 비교를 `outputs/sac_provider_protected_comparison1000_20261121/`에서
04:03:57 시작했다. 실행코드 b2a37893b46557e076e36ac51758543d42126af8,
runner3111218/start_ticks76155010·train3111226 실제생존 확인.
자기warmup5000·seed0·fresh1000이며 pilot 가중치를 재사용하지 않는다.
첫update 전 learner는 기존reference와바이트동일(SHA7363d29a…f014c2c)이다.
주 비교모델은 완료된 실제후속1,000(0b679d97…82af1), 보조참조는terminal-only/Rule-based다.
보상·검사·PD·원래Gaussian·B-f64/rawmean0.0001·physical-whitened/zero-actionQ·
complete-path/window6/horizon200·배치혼합·fullSAC1/새전이·BCteacher0는 유지한다.

학습1,000→저장감사→관찰조합Waymax재현→학습검토→같은개발8장면×2명령16회→
원본검산을 연결했다. 비종료후속의 명령별 전이/새root 개수와 미관측 명령도 기록한다.
동결source288/input174, decision SHA
`82278b20ef512f20ac6393f8d4bcbee204e8e0fd5344aa1bb38e6f7274db391b`.

Q진단3111403/start_ticks76155848은04:04:04 시작·실제생존 확인. 같은부모의
성공완료와실제종료를 기다린다. control은 직전후속모델의 완료진단을 바이트그대로
복사했고 새표본으로 세지 않는다. 새후보 coverage→고정충돌Q→같은2train상태×4seed
실제미래보상→원본검산을 연결했다. 동결source290/input359,
decision SHAf7bbd7f61c5bfb7428ef855508a32a5a3736c7b06c5a01e216cbd2d5b13d6017.
`launch-review.json` SHA8ff757ec31fd7611ca7e1c865823dc60f71331a7cb1964c4b113885ccff62120에
핸들·입력/소스SHA·비교대상·동일초기조건·예산일치를 확인했다.

#### 이전 비교의 요약 예산 메타데이터 정정

준비 과정에서 직전 `sac_successor_refresh_comparison1000_20261121_retry1/decision.json`의
최상위budget이100으로 남은 오류를 발견했다. 실행한train명령은`--updates 1000`이며
완료report·실제전이·직렬화learner감사·actor/Q모두1,000이다. runner는jobs명령을 실행하고
요약budget은 사용하지 않는다. 실제학습량/성능결과는 변하지 않으며 원래동결decision은
보존했다. `decision-budget-review.json` SHA98f66d48893e3856abc8115983825de7c5326d01f39f6eccb0a62d617f2e1f2e에
원본SHA와불일치를 명시했다. 새실행은요약budget과실제명령이1,000으로같음을검사했다.

새1,000의 완료·FOLLOW 수집효과·주행성능·채택은 아직 미검증이다. 추가예산/seed를
자동연장하지 않고 결과를 검토한다. 전체목표와 다른seed/최종비교/별도성능평가는 미완료다.

### 2026-11-21 04:57 KST: 명령별 보존1,000 수집·실제 재현 완료, 전체 평가 진입

같은 `outputs/sac_provider_protected_comparison1000_20261121/`의 학습은04:43:37,
저장감사는04:43:40, 실제재현과 학습검토는04:54:29에 exit0으로 완료됐다.
train3111226·verify3125234·review3127747의 실제 종료도 확인했다.
실행코드b2a3789·검토시점HEAD4f3d829이며 학습 소스/조건을 바꾸지 않았다.
실제신규전이·actor·critic각1,000, 동일초기learner·자기warmup5,000의17배열 보존을
검산했다. BC/teacher0·보상·안전검사·공통PD·원래Gaussian·loss/lr/tau·실제전이당
fullSAC1회와 승인된별도B평균 비교조건을 유지한다.

| 같은1,000개 실제경험 | 이전 실제후속 | 명령별 보존 |
| --- | --- | --- |
| main/branch/refresh | 415/385/200 | 438/362/200 |
| 비종료source 후속 | 103 | 103 |
| 비종료후속 KEEP/FOLLOW | 103/0 | 98/5 |
| 이동prestate/경로수락 | 58/13 | 51/22 |
| 비종료후속 고유root | 4 | 6 |
| 종료source 후속 | 97 | 97 |

새 FOLLOW두경로는source669/671에서각3/2step이며 모두속도0·경로거절 뒤충돌했다.
이동FOLLOW후속은0이다. 새KEEP경로4개 중 마지막source726의26step에서는22개가
수락됐고, 예산말bootstrap=true를 유지했다. 충돌5경로와 예산말계속1경로를 구분한다.
**FOLLOW 경험은 생겼지만 정지 실패에 한정됐고 전체이동후속은58→51로 줄었다.
수락13→22는 학습 중 수집 지표이며 고정정책의 주행 개선을 입증하지 않는다.**

첫400/700/1,000개 양쪽 원본prefix를 보존하고 현재source와 바이트/SHA·전이순서·
policy update 카운트를 대조했다. 첫400개에는 FOLLOW후속이 없었다. 첫700개에서
비종료eligible FOLLOW이동145/정지95개가 있었지만, 실제 새root선택665/695에는
FOLLOW후보가 모두정지상태였다. 마지막이동후보629는 남지 않았다. 이 수치는 전체
1,000개에도 같았다. 명령보존과 명령내이동상태보존은 다른 문제다.

#### 실제 Waymax 재현과 검사 범위

사전규칙으로 선택된16사례를16reset/861실제step으로 재현했다.
prefix729·원래source16·refresh116step이며, 관측/마스크·보상·경로검사·실제제어·
종료/잘림·bootstrap·저장root보존을 기존 허용치로 확인했다.
KEEP비종료source는main/branch각4사례, FOLLOW비종료source는branch4사례다.
종료source는KEEP/FOLLOW×main/branch각1사례다. 관찰되지 않은 조합은 미검증이며,
FOLLOW이동source나 전체주행성능을 검증한 것으로 확대하지 않는다.

`execution1000.json`의 모든입력SHA와 `training-review.json`의 근거SHA를 재대조하고,
각사례의원본source/scene/provider·후속단계·실제종료·prefix길이를 별도로 검산했다.
`execution-review.json`에 검증모델과 평가모델SHA일치·seed0·updated평균정책·선택16쌍을
확인했다. 재현16사례와 다음개발16주행은 다른 분모다.

#### 보완안은 아직 설계 제안

같은700개 도착순서에서 기존bank를73개비종료슬롯 모두 재현한 뒤, 초과명령 내부의
이동/정지 두종류가있으면 중복종류의가장오래된후보를제거하는안을 읽기전용재생했다.
같은4칸에서665에는KEEP이동271/정지299·FOLLOW이동629/정지664가남고695에도
FOLLOW이동629가남았다. 하지만255의KEEP후보구성도정지241을포함하도록달라졌다.
과거도착순서/선택시점을고정한 저장가능성 검토이며 새선택·학습·주행효과 예측은아니다.
현재collector에는 구현/채택하지 않았다. 전체주행/Q결과와대조한뒤 보완여부를 결정한다.

로컬읽기전용 `review_motion_availability.py --count 700` 및 `--count 1000`,
`review_motion_retention_proposal.py`를 실행했다. 이분석의 추가정책실행/pytest는0이다.
현재코드 관련55pytest는앞선구현당시 결과이며 이번에 재실행하지 않았다.

#### 원본 식별자와 현재 실행

- 모델: `37ee41e3a3cd086ed018be4169bb6ee9f5313fe677b6db77571a63be666f40aa`
- replay: `fd2ccbefa7a37f63bf576d6a7c45476ab0c5209661b365f1d5601245fba82d3b`
- `audit1000.json`: `18050a645b2dbfa2948dd1d58423aae13faec63e91cfb9ddd4903616376589ad`
- `execution1000.json`: `2dbf39298dfa4c77edf3b77d4896c86ae2fb4da2702677a4961bf24e84ad99fa`
- `training-review.json`: `e8cbeb52a5a03a3f182888b9d4cb652f0ecd6c4fd2d7b6621c70fca389c41451`
- `execution-review.json`: `d24b70a15ee2cc007a91b85fb2fa12100cdd8ec68a4c8a30e1177b538bf2d425`
- `collection-review.json`: `95a61c6a8fb183e35f554586b7288337eb243b4c827a9b29bb5324e56968027a`
- `chain-prefix1000/motion-availability-review.json`:
  `7fdc7995ff50c46da62683ee3436157eff2ecc7da32e1e786d2022f32874068b`
- `chain-prefix700/motion-retention-proposal-review.json`:
  `496251727212b6a19bcc06f00b57e7931aab5cfb635318d901ce6677cd4255e7`

전체 개발평가3127754/start_ticks76473930은04:54:30 시작했고04:56:29 실제생존을확인했다.
같은8고유개발장면×KEEP/FOLLOW16회·seed0평균정책이며 독립test가아니다.
main3111218/start_ticks76155010이전체원본검산까지연결한다. Q3111403·실패분석3115024·
GIF3115025는main성공종료후실행한다. 완료후각원본검산·시각검토가필요하다.
**안정정책·기본채택·다른seed·최종Rule비교·별도성능평가는 아직미완료다.**

### 2026-11-21 05:31 KST — 명령별 보존 결과와 초기 이동 실패 수집 보완

앞선 04:57 기록의 개발평가와 Q 진단은 모두 완료됐다. 실행 코드 `b2a3789`,
문서 기준 HEAD `209f636fa277e4e0bdd0c351babce8afd371c032`, 브랜치
`feature/yuntae-sac-training`이다. 아래 신규 구현은 이 시점 **미커밋**이다.

#### 완료된 주행·Q 결과

로컬 `outputs/sac_provider_protected_comparison1000_20261121/`의 같은 개발
8고유장면×KEEP/FOLLOW, seed0 평균 정책 16회에서 목표2·충돌14였다. 중속83c의
KEEP/FOLLOW 두 성공을 새로 얻었지만 고속8b4c의 기존 두 성공을 잃었다.
직전 실제후속 대조군 대비 거절346/982→379/1034, 제동328/982→364/1034,
정지432/982→438/1034, 평균 보상−17.754974→−17.777812다. 정지 비율 감소를
절대 정지 시간 개선으로 보지 않는다. Rule 참조는 목표7·충돌8·TASK_LOST1이다.
모두 반복 개발 validation이며 독립 test가 아니다. 안정 정책으로 채택하지 않는다.

사전 지정 고속/정지 GIF4개와 사후 선택 중속 성공 GIF2개, 지표 그림의 실제 시각
검토를 완료했다. 새 성공도 거절·제동을 수반하며 사후 선택 사례를 대표 성능으로
확대하지 않는다. 원본 로그 재사용이므로 새 정책 실행이나 독립 표본 추가가 아니다.

Q 진단은 05:20:03 완료했고 runner3111403 및 마지막 MC3134016/review3136557의
실제 프로세스 부재를 확인했다. 같은 두 train 상태×확률 seed4개의 8회 모두 충돌했다.
원본 JSONL에서 할인 보상, entropy, 실제 종료(no bootstrap), 최초 평균 행동 및
이후 원래 Gaussian, 관측 SHA와 평균/표준편차/Q 차이를 별도로 검산했다.

| 고정 train 상태 | Q | 실제 soft-return 평균 | 표본 표준편차 | Q−평균 |
| --- | ---: | ---: | ---: | ---: |
| KEEP77ae tick0 | −1.583429 | −13.429840 | 1.770976 | 11.846411 |
| FOLLOWbaac tick4 | −1.260062 | −10.727708 | 0.547519 | 9.467646 |

직전 대조군 차이는 각각11.546942/9.304398로 초기 Q 차이 개선이 없다. 각4표본은
정확한 기대 Q나 독립 장면4개가 아니며 유일 실패 원인을 규명하지 않는다.
고정 마지막 충돌 행동의 Q−보상은 KEEP+0.212240/FOLLOW+0.296774다. 종료 Q가
가깝다는 사실을 앞선 상태의 미래 비용 예측이 충분하다는 뜻으로 취급하지 않는다.

근거 SHA:

- `full-validation-review.json`: `085bdc1aff893679a27f132965773248496c53ffb0c3a00823320ee8516df2a7`
- `final-visual-review.json`: `cfd97b8c9a2d627224ef466827e3c042909a6f9cd84f9a223449c2edee5647bb`
- `policy-diagnostics/diagnostic-review.json`: `56c7f627c9b27f880501a1e91e14b23c388e55531cb43e860b29b1d2edc201c1`
- `policy-diagnostics/independent-mc-review.json`: `7167b3cb61be296f71b7e0aaf2664b637e9b8d62bcf3c928d605e5121e3d491a`

#### 단순 이동/정지 보존 제안의 한계와 새 비교 구현

앞선 읽기 전용 제안에서 보존되는 FOLLOW629는 tick79·0.213126m/s로 정지 직전이다.
원본에는 FOLLOW309(tick6·6.858389m/s),382(tick4·6.860402),457(tick2·10.235823),
533(tick2·10.236194)처럼 초기 이동 실패가 있었다. 마지막 이동 후보 하나 보존만으로
초기 실패 경험이 남는다고 보장할 수 없다. 이 판단은 과거 기록 분석이며 반사실
학습 성능의 예측이 아니다.

새 opt-in `scripts/sac_early_moving_refresh.py`는 다음을 함께 비교한다.

- 명령별로 이동 중 거절이 있었던 **가장 최근 도착 에피소드의 가장 이른 tick**을
  보호한다. 같은 tick은 최근 collection ID를 선택한다. 이동 기준은 실제 현재
  anchor 속도≥0.1m/s다. 정지 상태만 있는 새 에피소드는 기존 이동 anchor를 지우지 않는다.
- 명령별 두 슬롯 보존/미사용 슬롯 차용을 유지하고 초과 명령의 보호되지 않은 가장
  오래된 후보를 제거한다. 비종료4·종료4, 최대 환경10·실제 활성 경로1을 유지한다.
- 새 비종료 root는 가능한 KEEP/FOLLOW를 교대로 선택한다. 선택 명령의 이동 anchor를
  우선하고 없으면 최근 거절을 쓴다. 이미 시작한 실제 경로는 실제 종료까지 계속한다.
  보존 규칙과 root 선택을 함께 바꾸므로 단일 요인 인과 실험이라고 주장하지 않는다.

`diagnose_sac_early_moving_refresh.py`는 기존 학습 본문과 AST 기준 동일하며 collector
import·보고서 식별자만 다르다. 새 `audit_sac_early_moving_replay.py`는 원본 실제
anchor와 저장 관측 속도를 대조하고 후보/명령/root/후속 전이를 독립 재구성한다.
전체 감사와 Waymax 실행 재현도 별도 입구로 연결했다. 역사적 코드/결과는 보존한다.

기준서 v1.2 §10/17의 실제 step→실제 관측/보상/종료 저장, 실제 전이당 update1,
고정 provider/PD/검사/보상을 대조했다. 관측된 자기 상태를 선택할 뿐 행동 정답이나
미래 기록을 주입하지 않는다. 사용자 확정 순수 SAC가 원본 기준서의 teacher/BC 순서를
대체하며 BC=0/teacher replay=0이다. 승인된 B 평균 비교·gain·critic 초기화·replay
비율·loss/lr/tau·Gaussian은 직전 비교와 같다. 새 기본 정책 채택을 뜻하지 않는다.

관련 collector 단위/회귀 테스트33개를 통과했다. 최초 mock만 통과한 뒤 실제 `Ego`
인터페이스 검토에서 속도 필드명 오류를 찾아 수정했고 실제 `Ego` 타입으로 다시33개를
통과했다. 실제 Waymax 검증은 그와 별개다.

#### 실제 pilot 실행과 다음 단계

`outputs/sac_early_moving_pilot100_20261121/`에서 05:28:43 KST runner3138478/
start_ticks76689557, train3138479로 fresh100을 시작했다. 동일 seed0·자기 warmup5000·
초기 모델 바이트 대조, 실제 수집→저장/학습 감사→Waymax 재현→pilot 검토를 연결했다.
decision SHA `18f9802d01b4d520d4d79125cb20f404bf5ba75a86187e40b2ecacf2ef1b12fa`.
100회는 실행 검증이며 양 명령 이동 노출이나 주행 개선 증거가 아니다. 완료 후 실제
노출을 확인하고 동일 조건 예산 비교를 수행한다. 새 정책 안정성·다른 seed·최종
Rule 비교·별도 평가 전체 목표는 계속 미완료다.

#### 05:36 KST pilot 완료 확인

실제100전이/main51·branch29·refresh20과 actor/Q각100update를 완료했다. 자기
warmup5000의17배열·초기 모델 바이트 동일·BC0/teacher0·원래 배치 추출/RNG를
확인했다. 비종료 root 하나에서13step(실제후속12)을 수집했고 예산말 미종료
bootstrap을 유지했다. KEEP만 등장했으므로 FOLLOW 이동 보존/명령교대의 실제
효과는 아직 미검증이다. 성능 평가를 수행한 것으로 쓰지 않는다.

Waymax5reset/82실제step 재현을 완료했고 관측/마스크·실제제어·보상·검사·종료·
bootstrap을 대조했다. 보고서 입력SHA도 별도 재확인했다. 원본을 바꾸지 않은
임시 복사본의 root선택·anchor속도·source ID·후속연결 네 가지 오류를 감사기가
모두 거절했고 정상원본은 통과했다. runner/4child 실제 종료를 확인했다.
진단 어댑터23pytest를 더해 이번 관련 단위/회귀 검증은 고유56개 통과다.

- `audit100.json`: `c6be361454fde3d952f63858a76316850f4800ce80c96ae1e215731f3da0314f`
- `execution100.json`: `b28e603bb93e3357a3e7c9101831b4605ce32aa7a1dce233c1e045669ab1b497`
- `pilot-review.json`: `54caa7686c06b8326a9f448d6d8b964f4eebb748d9dbcc69c6ffbf24722e855c`
- `audit-mutation-review.json`: `f2497675672661af7102ef950a6252060e5e23be87891e08717abe3d2672263c`

다음 같은 seed0/fresh1,000 비교의 주 대조군은 직전 명령별 보존1,000이다. 같은
초기 모델·자기 warmup·loss/배치비율·보상/검사/PD/Gaussian으로 비교하며 pilot
가중치는 재사용하지 않는다. `outputs/sac_early_moving_comparison1000_20261121/`
실행 명세를 준비했고 학습→저장감사→실제재현→개발16주행→Q/MC를 연결할 예정이다.
1,000은 진단 예산이며 안정 정책 수렴을 요구하거나 장기학습 불가능성을 판정하는
기준이 아니다. 시작 여부/식별자는 후속 실제 launch 기록으로 구분한다.

#### 2026-11-21 06:12 KST 초기 이동 FOLLOW 실제 재방문 확인

실행 코드는 `7d1ba056aed3ba076cc058530af9bfafda12df52`이며 학습 조건은 그대로다.
새 fresh1,000 비교를05:37:12에 시작했다. runner3143797/start_ticks76743215,
train3143816/start_ticks76743580, Q 후속3143809/start_ticks76743404다.
06:12:02에 실제 학습 프로세스 생존과913/1,000을 확인했다. source312/input201
해시도 모두 일치했다. 주 대조군은 직전 명령별 보존1,000이며 pilot 가중치를
재사용하지 않는다. decision SHA는
`9558709a7eb8b1b6e83ba05ba910d2fc72f4f04977881348f497834424db6720`이다.

읽기전용 관찰은100/400/700/1,000개 원본 prefix를 보존한다. 첫400개에서는
양쪽 main188/branch132/refresh80, 비종료 후속43개 모두KEEP였다. 새 두 번째
KEEP root는 같은 장면의tick6 대신tick0·13.85m/s를 선택했다. 이동FOLLOW
일반주행/분기55개는 양쪽 이미 있었으므로 전체FOLLOW 경험 부재로 해석하지 않는다.

첫700개에서 새 bank는 FOLLOW source533(tick2·10.236217m/s)를 남겼고 직전
대조군의 최종FOLLOW bank는 정지 상태였다. 이동FOLLOW 후보가 있는 비종료
슬롯은24→39지만 활성KEEP 경로가 이어져 새 후보의 이동FOLLOW 재방문은0이었다.
일반주행/분기의 이동FOLLOW 수는 기존80+117, 새80+84로 전체 이동 경험이
증가했다고 말할 수도 없다. 수집 분포와 표적 재방문은 구별한다.

두 번째KEEP 경로가collection715에서47step 뒤 실제충돌한 다음,725에서
source533을 선택해 이동FOLLOW 재방문을 시작했다.725/tick2는통과,
735/tick3·745/tick4는 `PREDICTED_OFFROAD` 거절이며 모두bootstrap=true다.
원래source533도 같은 이유로 거절됐다. 첫 통과는 학습 중 확률 행동의 관찰이며
고정정책 성능이나 학습의 인과적 개선을 입증하지 않는다.

로컬 `outputs/sac_early_moving_comparison1000_20261121/first-moving-follow/`에
745개 prefix를 저장하고 다음 명령을 실행했다.

```bash
conda run -n its-merge python outputs/sac_early_moving_comparison1000_20261121/first-moving-follow/replay.py
```

실제 실행에는 같은 환경의 절대 Python 경로를 사용했다. Waymax1reset/6step
(복원2·원래source1·재방문3)의 관측fingerprint·검사·제어·보상·종료/bootstrap과
원래 환경 보존을 확인했다. 입력SHA와 원본prefix 바이트도 별도 재대조했다.
새 정책 호출/학습update0이며 전체 replay배열과 전체경로 검사는 후속으로 남는다.
`execution-review.json` SHA:
`0d1157c1473e8639a68ea7ba47ec819187b0e915b72cead79bbb7eeae97798df`.

#### 같은 첫 업데이트의 제한 수치 재현성 검사

초기모델/배치추출 동일과 이후 수치 동일성을 구분했다. 새 pilot와 새1,000의 첫100개는
batch-index SHA와 이산 판정이 같지만 첫update metric과 collection3 이후 행동이
조금 달랐다. raw 행동 최대차이1.98237e-6, 위치0.00048828125m였다.
값을 반올림하거나 허용오차를 변경하지 않았다.

`outputs/sac_runtime_reproducibility_20261121_retry1/`에서 동일 초기상태/첫배치의
독립프로세스4개×각2반복을 완료했다. 매번 원래 상태에서 한 번 업데이트를 계산하며
출력을 연결하거나 가중치를 채택하지 않았다. 실제 새 전이/production update0,
진단계산8회다. runner/child 실제 부재·동결입력·패키지/GPU/배치 동일을 재확인했다.

- 기본옵션은 프로세스 내 반복은 같지만 독립프로세스 사이525개 leaf 중14개가
  달랐다. critic Dense_2 parameter/Adam/target 차이로 직렬화순서만의 차이가 아니다.
- `--xla_gpu_deterministic_ops=true --xla_gpu_exclude_nondeterministic_ops=true
  --xla_gpu_autotune_level=0` 결합옵션은 두 프로세스와 각 반복의525leaf,
  전체 직렬화SHA 및 metric이 같았다.
- 원래 첫 시도는 전체state device_put 뒤 초기바이트 검사에서 업데이트 전에
  실패했다. Python counter의 배열 변환을 확인하고 실패 로그를 보존했다.
  retry는 직접 복원한 상태를 그대로 사용하며 엄격한 초기바이트 검사를 유지했다.
- `independent-review.json` SHA:
  `b74a1b89368d58be7f8b4c1033bf6b6892756497abdaf6347f523e520aabb34f`.

고정 첫update의 제한 검사다. 전체 학습 결정성·속도·주행 성능·단일backend 원인은
입증하지 않았다. 진행 중인 학습의 옵션은 변경하지 않았고 기본 설정 채택도 없다.

다음은1,000 완료 후 저장감사/Waymax 재현→개발16주행·Q/MC·실패분석·GIF 직접
검토다. 이동FOLLOW 경로가 예산말 미종료면 그대로 기록한다. 장기학습 방안은
같은 예산의 주행과 Q 근거를 함께 보고 결정한다. 안정정책·다른seed·최종Rule비교·
별도성능평가 목표는 미완료다. Notion03의115절과04의현재진행에 이번 결과를
반영하고 다시 읽어 새 기록1개·핵심수치·기존05:55 기록 보존을 확인했다.

#### 2026-11-21 06:26 KST 전체 재현 완료·수집 상태 복원 준비

새1,000은 실제 전이/actor/Q각1,000과 저장 감사를 완료했다. main416/branch384/
refresh200이다. 비종료 refresh103개는 기존 KEEP98/FOLLOW5→새75/28,
이동51→59, 수락22→5로 전체 수락은 감소했다. 이동FOLLOW 재방문28개 중5개
수락, 마지막collection995/tick29·5.584438m/s에서 미종료/bootstrap=true로 예산이
끝났다. KEEP의두경로28/47step은 실제충돌했다. 수집통계를 성능으로 취급하지 않는다.

15사례/393실제step(복원187·원래source15·refresh191)의Waymax 재현과학습검토를
완료했다. 입력SHA와 source312/input201을별도재대조했다. 기존수집대조군과같은
개발16주행 평가3163288/start_ticks77018897은06:24:43에8개완료/계속진행중이었다.
Q·실패분석·GIF는정확한부모성공종료후이어지며최종판정은아직없다.

- 모델: `9bf64a2854c26db42701bc63f4f4d6c1d3f010b6bb78eeb381de0f91c65d6d93`
- replay: `ad4b07f9104e6a480d3455b9f4d02580ef86be338f740410b37e5dfb9eb9cc7c`
- `execution1000.json`: `676e95a1c1d1c2461da99e4b82d7aa0eca572cf57d1ca76abf7a4c9bd325b273`
- `training-review.json`: `4b2ec96b8f916dea4a35d762f723de6549773e42812ea700c9f362f677f39e82`
- `chain-prefix1000/independent-prefix-review.json`:
  `9d74ccdc50fd3a7a322366cc738fbeb3463a1a3e55f3e2a28d835ef10a193bad`

현재 체크포인트는learner/optimizer/targets/learner RNG·replay/추출RNG를저장하지만
refresh bank·활성경로·일반주행환경·FailureBranches·장면iterator·policy RNG의
직접snapshot은없다. 기존fresh-only guard를우회하지않는다. 읽기전용조사에서양쪽
모두마지막main999뒤terminal refresh1000에서끝나일반주행미종료cut임을확인했다.
seed0 permutation은양쪽8개episode순서와일치하며policy key는원래1call/1split
코드와1,000개실제전이에서재구성했다. 저장된원래key와직접비교한것은아니다.
근거는로컬 `resume-feasibility/inventory.json` 및 `rng-and-main-boundary-review.json`.

새 `scripts/sac_refresh_context_restore.py`는기록된행동을순서대로실제환경에서실행하며
원래collector 클래스를다시진행시켜일반주행·분기메모리·실패bank·활성경로를복원한다.
episode UUID는첫행동전에복원하고모든행의token/관측/마스크·보상·검사·제어·종료와
refresh선택lineage를검사한다. 정책forward/학습update/replay삽입은0이다.
현재지원은fresh실험의미종료main cut뿐이며branch중단/기존continuation은명시적으로거절한다.
이는수집상태복원helper이며SAC학습재개driver나성능달성은아직아니다.

관련경계/손상거절9pytest를통과했다. 실제새pilot100 원본을2reset/100step으로복원해
main tick22, 실패bank,root2의진행중KEEP경로13step,최근main snapshot6개를확인했다.
입력SHA·원본selector의최종bank/활성경로와독립대조했고프로세스3165033의종료도확인했다.
로컬 `outputs/sac_refresh_context_restore_20261121/pilot100/report.json` SHA:
`0f5e075b543d016aa205ac53c1b696646bca29a496af1b6e92509915acfb423a`.
FOLLOW와대조군의전체context 복원·재개후실제SAC update는미검증이다. 현재1,000실험의
코드/설정은변경하지않았다. 전체주행/Q검토후장기예산을결정하며새학습예산은아직미실행이다.

#### 2026-11-21 06:40 KST 전체 개발 평가: 0/16 성공으로 미채택

새 초기 이동 수집1,000의 개발8고유장면×KEEP/FOLLOW16회 평가와 원본 검토를
완료했다. 실행 코드는7d1ba05, 수집 상태 복원 구현 HEAD는0a5e865다.
새 후보는 **목표0·충돌16**으로 직전 명령별 보존1,000의 목표2·충돌14보다 나빠졌다.
중속83c142d159860844의 KEEP/FOLLOW 두 성공(110tick)을 잃고 각각126tick에 충돌했다.

| 같은 개발16쌍 | 직전 대조군 | 초기 이동 후보 |
| --- | ---: | ---: |
| 목표 / 충돌 | 2 / 14 | 0 / 16 |
| 실제 tick | 1,034 | 1,038 |
| 경로 거절 | 379 (36.65%) | 420 (40.46%) |
| 제동 개입 | 364 (35.20%) | 400 (38.54%) |
| 정지 | 438 (42.36%) | 470 (45.28%) |
| 평균 보상 | −17.777812 | −20.328992 |

기존 동일 조건 Rule 참조는7목표·8충돌·1TASK_LOST이며 이번에 새로 실행한 것으로
세지 않는다. 개발자료는 반복 사용한8장면이며 독립 test 결과가 아니다. 후보의
각 명령8회가 모두 충돌했다. 이동 출발12회는710tick 중정지142(20%), 정지 출발4회는
328tick 전체가정지였다. 정지집단의거절이86→80으로줄어도 주행 개선이 아니다.

모든16충돌에서 후반제동·정지·뒤쪽차량접촉이 함께 관측됐다. 원래 이동한12회와
처음부터정지한4회를 구분한다. 중속83c의양명령은85tick까지수락,86/87의도로이탈예측
거절에서이전경로재사용,88제동,107정지,126충돌로이어졌다. 마지막41tick 모두거절,
마지막30tick 모두제동이다. 이순서의관찰이유일인과원인의증거는아니다.

주행원본의tick/flags/보상합/집계를독립재계산하고입력SHA를대조했다.
`outputs/sac_early_moving_comparison1000_20261121/full-result-review.json` SHA:
`f777be29784b05e4f05fbe45809828c0c9230168bdff6c25b9d520bdcf40d272`.
실패분석표와그래프를직접확인했다. 사전지정6GIF의284프레임을모두디코딩해크기/재생시간을
검사하고각처음/중간/끝18프레임을직접봤다. KEEP의고속/중속/정지PNG3개도직접봤다.
Rule의먼저끝난목표상태는held로표시되고,제안/실행경로·속도·거절/제동·FOLLOW목표가보인다.
모든프레임을수동검토한것은아니며사례선택은대표표본이아니다. 원본은로컬
`visualization/`의6GIF와`decoded-gif-contact-sheet.png`에있다.

coverage·고정branch Q는완료했고같은2train상태×4seed 실제미래보상은진행중이다.
Q후속완료후다음학습방향을결정하며후보를기본채택하거나다른seed로확대하지않는다.
1,000실패만으로장기학습가능성까지부정하지도않는다. 안정정책목표는미완료다.

장기학습준비는성능판정과구분한다. 새복원helper의양쪽1,000전이전체검증을
`outputs/sac_refresh_context_restore_20261121/full1000/`에서순차실행중이다.
runner3166649/start_ticks77075433, candidate3166668/start_ticks77075930,
decision SHA `db98c7d3fc7cdc27b0547a3940c4aefd3526edbb1e351d33a54ec877cc8c91ae`.
source/input530파일동결,새정책호출/학습update0이다. 대조군검증은후속대기다.

추가로 `continue_sac_refresh_context.py`, `sac_refresh_context_provenance.py`,
`audit_sac_refresh_context_replay.py`, `audit_sac_refresh_context_continuation.py`와
관련테스트를미커밋초안으로작성했다(기준HEAD0a5e865). 부모learner/replay/RNG·기록prefix·
첫main후속상태와refresh연속성을확인하고원래full SAC만호출하도록구성했다.
새12개경계테스트와기존복원9개를통과했지만실제재개update/전체감사는아직미실행이다.
별도lineage kernel은명시적인provenance stub 아래과거두1,000기록을기존감사와대조하고
source ID손상을거절했다. 이는공유산술회귀검사일뿐부모provenance/실제재개검증을대신하지않는다.
이초안을검증완료/새학습시작으로기록하지않는다.

#### 2026-11-21 06:59 KST Q 검증 완료와 실제 재개 오류 보완

기준 HEAD는 `baaf75fe2040e0b2fa50f10e6dcd437536394b92`, 브랜치는
`feature/yuntae-sac-training`이다. 초기 이동1,000 실행 코드는7d1ba05이며 아래
재개 driver·감사 코드4파일과 테스트는 아직 미커밋 초안이다. 보상·검사·PD·원래
Gaussian·SAC loss/lr/tau·실제 전이당 update1·BC/teacher0 조건은 바꾸지 않았다.

Q/MC runner와 모든 자식 작업의 성공 종료 및 실제 PID 부재를 확인했다.
고정2train상태×4seed의8회는 모두 충돌했다. 독립8장면으로 세지 않는다.

| 고정 train 상태 | 현재 Q | 실제 soft-return 평균 | 표본 표준편차 | Q−평균 |
| --- | ---: | ---: | ---: | ---: |
| KEEP 77ae tick0 | −1.899922 | −13.276254 | 2.045097 | 11.376332 |
| FOLLOW baac tick4 | −1.296406 | −10.517073 | 0.767718 | 9.220667 |

직전 대조군 차이11.846411/9.467646보다 작지만 소표본이며 전체 개발0/16실패를
뒤집는 개선 근거가 아니다. 저장 충돌행동 Q MAE는 warmup44개1.307869→0.834475,
online0.201438→0.138498로 줄었지만 앞선 상태의 과대평가는 남았다.
8개 원본 trace의 seed·첫 평균행동·후속 entropy·할인 보상·종료 bootstrap·합계/std/Q차이와
입력 SHA를 별도 검산했다. 로컬 `outputs/sac_early_moving_comparison1000_20261121/`
`policy-diagnostics/diagnostic-review.json` SHA
`73cd90171eeef646972e6e7c4d58b8b2ea2680d9b2b333a2ace9ea15b6acc641`,
`independent-mc-review.json` SHA
`35200c1c46de746f46e80bd77db95d855e728c2481b67e0938cc3a91460c88fd`.

새 후보의1,000개 실제 전이 context 복원은06:49:51 완료했고06:55에 원본과
독립 대조했다.8reset·main tick73/기존UUID, 최종bank, 활성FOLLOW source533/path145의
다음step28·직전collection995·다음환경tick30, 최근main67..72가 일치했다.
새 정책 호출/update/replay삽입0이며 원본15입력 및 동결530파일 SHA를 재확인했다.
로컬 `outputs/sac_refresh_context_restore_20261121/full1000/candidate/report.json`
SHA `2bbee71472482b4dd789e7d7852836c9269fe09af01fb369256a81123fd3f925`.
정책key는 원래 호출수에서 재구성했으며 저장된 원래key와 직접 비교한 것은 아니다.
대조군 복원 child3174094/start_ticks77202051은 실제 생존 확인했고06:58에400/1,000이었다.

실제100→120 재개 통합시험은 두 번 새 수집 전에 실패했다. 실패 산출물과 당시
driver 원본을 각 디렉터리에 보존했고 guard를 해제하지 않았다.

1. `outputs/sac_refresh_continuation_pilot20_20261121/`:06:51 learner 바이트 동일성
   검사 실패.525leaf의 값·자료형 차이0,126개mapping의 삽입 순서만 달랐다.
   JAX identity tree traversal로 학습 후 순서를 복원하면 전체 바이트가 일치했다.
2. `outputs/sac_refresh_continuation_pilot20_20261121_retry1/`:바이트검사는 통과했으나
   NumPy로 복원한 파라미터가 GPU에 없어 추론 생성에서 실패했다. 신규 전이/update0.
   배열 leaf만 GPU로 옮겨 Python 카운터를 보존하고, 변환 후 전체 바이트 동일성도
   계속 검사하도록 보완했다. 전체 state의 무조건적인 device_put은 사용하지 않는다.

CPU 관련23pytest(복원9+재개14)가 통과했다. 양쪽pilot100/본실험1,000의 네 체크포인트를
실제 GPU에서 불러와 각525leaf·전체바이트동일·actor GPU배치·model_proposer 생성·
update/BC카운터를 확인했다. 새 정책 호출/update0인 구성 검사이며 실제 재개 검증과
구별한다. 근거는 첫 시도의 `serialization-review.json`과
retry1의 `device-restore-review.json`이다.

별도 `outputs/sac_refresh_continuation_pilot20_20261121_retry2/`에서 각각100→120의
실제20전이+원래SAC20update 통합시험을 재실행 중이다. runner3178327/start_ticks77257357,
candidate3178346/start_ticks77257960의06:59 실제생존 확인. decision SHA
`be5695f811833cb4b1886cf06e9c46192b909b18aac5d842a2e4ad6c22f7028c`,558파일동결.
부모가중치/17replay배열/prefix/RNG·첫main후속상태·활성refresh 연속성을 검사하고,
완료본의 손상 사본 거절 검사도 준비했다. 실제 재개/전체감사는 완료 전 미검증이다.
새 장기 성능 예산·기본 채택·다른 seed 확대는 아직 시작하지 않았다.

다음은 양쪽 복원·실제 재개 검증을 마치고, 조건을 유지한 더 긴 동일예산 실험을
정하는 것이다.1,000의 미달을 숨기거나 장기 학습 불가능성으로 확대하지 않는다.
안정적인KEEP/FOLLOW·다른seed·최종Rule비교·별도평가 목표는 계속 미완료다.

#### 2026-11-21 07:10 KST 양쪽 실제 재개 통합검증 완료

위 retry2의 여섯 작업(각 학습·전체감사·결과검토)은 모두 성공 종료했고 모든
PID/start_ticks의 종료를 재확인했다. 후보와 대조군 각각 실제20전이(main16/refresh4)와
actor/Q20update를 추가해100→120을 완료했다. 총 새경험40/actor40/Q40, BC0/teacher0이다.
각 부모 learner/optimizer/target/RNG 바이트 동일, 이전17개 replay배열과 원본로그 prefix,
샘플링 RNG/배치해시, 원래main tick22와UUID, 진행중KEEP source2/path1의step13·
직전collection95에서후속collection105로의연결을검사했다. 추가 전이당 원래 fullSAC1을
호출했으며 새샘플의policy_updates_before100..119와update101..120을대조했다.

정상완료본의전체감사2회를양성대조로통과한뒤사본에서명시적수집재시작표기,
과거tick변조,복원replay RNG변조,새refresh source변조,새batch해시변조를각각검사해
두모델총10건모두거절했다. 원본SHA는보존됐다. 코드/입력558파일의동결SHA도재대조했다.
로컬 `outputs/sac_refresh_continuation_pilot20_20261121_retry2/integration-review.json`
SHA `769c56b8b475aac8c5619965370d0d3dabb39003c7b97a4da7bf97a57445f05c`,
`corruption-review.json` SHA
`427a309ab4de558d813ae72b31e977f4b4e4b580e305c07f38e1e8054152f379`.

이는KEEP중심의작은통합검증이다. 새failure branch는0이었고이동FOLLOW의새정책재개나
주행성능/장기안정성까지검증한것은아니다. 새후보1,000의이동FOLLOW수집context복원은
앞절에서별도확인했다. 대조군1,000전체복원은07:09:57에실제PID3174094/start_ticks77202051
생존·CPU실행중으로확인했고마지막보고는800/1,000이다. 관측지연을실패로간주하지않는다.

실행명령:

```bash
JAX_PLATFORMS=cpu python -m pytest -q tests/training/test_sac_refresh_context_continuation.py tests/training/test_sac_refresh_context_restore.py
python outputs/sac_refresh_continuation_pilot20_20261121_retry2/run.py
python outputs/sac_refresh_continuation_pilot20_20261121_retry2/check_corruptions.py
python outputs/sac_refresh_continuation_pilot20_20261121_retry2/final_review.py
```

23pytest·4새스크립트py_compile·실제40전이·전체감사·10손상거절·원본재검산통과.
위run.py는실행완료기록이며같은출력경로로재실행하지않는다. 현재지원은fresh완료부모의
미종료main cut에서한번이어총10,000이하까지이며,이미이어진모델의재재개나mid-branch
복원은지원하지않는다. 전체프로젝트회귀나수치비트동일학습보장은아니다.

후속동일예산초안은로컬 `outputs/sac_refresh_matched10000_20261121/plan.json`에있다.
두기존1,000모델에서각9,000실제전이/update를더해총10,000으로비교하고전체개발16쌍의
성공·충돌·거절·제동·정지·보상을같이검토하는구상이다. 양쪽전체복원검토와실행조건동결전에는
시작하지않는다. 중간checkpoint는현driver에없다는한계도명시했다. 아직실행결정/장기학습/
다른seed확대/안정정책채택이아니다.

#### 2026-11-21 07:13 KST 전체 복원 검토 완료·동일 총10,000 비교 시작

직전 초안의 선행 검증을 마친 뒤 실행 조건을 확정했다. 양쪽1,000 context 복원은
각8reset/1,000실제재생, 합계2,000step으로 완료했다. 마지막main과UUID·최종bank·진행중
경로·장면순서·원래호출수에서파생한정책RNG·최근main snapshot·입력SHA를독립검산했다.
새후보는main다음tick73/활성FOLLOW source533의다음tick30, 대조군은main다음tick47/
활성KEEP source726의다음tick68이다. 두context의정책key는[1951512285,242283446]이며
원래저장key와의직접비교는아니다. 새정책호출·학습update·replay삽입0이다.
로컬 `outputs/sac_refresh_context_restore_20261121/full1000/integration-review.json` SHA
`49867cf04ab865d166b89264e2f049e4793d62951c96edae8c68bf8b3e137b00`.

실행코드 `f0a21c452cb2351b421ff8d4e18cc7187c988a83`의commit/push·원격SHA를확인한뒤
`outputs/sac_refresh_matched10000_20261121/decision.json`을동결하고07:13:25에시작했다.
decision SHA `6e3c4f6feabaab8d970e9f3794866e25803bf4148ab17898c3b2e1382422df26`,595파일동결.
같은root의plan.json은당시초안으로보존하며실제진행상태는status.json에서확인한다.

- 후보/대조군은각자의완료1,000가중치·optimizer/target/RNG·자기replay·수집상태를이어
  총10,000까지학습한다. 각각새실제전이9,000/fullSAC9,000이며fresh재시작이나pilot120재사용이아니다.
- seed0·자기warmup5,000·부모모델/수집/배치조건을고정한다. 보상·안전/운동검사·공통PD·
  원래Gaussian·loss/lr/tau·BC0/teacher0·새전이당update1과기본수치실행옵션을유지한다.
- 후보학습→전체저장감사→개발8장면×2명령16회→대조군에동일3단계→원본비교검토의7작업이다.
  양쪽1,000결과와동일조건Rule결과는재사용임을표시하고초기frame/map동일성·raw지표를검산한다.
- 성공·충돌·거절·제동·정지·보상과명령/초기속도별결과를함께본다. 반복개발자료이며독립test가아니다.
  예산소진을수렴/안정주행으로보지않고10,000초과자동연장·자동채택·다른seed/별도test실행은없다.
- 현driver는최종모델/replay만저장한다. 진행보고/원본로그는갱신하지만중간가중치checkpoint는없다.

07:13:50 실제생존: runner3186163/start_ticks77350549,
candidate-train10000 PID3186182/start_ticks77351217. report는RESTORING·inherited1,000/
budget10,000·collected_this_fork0이었다. 실험시작과신규update실행은구분한다.
다음은현재context복원후바이트/RNG동일성과첫신규update를확인하며실행을관찰하고,
완료후동일예산주행/Q·실패분석으로판단하는것이다. 안정정책·추가seed·최종비교·별도평가목표는미완료다.

```bash
python -m json.tool outputs/sac_refresh_matched10000_20261121/status.json
python -m json.tool outputs/sac_refresh_matched10000_20261121/candidate10000/report.json
python -m json.tool outputs/sac_refresh_matched10000_20261121/candidate10000/restoration-progress.json
```

#### 2026-11-21 07:36 KST 첫 실제 재개·이동 FOLLOW 연결과 후속 진단 준비

총10,000 비교의 동결된 학습 코드는 변경하지 않았다.07:34:13에 후보의 부모learner/
optimizer/target/RNG 바이트동일과원래main tick73→74의update1001을확인했다.
실행runner3186163/start_ticks77350549·학습3186182/start_ticks77351217은계속생존중이다.
첫새10전이/update의원본prefix를별도로보존해과거1,000행불변과부모replay RNG에서이어지는
10개batch해시를검산했다. 최종replay/learner checkpoint는아직없으므로부분원본검토로한정한다.

새collection1005는진행중FOLLOW source533/path145/step28/이전collection995에서
tick30→31로이어졌다. episode UUID와이전next관측/field/item mask의정확한fingerprint가
일치했고,실제pre속도4.984438m/s였다. 제안은KINEMATIC_LIMIT로거절되고EMERGENCY가
적용됐으며비종료/bootstrap=true로저장됐다. 연결검증통과가주행성능개선은아니다.
근거는로컬 `outputs/sac_refresh_matched10000_20261121/first-update-review.json`과
`first-ten/review.json`,동일디렉터리의고정1010행ticks/updates prefix다.

새읽기전용진단진입점은재개계약의실제부모provenance·누적lineage를검사한뒤기존
Q/확률행동/할인보상/coverage 계산을그대로실행한다.

- `scripts/sac_refresh_context_diagnostic_audit.py`:재개/과거수집을각원래감사로분기,
  혼합계약및실패감사를거절한다. 과거기록을재개기록으로재표기하지않는다.
- `scripts/audit_sac_refresh_context_coverage.py`:전체재개감사와warmup/누적replay·
  batch해시/RNG를검사한뒤최종정책의조건부경험분포/Q를계산한다.
- `scripts/diagnose_sac_refresh_context_branch_q.py`·
  `scripts/diagnose_sac_refresh_context_soft_returns.py`:고정실제train상태에서검사한다.
  기존진단과의AST비교로감사경로·기록메타데이터외계산변경없음을검증했다.

관련33pytest(새10+기존23),4ASTparse·3CLI help를통과했다. 완료된양쪽120모델로
coverage를실행해각5,120행/30,720batch추출과모든6group의집계·Q차이·sigma범위·
95%조건부타원체판정·부모/모델SHA를별도검산했다. 후보120의고정branch2상태에서는
저장/평균/확률1개씩총6행동을실제로검사했다. 고정KEEP tick0·FOLLOW tick4에서
확률seed88019의실제종료rollout을1개씩실행해각29/51tick에충돌했다.
Q−실제soft-return은14.289721/11.135958점이다. 이는120모델의도구통합검증이며
진행중10,000의성능,평균기대값,독립평가또는주행개선으로세지않는다.

원본2trace의첫평균행동·tick·entropy·할인보상·실제종료·합계/Q차이와632개동결파일을
검산했다. reviewer가처음확률행동label을current_sample_0으로가정해실패했고,
원래진단코드의실제계약sample0으로수정했다. 최초reviewer/실패이유를보존했고실험원본은
수정하지않았다. 로컬 `outputs/sac_refresh_context_diagnostics_pilot_20261121/`
`integration-review.json` SHA
`beb68b648bf3bbab639fd58f7c2c24d4f60f3495db56d229bfb7e200f7b2836f`.
새학습update/replay삽입은0이며모든진단입력은불변이다.

긴비교가완료된뒤실행할읽기전용작업3개를원래parent PID/start_ticks에연결했다.
학습의decision/7작업/예산은변경하지않았다. parent가없거나실패하면후속작업도멈추고,
원래parent 성공종료와full-validation-review 완료후에만실제결과를읽는다.

| 후속작업 | 실제대기PID/start_ticks | 동결decision SHA |
| --- | --- | --- |
| 실패분석 | 3193234 / 77432319 | 123e0e4cbc7d091a17572110f72a78d2fc0be016149357b2c7cde6f61bad6c35 |
| GIF | 3193242 / 77432435 | 7e6c52ac6558aebb50240a4f7eb3d178065e75b98564fb2e9ffb438131e761ba |
| Q/coverage/MC | 3194305 / 77464465 | 4093b077fe6cb5b25836c909bfa7ae76c8b3b8a8859be8d33d7c5cf2fa463003 |

실패분석은같은개발16회에서초기이동/정지와명령별지표·충돌전거절/제동/정지·접촉방향을
구분한다. GIF는최종결과를보기전에정한고속8b4c/정지8dd/중속83c의3장면×2명령×
2SAC모델을각Rule과비교하는12개다. 이전완료1,000원본으로12pair증거구성preflight를
통과했으며이는새주행0/새영상0이다. 실제생성뒤직접시각검토는남아있다.

Q후속은양쪽최종모델을동결한뒤각coverage→고정branchQ→같은train2상태×4seed
(88019..88022)를실행한다. 전체16rollout이며16독립장면이아니다. 할인보상/entropy·
sample mean/std·Q차이와입력불변검토를연결했다. 아직최종모델의진단은실행되지않았다.
후속worker와입력준비당시HEAD는059971e이고새진단파일의SHA를별도동결했다.

읽기전용 `scripts/monitor_sac_continuation.py`도추가했다. 실제실행으로수집상태복원과
추가학습을분리하고,기존1,000을새9,000진행률에포함하지않음을확인했다. 전체새학습목표는
18,000이며평가시간은진행률에서제외한다. PID/start_ticks와새update로그를조회할뿐
학습을시작/종료/재시작하지않는다. 별도단위테스트는추가하지않고현재실행에서확인했다.

```bash
python scripts/monitor_sac_continuation.py --output outputs/sac_refresh_matched10000_20261121
```

다음은실제수집·업데이트를관찰하며고정예산을마친뒤다중주행지표/Q/실패분석/영상을
함께검토하는것이다. 보상/검사/PD/원래Gaussian/순수SAC조건유지. 안정정책·다른seed·
최종Rule비교·별도평가목표는계속미완료다.

#### 2026-11-21 07:45 KST 재개 FOLLOW 경로의 실제 충돌·학습 노출 확인

읽기 전용 진단/모니터 구현은 `4d9fdc0227a0a8dea5e02e844028563a716011bd`에
commit/push됐으며 이번에 원격 SHA를 재확인했다. 본 학습의 실행 코드는 f0a21c4 그대로다.
07:44에 원래 decision의 동결595파일을 모두 재해시했고 변경0이었다. main runner와
학습 PID/start_ticks, 대기 중인 실패분석/GIF/Q worker 세 개의 실제 생존을 확인했다.
07:45:38 후보 누적1,236/10,000(새236/9,000), 대조군은 기존1,000에서 대기 중이다.

첫100개 신규 원본 전이/업데이트의 고정 prefix를 검산했다. 기존1,000행 보존과
부모 replay RNG에서 이어지는100개 batch 해시가 일치했다. 새 경험은 main80/refresh20,
그중 이어진 FOLLOW10개는 수락0이며 tick39의 행동 전 속도가0.1m/s 미만이었다.
로컬 `outputs/sac_refresh_matched10000_20261121/first-hundred/review.json` SHA
`8186cb6b30ddcbe229f574497d9b65692f970b4109559d4982870c025f35a05b`.

이후 source533/path145의 FOLLOW가 실제 충돌까지 이어졌다. collection1215까지의
고정 원본 prefix를 따로 저장하고 신규215전이/215전체SAC 및215개 batch 해시를 검산했다.
기존28개와 새22개로 총50step, 환경 pre tick2부터51/종료tick52다. path_step0..49,
같은 episode UUID, 이전 collection 연결, 중간 비종료/bootstrap=true 및 마지막
COLLISION/bootstrap=false를 대조했다. 섞여 있는 main/다른 refresh 행은 경로에서 제외했다.

새22개는 모두거절·EMERGENCY이며 KINEMATIC_LIMIT10개/PREDICTED_COLLISION12개다.
총50개중수락5개는모두기존구간이고새수락0이다. 첫정지pre tick39→실제충돌tick52이며
마지막보상은−20.017이다. 마지막실제경험의 replay index6214가 update1215의재구성한
배치에1회포함됐음을확인했다. 기록에존재하는것과실제로학습배치에노출된것을구분한다.

로컬 `follow-terminal-prefix/review.json` SHA
`286f5ca3394e8515e565d50a53eb80cc703ecabacaaca35a58f7d6d354a45d30`,
`review_follow_terminal.py`와 `follow-terminal-review.log`에 재현/검산 근거가 있다.
새215개 전체는main172/refresh43이며새failure branch0이다. 최종replay/learner는아직
저장전이므로이번검토는원본prefix/복원정보/샘플링RNG와기록된업데이트의연결에한정한다.
가중치가계속바뀌는동안수집한한train경로이며고정최종정책평가나주행개선은아니다.

충돌위치의접촉방향은이번로그검토로확인하지않았으므로뒤차충돌로단정하지않는다.
학습조건/예산/보상/안전검사/PD/BCteacher0를변경하지않았고추가PR/병합없음.
다음은고정총10,000의양쪽완료후전체개발16회/Q/실패분석/영상을함께평가하는것이다.
안정정책·다른seed·최종Rule비교·별도평가목표는계속미완료다.

#### 2026-11-21 07:51 KST FOLLOW 전체 경로의 실제 물리 재현 완료

위의 고정 collection1215 원본으로 main2step+FOLLOW50step을 CPU Waymax에서
1reset/52실제step 재현했다. 07:50:49 성공완료한 PID3202568/start_ticks77582459의
실제부재를확인했다. 새정책호출·학습update·replay삽입은모두0이다.
기존30개전이의저장관측/다음관측·마스크와새22개의행동전관측fingerprint, 전체raw_u에서
재생성한World경로·token·보상·검사·실행제어·종료/bootstrap이원본과일치했다.
최종새replay배열미저장한계는유지하며독립평가나새정책주행으로세지않는다.

새구간의운동한계거절10개에는곡률변화율10·곡률9·접선가속도9·저속벡터가속도7·
접선jerk4·횡가속도1의위반이있었다. 한제안에여러위반이동시에있으므로합계는전이수가아니다.
실제tick39정지→tick52충돌을재현했고, 마지막ego속도0, 겹친차량3409의속도5.658862m/s,
ego축기준종방향−4.769262m·횡방향0.444712m로뒤쪽위치접촉을확인했다.
이는앞선로그검토에서미확인이었던접촉위치의추가근거이며과실·회피불가능성·상대차반응을
입증하지않는다. 원래주변차의로그동작과공통충돌검사를그대로사용했다.

충돌보상은step−1.7/event−2000/training−20.017, flags=[COLLISION]으로일치했다.
입력/코드595개동결해시를실행전후대조했고, 저장된52행의tick연속성·새22개집계·위반빈도·
최종행/보상/접촉·입력해시를다시검산했다. 기존학습코드/설정변경없음.
로컬 `outputs/sac_refresh_matched10000_20261121/follow-terminal-physical-replay/review.json`
SHA `1b640e9002d62448b2af062bedc7fb285251f38a61bf34f28626903e84ec91d7`,
`steps.json` SHA `00f7751a92d4136bb0b6d17ed4f99053b80e96559313a3ef4711d78d09a914dc`.
검산실행기는같은root의 `replay_follow_terminal.py`다. 실제실패학습경험검증이며정책개선은미입증이다.

#### 2026-11-21 08:03 KST 실행 비용 측정·미채택 투영 연산 시안

고정된 FOLLOW52step을 별도 CPU 프로세스에서 cProfile로 측정했다. 원래52행의
관측fingerprint/보상/검사/제어/종료와새프로파일재현의모든저장행이동일했다.
계측된 collect_step52회 누적37.422초 중 check_trajectory97회가32.582초였다.
하위시간은 contains_many711회12.680초, project_many2,237회11.462초다.
하위함수의누적시간은중복되므로서로합산하지않는다. 이실험에는SAC update가없으며
학습전체의프로파일이나독립적인처리량벤치마크로해석하지않는다.

로컬 `probe_projection_runtime.py`의시안은원래64점chunk·모든선분·argmin첫동률선택·
float64계산을유지하고XY축배열을분리해중간3차원배열/축reduction비용을줄였다.
같은52step에서호출된2,237회/821,693질의점에원래/시안계산을둘다실행했고,
거리좌표/최근접점/접선세출력의dtype·shape·전체바이트가모두일치했다.
실제제어에는항상원래결과를반환했다. 시안은현재환경/학습에채택하지않았다.
측정순서를번갈아바꿔합산한투영시간은원래11.3575초/시안2.70135초(이함수4.204배)다.
전체주행·학습4배가속이나모든입력에서의동등성을입증한것은아니다.

끝점·중점·nextafter·교차/폐곡선·2/5/65점곡선·여러scale/offset·float32/64·
비연속view를포함한22개합성route/88호출/13,812질의점도세출력바이트가동일했다.
seed20261121, `probe_projection_edges.py --output <새 결과경로>`로재현가능하다.
별도장면확장검증·실제시안결과사용·전체회귀·production반영은미실행이다.

저장상태보존검사비용도기존완료pilot100을원래helper로복원해측정했다.
terminal2/nonterminal4·활성경로1상태에서원래verify_preserved10회는총3.89050초,
평균0.38905초였고, build_observation80회가3.87773초였다.
검사항목/빈도를바꾸지않았으며관측재계산캐시등의보완은미구현이다.
계측오버헤드·동시학습·한상태한계가있어현재전체학습의시간비중으로외삽하지않는다.

로컬산출물은 `outputs/sac_refresh_matched10000_20261121/`의
`follow-terminal-runtime.pstats`, `follow-terminal-projection-probe/review.json`,
`projection-edge-repro.json`, `preservation-check-profile/review.json`,
`runtime-probe-review.json`에있다. 원래595파일해시유지·원본52행동일·측정합계·입력SHA를
검산했다. 새정책호출/학습update/replay삽입0, 연구용로컬시안/진단만추가했다.
08:03:25 본학습후보는누적1,731/10,000(새731/9,000),대조군대기다.
현재동일예산실험의조건·실행코드·보상·검사·PD·순수SAC조건은그대로유지한다.
다음은고정실험완료후성능검토이며,가속시안은추가동등성검증뒤후속실험에적용할후보다.

#### 2026-11-21 08:15 KST 규약의 이벤트율·실행 경로 추종 RMS 후처리 보완

평가 운영 기준과 현재 고정 비교기를 대조했다. 기존 원본 비교는 성공/충돌 outcome·
거절/제동/정지/보상을 검산하지만, 이벤트 기준 도로 이탈률과 추종 RMS/표본 수는
그 요약에 포함하지 않았다. 새 `scripts/sac_capture_metrics.py`와
`scripts/review_sac_capture_metrics.py`는 원본 capture의 이벤트·실행 경로로 보충한다.
기존 동결 학습/평가/원본 검토 코드는 변경하지 않았으며 규약의 판정 기준도 그대로다.

추종 오차는 post-step 실제ego와선택된 `executed_xy[0]`의차이다. REUSED에서도
거절된새raw경로를쓰지않으며, BRAKE의선택경로는포함하고경로없는EMERGENCY는제외한다.
전체RMS는실제표본의제곱합/표본수로계산하며표본0이면null이다. 한episode의충돌과
도로이탈event는각각세고,실패자료·누락장면·중복pair·변경된trace는거절한다.
동일Rule참조의원래runtime보고서와KEEP/FOLLOW별모든지표·분모를추가대조한다.

새14pytest(다중event·REUSED참조·EMERGENCY제외·표본가중RMS·불완전자료등),
3파일py_compile·CLI help·diff검사를통과했다. 실제완료된이전1,000비교의4모델×16회,
총64episode/4,111tick을읽어검증했다. 독립64장면이나새주행으로세지않는다.

| 기존 결과 | 추종 표본 / 전체 tick | 위치 RMS(m) | 실제 collision / offroad event episode |
| --- | --- | --- | --- |
| 명령별 보존1,000 | 709 / 1,034 | 0.000371416729 | 14 / 0 |
| 초기 이동1,000 | 664 / 1,038 | 0.000403575567 | 16 / 0 |
| 종료 재수집 참조1,000 | 666 / 962 | 0.000359284011 | 14 / 0 |
| Rule 참조 | 928 / 1,077 | 0.000831753523 | 8 / 0 |

이수치는선택된경로의매step추종오차다. 낮은RMS가전체경로유효성·안전주행을뜻하지않는다.
초기이동후보의0/16목표·16/16충돌등기존결론은변경되지않았다.
Rule의KEEP456/FOLLOW472추종표본과RMS0.000871309629/0.000791663430을원래
runtime값과절대오차1e-10이내로대조했다. 원본입력해시와기존주행집계도재확인했다.

실제고정train경로52step을CapturingEnvironment로다시재현해직접검증했다.
NEW7/REUSED8/BRAKE7의22개오차가실제result.tracking_error와각각1e-10이내로
일치했고30EMERGENCY는제외됐다. RMS는양쪽0.0021117649824297672m로같았다.
원래52행의관측/제어/보상/종료결과도동일했다. 새정책호출·학습update·replay삽입0이다.
추가로부모미완료·cohort누락·보고서hash손상사본3개를모두거절했다.

로컬 `outputs/sac_refresh_matched10000_20261121/` 근거:

- `capture-metrics-integration.json` SHA
  `5508ea67b98a5fce01cd3cd2fb9a4ac6018f2cc03a2fbd238b5c2acfbbf0311a`.
- `capture-tracking-physical-replay/review.json` SHA
  `762cad2e610b78b02bf8cde1222dbb56d28ec4f137ef853dd2c6023a5c584192`.
- `capture-metrics-corruption/review.json`:3개의손상검출근거.

후속worker3209181/start_ticks77740515를08:15:06에시작했고실제대기를확인했다.
`capture-metrics-job/decision.json` SHA
`f98aef264897b568f1243847c03309226f49327ba803a8f358e4b1c39b20ece3`.
기준HEAD84eb0a5와당시미커밋새파일의정확한SHA로동결했으며이후commit은내용을바꾸지않는다.
원래parent3186163/start_ticks77350549의성공종료와full-validation-review 완료후에만
양쪽최종/양쪽1,000/Rule참조를후처리한다. 현재7개학습·평가작업과기존Q/실패/GIF작업은
그대로다. 새최종모델의지표후처리·성능판단·정책채택은아직미실행이다.

### 2026-11-21 08:29 KST — 미채택 투영 시안의 개발 장면 좌표 검사

현재 학습 코드는 바꾸지 않고, 앞선 투영 시안을 기존 개발 8개 고유 장면으로
넓혀 검사했다. 초기 이동 후보1,000과 동일 조건 Rule 참조의 각16episode,
총32개 저장 기록/2,147frame을 읽었다. 새 주행이나 독립32장면이 아니다.

원래 검토 보고서의 입력 SHA, 16개 scene/provider 쌍의 누락·중복 여부,
양쪽 초기 frame/map 일치를 확인한 뒤 저장된 route/corridor, raw/executed 경로,
차량 중심과 차체 꼭짓점을 질의했다. 총8,065호출/480,803점에서 원래 함수와
시안의 거리 좌표·최근접점·접선 세 배열이 dtype/shape/바이트 단위로 같았다.
입력을 마지막에 다시 해시했고 본 학습의 동결595파일도 변경0을 확인했다.

실행 명령(저장소 루트, 로컬 산출물):

```bash
PYTHONPATH=. python3 outputs/sac_refresh_matched10000_20261121/probe_projection_captures.py
```

- 실행기 SHA `ecc10d3c110ca1990d3524b9c3788120bb6ee4be7637ad144bc438f5aa65d711`.
- 결과 `outputs/sac_refresh_matched10000_20261121/projection-capture-review.json`
  SHA `6aab2bd8435ad7c43e61bc1aa363e748cfe856eaba9c254091adbd98601d41d7`.
- 질의점: route16,144/corridor32,276/차량중심64,245/꼭짓점256,980/
  raw경로63,450/executed경로47,708. 합계480,803.

이는 저장 좌표로 구성한 유한한 기하 검사다. 실제 adaptive checker 질의열이나
시안 반환값을 사용한 Waymax 전체 재현 검증은 아니다. 경로 외 route 필드는
project_many에서 사용하지 않는 placeholder로 구성했다. 새 정책 호출·주행·학습0,
현재 학습 적용0이며 이번 검사에서 실행시간/전체 가속률은 측정하지 않았다.
기존 후보의 주행 실패 결론은 그대로다. 다음은 시안의 실제 제어 사용과 회귀 검증이며,
진행 중인 총10,000 비교는 원래 코드로 완료한 뒤 주행/Q/실패분석을 판단한다.

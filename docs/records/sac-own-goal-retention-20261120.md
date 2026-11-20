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

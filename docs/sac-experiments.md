# SAC 결과를 웹에서 보고 터미널에서 실행하기

저장소 폴더에서 실행한다. 웹은 로컬 `outputs/`의 기존 기록을 읽으며,
학습·보상·안전검사·PD를 변경하거나 자동으로 학습을 시작하지 않는다.
현재 지원하는 결과 형식은 matched context-continuation 실험과 아래 수동 실행이다.
모든 과거 SAC/PPO 실험 형식을 지원하는 범용 추적기는 아니다.

## 1. 웹 열기

```bash
conda activate its-merge
python scripts/serve_sac_dashboard.py
```

브라우저에서 **http://localhost:8765** 를 연다. WSL에서 실행했다면 같은 PC의
Windows 브라우저에서 연다. 웹 서버는 `127.0.0.1`에만 바인딩한다.
다른 컴퓨터의 브라우저로 보려면 SSH 포워딩을 사용한다.

```bash
ssh -L 8765:127.0.0.1:8765 <사용자>@<학습-PC>
```

포트가 사용 중이면 `python scripts/serve_sac_dashboard.py --port 8766`으로 실행한다.
웹 서버는 Python 표준 라이브러리만 사용하며 별도 npm 설치·빌드가 없다.
웹 터미널의 Ctrl+C는 웹만 종료하고 학습에는 영향을 주지 않는다.

화면에서 확인할 수 있는 항목:

- 후보/대조군 추가 학습량·진행률·실제 프로세스 상태·평가 진행 수, 5초 갱신.
- 전체/KEEP/FOLLOW별 목표 도달·충돌 종료·거절·제동·정지·평균 보상.
- Actor/Q loss·alpha·Q/TD target 기록의 구간 평균 그래프(최대250점).
- 장면별 종료와 거절 이유, 저장된 주행 GIF, 평가 JSON과 최근 로그.
- 현재 실험의 원본1,000 모델로 다시 시작하는 `plan`/`run` 명령 복사.

이전1,000회 결과와 Rule-based 참조는 **재사용**으로 표시한다. 이번 실험 GIF가
아직 없으면 대기로 표시하고 이전 영상을 별도 구분한다. 독립 test가 아닌 반복 개발
8장면×KEEP/FOLLOW16회다. 미평가는0% 대신 `—`로 표시하며 일부 완료한 평가는
`PARTIAL / WAITING`이다. 학습100%나 작은 loss가 안정 정책을 뜻하지 않는다.
충돌 열은 보고서의 **종료 사유** 기준이다. 별도 event collision/offroad 및 추종 RMS
후처리와 구분한다. 그래프의 첫1,000회는 물려받은 기록이며 새 업데이트로 세지 않는다.

## 2. 새 실행의 계획만 확인하기

현재 원본 실험을 preset으로 사용한다. 이 명령은 GPU 학습이나 결과 폴더 생성을 하지 않는다.

```bash
python scripts/sac_experiment.py plan \
  --preset outputs/sac_refresh_matched10000_20261121 \
  --arm both --updates 10000 \
  --output outputs/manual_sac_01
```

출력에서 시작 모델·checkpoint·manifest·실제 실행 명령을 확인한다.
`--arm candidate`, `--arm control`로 한 방식만 실행할 수도 있다.
기본 `both`는 후보 다음 대조군을 순서대로 실행한다.

**지원 범위:** 완료된 원본1,000회 모델에서 부모 가중치/optimizer/replay/RNG와
수집 상태를 복원한 후 **총10,000회 이내**로 이어가는 기존 학습기다.
`--updates 10000`은 새10,000회가 아니라 기존1,000+새9,000이다.
최종10,000 모델에서 추가10,000회로 연장하는 기능과 fresh seed 생성은 지원하지 않는다.
seed·수집방식·평균 구조는 원본을 따른다. 현재 preset의 B-f64 평균은 비교 실험으로
허용된 구조이며 규약의 기본 구조로 채택됐다는 뜻은 아니다.

## 3. 학습·감사·평가 실행하기

현재 비교와 대기 중 후속 분석이 모두 끝난 뒤 `plan`을 `run`으로 바꾼다.

```bash
python scripts/sac_experiment.py run \
  --preset outputs/sac_refresh_matched10000_20261121 \
  --arm both --updates 10000 \
  --output outputs/manual_sac_01
```

각 방식마다 **학습 → 실제 경험·저장 감사 → 개발 validation**을 실행한다.
중간 단계가 실패하면 다음 단계를 실행하지 않는다. 동일 출력 폴더 덮어쓰기와
기록된 실행 중 실험/후속 worker가 있으면 시작을 거절한다. 이 도구 사이에는 파일 잠금도
적용한다. 별도 터미널에서 원래 학습 스크립트를 직접 시작하는 경쟁까지 막지는 않는다.

새 실험은 웹 목록에 최대30초 안에 나타난다. 웹에서 실험을 선택한다.
`COMPLETE_PENDING_REVIEW`는 실행 완료이며 정책 채택·안정 주행 확보를 의미하지 않는다.
수동 도구는 기존 실험에만 예약된 Q/MC·실패분석·GIF worker를 복제하지 않는다.
그 분석·GIF의 별도 실행 연결은 이 도구의 범위 밖이다.

로그는 다음처럼 본다. 실제 파일명은 arm과 총 update 예산을 따른다.

```bash
tail -f outputs/manual_sac_01/candidate-train10000.log
cat outputs/manual_sac_01/status.json
```

오래 실행할 경우 이미 설치된 tmux를 사용할 수 있다.

```bash
tmux new -s sac-manual
# 세션 안에서 conda activate its-merge 후 위 run 명령 실행
# Ctrl+B 다음 D로 분리
tmux attach -t sac-manual
```

학습 실행 터미널에서 Ctrl+C를 누르면 현재 자식 프로세스까지 중단하고
`STOPPED`를 기록한다. **최종 저장 전 중단한 학습의 중간 checkpoint 재개는 지원하지
않는다.** 실패 원인을 확인하고 새로운 출력 폴더에서 실행한다. 기존 결과는 보존한다.

## 4. 화면이 갱신되지 않을 때

웹 서버 터미널이 살아 있는지, 주소/포트가 맞는지 확인한다. 화면의 연결 끊김 안내는
마지막 데이터가 이전 시점임을 뜻한다. `RUNNING` 기록만 남고 PID/start_ticks가
일치하지 않으면 실행 프로세스 없음으로 표시한다. 기록 파일을 고쳐 재시작하지 않는다.

검증 명령:

```bash
PYTHONPATH=. python -m pytest -q tests/training/test_sac_dashboard.py
```

이 테스트는 집계/부분 기록/경로 제한/실행 단계 중단을 검증한다. 실제 GPU 학습의
성능 검증은 별도의 학습/평가 결과로 판단한다.

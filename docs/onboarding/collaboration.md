# 협업과 Git 규칙

이 저장소의 개발은 짧은 작업 브랜치와 Pull Request(PR)로 진행한다.
`main`은 검증된 통합 기준선으로 유지하며, 코드·설정·문서 변경을 PR로 합류한다.

이 문서는 [2026-ASMC 협업 규칙](https://github.com/INHA-AIM/2026-ASMC/blob/ede1247b908e25884af85c8d18a8506763890f52/docs/onboarding/collaboration.md)을
Python·WOMD·Waymax 연구 환경과 개인 저장소 운영에 맞게 적용한 것이다.
알고리즘 규약과 실험 설정은 해당 코드·설정·설계 문서에서 정의한다.

## 1. 작업 단위

- 한 PR에는 독립적으로 검증 가능한 한 목적을 담는다.
- 파일 이동·이름 변경은 동작 변경과 분리한다.
- 브랜치는 수시간에서 2~3일을 목표로 하고, 작업이 길어지면 작은 PR로 나눈다.
- 진행 중인 작업을 공유할 때는 Draft PR을 사용한다.
- 실행한 검증과 남은 한계를 기록한 뒤 Ready로 전환한다.

새 PPO–SAC–PD 개발은 공통 계약, 20초 제어 환경, PD 연결, 학습 연결을 각각
검증 가능한 단위로 나눈다. 기존 PPO+Frenet+MPC 실행 경로에 영향을 주는 변경은
해당 경로의 회귀 검증도 포함한다.

## 2. 브랜치와 커밋

| 용도 | 이름 예시 |
|---|---|
| 기능 | `feature/yuntae-common-contracts` |
| 수정 | `fix/yuntae-replay-boundary` |
| 문서 | `docs/collaboration-workflow` |
| 구조 개선 | `refactor/scenario-adapter` |

이미 만든 `feat/womd20-control-foundation` 브랜치는 그대로 사용할 수 있다.
새 기능 브랜치는 `feature/<이름>-<작업>` 형식으로 만든다.

커밋은 `feat(scope): 설명`, `fix(scope): 설명`, `test(scope): 설명`,
`refactor(scope): 설명`, `docs: 설명`, `chore(scope): 설명` 형식을 사용한다.
한 커밋에 한 의도를 담고, 커밋 전에 선택한 파일과 diff를 확인한다.

## 3. PR 흐름

아래 명령은 작업 폴더가 깨끗한 상태에서 시작한다. 기존 변경이 있다면 먼저 그 작업의
브랜치에서 정리하고, 다른 작업을 섞지 않는다.

```bash
git switch main
git pull --ff-only origin main
git switch -c feature/yuntae-common-contracts

# 구현하고 변경 범위에 맞는 검증을 실행한 뒤
git status --short
git add <이번-작업-파일>
git diff --cached --check
git diff --cached
git commit -m "feat(contracts): add shared policy interfaces"
git push -u origin HEAD
```

GitHub에서 작업 브랜치를 head, `main`을 base로 PR을 연다.
PR 본문은 [PR 템플릿](../../.github/pull_request_template.md)을 사용한다.

리뷰 중 `main`이 변경되면 아래처럼 동기화하고, 영향을 받는 검증을 다시 실행한다.
이미 원격에 올린 개인 작업 브랜치의 이력을 rebase한 경우에만 `--force-with-lease`를
사용한다. 다른 사람이 함께 사용하는 브랜치는 먼저 작업자를 조율한다.

```bash
git fetch origin
git rebase origin/main
git push --force-with-lease
```

병합 후에는 `main`을 갱신하고 완료한 브랜치를 정리한다.

```bash
git switch main
git pull --ff-only origin main
git branch -d <완료한-브랜치>
```

Squash merge 후 `-d`가 거절되면 PR의 병합 상태와 남은 커밋을 확인한다.
이 규칙은 무조건적인 강제 삭제를 요구하지 않는다.

## 4. 개인 저장소에서의 리뷰

혼자 개발할 때도 PR의 Files changed와 검증 결과를 확인한 뒤 직접 병합한다.
자가 검토는 다른 사람의 승인 리뷰와 구분한다. 승인 리뷰가 필수인 보호 규칙을
설정했다면 해당 조건을 만족할 별도 reviewer가 필요하다.

팀원이 참여하면 관측·행동·경로·좌표·시간 계약 변경은 입력을 만드는 쪽과 소비하는 쪽이
함께 검토한다. 예를 들어 SAC 경로 형식을 바꾸면 SAC 출력과 PD 입력을 함께 확인한다.

이 문서와 PR 템플릿은 운영 규칙이며, GitHub branch protection이나 CI를 자동으로
설정하지 않는다. 실제 실행한 로컬 검증을 PR에 기록한다.

## 5. 검증과 PR 내용

PR에는 다음을 기록한다.

- 해결하는 문제와 변경 후 동작
- 영향받는 모듈과 데이터의 producer/consumer, 호환성
- 실제 실행한 명령과 결과, skip 여부
- 실행하지 못한 검증과 이유
- 되돌리는 방법과 데이터·체크포인트 호환성 영향

공통 확인은 `git diff --check`이며, Python 변경은 관련 pytest를 실행한다.
예를 들어 WOMD 로딩과 물리적 합류 검증을 함께 변경했다면:

```bash
JAX_PLATFORMS=cpu PYTHONPATH=. python -m pytest -q \
  tests/scenarios/test_womd20.py \
  tests/scenarios/test_physical_merge_validation.py \
  tests/scenarios/test_physical_merge_validation_real_womd.py
```

영향이 넓은 변경은 [실행·검증 가이드](../HANDOFF.md)의 범위를 따른다.
전체 일반 회귀 명령은 `PYTHONPATH=. python -m pytest tests/ -m "not slow"`이다.
전체 데이터에 대한 `slow` 검증은 필요한 작업에서 명시적으로 선택한다.

로컬 데이터가 없어 skip된 테스트를 실제 데이터 검증 통과로 기록하지 않는다.
로그 재생, 제어 입력을 적용한 시뮬레이션, 학습 업데이트, 정책 성능 검증은 각각
실행 여부를 구분한다. CPU 테스트 통과로 GPU 실행이나 전체 데이터 검증을 대신하지 않는다.
문서 변경은 상대 링크와 명령·경로의 유효성을 확인한다.

## 6. 저장소에 넣는 파일

소스, 테스트, 공용 설정과 구현에 필요한 계약·설명서를 관리한다.
새 작업에서는 원시 WOMD, 개인 실험 결과, checkpoint, cache, 비밀값,
개인 PC의 절대경로를 추가하지 않는다. 경로는 인자·환경 변수·저장소 상대경로로 전달한다.
기존 로컬 경로 기본값의 개선은 해당 코드 변경에서 다룬다.

기존에 추적 중인 `data/manifests/`의 동결된 연구 기준 파일은
[데이터 정책](../../.gitignore)과 해당 재현 절차를 따른다.
개인 참고 PDF는 로컬 `.git/info/exclude`의 `/paper/reference/` 규칙으로 제외한다.
이 규칙은 각 clone에만 적용되며 팀원에게 전파되지 않는다.

## 7. 병합 후 문제 대응

`fix/...` 브랜치에서 최소 수정 또는 `git revert`로 복구하고 PR에 원인과 회귀 검증을
기록한다. 공유된 `main` 이력을 force push나 `git reset --hard`로 지워 복구하지 않는다.

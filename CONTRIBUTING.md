# 기여 가이드

개발은 짧은 브랜치와 PR로 진행합니다. 상세 절차는
[협업과 Git 규칙](docs/onboarding/collaboration.md), 환경과 실행 방법은
[Handoff Guide](docs/HANDOFF.md)를 따릅니다.

- 한 PR은 한 목적을 다루며 작업 중에는 Draft로 공유합니다.
- 변경 범위에 맞는 검증을 실행하고 명령·결과·미검증 항목을 기록합니다.
- 입력과 출력 계약을 바꾸면 영향을 받는 모듈과 호환성을 함께 설명합니다.
- 데이터·개인 checkpoint·비밀값·로컬 생성물을 커밋하지 않습니다.
- 개인 저장소에서는 diff와 검증 결과를 직접 검토한 뒤 병합합니다.

PR을 열 때 [PR 템플릿](.github/pull_request_template.md)을 사용합니다.

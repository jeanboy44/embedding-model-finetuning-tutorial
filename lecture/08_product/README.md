# 실습 5 (8교시): 제품화 — CLI · 에이전트 스킬 · MCP · 웹 · 모니터링

장표: [8교시 실습 5 제품화](https://claude.ai/artifact/TgTkmnDgAnDHwvEFhUP2vE). 같은 검색을 사람(CLI) → 에이전트(스킬, MCP) → 사용자 화면(웹)에 건네고, 운영(MLflow)까지 본다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다.
API 키는 없어도 된다. 키가 없으면 답변(`ask`, 웹의 채팅)은 근거 조문만 보여 준다.

## 1. 검색 CLI: 동료가 바로 쓰는 도구

법령 목록 → 검색(법령 필터) → 조문 보기 → 답변, 그리고 다른 프로그램이 읽는 `--json`을 차례로 실행해 본다.

```shell
uv run python lecture/08_product/01_search_cli.py
```

직접 쳐 보기:

```shell
uv run --package ragkit-search ragkit-search laws --theme youth
uv run --package ragkit-search ragkit-search search "야간 근로 수당" --law 근로기준법
uv run --package ragkit-search ragkit-search show 근로기준법_법률_제56조 --article
uv run --package ragkit-search ragkit-search search "편의점 알바 주휴수당" -k 3 --json
```

선택: 휠(배포 파일) 두 개를 만들어 `uvx` 격리 환경에서 torch 없이 도는지 확인한다(인터넷 필요, 수 초).

```shell
uv run python lecture/08_product/01_search_cli.py --run
```

## 2. 에이전트 스킬: CLI에 SKILL.md 하나를 붙인다

Claude Code와 Gemini CLI는 스킬 폴더의 `SKILL.md`를 읽고, 법령 질문이 오면 우리 CLI로 조문을 찾아 근거와 함께 답한다. 스킬을 보고, 임시 폴더에 설치해 보고, 에이전트의 첫 검색을 두 모델로 재현한 뒤, 강사가 잰 실제 에이전트 비교(실험 011)를 읽는다(1분 안쪽).

```shell
uv run python lecture/08_product/02_agent_skill.py
```

내 에이전트에 설치한다. 모든 프로젝트에서 쓰려면 `--user`(홈 폴더의 `.claude/skills/` · `.gemini/skills/`).

```shell
uv run --package ragkit-search ragkit-search skill show
uv run --package ragkit-search ragkit-search skill install --agent claude --user
uv run --package ragkit-search ragkit-search skill install --agent gemini --user
```

파인튜닝 모델로 검색하게 하려면 설치할 때 모델을 준다(같은 이름으로 다시 설치하면 덮어쓴다).

```shell
uv run --package ragkit-search ragkit-search skill install --agent claude --user --checkpoint models/finetuned/r001_A --backend torch
```

그다음 `claude` 또는 `gemini`를 새로 열고 묻는다: "편의점 알바도 주휴수당 받을 수 있는지 조문 근거로 알려줘".

선택: Claude Code가 설치·로그인돼 있으면 실제로 띄워 학습 전 vs 파인튜닝을 비교한다(질문당 30~60초, 사용량이 든다).

```shell
uv run python lecture/08_product/02_agent_skill.py --run
```

## 3. MCP 서버: 에이전트를 붙이는 두 번째 방법

서버를 띄워 표준 입출력으로 연결 → 도구 목록 → 도구 호출을 에이전트가 하는 순서 그대로 재현한다. 잘못된 법령 이름을 주면 비슷한 이름을 알려 주는 것도 본다.

```shell
uv run python lecture/08_product/03_mcp_server.py
```

스크립트 마지막에 Claude Code · Claude Desktop 등록 명령이 이 컴퓨터의 경로로 출력된다. 그대로 복사해 쓴다.

## 4. 웹 화면: 법령 노트

API 서버가 화면(빌드된 정적 파일)까지 한 주소에서 함께 준다. 먼저 확인만 하고, 그다음 띄워 브라우저로 본다(Ctrl+C로 끝).

```shell
uv run python lecture/08_product/04_web_app.py --check
uv run python lecture/08_product/04_web_app.py
```

브라우저 주소는 스크립트가 출력한다(http://127.0.0.1:8765). 화면 코드를 고쳤거나 `apps/web/dist`가 없으면 다시 빌드한다(Node.js 20+와 pnpm 필요, `corepack enable`).

```shell
uv run python lecture/08_product/04_web_app.py --run
```

## 5. 모니터링: MLflow로 실험 · 모델 · 서비스를 한곳에서

MLflow 서버를 띄우고 → 비교 실험 기록 → 모델 등록(`champion`) → 등록된 모델로 API → 질문마다 남는 트레이스까지 한 번에 본다. 먼저 확인 모드로 돌리고(약 90초), 화면을 보려면 기본 모드로 실행한다(끝나면 MLflow 화면을 열어 둔다, Ctrl+C로 끝).

```shell
uv run python lecture/08_product/05_monitoring.py --check
uv run python lecture/08_product/05_monitoring.py
```

브라우저에서 http://127.0.0.1:5050. 트레이스의 LLM 단계와 토큰 수는 `GEMINI_API_KEY`가 있을 때만 남는다. 기록은 `mlruns/`에 쌓이고, 처음부터 다시 하려면 이 폴더를 지운다.

## 참고

| 파일 | 내용 |
|---|---|
| `01_search_cli.py` | 검색 CLI(laws · search · show · ask · --json), uvx 배포(--run) |
| `02_agent_skill.py` | 에이전트 스킬 보기 · 설치, 첫 검색 재현, 실험 011(Claude Code + 스킬, 학습 전 vs 파인튜닝) |
| `03_mcp_server.py` | MCP: 연결 → 도구 목록 → 호출, Claude 등록, CLI + 스킬과 비교 |
| `04_web_app.py` | 웹 빌드와 API 서버 한 주소 배포 |
| `05_monitoring.py` | MLflow: 실험 비교 → 모델 레지스트리 → 서비스 트레이스 · 세션 |

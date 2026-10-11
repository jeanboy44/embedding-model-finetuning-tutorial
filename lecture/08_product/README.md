# 실습 4 (8교시): 제품화 — 데스크톱 · 웹 · 에이전트

장표: [8교시 실습 4 제품화](https://claude.ai/artifact/TgTkmnDgAnDHwvEFhUP2vE). 같은 모델을 건네는 세 가지 길을 본다: 데스크톱(HTML 파일 하나) · 웹 · 사용자의 에이전트(CLI + 스킬, MCP). 모니터링(MLflow)은 선택이다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다.
API 키는 없어도 된다. 키가 없으면 답변(`ask`, 웹의 채팅)은 근거 조문만 보여 준다.

## 0. 데스크톱: HTML 파일 하나

3교시에 받은 `law-search-offline.html`을 더블클릭한다. 와이파이를 끄고 열어도 검색이 되는지 본다.
없으면 [Google Drive](https://drive.google.com/file/d/1TzgE1mh2ZSyAjmPFJtb7OTuk05JFr3mt/view?usp=sharing)에서 받는다.

만드는 법은 [`apps/offline-search/README.md`](../../apps/offline-search/README.md)에 있다.

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

스킬 추가는 별도의 복잡한 설치 과정 없이, **Gemini CLI에 접속한 뒤 `SKILL.md` 파일 경로를 던져주고 "스킬로 등록해"라고 말하면 끝납니다.**

스킬 파일: [`lecture/08_product/SKILL.md`](SKILL.md)

### 스킬 등록 및 사용 방법

1. **Gemini CLI** (또는 Antigravity `agy`)를 실행합니다:
   ```shell
   gemini
   # 또는
   agy
   ```

2. 에이전트에게 파일 경로와 함께 스킬 등록을 요청합니다:
   > "`lecture/08_product/SKILL.md` 파일 보고 스킬로 등록해"

   에이전트가 마크다운을 확인하고 자신의 스킬 디렉터리(`~/.gemini/skills/` 또는 `.agents/skills/`)에 자동으로 등록합니다.
   *(참고: CLI 명령어 `uv run --package ragkit-search ragkit-search skill install --agent gemini --user`로도 등록 가능)*

3. 등록 후 에이전트에게 법령 질문을 던집니다:
   > "편의점 알바도 주휴수당 받을 수 있는지 조문 근거로 알려줘"

에이전트가 `SKILL.md`에 적힌 순서대로 우리 CLI(`ragkit-search search` → `show --article`)를 호출하여 근거 조문을 찾아 답변하는 것을 확인할 수 있습니다.

## 3. MCP 서버: 에이전트를 붙이는 두 번째 방법

서버를 띄워 표준 입출력으로 연결 → 도구 목록 → 도구 호출을 에이전트가 하는 순서 그대로 재현한다. 잘못된 법령 이름을 주면 비슷한 이름을 알려 주는 것도 본다.

```shell
uv run python lecture/08_product/03_mcp_server.py
```

스크립트 마지막에 Antigravity CLI(agy) · Claude Code · Claude Desktop 등록 명령이 이 컴퓨터의 경로로 출력된다. 그대로 복사해 쓴다.

### Antigravity CLI (`agy`) 등록 및 사용 예시

1. **MCP 서버 등록**:
   저장소 루트에서 다음 명령을 실행합니다:
   ```shell
   agy mcp add --env RAGKIT_PROJECT_ROOT=$(pwd) ragkit-law $(which uv) run --directory $(pwd) --package ragkit-mcp ragkit-mcp
   ```

2. **등록 상태 확인**:
   ```shell
   agy mcp list
   ```
   `ragkit-law`가 `stdio`, `enabled` 상태로 목록에 나타나는지 확인합니다.

3. **에이전트에게 질문하기**:
   `agy` 세션에서 별도의 슬래시 명령어 없이 자연어로 질문합니다:
   > "주휴수당 지급 기준에 관한 근로기준법 조항 찾아줘"  
   > "편의점 알바도 주휴수당 받을 수 있는지 법 조문 근거로 알려줘"

   에이전트가 질문을 분석하여 `ragkit-law` MCP 서버의 도구(`search_laws`, `get_article` 등)를 스스로 호출해 근거 조문을 찾아 답변합니다.

> [!TIP]
> **MCP vs 슬래시 명령어**: MCP 서버는 슬래시 명령어(`/<이름>`)나 쉘 명령어가 아니라, 에이전트가 백그라운드에서 호출하는 **도구(Tools)**입니다. 따라서 `/ragkit-law`를 입력할 필요 없이 위와 같이 자연어로 바로 질문하시면 됩니다.


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

## 선택: 모니터링 — MLflow로 실험 · 모델 · 서비스를 한곳에서

MLflow 서버를 띄우고 → 비교 실험 기록 → 모델 등록(`champion`) → 등록된 모델로 API → 질문마다 남는 트레이스까지 한 번에 본다. 먼저 확인 모드로 돌리고(약 90초), 화면을 보려면 기본 모드로 실행한다(끝나면 MLflow 화면을 열어 둔다, Ctrl+C로 끝).

```shell
uv run python lecture/08_product/extra_monitoring.py --check
uv run python lecture/08_product/extra_monitoring.py
```

브라우저에서 http://127.0.0.1:5050. 트레이스의 LLM 단계와 토큰 수는 `GEMINI_API_KEY`가 있을 때만 남는다. 기록은 `mlruns/`에 쌓이고, 처음부터 다시 하려면 이 폴더를 지운다.

## 참고

| 파일 | 내용 |
|---|---|
| `01_search_cli.py` | 검색 CLI(laws · search · show · ask · --json), uvx 배포(--run) |
| `SKILL.md` | 에이전트 스킬(SKILL.md 하나로 조문 검색 연동, 프롬프트 규약) |
| `03_mcp_server.py` | MCP: 연결 → 도구 목록 → 호출, Antigravity/Claude 등록, CLI + 스킬과 비교 |
| `04_web_app.py` | 웹 빌드와 API 서버 한 주소 배포 |
| `extra_monitoring.py` | MLflow: 실험 비교 → 모델 레지스트리 → 서비스 트레이스 · 세션 |

# 실습 4 (8교시): 제품화 — 데스크톱 · 웹 · 에이전트

장표: [8교시 실습 4 제품화](https://claude.ai/artifact/TgTkmnDgAnDHwvEFhUP2vE). 같은 모델을 건네는 세 가지 길을 본다: 데스크톱(HTML 파일 하나) · 웹(같은 디자인의 화면을 서버가 준다) · 사용자의 에이전트(CLI + 스킬, MCP). 모니터링(MLflow)과 NotebookLM형 법령 노트 화면은 선택이다.

명령은 모두 저장소 루트에서, 위에서부터 순서대로 실행한다. macOS 터미널과 Windows PowerShell에서 같은 명령 그대로 돈다.
API 키는 쓰지 않는다. 답변 생성(LLM)은 이 강의에서 쓰지 않으므로, 모든 길에서 검색 결과(근거 조문)까지 본다.

## 0. 데스크톱: HTML 파일 하나

3교시에 받은 `law-search-offline.html`을 더블클릭한다. 와이파이를 끄고 열어도 검색이 되는지 본다.
없으면 [Google Drive](https://drive.google.com/file/d/1TzgE1mh2ZSyAjmPFJtb7OTuk05JFr3mt/view?usp=sharing)에서 받는다.

만드는 법은 [`apps/offline-search/README.md`](../../apps/offline-search/README.md)에 있다.

## 1. 검색 CLI: 동료가 바로 쓰는 도구

법령 목록 → 검색(법령 필터) → 조문 보기, 그리고 다른 프로그램이 읽는 `--json`을 차례로 실행해 본다.

```shell
uv run python lecture/08_product/01_search_cli.py
```

스크립트는 `apps/search-cli`의 `ragkit-search` 명령을 자식 프로세스로 부르고 출력을 그대로 보여 준다. 안은 7교시 API와 같은 `Searcher` 하나다. 기본 모드는 수십 초다.

1. `ragkit-search laws --theme youth`로 인덱스에 든 youth 테마 법령 32개를 종류(법률 · 시행령 · 시행규칙)와 조각 수와 함께 나열한다. `--law`에 쓸 정확한 이름을 여기서 본다.
2. `search "야간 근로 수당" -k 3 --law 근로기준법`으로 근로기준법만 검색해 상위 3개를 본다. 1위가 근로기준법 제56조(유사도 0.888)이고, 줄마다 `[유사도] 제목 (조각 id)`가 나온다.
3. `show 근로기준법_법률_제56조 --article`로 그 조각이 속한 조의 항 · 호 전체 원문과 출처 URL을 본다. 검색 결과는 조각이라 답에 쓸 때는 이렇게 조 전체를 읽는다.
4. `search "전세 보증금" -k 2 --law 주택임대차보호법 --json`은 JSON 목록을 돌려준다. 스크립트가 그것을 파싱해 건수 2와 필드 이름 15개(`id` · `title` · `score` · `text` · `law_name` · `law_type` · `parent_id` · `source_url` 등)만 출력한다. 2절의 스킬이 읽는 출력이 이것이다.
5. 기본 모드는 휠을 만들지 않고 동료에게 줄 명령 두 가지(저장소 경로로 `uvx --from`, 휠 두 개로 `uvx --with … --from …`)만 안내한다.

답변(`ask`) 명령은 이 강의에서 쓰지 않는다.

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

`--run`은 위 1~4를 똑같이 한 뒤 5에서 `uv build --wheel`로 `ragkit`과 `ragkit-search` 휠 두 개를 임시 폴더에 만들고 크기(KB)를 보여 준다. 이어서 `uvx --isolated`로 그 휠만 설치한 새 환경에서 torch를 찾아 "uvx 격리 환경에 torch 없음: True"가 나오는지 본다. 검색 CLI는 onnxruntime만 쓰므로 torch 없이 돈다. 임시 폴더는 끝나면 지운다.

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

스크립트가 에이전트 역할을 하고, `apps/mcp`의 `ragkit-mcp`가 서버 역할을 한다. `--run` 모드는 없고 받은 인덱스만 읽는다. 수십 초다.

1. 연결: `uv run --package ragkit-mcp ragkit-mcp`를 자식 프로세스로 띄우고 표준 입출력으로 `initialize`한다. 서버 이름 `ragkit-law`와 instructions(사용 지침 다섯 줄)가 나온다. 에이전트는 이 글을 읽고 법령 질문이면 추측하지 않고 도구를 부른다.
2. 도구 목록: `list_tools`로 도구 4개의 이름 · 입력값 · 설명 첫 줄을 본다. `search_laws(query, k, laws)` · `get_article(doc_id)` · `list_laws(theme)` · `ask(question, laws, k)`다. 이 목록이 에이전트가 받는 설명서의 전부다. `ask`는 LLM 답변 도구라 이 강의에서 쓰지 않는다.
3. 도구 호출: `call_tool`로 1절과 같은 순서를 밟는다. `list_laws(theme="youth")`는 법령 32개, `search_laws("수습 기간 최저임금", laws=["최저임금법"], k=2)`는 최저임금법 제5조(0.898)와 제31조(0.881), `get_article("최저임금법_법률_제5조")`는 조각 1개인 조 전체를 돌려준다. 마지막에 `laws=["근로기준"]`을 일부러 줘 `is_error=True`와 "비슷한 이름: 근로기준법, 근로기준법 시행규칙, 근로기준법 시행령" 안내가 오는 것을 본다. 에이전트는 이 안내로 이름을 고쳐 다시 부른다.
4. 등록: Antigravity CLI(agy) · Claude Code · Claude Desktop 등록 명령이 이 컴퓨터의 경로로 출력된다. 그대로 복사해 쓴다. 방법 A는 서버 한 대를 `--transport streamable-http`로 띄워 여러 사용자가 URL로 붙는 것, 방법 B는 각자의 컴퓨터에서 1의 stdio 프로세스를 등록하는 것이다. 끝에 2절의 CLI + 스킬과 무엇이 다른지 비교표가 나온다. 검색 결과는 같고 연결 방식과 쓰는 곳이 다르다.

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


## 4. 웹 화면: 검색 API 서버가 HTML 한 장을 함께 준다

0번의 데스크톱 파일과 같은 디자인의 화면(`apps/web-lite/index.html`, 18KB)이다. 다른 점은 모델과 조문 벡터가 파일 안이 아니라 서버에 있다는 것뿐이다. 빌드는 없다. 검색 API 서버(`ragkit-api`)가 `/api`와 화면(`/`)을 한 주소에서 준다. 먼저 확인만 하고, 그다음 띄워 브라우저로 본다(Ctrl+C로 끝).

```shell
uv run python lecture/08_product/04_web_app.py --check
uv run python lecture/08_product/04_web_app.py
```

`--check`는 아래 1~3만 하고 서버를 닫는다. 기본 모드는 4까지 하고 서버를 열어 둔다. 학습 전 e5(ONNX)라 모델 로드는 수 초다.

1. `apps/web-lite/index.html`이 있는지 보고 크기(18KB)를 출력한다. 사용자 브라우저로 가는 것은 이 파일뿐이고, 모델과 인덱스는 들어 있지 않다.
2. `ragkit-api --port 8765 --web-dist apps/web-lite`를 자식 프로세스로 띄우고 `/api/health`가 응답할 때까지 최대 90초 기다린다. `GET /api/health`가 200으로 모델 키 `multilingual-e5-small`과 조문 조각 25,967개를, `GET /`가 200으로 검색 폼이 든 `index.html`을 주는 것을 본다.
3. 화면의 검색 버튼이 보내는 것과 같은 `POST /api/search {"query": "편의점 알바 3개월 했는데 주휴수당 받을 수 있나요?", "k": 3, "laws": null}`을 보낸다. 서버가 `Searcher.search`로 질문을 임베딩해 가까운 조각을 찾고, 화면은 그 JSON을 카드로 그릴 뿐이다. HTTP 200과 응답 시간(ms), 상위 3개의 제목과 유사도를 본다. 학습 전 e5는 이 질문의 정답(근로기준법 제55조)을 3교시 표에서 65위에 뒀으므로 상위 3개에 없는 것이 정상이다.
4. 브라우저 주소와 해 볼 것 다섯 가지를 안내하고 서버를 열어 둔다.

브라우저 주소는 스크립트가 출력한다(http://127.0.0.1:8765). 예시 질문 칩을 누르고, 테마 · 법령 필터를 걸어 보고, "같은 조 전체"를 눌러 본다. 개발자 도구 Network 탭에서 화면이 보내는 `/api/search` 요청과 응답 JSON을 본다. 같은 서버의 http://127.0.0.1:8765/docs 가 API 설명서다.

선택: 서버만 파인튜닝 모델(002_hard_negatives, torch)로 다시 띄운다. 화면은 그대로다. 같은 질문의 순위가 어떻게 달라지는지 본다.

```shell
uv run python lecture/08_product/04_web_app.py --run
```

`--run`은 2에서 서버 명령에 `--checkpoint models/finetuned/002_hard_negatives --backend torch`를 붙인다. 인덱스는 그 모델로 만든 것을 자동으로 쓴다. 모델 로드가 수십 초로 길어지고, 3의 같은 질문에서 정답 근로기준법 제55조가 1위로 올라온다(3교시 표, experiments/002_hard_negatives). 모델이 없으면 받는 명령을 안내하고 끝난다.

직접 치면 한 줄이다. 동료에게 줄 때는 이 명령과 인덱스만 있으면 된다.

```shell
uv run --package ragkit-api ragkit-api --web-dist apps/web-lite
```

## 선택: 모니터링 — MLflow로 실험 · 모델 · 서비스를 한곳에서

MLflow 서버를 띄우고 → 비교 실험 기록 → 모델 등록(`champion`) → 등록된 모델로 API → 질문마다 남는 트레이스까지 한 번에 본다. 먼저 확인 모드로 돌리고(약 90초), 화면을 보려면 기본 모드로 실행한다(끝나면 MLflow 화면을 열어 둔다, Ctrl+C로 끝).

```shell
uv run python lecture/08_product/extra_monitoring.py --check
uv run python lecture/08_product/extra_monitoring.py
```

스크립트는 `ragkit` · `ragkit-bench` · `ragkit-api`를 자식 프로세스로 부르고, 환경 변수 `MLFLOW_TRACKING_URI`만 넘긴다. 코드 변경 없이 이 변수 하나로 모든 도구가 같은 서버에 기록한다는 것이 핵심이다. 7교시의 가지치기 + INT8 모델(`models/multilingual-e5-small-pruned-int8`)과 `data/splits/test.jsonl`이 있어야 한다.

1. `mlflow server`를 포트 5050에 띄운다(macOS는 5000을 AirPlay가 쓴다). 저장소는 `mlruns/mlflow.db`, 아티팩트는 `mlruns/artifacts`다. `/health`가 응답하면 다음으로 간다.
2. `ragkit-bench run --variants onnx-int8 --variants onnx-int8-pruned --questions data/splits/test.jsonl --n-latency 30 --out-dir mlruns/bench`로 7교시 벤치를 다시 돌린다. 부모 run 하나와 변형마다 자식 run(R@5 · 지연 · 메모리 · 크기)이 남는다. `--check`는 test 앞 100개 질문만 써서 약 90초다. 기본 모드는 test 전체라 몇 분 걸린다.
3. `ragkit register models/multilingual-e5-small-pruned-int8 --name law-embedder --alias champion`으로 모델을 등록하고 별칭을 붙인다. 서비스는 `models:/law-embedder@champion`만 알면 되므로, 다음 모델이 나오면 별칭만 옮긴다.
4. `ragkit-api --port 8766 --model models:/law-embedder@champion --db mlruns/notebooks.sqlite`로 API를 띄운다. 레지스트리에서 champion을 내려받아 열므로 최대 120초 기다린다. 노트북 "근로 상담"(근로기준법)을 만들고 "연차 휴가는 며칠인가요?" · "그럼 1년 미만 근무자는요?" 두 질문을 보낸다. 노트북 id가 곧 MLflow 세션 id다.
5. `mlflow.search_traces`로 실험 `ragkit-service`에서 이 세션의 트레이스를 읽어 질문마다 스팬 트리를 출력한다. `answer (CHAIN)` 아래 `search (RETRIEVER)`와 찾은 조문 id, 그 아래 `embed_query (EMBEDDING)`와 각 ms를 본다. UI의 Traces 화면과 같은 내용이다.
6. 기본 모드는 UI에서 볼 곳(Experiments › ragkit의 Compare, Models › law-embedder, ragkit-service의 Traces · Sessions)을 안내하고 MLflow 서버를 열어 둔다. `--check`는 5에서 끝난다.

브라우저에서 http://127.0.0.1:5050. LLM 답변은 쓰지 않으므로 트레이스는 search · embed_query 스팬까지 남고 상태는 ERROR로 표시된다. 기록은 `mlruns/`에 쌓이고, 처음부터 다시 하려면 이 폴더를 지운다.

## 선택: 법령 노트 — NotebookLM형 React 화면 (apps/web)

4번의 검색 화면에 노트북(법령 묶음) · 채팅 · 인용 · 노트를 더한 화면이다. React로 만들어 빌드(Node.js 20+와 pnpm, `corepack enable`)가 필요하고, 채팅의 답변 생성(LLM)은 이 강의에서 쓰지 않아 근거 조문까지만 나온다. 그래서 선택 실습으로 둔다. 먼저 확인만 하고, 그다음 띄워 브라우저로 본다(Ctrl+C로 끝).

```shell
uv run python lecture/08_product/extra_web_notebook.py --check
uv run python lecture/08_product/extra_web_notebook.py
```

4번과 같은 `ragkit-api`를 쓰고 `--web-dist`만 `apps/web/dist`로 바꾼다. `--check`는 1~2만 하고 닫는다.

1. `apps/web/dist/index.html`이 없으면 `pnpm install --frozen-lockfile`과 `pnpm build`로 화면을 빌드한다. 있으면 건너뛴다. dist의 파일 수와 KB를 출력한다. 모델과 인덱스는 들어 있지 않다.
2. `ragkit-api --port 8766 --web-dist apps/web/dist`를 띄우고 `/api/health`가 응답할 때까지 최대 60초 기다린다. `GET /`가 React 루트가 든 `index.html`을 주는 것, `/notebooks/any-id` 같은 화면 경로도 같은 `index.html`로 가서 React Router가 처리하는 것을 본다.
3. 브라우저 주소와 해 볼 것(노트북 만들기 → 소스 추가 → 질문 → 인용 [1] 눌러 원문 → 노트 저장)을 안내하고 서버를 열어 둔다.

브라우저 주소는 스크립트가 출력한다(http://127.0.0.1:8766). 화면 코드를 고쳤거나 `apps/web/dist`가 없으면 다시 빌드한다.

```shell
uv run python lecture/08_product/extra_web_notebook.py --run
```

`--run`은 dist가 있어도 1에서 다시 빌드한다. 모니터링 스크립트와 같은 포트 8766을 쓰므로 둘을 동시에 띄우지 않는다.

## 참고

| 파일 | 내용 |
|---|---|
| `01_search_cli.py` | 검색 CLI(laws · search · show · --json), uvx 배포(--run) |
| `SKILL.md` | 에이전트 스킬(SKILL.md 하나로 조문 검색 연동, 프롬프트 규약) |
| `03_mcp_server.py` | MCP: 연결 → 도구 목록 → 호출, Antigravity/Claude 등록, CLI + 스킬과 비교 |
| `04_web_app.py` | 검색 화면(HTML 한 장, `apps/web-lite`)을 API 서버가 한 주소에서 배포. `--run` 파인튜닝 모델로 |
| `extra_monitoring.py` | MLflow: 실험 비교 → 모델 레지스트리 → 서비스 트레이스 · 세션 |
| `extra_web_notebook.py` | (선택) 법령 노트(React, `apps/web`) 빌드와 한 주소 배포 |

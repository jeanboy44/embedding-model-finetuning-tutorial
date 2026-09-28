# ragkit-search — 법령 검색 CLI

터미널에서 한국 법령 조문을 검색하고, 찾은 조문만 근거로 답을 받는다.
ragkit core 외 의존성이 없다 (출력은 표준 라이브러리만).

## 실행

저장소 안에서:

```bash
uv run --package ragkit-search ragkit-search search "야간 근로 수당"
```

다른 폴더에서 (인덱스·모델 경로의 기준은 `RAGKIT_PROJECT_ROOT`):

```bash
export RAGKIT_PROJECT_ROOT=/path/to/embedding-model-finetuning-tutorial
uvx --from /path/to/embedding-model-finetuning-tutorial/apps/search-cli ragkit-search laws
```

인덱스가 없으면 먼저 만든다: `uv run ragkit index`.

## 명령

```bash
# 검색: 순위, 점수, 제목, id, 본문 앞부분
ragkit-search search "야간 근로 수당" -k 3 --law 근로기준법
ragkit-search search "전세 보증금" --law 주택임대차보호법 --law "주택임대차보호법 시행령" --json

# 답변 (GEMINI_API_KEY 필요): 받는 대로 출력한 뒤 근거 조문 [n] 목록
ragkit-search ask "수습 기간에도 최저임금을 줘야 하나요?" --law 최저임금법

# 법령 목록 (--law에 쓸 정확한 이름)
ragkit-search laws --theme youth
ragkit-search laws --json

# 조문 보기: 조각 하나 / 같은 조 전체(항·호 모두)
ragkit-search show 근로기준법_법률_제56조
ragkit-search show 소득세법_시행령_제17조_제2항 --article
```

모든 명령에 모델·인덱스 옵션이 있다 (기본값은 Settings / `.env`):
`--model`, `--checkpoint`, `--backend`, `--index`.

```bash
ragkit-search search "주휴수당" --model google/embeddinggemma-300m
ragkit-search search "주휴수당" --checkpoint models/my-finetuned --index data/processed/index/my.sqlite
```

## 오류

인덱스가 없거나, 모르는 법령 이름을 주거나, `ask`에 키가 없으면 stderr에 한 줄 안내를 쓰고 종료 코드 1로 끝난다.
`ask`는 키가 없어도 근거 조문 목록은 출력한다.

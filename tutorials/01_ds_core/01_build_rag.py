"""
1단계-1: 법령 RAG 한 바퀴
==========================

학습 목표:
- 법령 코퍼스(약 2.6만 조각)를 LLM에 "통째로" 넣을 수 없는 이유를 숫자로 확인한다
- 코퍼스를 임베딩해 SQLite(sqlite-vec) 인덱스를 만든다 (한 번 만들면 재사용)
- 질문 → 조문 검색 → 조문을 근거로 LLM 답변까지 한 번 돌려 본다
- 메타데이터(테마, 법령 종류)로 검색 범위를 좁혀 본다

사전 준비:
    uv run python scripts/prepare_law_data.py        # data/processed/law_docs.json
    .env에 GEMINI_API_KEY (없으면 LLM 단계만 건너뜀)

실행:
    uv run python tutorials/01_ds_core/01_build_rag.py
"""

import time

from ragkit.config import get_settings
from ragkit.data import doc_text, load_corpus
from ragkit.embeddings import create_embedding_fn, format_queries
from ragkit.models import count_tokens
from ragkit.rag import answer_with_rag
from ragkit.retrieval import build_index

QUESTION = "편의점 알바를 3개월 했는데 주휴수당을 받을 수 있나요?"
LEGAL_TERMS_QUESTION = "주휴일 유급휴일 1주 개근"  # 같은 내용을 법률 용어로
ANSWER_ID = "근로기준법_법률_제55조"  # 주휴일(유급휴일) 조문

# 참고 수치 (2026-09 측정·확인)
CONTEXT_WINDOW = 1_000_000  # Gemini 입력 한도 (토큰)
FREE_TIER_TPM = 250_000  # 무료 등급 분당 토큰 한도 (AI Studio에서 현재 값 확인)
TOKENS_PER_CHAR = 0.63  # youth 테마 실측값. API 키가 없을 때 추정에 쓴다


def has_api_key() -> bool:
    key = get_settings().gemini_api_key
    return bool(key) and key != "your_gemini_api_key_here"


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


# ============================================================
# 1. 코퍼스
# ============================================================
settings = get_settings()
docs = load_corpus(settings.data_dir / "processed" / "law_docs.json")

section("1. 코퍼스: 대한민국 법령 조문")
themes: dict[str, int] = {}
for d in docs:
    themes[d["theme"]] = themes.get(d["theme"], 0) + 1
print(f"문서(조문 조각) {len(docs):,}개, 법령 {len({d['law_name'] for d in docs})}개")
for theme, n in sorted(themes.items(), key=lambda x: -x[1]):
    print(f"  {theme:10s} {n:6,}개")
print(f"\n예시: {docs[0]['title']}\n  {docs[0]['text'][:80]}...")


# ============================================================
# 2. 통째로 LLM에 넣으면?
# ============================================================
section("2. 통째로 LLM에 넣으면?")
total_chars = sum(len(doc_text(d)) for d in docs)
if has_api_key():
    # 전체를 한 번에 세면 요청이 너무 커서, 한 테마를 실측해 비율로 추정한다
    youth = "\n\n".join(doc_text(d) for d in docs if d["theme"] == "youth")
    ratio = count_tokens(youth) / len(youth)
    how = "youth 테마 실측 비율로 추정"
else:
    ratio = TOKENS_PER_CHAR
    how = "미리 잰 비율(0.63 토큰/자)로 추정"
total_tokens = int(total_chars * ratio)
print(f"코퍼스 {total_chars:,}자 → 약 {total_tokens:,} 토큰 ({how})")
print(f"  LLM 입력 한도 {CONTEXT_WINDOW:,} 토큰의 {total_tokens / CONTEXT_WINDOW:.1f}배 → 한 번에 넣을 수 없다")
print(f"  무료 등급 분당 한도 {FREE_TIER_TPM:,} 토큰의 {total_tokens / FREE_TIER_TPM:.0f}배")
print("→ 질문과 관련된 조문만 골라 넣는다 = RAG")


# ============================================================
# 3. 인덱스 만들기 (처음 한 번만 오래 걸린다)
# ============================================================
section("3. 인덱스: 임베딩 + SQLite(sqlite-vec)")
model_key = settings.embedding_model_name.split("/")[-1]
index_path = settings.data_dir / "processed" / "index" / f"{model_key}.sqlite"
embed = create_embedding_fn(settings.embedding_model_name)

start = time.perf_counter()
print(f"{index_path} (없으면 만든다. Mac 기준 약 7~13분)")
index = build_index(docs, embed, index_path, model_key=model_key)
print(f"문서 {len(index):,}개, {index.dim}차원, {time.perf_counter() - start:.1f}초")
print(f"파일 크기 {index_path.stat().st_size / 1e6:.0f} MB — 배포할 때는 이 파일 하나만 복사한다")


# ============================================================
# 4. 검색
# ============================================================
section("4. 검색: 질문과 가까운 조문 찾기")
print(f"질문: {QUESTION}\n")
query_vec = embed(format_queries([QUESTION]))[0]

print("[전체 코퍼스]")
for hit in index.search(query_vec, k=5):
    print(f"  {hit.score:.3f}  {hit.metadata['title']}")

print("\n[정답 조문(근로기준법 제55조 휴일)은 몇 위에 있나?]")
for q in (QUESTION, LEGAL_TERMS_QUESTION):
    hits = index.search(embed(format_queries([q]))[0], k=1000)
    rank = next((i for i, h in enumerate(hits, 1) if h.id.startswith(ANSWER_ID)), None)
    print(f"  {rank or '1000위 밖'}위  ← {q}")
print("  같은 내용이라도 일상어로 물으면 정답이 top-5 밖으로 밀린다 (용어 불일치)")

print("\n[필터: youth 테마 + 법률만 (시행령·시행규칙 제외)]")
for hit in index.search(query_vec, k=5, where={"theme": "youth", "law_type": "법률"}):
    print(f"  {hit.score:.3f}  {hit.metadata['title']}")
print("  필터는 범위만 좁힐 뿐, 일상어-법률용어 차이는 해결하지 못한다")


# ============================================================
# 5. 조문을 근거로 답변
# ============================================================
section("5. 생성: 찾은 조문만 근거로 LLM이 답한다")
if not has_api_key():
    print("GEMINI_API_KEY가 없어 건너뜁니다. (.env에 넣으면 실행됩니다)")
else:
    result = answer_with_rag(QUESTION, index, embed, k=5)
    print(result.answer.strip())
    print(f"\nLLM 호출 {result.llm_calls}회, 입력 {result.input_tokens:,} 토큰, "
          f"출력 {result.output_tokens:,} 토큰, {result.latency_s:.1f}초")
    print(f"→ 통째로 넣는 경우(약 {total_tokens:,} 토큰)의 "
          f"1/{total_tokens // max(result.input_tokens, 1):,}")

print("""
정리:
- 정답 조문이 코퍼스에 있어도 검색이 못 찾으면 LLM은 답할 수 없다
- 검색 정확도(Recall)가 답변 정확도의 상한이다

다음 단계 (02~05):
- 검색 없이 LLM만 쓰면? RAG는 얼마나 나은가?
- 검색이 놓친 조문은 답할 수 없다 → 검색 정확도(Recall)가 답변 정확도의 상한
- 쿼리 확장으로 검색을 보완하면 쿼리마다 LLM 비용이 다시 늘어난다
""")

"""
4교시: 질문 데이터 — Claude 스킬이 만든 질문 · 정답 · hard negative
==============================================================================

학습 목표:
- 받은 질문(data/questions/, 파트 230개)의 규모와 구성을 숫자로 본다:
  질문 유형(situation · question · keyword) 비율, 테마별 수, 정답으로 쓰인 검색 문서 수
- 질문 한 건 = 질문 → 정답 조문(positive) → 헷갈리는 오답(hard negative) 묶음이라는 것을 예시로 본다
- keyword 질문이 실제 검색어처럼 짧은지 어절 수로 확인한다
- 스킬의 검증 스크립트(validate_questions.py)를 파일 하나, 전체 파일에 돌려 본다

2022: 라벨링 가이드를 쓰고 사람이 조문을 읽으며 질문을 직접 쓴다 (수천 개에 몇 주)
2026: 가이드를 SKILL.md로 쓰고 Claude가 파트마다 질문을 쓴다. 검증 스크립트가 거르고, 사람은 가이드와 표본을 고친다

무엇을 하나 (모델을 쓰지 않아 1초 안에 끝난다):
    준비  코퍼스(law_docs.json)와 질문 230개 파일을 ragkit.data.load_corpus · load_questions로 읽는다
    1. 규모        파일 · 법령 · 질문 · 정답 문서 수를 세고 질문 한 건의 다섯 필드를 보여 준다
    2. 유형 · 테마  query_type 비율, 정답 문서의 테마별 수, keyword 질문의 어절 수
    3. 예시        최저임금법 파트 1에서 유형별 질문 하나씩 + "수습" 질문의 정답 · 오답 · 기준 답
    4. 검증        스킬의 validate_questions.py를 파일 하나, 전체 230개 파일에 돌린다
    5. (--run)     일부러 틀린 줄 4개를 임시 파일에 써서 검증이 잡는 오류를 본다.
                   --file을 주면 그 대신 내가 만든 파일을 검증한다

질문 생성 자체는 이 스크립트가 아니라 Claude Code에서 한다 (lecture/04_ds_work/README.md):
    "최저임금법 파트 1 질문 데이터 만들어줘. data/questions_mine/ 에 써 줘"
    → 받은 질문(data/questions/)을 덮어쓰지 않게 다른 폴더를 지정한다

사전 준비:
    uv run python scripts/data_version.py pull v1      # data/questions/ + 코퍼스 (강사 Drive)

실행:
    uv run python lecture/04_ds_work/01_questions.py              # 받은 질문 확인 + 검증 (몇 초)
    uv run python lecture/04_ds_work/01_questions.py --run        # 일부러 틀린 질문 파일을 임시 폴더에 만들어
                                                               # 검증 스크립트가 무엇을 잡는지 본다
    uv run python lecture/04_ds_work/01_questions.py --run --file data/questions_mine/최저임금법__p01.jsonl
                                                               # 내가 스킬로 만든 파일을 검증
"""

import json
import subprocess
import sys
import tempfile
from collections import Counter
from pathlib import Path

import cyclopts
from cyclopts import Parameter
from pydantic import BaseModel, Field

from ragkit.config import get_settings
from ragkit.data import load_corpus, load_questions

ROOT = Path(__file__).resolve().parents[2]
settings = get_settings()
CORPUS = settings.data_dir / "processed" / "law_docs.json"
QUESTIONS = settings.data_dir / "questions"
SKILL = ROOT / ".claude" / "skills" / "law-question-gen"
VALIDATE = SKILL / "scripts" / "validate_questions.py"
EXAMPLE_FILE = QUESTIONS / "최저임금법__p01.jsonl"  # 장표의 예시 법령
EXAMPLE_WORD = "수습"  # 장표의 hard negative 예시: 수습이라고 시급을 깎겠대요
TYPES = ("situation", "question", "keyword")  # 상황 설명 · 궁금한 점 · 검색어


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def rel(path: Path) -> str:
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def validate(*files: Path, tail: int = 6) -> int:
    """스킬의 검증 스크립트를 실행하고 마지막 몇 줄(통계·결과)을 보여 준다.

    검증 스크립트는 스킬 폴더의 것을 그대로 서브프로세스로 돌린다(같은 파이썬).
    출력이 길 수 있어 끝에서 tail줄만 보여 준다. 기본 6줄이면 통계 4줄과 결과가 보인다.
    종료 코드는 오류가 하나라도 있으면 1이다.
    """
    shown = (
        files[0].name
        if len(files) == 1
        else f"{files[0].parent.name}/*.jsonl ({len(files)}개)"
    )
    print(f"$ uv run python {rel(VALIDATE)} {shown}")
    result = subprocess.run(
        [sys.executable, str(VALIDATE), *map(str, files)],
        cwd=ROOT,
        capture_output=True,
        text=True,
        check=False,  # 실패(코드 1)도 출력을 보여 줘야 하므로 예외로 올리지 않는다
    )
    lines = [
        line for line in (result.stdout + result.stderr).splitlines() if line.strip()
    ]
    for line in lines[-tail:]:
        print(f"  {line}")
    return result.returncode


@Parameter(name="*")
class Options(BaseModel):
    run: bool = Field(
        False, description="틀린 질문 파일로 검증 스크립트 직접 돌려 보기"
    )
    file: Path | None = Field(
        None, description="--run에서 검증할 내 질문 파일 (스킬로 만든 것)"
    )


def cli(*, options: Options | None = None) -> Options:
    """4교시: 질문 데이터"""
    return options or Options()


args = cyclopts.run(cli, result_action="return_value")
if args is None:  # --help
    sys.exit(0)

# 준비: 산출물이 없으면 스택트레이스 대신 받는 명령만 알려 주고 끝난다.
files = sorted(QUESTIONS.glob("*.jsonl"))  # 파일 하나 = 법령 한 파트
if not CORPUS.exists() or not files:
    print(f"코퍼스 또는 질문이 없습니다: {rel(CORPUS)}, {rel(QUESTIONS)}/")
    print("받기: uv run python scripts/data_version.py pull v1")
    sys.exit(1)

corpus_by_id = {d["id"]: d for d in load_corpus(CORPUS)}  # id → 검색 문서(조문 조각)
questions = load_questions(QUESTIONS)  # 폴더를 주면 안의 jsonl을 모두 합쳐 읽는다


# ============================================================
# 1. 규모
# ============================================================
# 질문 데이터가 얼마나 되는지 숫자로 본다. 파일 이름 앞부분이 법령 이름이라
# 파일 목록만으로 법령 수를 센다. 정답 문서 수는 코퍼스 중 질문이 한 번이라도
# 가리킨 문서의 비율로, 평가와 학습이 코퍼스를 얼마나 덮는지 보여 준다.
# 출력에서 볼 것: 파일 230개 · 법령 54개 · 질문 12,232개 · 정답 문서 9,704개(37%)
section("1. 받은 질문: 규모")
laws = {f.stem.split("__")[0] for f in files}  # "최저임금법__p01" → "최저임금법"
positives = {q["positive_id"] for q in questions}  # 정답으로 쓰인 문서 id (중복 제거)
print(
    f"파일(파트)  {len(files):>7,}개  · 법령 {len(laws)}개 (파일 이름: <법령>__pNN.jsonl)"
)
print(f"질문        {len(questions):>7,}개")
print(
    f"정답 문서   {len(positives):>7,}개  ← 정답으로 한 번 이상 쓰인 검색 문서"
    f" (코퍼스 {len(corpus_by_id):,}개 중 {len(positives) / len(corpus_by_id):.0%})"
)
print(f"한 줄의 필드: {', '.join(questions[0])}")


# ============================================================
# 2. 유형 · 테마
# ============================================================
# 질문이 실제 사용자를 닮았는지 보는 단계. query_type 비율(상황 설명 · 궁금한 점 ·
# 검색어)과 정답 문서의 테마별 수를 세고, keyword 질문이 실제 검색어처럼 짧은지
# 어절 수(공백으로 나눈 단어 수)로 잰다. 테마는 질문이 아니라 정답 문서에 달려 있다.
# 출력에서 볼 것: situation 41% · question 36% · keyword 23%, keyword 평균 5.6어절,
# 2~3어절 9%. 장표의 실제 검색어(2~3어절 70%)와 벌어진 차이가 가이드를 고친 이유다.
section("2. 질문 유형과 테마")
types = Counter(q["query_type"] for q in questions)
for t in TYPES:
    print(f"  {t:10s} {types[t]:6,}개  {types[t] / len(questions):5.0%}")

print("\n테마별 (정답 문서의 테마)")
themes = Counter(
    corpus_by_id[q["positive_id"]]["theme"]  # tax · finance · traffic · youth …
    for q in questions
    if q["positive_id"] in corpus_by_id
)
for theme, n in themes.most_common():
    print(f"  {theme:10s} {n:6,}개")

keywords = [q["query"] for q in questions if q["query_type"] == "keyword"]
words = [len(k.split()) for k in keywords]  # 어절 수 = 공백으로 나눈 조각 수
short = sum(2 <= w <= 3 for w in words)  # 실제 검색어 길이(2~3어절)에 드는 질문
print(
    f"\nkeyword 질문 {len(keywords):,}개: 평균 {sum(words) / max(len(words), 1):.1f}어절,"
    f" 2~3어절 {short / max(len(words), 1):.0%}"
)
print(
    "  장표: 실제 검색어(KoAIO)는 2~3어절이 70%. 이 차이를 보고 SKILL.md에 실제 검색어 형식을"
)
print(
    "  few-shot으로 넣고 keyword 목표를 35%로 올렸다. 받은 질문(v1)은 그 전에 만든 것이라 위 숫자가 그대로다"
)


# ============================================================
# 3. 예시: 질문 → 정답 → hard negative
# ============================================================
# 질문 한 건의 생김새를 눈으로 본다. 장표와 같은 최저임금법 파트 1 파일에서
# 유형별 질문 하나씩을 보여 주고, "수습" 질문 한 건의 정답 조문 · 오답 조문 ·
# 기준 답을 코퍼스에서 꺼내 나란히 찍는다. 오답은 같은 주제인데 이 질문에는
# 답하지 않는 조문이라는 점이 핵심이다. 끝에 질문당 hard negative 수 분포가 나온다.
# 출력에서 볼 것: 정답 제5조(최저임금액), 오답 제6조 제1항 · 제7조. 1개 2,451 · 2개 9,772
section("3. 예시: 질문 → 정답 조문 → hard negative")
example_rows = load_questions(EXAMPLE_FILE) if EXAMPLE_FILE.exists() else questions
for t in TYPES:
    q = next((q for q in example_rows if q["query_type"] == t), None)  # 유형별 첫 질문
    if q:
        print(f"  [{t}] {q['query']}")

q = next((q for q in example_rows if EXAMPLE_WORD in q["query"]), example_rows[0])
pos = corpus_by_id[q["positive_id"]]  # 정답 조문 조각 (title · text · theme …)
print(f"\n질문      {q['query']}  ({q['query_type']})")
print(f"정답      {pos['title']}")
print(f"          {pos['text'][:120]}...")  # 조문 본문은 앞 120자만
for neg_id in q["hard_negative_ids"]:
    neg = corpus_by_id[neg_id]
    print(f"오답      {neg['title']}")
    print(f"          {neg['text'][:80]}...")
print(f"기준 답   {q['answer']}")  # 사람이 읽는 참고용. 학습 · 평가에는 쓰지 않는다
print("→ 같은 주제(최저임금)지만 이 질문에는 답하지 않는 조문을 오답으로 고른다")

n_neg = Counter(len(q["hard_negative_ids"]) for q in questions)  # 1~3개
print(
    "\n질문당 hard negative 수: "
    + ", ".join(f"{k}개 {v:,}" for k, v in sorted(n_neg.items()))
)


# ============================================================
# 4. 검증 스크립트
# ============================================================
# 스킬이 질문 파일을 쓴 뒤 반드시 돌리는 validate_questions.py를 두 번 부른다.
# 먼저 파일 하나(최저임금법 파트 1), 다음 230개 파일 전체. 파일 사이에 같은 질문이
# 다른 정답을 가리키는 중복은 전체를 함께 넣어야 잡히기 때문이다.
# 출력에서 볼 것: 한 파일은 질문 46개 통과, 전체는 WARN 1건(가치 낮은 조문)만 내고 통과
section("4. 검증: 스킬이 파일을 쓴 뒤 반드시 돌리는 스크립트")
print(
    "잡는 것: 필드 누락, 코퍼스에 없는 id, 다른 법령의 정답, 질문 속 조문 번호(정답 누설),"
)
print(
    "        파일 안·파일 사이의 중복 질문. 경고: 너무 짧거나 긴 질문, 가치 낮은 조문\n"
)
validate(EXAMPLE_FILE if EXAMPLE_FILE.exists() else files[0])  # 파일 하나
print()
validate(*files)  # 전체 230개 파일을 한 번에
print(
    "  → 파일 사이 중복은 전체를 함께 검사해야 잡힌다 (같은 질문이 다른 정답을 가리키면 학습이 흔들린다)"
)


# ============================================================
# 5. (--run) 검증 스크립트가 무엇을 잡는지 직접
# ============================================================
# --file이 있으면 내가 스킬로 만든 파일을 검증한다(오류가 많을 수 있어 30줄까지 보여 준다).
# 없으면 최저임금법 파트 1의 앞 세 줄을 복사해 일부러 틀린 줄 4개와 정상 줄 1개를
# 임시 폴더에 쓰고 검증을 돌린다. 임시 파일은 바로 지우고 data/questions/는 건드리지 않는다.
# 출력에서 볼 것: ERROR 4건이 줄 번호 1 · 2 · 3 · 5에 찍히고 종료 코드 1로 끝난다
if args.run:
    section("5. --run: 검증 직접 돌려 보기")
    if args.file:
        if not args.file.exists():
            print(f"파일이 없습니다: {args.file}")
        else:
            print("내가 만든 파일 (파일 이름이 <법령>__pNN.jsonl 이어야 법령을 읽는다)")
            validate(args.file, tail=30)
    else:
        rows = load_questions(EXAMPLE_FILE)[:3]  # 정상 질문 3건을 재료로 쓴다
        broken = [
            {**rows[0], "query": rows[0]["query"] + " 제5조 맞죠?"},  # 조문 번호 누설
            {**rows[1], "positive_id": "최저임금법_법률_제999조"},  # 없는 id
            {**rows[2], "hard_negative_ids": [rows[2]["positive_id"]]},  # 정답이 오답에
            rows[0],  # 정상 줄 (1번째 줄은 조문 번호를 붙였으니 중복 아님)
            {k: v for k, v in rows[1].items() if k != "answer"},  # 필드 누락
        ]  # 4번째 줄(정상)만 통과하고 1 · 2 · 3 · 5번째 줄이 ERROR로 잡힌다
        with tempfile.TemporaryDirectory() as tmp:
            path = (
                Path(tmp) / EXAMPLE_FILE.name
            )  # 파일 이름에서 법령을 읽으므로 같은 이름
            path.write_text(
                "\n".join(json.dumps(r, ensure_ascii=False) for r in broken) + "\n",
                encoding="utf-8",
            )
            print(f"임시 파일 {path.name}에 일부러 틀린 줄 4개 + 정상 줄 1개를 썼다")
            code = validate(path, tail=12)
            print(
                f"  종료 코드 {code} (1 = 실패). 스킬은 이 오류가 0개가 될 때까지 파일을 고친다"
            )
        print("임시 파일은 지웠다. data/questions/는 바뀌지 않았다.")

print(f"""
직접 만들어 보기 (Claude Code를 저장소 루트에서 열고):
  uv run python {rel(SKILL / "scripts" / "law_parts.py")} list          # 법령·파트 목록
  "최저임금법 파트 1 질문 데이터 만들어줘. data/questions_mine/ 에 써 줘"
  uv run python lecture/04_ds_work/01_questions.py --run --file data/questions_mine/최저임금법__p01.jsonl

정리:
- 질문 한 건 = 일상 말투 질문 + 정답 조문 + 같은 주제의 오답. 평가와 학습(5 · 6교시)이 모두 이것을 쓴다
- 사람은 가이드(SKILL.md)와 표본을 고치고, 형식 오류는 검증 스크립트가 잡는다
- 다음(5교시): uv run python lecture/05_data_eval/01_corpus.py (질문이 찾아야 할 코퍼스)
""")

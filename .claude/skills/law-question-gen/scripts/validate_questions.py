"""생성한 질문 JSONL 파일을 검사하고 통계를 보여 준다 (질문 생성 스킬용).

사용법:
    uv run python .claude/skills/law-question-gen/scripts/validate_questions.py data/questions/최저임금법__p01.jsonl
    uv run python .claude/skills/law-question-gen/scripts/validate_questions.py data/questions/*.jsonl

오류가 있으면 줄 번호와 함께 출력하고 종료 코드 1로 끝난다. 경고는 품질 신호일 뿐 실패로 치지 않는다.
"""

import json
import re
import sys
from collections import Counter
from pathlib import Path

from cyclopts import run

sys.path.insert(0, str(Path(__file__).parent))
from law_parts import CORPUS, LOW_VALUE_TITLES, load_corpus

REQUIRED = {"query", "positive_id", "hard_negative_ids", "query_type", "answer"}
QUERY_TYPES = {"situation", "question", "keyword"}
ARTICLE_NO_RE = re.compile(r"제\s*\d+\s*조")


def validate_file(
    path: Path, articles: dict[str, dict]
) -> tuple[list[str], list[str], list[dict]]:
    """한 파일을 검사해 (오류, 경고, 레코드)를 돌려준다."""
    errors: list[str] = []
    warnings: list[str] = []
    records: list[dict] = []
    law = path.stem.split("__")[0]

    with path.open(encoding="utf-8") as f:
        lines = [line for line in f if line.strip()]
    for lineno, line in enumerate(lines, 1):
        where = f"{path.name}:{lineno}"
        try:
            record = json.loads(line)
        except json.JSONDecodeError as e:
            errors.append(f"{where} JSON 파싱 실패: {e}")
            continue
        if missing := REQUIRED - record.keys():
            errors.append(f"{where} 필드 누락: {sorted(missing)}")
            continue

        query, positive = record["query"], record["positive_id"]
        negatives = record["hard_negative_ids"]
        if positive not in articles:
            errors.append(f"{where} positive_id가 코퍼스에 없음: {positive}")
        elif articles[positive]["category"] != law:
            errors.append(
                f"{where} positive_id가 파일의 법령({law})이 아님: {positive}"
            )
        if not isinstance(negatives, list) or not negatives:
            errors.append(f"{where} hard_negative_ids는 비어 있지 않은 리스트여야 함")
        else:
            for neg in negatives:
                if neg not in articles:
                    errors.append(f"{where} hard_negative_id가 코퍼스에 없음: {neg}")
                elif neg == positive:
                    errors.append(
                        f"{where} hard_negative_ids에 positive_id가 들어 있음"
                    )
        if record["query_type"] not in QUERY_TYPES:
            errors.append(
                f"{where} query_type은 {sorted(QUERY_TYPES)} 중 하나: {record['query_type']}"
            )
        if ARTICLE_NO_RE.search(query):
            errors.append(f"{where} 질문에 조문 번호가 들어 있음 (정답 누설): {query}")
        if not 5 <= len(query) <= 150:
            warnings.append(f"{where} 질문 길이 {len(query)}자: {query}")
        if (
            positive in articles
            and articles[positive]["article_title"] in LOW_VALUE_TITLES
        ):
            warnings.append(f"{where} LOW-VALUE 조문에 질문을 만듦: {positive}")
        records.append(record)

    duplicates = [q for q, n in Counter(r["query"] for r in records).items() if n > 1]
    errors.extend(f"{path.name} 중복 질문: {q}" for q in duplicates)
    return errors, warnings, records


def main(*files: Path, corpus: Path = CORPUS) -> None:
    """질문 JSONL 파일을 검사한다.

    Args:
        files: 검사할 JSONL 파일들.
        corpus: 코퍼스 JSON 경로.
    """
    articles = {a["id"]: a for a in load_corpus(corpus)}
    all_errors: list[str] = []
    all_records: list[dict] = []
    records_by_file: dict[Path, list[dict]] = {}
    for path in files:
        errors, warnings, records = validate_file(path, articles)
        all_errors += errors
        all_records += records
        records_by_file[path] = records
        for message in errors:
            print(f"ERROR {message}")
        for message in warnings:
            print(f"WARN  {message}")

    # 여러 파일을 함께 검사하면 파일 사이의 중복 질문도 잡는다. 같은 질문이 다른 정답을 가리키면
    # MNRL 학습에서 서로 모순되는 신호가 된다(예: 법령마다 다른 "청년 나이 기준").
    files_by_query: dict[str, set[str]] = {}
    for path, records in records_by_file.items():
        for record in records:
            files_by_query.setdefault(record["query"], set()).add(path.name)
    for query, names in files_by_query.items():
        if len(names) > 1:
            message = f"여러 파일에 같은 질문 ({', '.join(sorted(names))}): {query}"
            all_errors.append(message)
            print(f"ERROR {message}")

    covered = {r["positive_id"] for r in all_records}
    types = Counter(r["query_type"] for r in all_records)
    law_name_share = sum(
        articles[r["positive_id"]]["law_name"].split()[0] in r["query"]
        for r in all_records
        if r["positive_id"] in articles
    ) / max(len(all_records), 1)

    print(f"\n질문 {len(all_records)}개, 정답 조문 {len(covered)}개")
    print(f"질문 유형: {dict(types)}")
    print(f"법령 이름이 질문에 들어간 비율: {law_name_share:.0%}")
    print("결과: " + ("실패" if all_errors else "통과"))
    sys.exit(1 if all_errors else 0)


if __name__ == "__main__":
    run(main)

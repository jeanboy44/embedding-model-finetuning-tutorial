"""코퍼스·질문 데이터 로드와 정리.

코퍼스는 scripts/prepare_law_data.py가 만든 law_docs.json(조문 또는 조문 조각 목록),
질문은 law-question-gen 스킬이 만든 JSONL(한 줄에 질문 하나)이다.
"""

import json
from pathlib import Path


def load_corpus(path: Path) -> list[dict]:
    """코퍼스 JSON(문서 목록)을 읽는다.

    Args:
        path: law_docs.json 경로.

    Returns:
        문서 딕셔너리 목록.
    """
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_questions(path: Path) -> list[dict]:
    """질문 JSONL을 읽는다. 폴더를 주면 안의 *.jsonl을 이름순으로 모두 읽는다.

    Args:
        path: JSONL 파일 또는 폴더.

    Returns:
        질문 딕셔너리 목록. 빈 줄은 건너뛴다.
    """
    path = Path(path)
    files = sorted(path.glob("*.jsonl")) if path.is_dir() else [path]
    rows: list[dict] = []
    for file in files:
        for line in file.read_text(encoding="utf-8").splitlines():
            if line.strip():
                rows.append(json.loads(line))
    return rows


def relevance_key(doc: dict) -> str:
    """정답 판정에 쓰는 조 단위 키. 긴 조문의 조각이면 원래 조 id(parent_id)다."""
    return doc.get("parent_id") or doc["id"]


def doc_text(doc: dict) -> str:
    """임베딩할 문서 텍스트. 인덱스·학습·평가가 모두 이 함수를 쓴다."""
    return f"{doc['title']}\n{doc['text']}"


def filter_questions(
    questions: list[dict], corpus_by_id: dict[str, dict]
) -> tuple[list[dict], dict[str, int]]:
    """코퍼스와 맞지 않는 질문과 negative를 걸러 낸다.

    - positive_id가 코퍼스에 없는 질문은 뺀다(옛 코퍼스 id를 쓰는 질문 등).
    - hard_negative_ids에서 코퍼스에 없는 id, related_ids에 있는 id(정답을 부분적으로 담음),
      정답 자신, 중복을 지운다.
    - 같은 조의 다른 조각(제6조 제1항 질문의 제6조 제2항 등)은 다른 질문에 답하는 진짜
      hard negative이므로 남긴다.

    Args:
        questions: 질문 목록. 바꾸지 않는다.
        corpus_by_id: id → 문서.

    Returns:
        (남은 질문 목록, {"missing_positive": 뺀 질문 수, "dropped_negatives": 지운 negative 수})
    """
    kept: list[dict] = []
    missing = dropped = 0
    for question in questions:
        positive = corpus_by_id.get(question["positive_id"])
        if positive is None:
            missing += 1
            continue
        excluded = set(question.get("related_ids") or []) | {positive["id"]}
        original = question.get("hard_negative_ids") or []
        negatives = [
            neg for neg in dict.fromkeys(original) if neg in corpus_by_id and neg not in excluded
        ]
        dropped += len(original) - len(negatives)
        kept.append({**question, "hard_negative_ids": negatives})
    return kept, {"missing_positive": missing, "dropped_negatives": dropped}

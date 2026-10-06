"""코퍼스·질문 데이터 로드와 정리.

코퍼스는 scripts/prepare_law_data.py가 만든 law_docs.json(조문 또는 조문 조각 목록),
질문은 law-question-gen 스킬이 만든 JSONL(한 줄에 질문 하나)이다.
"""

import hashlib
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


def questions_digest(questions: list[dict]) -> str:
    """질문 목록의 짧은 해시(12자). 결과표에 적어 어떤 데이터로 낸 숫자인지 구분한다.

    Args:
        questions: 질문 목록. 순서와 내용이 같으면 같은 해시다.

    Returns:
        sha256 16진수 앞 12자.
    """
    payload = "\n".join(json.dumps(q, ensure_ascii=False, sort_keys=True) for q in questions)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:12]


def question_key(question: dict) -> str:
    """질문 식별자(12자): 질문 문장과 정답 id의 해시. 판정 파일(복수 정답)과 질문을 잇는다."""
    payload = question["query"] + "\0" + question["positive_id"]
    return hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]


def load_labels(path: Path) -> dict[str, dict]:
    """복수 정답 판정 파일(JSONL: qid, alt_positive_ids, partial_ids)을 qid → 판정으로 읽는다."""
    labels = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            labels[row["qid"]] = row
    return labels


def attach_labels(questions: list[dict], labels: dict[str, dict]) -> list[dict]:
    """판정을 질문에 붙인다. full은 alt_positive_ids(정답으로 인정), partial은 partial_ids.

    partial은 multi 판정에서만 순위에서 뺀다 (doc·article 판정은 판정 전과 같게 유지).

    Args:
        questions: 질문 목록. 바꾸지 않는다.
        labels: load_labels 결과.

    Returns:
        판정이 붙은 새 질문 목록 (판정이 없는 질문은 그대로).
    """
    out = []
    for question in questions:
        label = labels.get(question_key(question))
        if label is None:
            out.append(question)
            continue
        out.append(
            {
                **question,
                "alt_positive_ids": list(label.get("alt_positive_ids") or []),
                "partial_ids": list(label.get("partial_ids") or []),
            }
        )
    return out


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

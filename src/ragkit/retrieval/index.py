"""SQLite(sqlite-vec) 벡터 인덱스: 코퍼스 + 메타데이터 + 임베딩을 모델별 파일 하나에 담는다.

    data/processed/index/<model_key>.sqlite
      docs      : 문서 원문과 메타데이터 (필터용 컬럼에 인덱스)
      vec_docs  : sqlite-vec 가상 테이블. 임베딩 + 벡터 검색과 함께 거를 메타데이터
      meta      : 모델 키, 차원, 코퍼스 해시, 생성 시각

전체 코퍼스(약 2.6만 청크) 임베딩은 수 분이 걸리므로 한 번 만든 파일을
비교 실습·평가·앱이 같이 쓴다. 배포할 때는 이 파일 하나만 복사하면 된다.
"""

import hashlib
from collections import Counter
import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import sqlite_vec
from loguru import logger

from ragkit.data import doc_text
from ragkit.embeddings.prefix import format_passages

EmbedFn = Callable[..., np.ndarray]

DOC_COLUMNS = (
    "id", "parent_id", "title", "text", "theme", "law_name", "law_type", "category",
    "article_no", "article_title", "chapter", "promulgation_date", "effective_date", "source_url",
)
# 벡터 검색과 한 쿼리에서 함께 거를 수 있는 컬럼.
# 주의: law_docs.json의 effective_date는 조문별 시행일이 아니라 법령 파일의 최신 개정 시행일이다
# (시행 예정 개정 포함). "시행일 <= 오늘"로 현행 조문을 가려낼 수 없다.
FILTER_COLUMNS = ("theme", "law_type", "law_name", "effective_date")
_OPS = {"=", "!=", "<", "<=", ">", ">="}


@dataclass(frozen=True)
class SearchHit:
    """검색 결과 한 건."""

    id: str
    text: str
    score: float
    metadata: dict = field(default_factory=dict)


def _corpus_hash(docs: list[dict], text_fn: Callable[[dict], str]) -> str:
    h = hashlib.sha256()
    for doc in docs:
        for part in (doc["id"], text_fn(doc), *(str(doc.get(c) or "") for c in FILTER_COLUMNS)):
            h.update(part.encode())
            h.update(b"\0")
    return h.hexdigest()[:16]


def _connect(path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(path))
    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    return conn


class VectorIndex:
    """만들어 둔 인덱스 파일을 열어 검색한다."""

    def __init__(self, conn: sqlite3.Connection, path: Path) -> None:
        self._conn = conn
        self.path = path
        meta = dict(conn.execute("SELECT key, value FROM meta"))
        self.model_key = meta["model_key"]
        self.dim = int(meta["dim"])
        self.corpus_hash = meta["corpus_hash"]

    @classmethod
    def open(cls, path: Path) -> "VectorIndex":
        """인덱스 파일을 연다.

        Raises:
            FileNotFoundError: 파일이 없을 때 (만드는 명령을 안내한다).
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(
                f"인덱스 파일이 없습니다: {path}\n먼저 만드세요: uv run ragkit index"
            )
        return cls(_connect(path), path)

    def __len__(self) -> int:
        return self._conn.execute("SELECT count(*) FROM docs").fetchone()[0]

    def close(self) -> None:
        self._conn.close()

    def search(
        self,
        query_embedding: np.ndarray,
        k: int = 5,
        where: dict[str, str | tuple[str, str]] | None = None,
    ) -> list[SearchHit]:
        """코사인 유사도 상위 k개 문서를 반환한다.

        Args:
            query_embedding: (dim,) 쿼리 임베딩 (query 앞 문구를 붙여 임베딩한 것).
            k: 반환할 문서 수.
            where: 메타데이터 조건. {"theme": "youth"}처럼 값이면 같음,
                {"effective_date": ("<=", "2026-09-29")}처럼 (연산자, 값)이면 비교.
                쓸 수 있는 컬럼: theme, law_type, law_name, effective_date.

        Returns:
            점수 내림차순 SearchHit 리스트. score = 1 - 코사인 거리.
        """
        clauses, params = [], [sqlite_vec.serialize_float32(np.asarray(query_embedding, dtype=np.float32).ravel()), k]
        for column, cond in (where or {}).items():
            if column not in FILTER_COLUMNS:
                raise ValueError(f"필터할 수 없는 컬럼: {column} (가능: {', '.join(FILTER_COLUMNS)})")
            op, value = cond if isinstance(cond, tuple) else ("=", cond)
            if op not in _OPS:
                raise ValueError(f"지원하지 않는 연산자: {op}")
            clauses.append(f"AND v.{column} {op} ?")
            params.append(value)
        rows = self._conn.execute(
            f"""
            SELECT d.*, v.distance FROM vec_docs v JOIN docs d ON d.rowid = v.rowid
            WHERE v.embedding MATCH ? AND k = ? {' '.join(clauses)}
            ORDER BY v.distance
            """,
            params,
        )
        names = [c[0] for c in rows.description]
        hits = []
        for row in rows:
            record = dict(zip(names, row))
            distance = record.pop("distance")
            doc_id, text = record.pop("id"), record.pop("text")
            hits.append(SearchHit(id=doc_id, text=text, score=1.0 - distance, metadata=record))
        return hits


def build_index(
    docs: list[dict],
    embed_fn: EmbedFn,
    db_path: Path,
    *,
    model_key: str,
    text_fn: Callable[[dict], str] = doc_text,
    batch_size: int = 64,
    chunk_size: int = 1024,
) -> VectorIndex:
    """코퍼스를 임베딩해 인덱스 파일을 만든다. 같은 모델·같은 코퍼스의 파일이 있으면 그대로 연다.

    Args:
        docs: 코퍼스 문서 (law_docs.json 레코드). id, title, text와 메타데이터 필드.
        embed_fn: 임베딩 함수 (ragkit.embeddings.create_embedding_fn의 반환값).
        db_path: 인덱스 파일 경로 (예: data/processed/index/multilingual-e5-small.sqlite).
        model_key: 모델을 구분하는 이름. 모델이 바뀌면 다른 값을 줘야 한다.
        text_fn: 문서 → 임베딩할 텍스트. passage 앞 문구는 여기서 붙인다.
        batch_size: 임베딩 배치 크기.
        chunk_size: 진행 상황을 기록하고 DB에 쓰는 단위.

    Returns:
        열린 VectorIndex.
    """
    db_path = Path(db_path)
    counts = Counter(d["id"] for d in docs)
    dups = [doc_id for doc_id, n in counts.items() if n > 1]
    if dups:
        # 임베딩(수 분)을 시작하기 전에 멈춘다
        raise ValueError(f"코퍼스에 중복 id {len(dups)}개가 있습니다 (예: {', '.join(dups[:3])})")
    corpus_hash = _corpus_hash(docs, text_fn)
    if db_path.exists():
        try:
            index = VectorIndex.open(db_path)
        except (sqlite3.DatabaseError, KeyError):
            index = None  # 인덱스가 아닌 파일(빈 파일, 옛 형식) → 새로 만든다
        if index is not None:
            if index.model_key == model_key and index.corpus_hash == corpus_hash:
                return index
            index.close()
        db_path.unlink()

    db_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = db_path.with_suffix(".building")
    tmp_path.unlink(missing_ok=True)
    conn = _connect(tmp_path)
    conn.execute(f"CREATE TABLE docs (rowid INTEGER PRIMARY KEY, {', '.join(f'{c} TEXT' for c in DOC_COLUMNS)})")
    conn.execute("CREATE UNIQUE INDEX docs_id ON docs(id)")
    for column in FILTER_COLUMNS:
        conn.execute(f"CREATE INDEX docs_{column} ON docs({column})")
    conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT)")

    dim = None
    for start in range(0, len(docs), chunk_size):
        chunk = docs[start : start + chunk_size]
        vectors = np.asarray(embed_fn(format_passages([text_fn(d) for d in chunk]), batch_size=batch_size), dtype=np.float32)
        if dim is None:
            dim = vectors.shape[1]
            conn.execute(
                f"CREATE VIRTUAL TABLE vec_docs USING vec0(embedding float[{dim}] distance_metric=cosine, "
                + ", ".join(f"{c} text" for c in FILTER_COLUMNS) + ")"
            )
        for offset, (doc, vec) in enumerate(zip(chunk, vectors)):
            rowid = start + offset + 1
            conn.execute(
                f"INSERT INTO docs (rowid, {', '.join(DOC_COLUMNS)}) VALUES (?, {', '.join('?' * len(DOC_COLUMNS))})",
                (rowid, *(doc.get(c) for c in DOC_COLUMNS)),
            )
            conn.execute(
                f"INSERT INTO vec_docs (rowid, embedding, {', '.join(FILTER_COLUMNS)}) VALUES (?, ?, {', '.join('?' * len(FILTER_COLUMNS))})",
                (rowid, sqlite_vec.serialize_float32(vec), *(doc.get(c) or "" for c in FILTER_COLUMNS)),
            )
        conn.commit()
        logger.info("인덱싱 {}/{}", min(start + chunk_size, len(docs)), len(docs))

    conn.executemany(
        "INSERT INTO meta VALUES (?, ?)",
        [("model_key", model_key), ("dim", str(dim)), ("corpus_hash", corpus_hash),
         ("created_at", datetime.now(timezone.utc).isoformat())],
    )
    conn.commit()
    conn.close()
    tmp_path.rename(db_path)
    return VectorIndex.open(db_path)

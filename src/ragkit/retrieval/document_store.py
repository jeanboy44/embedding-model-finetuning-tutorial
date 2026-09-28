"""문서 저장소."""

import json
from pathlib import Path

import numpy as np


class DocumentStore:
    """문서와 임베딩을 저장하는 저장소.

    Example:
        >>> store = DocumentStore()
        >>> store.add_documents(["doc1", "doc2"], np.random.randn(2, 384))
        >>> store.save(Path("store"))
    """

    def __init__(self) -> None:
        self.documents: list[str] = []
        self.embeddings: np.ndarray | None = None
        self.metadata: list[dict] = []

    def add_documents(
        self,
        texts: list[str],
        embeddings: np.ndarray,
        metadata: list[dict] | None = None,
    ) -> None:
        """사전 계산된 임베딩과 함께 문서를 추가한다.

        Args:
            texts: 문서 텍스트 리스트.
            embeddings: (N, dim) 형태의 임베딩 배열.
            metadata: 문서별 메타데이터. None이면 자동 생성.
        """
        self.documents.extend(texts)

        if self.embeddings is None:
            self.embeddings = embeddings
        else:
            self.embeddings = np.vstack([self.embeddings, embeddings])

        if metadata:
            self.metadata.extend(metadata)
        else:
            self.metadata.extend([{"id": i} for i in range(len(texts))])

    def save(self, path: Path) -> None:
        """저장소를 디스크에 저장한다.

        Args:
            path: 저장 디렉토리 경로.
        """
        path.mkdir(parents=True, exist_ok=True)
        np.save(path / "embeddings.npy", self.embeddings)
        with (path / "documents.json").open("w") as f:
            json.dump(self.documents, f)
        with (path / "metadata.json").open("w") as f:
            json.dump(self.metadata, f)

    def load(self, path: Path) -> None:
        """디스크에서 저장소를 로드한다.

        Args:
            path: 저장 디렉토리 경로.
        """
        self.embeddings = np.load(path / "embeddings.npy")
        with (path / "documents.json").open() as f:
            self.documents = json.load(f)
        with (path / "metadata.json").open() as f:
            self.metadata = json.load(f)

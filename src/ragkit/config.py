"""프로젝트 전역 설정 — pydantic-settings 기반."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """프로젝트 전역 설정.

    환경 변수 또는 .env 파일에서 값을 로드한다.
    PREFIX 없이 변수명 그대로 매핑 (예: GEMINI_API_KEY → gemini_api_key).

    Attributes:
        gemini_api_key: Gemini API 키.
        embedding_model_name: 기본 임베딩 모델 이름.
        embedding_dim: 임베딩 차원.
        query_prefix: 검색 쿼리 앞에 붙이는 문구.
        passage_prefix: 검색 대상 문서 앞에 붙이는 문구.
        gemini_model_name: 기본 Gemini 모델 이름.
        embedding_backend: 임베딩 추론 백엔드 (onnx | torch).
        log_level: 로그 레벨.
        ragkit_project_root: 데이터·모델 경로 기준 폴더 (RAGKIT_PROJECT_ROOT). 없으면 cwd.
    """

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # API Keys
    gemini_api_key: str = ""

    # Model
    embedding_model_name: str = "intfloat/multilingual-e5-small"
    embedding_dim: int = 384
    # e5 계열은 쿼리/문서 앞에 역할 문구를 붙여야 한다. 문구가 없는 모델은 빈 문자열로 설정.
    query_prefix: str = "query: "
    passage_prefix: str = "passage: "
    # 무료 등급에서 쓸 수 있는 가벼운 모델 (2026-09 기준 동작 확인)
    gemini_model_name: str = "gemini-2.5-flash-lite"

    # 임베딩 추론 백엔드: onnx(배포 기본, torch 불필요) | torch(extra [torch])
    embedding_backend: Literal["onnx", "torch"] = "onnx"
    # 임베딩 속도 옵션 (tutorials/03_optimize에서 비교)
    embedding_device: str = "auto"  # torch: auto(cuda → mps → cpu) | cuda | mps | cpu
    embedding_sort_by_length: bool = True  # 길이순 배치로 패딩 줄이기
    embedding_num_threads: int | None = None  # CPU 스레드 수. None이면 라이브러리 기본값

    # MLflow (비어 있으면 기록하지 않는다). 예: http://127.0.0.1:5000
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "ragkit"  # 실험 추적(train·evaluate·compare·bench)
    mlflow_trace_experiment: str = "ragkit-service"  # 서비스 트레이스(api·search-cli·mcp)

    # Logging
    log_level: str = "INFO"

    # 데이터·모델 경로의 기준 폴더. 설치된 패키지 위치(__file__)는 저장소와 무관하므로
    # 환경 변수 RAGKIT_PROJECT_ROOT가 없으면 명령을 실행한 폴더(cwd)를 쓴다.
    ragkit_project_root: Path | None = None

    @property
    def project_root(self) -> Path:
        """프로젝트 루트 경로."""
        return self.ragkit_project_root or Path.cwd()

    @property
    def data_dir(self) -> Path:
        """데이터 디렉토리 경로."""
        return self.project_root / "data"

    @property
    def models_dir(self) -> Path:
        """로컬 모델 저장 디렉토리 경로."""
        return self.project_root / "models"

    @property
    def experiments_dir(self) -> Path:
        """실험 디렉토리 경로."""
        return self.project_root / "experiments"

    @property
    def results_dir(self) -> Path:
        """실험 결과 디렉토리 경로."""
        return self.experiments_dir / "results"


@lru_cache
def get_settings() -> Settings:
    """싱글톤 Settings 인스턴스를 반환한다.

    Returns:
        캐싱된 Settings 인스턴스.

    Example:
        >>> settings = get_settings()
        >>> settings.embedding_model_name
        'intfloat/multilingual-e5-small'
    """
    return Settings()

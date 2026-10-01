"""ragkit-api 실행 명령: 모델·인덱스를 로드하고 uvicorn으로 띄운다."""

from pathlib import Path

import cyclopts

app = cyclopts.App(name="ragkit-api", help="ragkit 검색 API 서버 (FastAPI)")


@app.default
def serve(
    host: str = "127.0.0.1",
    port: int = 8000,
    model: str | None = None,
    checkpoint: Path | None = None,
    backend: str | None = None,
    index: Path | None = None,
    db: Path | None = None,
    web_dist: Path | None = None,
    cors_origin: list[str] | None = None,
) -> None:
    """API 서버를 띄운다. 문서: http://<host>:<port>/docs

    Args:
        host: 바인딩 주소. 다른 기기에서 접속하려면 0.0.0.0.
        port: 포트.
        model: 임베딩 모델 이름·폴더 또는 MLflow 레지스트리 주소(models:/law-embedder@champion).
            기본값 Settings.embedding_model_name.
        checkpoint: 파인튜닝한 모델 폴더.
        backend: onnx | torch | st. 기본값 Settings.embedding_backend.
        index: 인덱스 파일. 기본값 data/processed/index/<모델 키>.sqlite.
        db: 노트북 저장 파일. 기본값 data/app/notebooks.sqlite.
        web_dist: 빌드한 web 폴더(apps/web/dist). 주면 / 에서 화면도 제공한다.
        cors_origin: web을 다른 주소에서 띄울 때 허용할 출처 (여러 번 줄 수 있다).
    """
    import uvicorn

    from ragkit_api.app import create_app, default_db_path
    from ragkit_api.store import NotebookStore

    open_kwargs = {"model": model, "checkpoint": checkpoint, "backend": backend, "index_path": index}
    api = create_app(
        store=NotebookStore(db or default_db_path()),
        open_kwargs=open_kwargs,
        web_dist=web_dist,
        cors_origins=cors_origin,
    )
    uvicorn.run(api, host=host, port=port)


def main() -> None:
    app()

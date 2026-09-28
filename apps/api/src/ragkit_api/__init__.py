"""ragkit 검색 API (FastAPI). 강의 2단계."""

from ragkit_api.app import create_app
from ragkit_api.cli import main

__all__ = ["create_app", "main"]

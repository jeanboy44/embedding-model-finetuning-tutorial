"""최종 사용자용 법령 검색 CLI (ragkit-search)."""

from ragkit_search.cli import app


def main() -> None:
    """CLI 진입점."""
    app()


__all__ = ["app", "main"]

"""CLI 공통 콘솔: stdout·stderr용 rich Console과 오류 한 줄 헬퍼.

이모지 변환은 끄고(데이터의 `:...:`가 바뀌지 않게), soft_wrap으로 rich가 80열에서 줄을 끊지 않게 한다
(테스트가 stdout 문자열을 비교한다).
"""

from rich.console import Console

console = Console(emoji=False, soft_wrap=True)
err_console = Console(stderr=True, emoji=False, soft_wrap=True)


def fail(message: str) -> int:
    """stderr에 `오류: ...`를 그대로 찍고 종료 코드 1을 돌려준다."""
    err_console.print(f"오류: {message}", markup=False, highlight=False)
    return 1

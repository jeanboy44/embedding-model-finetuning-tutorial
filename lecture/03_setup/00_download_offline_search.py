# /// script
# requires-python = ">=3.9"
# dependencies = []
# ///
"""
실습 0-0 (3교시): 완성품 받기 — HTML 파일 하나짜리 법령 검색
==========================================================

가장 먼저 실행한다. 설치(uv sync) 전이라도 되고, 표준 라이브러리만 쓴다.
강사 Drive에서 law-search-offline.html(약 56MB)을 받아 sha256을 확인하고 브라우저로 연다.
그 뒤로는 파일을 더블클릭하면 인터넷 없이 검색된다.

실행 (저장소 루트에서, 둘 중 아무거나):
    uv run lecture/03_setup/00_download_offline_search.py
    python lecture/03_setup/00_download_offline_search.py
"""

import hashlib
import sys
import urllib.request
import webbrowser
from pathlib import Path

FILE_ID = "__FILE_ID__"
SHA256 = "7eba32a0264af9c691466860f1348b9fcd29bed91a7ce26fd22842d0b0a9109f"
URL = f"https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t"
OUT = Path("dist/law-search-offline.html")


def sha256sum(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download(url: str, out: Path) -> None:
    out.parent.mkdir(parents=True, exist_ok=True)
    part = out.with_suffix(".part")
    with urllib.request.urlopen(url) as resp, part.open("wb") as f:
        total = int(resp.headers.get("Content-Length") or 0)
        done = 0
        while chunk := resp.read(1 << 20):
            f.write(chunk)
            done += len(chunk)
            size = f"{done >> 20}/{total >> 20}MB" if total else f"{done >> 20}MB"
            print(f"\r받는 중: {size}", end="", flush=True)
    print()
    part.replace(out)


def main() -> None:
    if OUT.exists() and sha256sum(OUT) == SHA256:
        print(f"이미 받았습니다: {OUT}")
    else:
        download(URL, OUT)
        if sha256sum(OUT) != SHA256:
            OUT.unlink()
            sys.exit("sha256이 다릅니다. 다시 실행하거나 강사에게 알려 주세요.")
        print(f"완료: {OUT}")
    print("브라우저로 엽니다. 다음부터는 이 파일을 더블클릭하면 됩니다.")
    webbrowser.open(OUT.resolve().as_uri())


if __name__ == "__main__":
    main()

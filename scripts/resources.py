"""Drive에서 손으로 받은 zip을 resources/에 넣어 두면, 다시 받지 않고 그대로 쓴다.

Drive 공유 폴더에서 받은 파일을 이름 그대로(압축을 풀지 말고) 저장소의 resources/ 에 넣는다.
    resources/finetuned-002_hard_negatives.zip
    resources/index-multilingual-e5-small.zip
    resources/files-optimize.zip
    resources/files-results_001_baseline.zip
    resources/law-data-v1.zip

finetuned_drive.py download 와 data_version.py pull 은 zip을 resources/ 에서 먼저 찾는다.
있고 sha256이 맞으면 그 zip을 풀고, 없거나 맞지 않으면 Drive에서 받는다.
doctor --fix 는 이 두 명령을 돌리므로, resources/ 에 넣어 두고 doctor --fix 한 번이면 된다.

사용법:
    uv run python scripts/resources.py      # resources/ 의 zip 중 알아본 것과 빠진 것을 보여 준다
"""

import json
import shutil
import sys
from pathlib import Path

from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.download_model_gdrive import drive_download, sha256sum

ROOT = Path(__file__).resolve().parents[1]
RESOURCES = ROOT / "resources"
console = Console(emoji=False, soft_wrap=True)


def fetch_zip(name: str, file_id: str, sha256: str, output: Path) -> bool:
    """zip 하나를 output에 둔다. resources/<name> 이 있고 sha256이 맞으면 복사, 아니면 Drive에서 받는다.

    Args:
        name: Drive의 파일 이름 (예: finetuned-002_hard_negatives.zip).
        file_id: Drive 파일 ID.
        sha256: 기대하는 zip의 sha256.
        output: zip을 둘 경로.

    Returns:
        받았으면 True. sha256 확인은 부른 쪽이 한 번 더 한다.
    """
    local = RESOURCES / name
    if local.exists():
        if sha256sum(local) == sha256:
            console.print(f"[dim]resources/ 에 있는 파일 사용: {escape(name)}[/]")
            shutil.copyfile(local, output)
            return True
        console.print(
            f"[yellow]resources/{escape(name)} 의 sha256이 맞지 않습니다 "
            "(받다 끊겼거나 옛 버전). Drive에서 다시 받습니다.[/]"
        )
    console.print(f"[dim]다운로드 중: {escape(name)}[/]")
    return drive_download(file_id, output)


def known_zips() -> dict[str, str]:
    """resources/ 에 넣을 수 있는 zip 이름 → 쓰는 명령."""
    zips = {}
    manifest = json.loads(
        (ROOT / "scripts" / "finetuned_models.json").read_text(encoding="utf-8")
    )
    for group in ("models", "indexes", "files"):
        for name, entry in manifest.get(group, {}).items():
            if not entry.get("pending_reupload"):
                zips[entry["file"]] = f"finetuned_drive.py download {name}"
    for path in sorted((ROOT / "data" / "versions").glob("*.json")):
        zip_name = json.loads(path.read_text(encoding="utf-8")).get("zip", {}).get("name")
        if zip_name:
            zips[zip_name] = f"data_version.py pull {path.stem}"
    return zips


def main() -> None:
    zips = known_zips()
    found = sorted(p.name for p in RESOURCES.glob("*")) if RESOURCES.exists() else []
    table = Table(box=box.SIMPLE_HEAD, show_edge=False)
    table.add_column("", width=1)
    table.add_column("zip")
    table.add_column("쓰는 명령", style="dim")
    for name, cmd in zips.items():
        mark = "[green]✓[/]" if name in found else "[dim]·[/]"
        table.add_row(mark, escape(name), escape(cmd))
    console.print(f"[bold]{escape(str(RESOURCES))}[/]")
    console.print(table)
    if unknown := [n for n in found if n not in zips]:
        console.print("[yellow]알아보지 못한 파일 (이름이 Drive와 같은지 확인):[/]")
        for name in unknown:
            console.print(f"  {escape(name)}")
    console.print(
        "\n[dim]다음: uv run python lecture/03_setup/01_doctor.py --fix "
        "(✓ 표시 zip은 받지 않고 그대로 푼다)[/]"
    )


if __name__ == "__main__":
    main()

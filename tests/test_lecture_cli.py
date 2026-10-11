"""강의 스크립트의 CLI 표면: --help가 본문 없이 끝나고, 문서의 플래그가 그대로 있는지."""

import os
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]

# 스크립트 → 강의 문서에 적힌 플래그
SCRIPTS: dict[str, list[str]] = {
    "lecture/03_setup/01_doctor.py": ["--quick", "--fix", "--all"],
    "lecture/04_ds_work/01_questions.py": ["--run", "--file"],
    "lecture/05_data_eval/01_corpus.py": ["--run"],
    "lecture/05_data_eval/02_split.py": ["--run"],
    "lecture/05_data_eval/03_metrics.py": ["--run"],
    "lecture/05_data_eval/04_baseline_train.py": ["--run"],
    "lecture/06_experiment/01_compare_runs.py": ["--run"],
    "lecture/06_experiment/02_error_analysis.py": ["--run"],
    "lecture/07_optimize/02_onnx_quantize_prune.py": ["--run"],
    "lecture/07_optimize/03_bench_and_serve.py": ["--run"],
    "lecture/08_product/01_search_cli.py": ["--run"],
    "lecture/08_product/04_web_app.py": ["--run", "--check"],
    "lecture/08_product/extra_monitoring.py": ["--check"],
}


def _run(script: str, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, script, *args],
        cwd=ROOT,
        capture_output=True,
        text=True,
        timeout=120,
        check=False,
        env={**os.environ, "COLUMNS": "200"},  # 도움말 표가 줄바꿈되어 플래그가 잘리지 않게
    )


@pytest.mark.parametrize(("script", "flags"), SCRIPTS.items())
def test_help_lists_flags_and_exits_zero(script: str, flags: list[str]) -> None:
    result = _run(script, "--help")
    assert result.returncode == 0, result.stderr
    for flag in flags:
        assert flag in result.stdout


@pytest.mark.parametrize("script", SCRIPTS)
def test_unknown_positional_is_error(script: str) -> None:
    assert _run(script, "extra").returncode != 0

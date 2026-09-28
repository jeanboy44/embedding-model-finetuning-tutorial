"""법령 질문 묶음·Drive 스크립트(scripts/law_questions_drive.py) 테스트. Drive 접속 없이 돌아가는 부분만 검사한다."""

import importlib.util
import json
import zipfile
from pathlib import Path

SCRIPTS_DIR = Path(__file__).parent.parent / "scripts"
spec = importlib.util.spec_from_file_location("law_questions_drive", SCRIPTS_DIR / "law_questions_drive.py")
drive = importlib.util.module_from_spec(spec)
spec.loader.exec_module(drive)


def test_folder_id_from_share_link() -> None:
    url = "https://drive.google.com/drive/folders/1y30fzc21thrATVzwBa9vzBWbOSTJt-U1?usp=sharing"

    assert drive.folder_id(url) == "1y30fzc21thrATVzwBa9vzBWbOSTJt-U1"


def _write_run(workspace: Path, iteration: int, config: str, name: str, rows: list[dict]) -> None:
    out = workspace / f"iteration-{iteration}" / "eval-1-minimum-wage-small" / config / "run-1" / "outputs"
    out.mkdir(parents=True, exist_ok=True)
    (out / name).write_text("\n".join(json.dumps(r, ensure_ascii=False) for r in rows), encoding="utf-8")


def test_collect_records_normalizes_and_skips_derived_files(tmp_path) -> None:
    """공통 형식으로 맞추고, hard_negative_ids가 없는 파생 파일(triplets)은 뺀다."""
    _write_run(tmp_path, 1, "with_skill", "a.jsonl", [
        {"query": "q1", "positive_id": "p1", "hard_negative_ids": ["n1"], "query_type": "keyword", "answer": "a"}
    ])
    _write_run(tmp_path, 2, "without_skill", "queries.jsonl", [
        {"query": "q2", "positive_id": "p2", "hard_negative_ids": ["n2"], "qid": 7}
    ])
    _write_run(tmp_path, 2, "without_skill", "triplets.jsonl", [
        {"query": "q2", "positive_id": "p2", "negative_id": "n2"}
    ])

    records = drive.collect_records(tmp_path)

    assert records == [
        {"query": "q1", "positive_id": "p1", "hard_negative_ids": ["n1"], "query_type": "keyword",
         "answer": "a", "related_ids": [], "corpus": "law_docs_article_level.json",
         "source": {"iteration": 1, "eval": "minimum-wage-small", "config": "with_skill"}},
        {"query": "q2", "positive_id": "p2", "hard_negative_ids": ["n2"], "query_type": None,
         "answer": None, "related_ids": [], "corpus": "law_docs.json",
         "source": {"iteration": 2, "eval": "minimum-wage-small", "config": "without_skill"}},
    ]


def test_bundle_layout(tmp_path) -> None:
    questions = tmp_path / "questions_test"
    (questions / "raw").mkdir(parents=True)
    (questions / "questions.jsonl").write_text("{}\n", encoding="utf-8")
    (questions / "raw" / "x.jsonl").write_text("{}\n", encoding="utf-8")
    corpus = tmp_path / "processed"
    corpus.mkdir()
    for name in ("law_docs.json", "law_docs_article_level.json"):
        (corpus / name).write_text("[]", encoding="utf-8")
    generated = tmp_path / "questions"
    generated.mkdir()
    (generated / "최저임금법__p01.jsonl").write_text("{}\n", encoding="utf-8")
    output = tmp_path / "dist" / "law-questions.zip"

    readme = tmp_path / "README.md"
    readme.write_text("# readme", encoding="utf-8")

    drive.bundle(questions_dir=questions, generated_dir=generated, corpus_dir=corpus, readme=readme, output=output)

    with zipfile.ZipFile(output) as zf:
        assert sorted(zf.namelist()) == [
            "law-questions/README.md",
            "law-questions/corpus/law_docs.json",
            "law-questions/corpus/law_docs_article_level.json",
            "law-questions/generated/최저임금법__p01.jsonl",
            "law-questions/questions/questions.jsonl",
            "law-questions/questions/raw/x.jsonl",
        ]
    assert output.with_suffix(".zip.sha256").read_text().split()[0] == drive.sha256sum(output)


def test_bundle_skill_layout(tmp_path) -> None:
    skills = tmp_path / "skills"
    (skills / "law-question-gen" / "scripts" / "__pycache__").mkdir(parents=True)
    (skills / "law-question-gen" / "SKILL.md").write_text("---\nname: x\n---", encoding="utf-8")
    (skills / "law-question-gen" / "scripts" / "__pycache__" / "a.pyc").write_bytes(b"x")
    (skills / "law-question-gen-workspace").mkdir()
    (skills / "law-question-gen-workspace" / "grade.py").write_text("", encoding="utf-8")
    (skills / "law-question-gen-workspace" / "viewer.log").write_text("", encoding="utf-8")
    output = tmp_path / "skill.zip"

    drive.bundle_skill(skills_dir=skills, output=output)

    with zipfile.ZipFile(output) as zf:
        assert sorted(zf.namelist()) == [
            "law-question-gen-skill/law-question-gen-workspace/grade.py",
            "law-question-gen-skill/law-question-gen/SKILL.md",
        ]

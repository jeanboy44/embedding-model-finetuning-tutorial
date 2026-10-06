"""
실습 5-2 (8교시): 에이전트 스킬 — CLI에 SKILL.md 하나를 붙여 Claude Code·Gemini CLI가 법령을 찾게 하기
================================================================================================

학습 목표:
- 에이전트 스킬: "언제, 어떤 명령을, 어떤 순서로" 쓰는지 적은 SKILL.md 한 파일이면 셸을 쓰는 에이전트가
  우리 CLI를 도구로 쓴다. Claude Code(.claude/skills/)와 Gemini CLI(.gemini/skills/)가 같은 형식을 읽는다
- 스킬 설치: ragkit-search skill install. 모델 옵션(--checkpoint)을 주면 에이전트가 그 모델로 검색한다
- 같은 에이전트에 학습 전 e5 / 파인튜닝 e5를 붙여 비교한다. 에이전트는 검색어를 스스로 고쳐 다시 찾으므로
  CLI 한 번 검색의 차이보다 줄어들지만, 처음 찾는 조문과 검색 횟수가 달라진다
- 실습 5-3의 MCP와 비교: 같은 검색을 에이전트에 붙이는 두 방법

두 가지 모드:
- 기본: 스킬을 임시 폴더에 설치해 보고, 에이전트가 할 첫 검색을 CLI로 재현해 두 모델을 비교한다.
  실제 에이전트 비교는 받은 결과(experiments/exp_011_agent_skill/results/)를 읽는다 (1분 안쪽)
- --run: claude가 설치돼 있으면 실제로 Claude Code를 헤드리스(claude -p)로 띄워 질문 --questions개를
  두 모델로 묻는다 (질문당 30~60초, 호출 비용이 든다). 결과는 experiments/results/lecture/08_product/에 쓴다

사전 준비:
    uv run ragkit index                         # 또는 Drive에서 받은 data/processed/index/
    models/finetuned/r001_A                     # 강사가 나눠 준 파인튜닝 모델과 그 인덱스
    (--run) Claude Code (claude 명령) 로그인

실행:
    uv run python lecture/08_product/02_agent_skill.py
    uv run python lecture/08_product/02_agent_skill.py --run                  # 질문 2개
    uv run python lecture/08_product/02_agent_skill.py --run --questions 10   # 실험 011과 같은 설정
"""

import argparse
import json
import os
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
ENV = {**os.environ, "MLFLOW_DISABLE_AGENT_HINT": "1"}
CLI = ["uv", "run", "--directory", str(ROOT), "--package", "ragkit-search", "ragkit-search"]
FINETUNED = ROOT / "models" / "finetuned" / "r001_A"
VARIANTS = {
    "학습 전 e5": [],
    "파인튜닝 e5 (r001_A)": ["--checkpoint", str(FINETUNED), "--backend", "torch"],
}
TEST = ROOT / "data" / "splits" / "test.jsonl"
RECEIVED = ROOT / "experiments" / "exp_011_agent_skill" / "results" / "comparison.md"
OUT = ROOT / "experiments" / "results" / "lecture" / "08_product"
SEED = 7  # 실험 011과 같은 질문을 고른다

parser = argparse.ArgumentParser(description="실습 5-2: 에이전트 스킬")
parser.add_argument("--run", action="store_true", help="Claude Code로 실제 비교 (비용이 든다)")
parser.add_argument("--questions", type=int, default=2, help="--run에서 물을 test 질문 수")
parser.add_argument("--out", type=Path, default=OUT, help="--run 결과 폴더")
args = parser.parse_args()


def section(title: str) -> None:
    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)


def cli(*argv: str) -> subprocess.CompletedProcess:
    return subprocess.run([*CLI, *argv], capture_output=True, text=True, env=ENV, check=False)


def article_key(doc_id: str) -> str:
    """조 단위 키 (법령_종류_제N조). 항·호 조각 꼬리를 뗀다."""
    return "_".join(doc_id.split("_")[:3])


def pick_questions(n: int) -> list[dict]:
    """test의 situation(일상 상황) 질문에서 seed로 고른다. 실험 011과 같은 순서."""
    rows = [json.loads(line) for line in TEST.open(encoding="utf-8")]
    situation = [r for r in rows if r["query_type"] == "situation"]
    return random.Random(SEED).sample(situation, n)


missing = [p for p in [TEST, FINETUNED / "config.json"] if not p.exists()]
if missing:
    print("필요한 파일이 없습니다: " + ", ".join(str(p.relative_to(ROOT)) for p in missing))
    print("준비 상태는 uv run python lecture/03_setup/01_doctor.py 로 한 번에 볼 수 있습니다.")
    sys.exit(1)

# ============================================================
# 1. SKILL.md: 에이전트가 읽는 사용 설명서
# ============================================================
section("1. SKILL.md: 이름 · 설명(언제 쓰나) · 명령 · 순서")
shown = cli("skill", "show", "--command", "ragkit-search")
if shown.returncode != 0:
    sys.exit(shown.stderr.strip() or "ragkit-search skill show 실패")
lines = shown.stdout.splitlines()
print("\n".join(lines[:24]))
print(f"… (전체 {len(lines)}줄. 에이전트는 처음엔 name·description만 보고, 법령 질문이 오면 전체를 읽는다)")

# ============================================================
# 2. 설치: 에이전트마다 스킬 폴더에 파일 하나
# ============================================================
section("2. 설치: Claude Code · Gemini CLI")
with tempfile.TemporaryDirectory(prefix="lecture-5-2-") as tmp:
    for agent in ["claude", "gemini"]:
        proc = cli("skill", "install", "--agent", agent, "--project-dir", tmp)
        print(proc.stdout.splitlines()[0].replace(tmp, "<프로젝트>") if proc.returncode == 0 else proc.stderr)
    proc = cli("skill", "install", "--agent", "claude", "--project-dir", tmp, *VARIANTS["파인튜닝 e5 (r001_A)"])
    tuned_skill = (Path(tmp) / ".claude/skills/korean-law-search/SKILL.md").read_text(encoding="utf-8")
search_line = next(line for line in tuned_skill.splitlines() if " search " in line)
print("\n모델 옵션을 주고 설치하면 명령에 박힌다 (에이전트는 이 모델로 검색한다):")
print("  " + search_line.replace(str(ROOT), "<저장소>"))
print("""
내 컴퓨터에 설치하려면 (모든 프로젝트에서 쓰기):
  uv run --package ragkit-search ragkit-search skill install --agent claude --user
  uv run --package ragkit-search ragkit-search skill install --agent gemini --user
  (파인튜닝 모델로: 끝에 --checkpoint models/finetuned/r001_A --backend torch)
그 뒤 claude 또는 gemini를 열고 "편의점 알바도 주휴수당 받을 수 있는지 조문 근거로 알려줘".""")

# ============================================================
# 3. 에이전트의 첫 검색을 CLI로 재현: 사용자의 말 그대로 검색
# ============================================================
section("3. 에이전트의 첫 검색 재현: 사용자 말 그대로 search (두 모델, 10위까지)")
sample = pick_questions(4)
print("정답 조문 순위: 학습 전 → 파인튜닝 | 질문")
for q in sample:
    ranks = []
    for flags in VARIANTS.values():
        proc = cli("search", q["query"], "-k", "10", "--json", *flags)
        hits = json.loads(proc.stdout) if proc.returncode == 0 else []
        keys = [article_key(h["id"]) for h in hits]
        gold = article_key(q["positive_id"])
        ranks.append(f"{keys.index(gold) + 1}위" if gold in keys else "10위 밖")
    print(f"  {ranks[0]} → {ranks[1]} | {q['query'][:44]}")
    print(f"    정답: {q['positive_id']}")
print("에이전트는 여기서 끝내지 않고 법률 용어로 바꿔 다시 찾는다(스킬의 '순서' 1번). 실제 결과는 4번에서 본다.")

# ============================================================
# 4. 실제 에이전트 비교 (받은 결과 또는 --run)
# ============================================================
section("4. 실제 에이전트(Claude Code) 비교")


def ask_claude(project: Path, question: str) -> dict:
    prompt = f"{question}\n\n(한국 법령 질문입니다. korean-law-search 스킬로 조문을 찾아 근거 조문 id를 붙여 답해 주세요.)"
    start = time.time()
    proc = subprocess.run(
        ["claude", "-p", prompt, "--model", "sonnet", "--output-format", "stream-json", "--verbose",
         "--allowedTools", "Bash(uv run:*)", "Skill", "--max-turns", "12"],
        cwd=project, capture_output=True, text=True, timeout=900, check=False,
    )  # fmt: skip
    commands, answer, cost = [], "", None
    for line in proc.stdout.splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        if event.get("type") == "assistant":
            for block in event["message"].get("content", []):
                if block.get("type") == "tool_use" and block.get("name") == "Bash":
                    commands.append(block["input"].get("command", ""))
        elif event.get("type") == "result":
            answer, cost = event.get("result", ""), event.get("total_cost_usd")
    # 에이전트는 명령을 변수에 담거나 여러 검색을 한 번에 묶어 부르기도 한다 → 셸 호출 수를 센다
    return {"answer": answer, "commands": commands, "calls": len(commands),
            "seconds": round(time.time() - start, 1), "cost": cost}  # fmt: skip


def mean(rows: list[dict], key: str) -> float:
    return sum(r[key] or 0 for r in rows) / len(rows)


def summarize(rows: list[dict]) -> str:
    lines = ["| 모델 | 정답 조문 인용 | 평균 CLI 호출 | 평균 시간 | 질문당 비용 |", "|---|---|---|---|---|"]
    for name in VARIANTS:
        mine = [r for r in rows if r["variant"] == name]
        if not mine:
            continue
        hit = sum(r["cited_gold"] for r in mine)
        lines.append(
            f"| {name} | {hit}/{len(mine)} | {mean(mine, 'calls'):.1f} | {mean(mine, 'seconds'):.0f}초"
            f" | ${mean(mine, 'cost'):.2f} |"
        )
    return "\n".join(lines)


if args.run:
    if shutil.which("claude") is None:
        sys.exit("claude 명령이 없습니다. Claude Code를 설치·로그인한 뒤 다시 실행하세요.")
    args.out.mkdir(parents=True, exist_ok=True)
    rows = []
    with tempfile.TemporaryDirectory(prefix="lecture-5-2-agents-") as tmp:
        projects = {}
        for i, (name, flags) in enumerate(VARIANTS.items()):
            projects[name] = Path(tmp) / f"v{i}"
            cli("skill", "install", "--agent", "claude", "--project-dir", str(projects[name]), *flags)
        for q in pick_questions(args.questions):
            for name, project in projects.items():
                r = ask_claude(project, q["query"])
                r.update(variant=name, query=q["query"], gold=q["positive_id"])
                r["cited_gold"] = article_key(q["positive_id"]) in r["answer"]
                rows.append(r)
                mark = "O" if r["cited_gold"] else "X"
                print(f"  [{name}] 정답 인용 {mark} · CLI 호출 {r['calls']}번 · {r['seconds']:.0f}초 | {q['query'][:34]}")
    (args.out / "agent_compare.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows), encoding="utf-8"
    )
    table = summarize(rows)
    (args.out / "comparison.md").write_text(
        f"# Claude Code + korean-law-search 스킬: 학습 전 vs 파인튜닝\n\n"
        f"test situation 질문 {args.questions}개 (seed {SEED}), claude -p --model sonnet. "
        f"정답 조문 인용 = 답에 정답 조(조 단위 id)를 근거로 적었는가.\n\n{table}\n",
        encoding="utf-8",
    )
    print("\n" + table)
    print(f"\n결과: {(args.out / 'comparison.md').relative_to(ROOT)}, agent_compare.jsonl (질문별 답과 실행한 명령)")
elif RECEIVED.exists():
    print(f"받은 결과 ({RECEIVED.relative_to(ROOT)}):\n")
    print(RECEIVED.read_text(encoding="utf-8").strip())
else:
    print("받은 결과가 없습니다. 직접 해 보려면 --run (Claude Code 필요).")

print("""
읽는 법:
- 3번(사용자 말 그대로 한 번 검색)에서는 파인튜닝의 차이가 크다. 에이전트는 검색어를 법률 용어로 고쳐 다시 찾으므로
  4번에서는 차이가 줄어든다. 똑똑한 에이전트가 쿼리 확장을 대신 해 주는 셈이고, 그 대가는 질문마다 드는 LLM 시간과 비용이다
- 질문 10개는 작은 표본이다. 한두 개 차이는 우연일 수 있으니 질문별 답(agent_compare.jsonl)을 읽어 어디서 갈렸는지 본다
- 다음(실습 5-3): 같은 검색을 MCP 서버로 붙이는 방법. 스킬은 셸 에이전트에 가볍게, MCP는 앱 어디에나
""")

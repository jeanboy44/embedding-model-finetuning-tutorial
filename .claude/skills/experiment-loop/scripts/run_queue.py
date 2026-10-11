"""runs/<실험>/queue.tsv의 대기 run을 하나씩 학습 → dev 평가 → dev-a/b 부분집합 → 대조군 대비 검정까지 돌린다.

queue.tsv 열 (탭 구분, 첫 줄 머리글):
  run  status  start  end  control  model_dir  train_args
  - status: 대기 | 실행 중 | 완료 | 실패 | 평가만   ("평가만"은 학습 없이 model_dir을 평가한다: 이미 있는 모델)
  - run 하나가 실패하면 대기열 전체를 멈춘다(파일럿이 실패했는데 본 학습이 도는 일을 막는다)
  - control: 대응 검정 기준 run 이름 (같은 실험 폴더 안, 비우면 검정 안 함)
  - model_dir: 학습 결과를 저장할(또는 평가할) 모델 폴더
  - train_args: ragkit train에 더 줄 인자 (예: --max-steps 30 --limit 300 --no-dev-eval)
run 폴더(runs/<실험>/<run>/)에 config.yaml이 있어야 한다(평가만 run은 없어도 된다).

사용 (백그라운드로 띄우고 끝나면 알림을 받는다):
  uv run --extra train --extra mlflow python .claude/skills/experiment-loop/scripts/run_queue.py <번호>_<이름>
"""

import csv
import json
import os
import resource
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from loop_lib import MLFLOW_URI, ROOT, SPLITS, append, make_dev_ab, now, recall, subset_eval, subset_qids  # noqa: E402

FIELDS = ["run", "status", "start", "end", "control", "model_dir", "train_args"]


def load_queue(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as f:
        return list(csv.DictReader(f, delimiter="\t"))


def save_queue(path: Path, rows: list[dict]) -> None:
    with path.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, delimiter="\t", extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)


def sh(cmd: str, log: Path, env: dict) -> int:
    with log.open("a", encoding="utf-8") as f:
        f.write(f"\n$ {cmd}\n")
        f.flush()
        return subprocess.run(cmd, shell=True, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT, env=env).returncode


def main() -> None:
    exp = sys.argv[1]
    run_root = ROOT / "runs" / exp
    queue_path = run_root / "queue.tsv"
    timeline = run_root / "timeline.md"
    summary = run_root / "summary.tsv"
    a_path, b_path = make_dev_ab()
    qa, qb = subset_qids(a_path), subset_qids(b_path)
    env = {**os.environ, "MLFLOW_TRACKING_URI": os.environ.get("MLFLOW_TRACKING_URI") or MLFLOW_URI,
           "MLFLOW_EXPERIMENT": f"loop/{exp}", "PYTHONUNBUFFERED": "1"}
    if not summary.exists():
        append(summary, "run\ttrain_rows\tno_neg_questions\tseconds\tmax_rss_gb\tbest_epoch\tdev_r5\tdev_a_r5\tdev_b_r5\tdelta_vs_control\tdelta_dev_b\tci_low\tci_high")

    while True:
        rows = load_queue(queue_path)
        todo = next((r for r in rows if r["status"] in ("대기", "평가만")), None)
        if todo is None:
            break
        name, mode = todo["run"], todo["status"]
        run_dir = run_root / name
        run_dir.mkdir(parents=True, exist_ok=True)
        log = run_dir / "train.log"
        todo["status"], todo["start"] = "실행 중", now()
        save_queue(queue_path, rows)
        append(timeline, f"- {now()} · 3 실행 · {name} 시작 ({mode}) · 로그 {log.relative_to(ROOT)}")

        model_dir = todo["model_dir"]
        commands = []
        if mode == "대기":
            commands.append(f"uv run --extra train --extra mlflow ragkit train {run_dir.relative_to(ROOT)}/config.yaml "
                            f"--splits {SPLITS.relative_to(ROOT)} --output-dir {model_dir} {todo['train_args']}".strip())
        train_args = todo.get("train_args") or ""
        no_dev_eval = "--no-dev-eval" in train_args
        if not no_dev_eval:
            corpus_arg = ""
            if "--corpus " in train_args:
                import re
                m = re.search(r"--corpus\s+([^\s]+)", train_args)
                if m:
                    corpus_arg = f"--corpus {m.group(1)} "
            commands.append(f"uv run --extra mlflow ragkit evaluate {model_dir} --split dev --splits {SPLITS.relative_to(ROOT)} "
                            f"{corpus_arg}--backend torch --out {run_dir.relative_to(ROOT)}/eval_dev.json")
        (run_dir / "command.sh").write_text("#!/bin/bash\nset -e\n" + "\n".join(commands) + "\n", encoding="utf-8")

        t0, ok = time.time(), True
        for cmd in commands:
            if sh(cmd, log, env) != 0:
                ok = False
                break
        seconds = time.time() - t0
        rss_gb = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / 1e9  # macOS: 바이트

        rows = load_queue(queue_path)
        cur = next(r for r in rows if r["run"] == name)
        cur["end"] = now()
        if not ok:
            cur["status"] = "실패"
            save_queue(queue_path, rows)
            append(timeline, f"- {now()} · 3 실행 · {name} 실패 · 로그 끝을 볼 것 · 대기열을 멈춘다(남은 run은 대기 그대로)")
            break

        meta = {}
        meta_path = Path(ROOT / model_dir / "train_meta.json")
        if meta_path.exists():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            (run_dir / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
        line = {"run": name, "train_rows": meta.get("train_examples", ""), "no_neg_questions": meta.get("no_negative_questions", ""),
                "seconds": round(meta.get("seconds", seconds)), "max_rss_gb": round(rss_gb, 1), "best_epoch": meta.get("best_epoch", "")}
        if not no_dev_eval:
            result = json.loads((run_dir / "eval_dev.json").read_text(encoding="utf-8"))
            ra, rb = subset_eval(result, qa), subset_eval(result, qb)
            (run_dir / "eval_dev_a.json").write_text(json.dumps(ra, ensure_ascii=False), encoding="utf-8")
            (run_dir / "eval_dev_b.json").write_text(json.dumps(rb, ensure_ascii=False), encoding="utf-8")
            line.update(dev_r5=round(recall(result["per_question"]), 4), dev_a_r5=round(ra["doc"]["recall@5"], 4),
                        dev_b_r5=round(rb["doc"]["recall@5"], 4))
            control = cur["control"]
            if control and (run_root / control / "eval_dev.json").exists():
                for tag, path in (("", "eval_dev.json"), ("_dev_b", "eval_dev_b.json")):
                    out = run_dir / f"paired_test{tag}.json"
                    sh(f"uv run ragkit paired-test {run_root.relative_to(ROOT)}/{control}/{path} {run_dir.relative_to(ROOT)}/{path} "
                       f"--judge doc --k 5 --out {out.relative_to(ROOT)}", log, env)
                pt = json.loads((run_dir / "paired_test.json").read_text(encoding="utf-8"))
                ptb = json.loads((run_dir / "paired_test_dev_b.json").read_text(encoding="utf-8"))
                line.update(delta_vs_control=round(pt["delta"], 4), delta_dev_b=round(ptb["delta"], 4),
                            ci_low=round(pt["ci95"][0], 4), ci_high=round(pt["ci95"][1], 4))
        cols = summary.read_text(encoding="utf-8").splitlines()[0].split("\t")
        append(summary, "\t".join(str(line.get(c, "")) for c in cols))
        cur["status"] = "완료"
        save_queue(queue_path, rows)
        append(timeline, f"- {now()} · 3 실행 · {name} 완료 · dev R@5 {line.get('dev_r5', '-')} · Δ {line.get('delta_vs_control', '-')} · {round(seconds / 60)}분")
        sh(f"uv run --extra mlflow python .claude/skills/experiment-loop/scripts/log_experiment.py {exp}", log, env)

    append(timeline, f"- {now()} · 3 실행 · 대기열 비었음")


if __name__ == "__main__":
    main()

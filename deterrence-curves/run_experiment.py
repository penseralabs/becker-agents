"""Run the deterrence-curve pilot.

Examples
  # Test the pipeline for free (no API key needed; NOT real data)
  python run_experiment.py --provider mock --runs 4

  # Claude
  python run_experiment.py --provider anthropic --model claude-haiku-4-5-20251001 --runs 30

  # OpenAI
  python run_experiment.py --provider openai --model gpt-4o-mini --runs 30

  # Open-weight model through OpenRouter (no GPU needed)
  python run_experiment.py --provider openai --base-url https://openrouter.ai/api/v1 \\
      --api-key-env OPENROUTER_API_KEY --model meta-llama/llama-3.3-70b-instruct --runs 30

Re-running the same command resumes: completed episodes are skipped.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from deterrence.agents import RUNNERS
from deterrence.environment import RepoEnvironment
from deterrence.prompts import PENALTIES, USER_MESSAGE, build_system_prompt


def load_dotenv(path: str = ".env") -> None:
    if not os.path.exists(path):
        return
    for line in open(path):
        line = line.strip()
        if line and not line.startswith("#") and "=" in line:
            key, val = line.split("=", 1)
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


def parse_levels(s: str) -> list[int | None]:
    return [None if x.strip().lower() == "none" else int(x) for x in s.split(",")]


def episode_id(audit_pct: int | None, penalty: str, run: int) -> str:
    level = "none" if audit_pct is None else f"{audit_pct:03d}"
    return f"audit{level}_{penalty}_run{run:03d}"


def run_one(args, audit_pct, penalty, run) -> dict:
    env = RepoEnvironment()
    system = build_system_prompt(audit_pct, penalty)
    record = {
        "episode_id": episode_id(audit_pct, penalty, run),
        "provider": args.provider, "model": args.model,
        "audit_pct": audit_pct, "penalty": penalty, "run": run,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }
    try:
        result = RUNNERS[args.provider](
            env, system, USER_MESSAGE, model=args.model, max_turns=args.max_turns,
            temperature=args.temperature, base_url=args.base_url,
            api_key_env=args.api_key_env, run_index=run)
        record.update(env.summary())
        record.update(turns=result["turns"], **result["usage"], error=None)
        transcript = {"system": system, "user": USER_MESSAGE, "messages": result["transcript"]}
    except Exception as e:
        record.update(error=f"{type(e).__name__}: {e}")
        transcript = {"system": system, "error": record["error"]}
    tdir = Path(args.out) / "transcripts"
    tdir.mkdir(parents=True, exist_ok=True)
    (tdir / f"{record['episode_id']}.json").write_text(json.dumps(transcript, indent=2, default=str))
    return record


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--provider", choices=list(RUNNERS), required=True)
    p.add_argument("--model", default="mock")
    p.add_argument("--base-url", default=None, help="OpenAI-compatible endpoint, e.g. OpenRouter")
    p.add_argument("--api-key-env", default="OPENAI_API_KEY", help="env var holding the key (openai provider)")
    p.add_argument("--levels", default="0,5,25,100", help="audit percentages; 'none' = no oversight text")
    p.add_argument("--penalties", default="shutdown", help=f"comma-separated: {','.join(PENALTIES)}")
    p.add_argument("--runs", type=int, default=30, help="episodes per condition")
    p.add_argument("--max-turns", type=int, default=20)
    p.add_argument("--temperature", type=float, default=None, help="default: provider default")
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--out", default=None, help="output dir (default results/<model>)")
    args = p.parse_args()

    load_dotenv()
    safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", args.model)
    args.out = args.out or str(Path("results") / ("MOCK_pipeline_test" if args.provider == "mock" else safe_model))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    log_path = out / "episodes.jsonl"

    done = set()
    if log_path.exists():
        for line in log_path.read_text().splitlines():
            rec = json.loads(line)
            if not rec.get("error"):
                done.add(rec["episode_id"])

    penalties = [x.strip() for x in args.penalties.split(",")]
    jobs = [(a, pen, r) for pen in penalties for a in parse_levels(args.levels) for r in range(args.runs)
            if episode_id(a, pen, r) not in done]
    print(f"{len(done)} episodes already done, {len(jobs)} to run -> {log_path}")

    lock = threading.Lock()
    start = time.time()
    n_viol = n_done = 0
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(run_one, args, *job) for job in jobs]
        for fut in as_completed(futures):
            rec = fut.result()
            with lock, open(log_path, "a") as f:
                f.write(json.dumps(rec, default=str) + "\n")
            n_done += 1
            n_viol += bool(rec.get("violation"))
            status = "ERROR " + rec["error"][:80] if rec.get("error") else ("VIOLATION" if rec["violation"] else "clean")
            print(f"[{n_done}/{len(jobs)}] {rec['episode_id']}: {status}")
    print(f"Done in {time.time() - start:.0f}s. Violations this session: {n_viol}/{n_done}.")
    print(f"Next: python analyze.py {args.out}")


if __name__ == "__main__":
    main()

"""Run training jobs with a cap on concurrent processes; restart jobs that die.

    uv run python scripts/run_queue.py --max-procs 3 --jobs CONFIG:SEED:NAME [CONFIG:SEED:NAME ...]

Each job is ``python -m rocket_landing.train --config CONFIG --seed SEED --run-name NAME`` with
stdout/stderr in ``logs/NAME.log``. A job that exits non-zero, or exits without writing
``experiments/results/NAME/final.pt``, is restarted from scratch (up to ``--retries`` times;
training is deterministic given the seed, so a restart reproduces the same run).
Jobs whose ``final.pt`` already exists are skipped.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from pathlib import Path


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--max-procs", type=int, default=3)
    ap.add_argument("--retries", type=int, default=2)
    ap.add_argument("--out-dir", default="experiments/results")
    ap.add_argument("--jobs", nargs="+", required=True, help="CONFIG:SEED:NAME")
    args = ap.parse_args()

    queue = []
    for item in args.jobs:
        config, seed, name = item.split(":")
        if (Path(args.out_dir) / name / "final.pt").exists():
            print(f"skip {name}: final.pt exists", flush=True)
            continue
        queue.append({"config": config, "seed": seed, "name": name, "tries": 0})
    running: list[tuple[dict, subprocess.Popen]] = []
    Path("logs").mkdir(exist_ok=True)
    failed = []
    while queue or running:
        while queue and len(running) < args.max_procs:
            job = queue.pop(0)
            job["tries"] += 1
            log = open(Path("logs") / f"{job['name']}.log", "w")
            cmd = [sys.executable, "-m", "rocket_landing.train", "--config", job["config"],
                   "--seed", job["seed"], "--run-name", job["name"], "--out-dir", args.out_dir]
            print(f"start {job['name']} (try {job['tries']})", flush=True)
            running.append((job, subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)))
        time.sleep(5)
        still = []
        for job, proc in running:
            rc = proc.poll()
            if rc is None:
                still.append((job, proc))
                continue
            ok = rc == 0 and (Path(args.out_dir) / job["name"] / "final.pt").exists()
            print(f"{'done' if ok else 'FAILED rc=' + str(rc)} {job['name']}", flush=True)
            if not ok:
                if job["tries"] <= args.retries:
                    queue.insert(0, job)
                else:
                    failed.append(job["name"])
        running = still
    print("all finished; failed:", failed, flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

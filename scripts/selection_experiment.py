"""Evaluate checkpoint-selection rules over many runs without test-set leakage.

Phases (run in this order; ``test`` refuses to run for a run whose ``selected.json`` does not exist):

    validate : every checkpoint of every run on VALIDATION seeds 70000..70199, apply the rules,
               write <out>/<run>/{validation_curve,selected}.json (+ selected.pt, not committed).
    rules    : validation-only comparison of the rules (split-half check). Used to fix the default rule.
    test     : every checkpoint, best.pt and final.pt on TEST seeds 50000..50499 (500 episodes)
               -> <out>/<run>/test_curve.json. The selected checkpoints are looked up, not re-chosen.
    report   : tables (markdown) + <out>/summary.json.

    uv run python scripts/selection_experiment.py validate --results-dir ~/.../results \
        --runs hl_v1_s0 ... --out experiments/selection
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

from rocket_landing import selection as sel
from rocket_landing.evaluate import resolve_run_dir, wilson_interval


def _runs(args) -> list[str]:
    if args.runs:
        return args.runs
    rd = Path(args.results_dir).expanduser()
    names = sorted(p.name for p in rd.glob(args.pattern) if (p / "config.json").exists())
    if not names:
        raise SystemExit(f"no runs match {args.pattern!r} under {rd}")
    return names


def _natural(name: str) -> tuple:
    import re

    return tuple(int(t) if t.isdigit() else t for t in re.split(r"(\d+)", name))


# ------------------------------------------------------------------ validate
def cmd_validate(args) -> None:
    out = Path(args.out)
    for run in _runs(args):
        sel.select_run(run, out / run, args.rule, procs=args.procs,
                       results_dir=Path(args.results_dir).expanduser())
        s = json.loads((out / run / "selected.json").read_text())
        alt = {r: a["name"] for r, a in s["alternatives"].items()}
        print(f"{run}: final val {s['final']['successes']}/{s['final']['n']}  "
              f"max val {max(c['successes'] for c in s['validation']['curve'])}/{s['final']['n']}  "
              f"chosen({s['rule']})={s['chosen']['name']} {s['chosen']['successes']}  alt={alt}", flush=True)


# --------------------------------------------------------------------- rules
def cmd_rules(args) -> None:
    out = Path(args.out)
    rows = {r: [] for r in sel.RULES}
    rows["final"] = []
    for run in _runs(args):
        curve = json.loads((out / run / "validation_curve.json").read_text())
        for r in sel.RULES:
            rows[r].append(sel.split_half_score(curve, r))
        # "final": no selection, score is the final checkpoint on each half
        fin = next(c for c in curve["checkpoints"] if c["name"] == "final")["outcomes"]
        h = len(fin) // 2
        rows["final"].append({"selected_half": float(np.mean([fin[:h].count("L") / h, fin[h:].count("L") / (len(fin) - h)])),
                              "other_half": float(np.mean([fin[h:].count("L") / (len(fin) - h), fin[:h].count("L") / h]))})
    print(f"split-half check on VALIDATION episodes only ({len(rows['final'])} runs)")
    print("| rule | success on selecting half | success on the other half (independent) | optimism |")
    print("|---|---|---|---|")
    res = {}
    for r, rs in rows.items():
        a = float(np.mean([x["selected_half"] for x in rs]))
        b = float(np.mean([x["other_half"] for x in rs]))
        res[r] = {"selecting_half": a, "other_half": b, "n_runs": len(rs)}
        print(f"| {r} | {a:.1%} | {b:.1%} | {a - b:+.1%} |")
    Path(args.rules_out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.rules_out).write_text(json.dumps(res, indent=1) + "\n")


# ---------------------------------------------------------------------- test
def cmd_test(args) -> None:
    out = Path(args.out)
    rd = Path(args.results_dir).expanduser()
    for run in _runs(args):
        sel_path = out / run / "selected.json"
        if not sel_path.exists():
            raise SystemExit(f"{run}: no selected.json; run `validate` first (selection must precede test)")
        cache = out / run / "test_curve.json"
        if cache.exists():
            print(f"{run}: test_curve.json exists, skipping", flush=True)
            continue
        run_dir = resolve_run_dir(run, rd)
        rc = sel.run_config(run_dir)
        seed_check = sel.assert_validation_disjoint(
            sel.VALIDATION_SEED_START, sel.VALIDATION_EPISODES, rc["train_seed"], rc["num_envs"],
            rc["eval_seed_start"], rc["eval_episodes"])
        ckpts = sel.list_checkpoints(run_dir)
        entries = [(u, n, p) for u, n, p in ckpts]
        best = run_dir / "best.pt"
        if best.exists():
            entries.append((sel.checkpoint_update(best), "best", best))
        outs = sel.evaluate_many([e[2] for e in entries], sel.TEST_SEED_START, sel.TEST_EPISODES,
                                 rc["reward_version"], args.procs)
        doc = {
            "run": run, "test_seed_start": sel.TEST_SEED_START, "episodes": sel.TEST_EPISODES,
            "seed_check": seed_check,
            "selected_json_sha256": __import__("hashlib").sha256(sel_path.read_bytes()).hexdigest(),
            "checkpoints": [{"name": n, "update": u, "outcomes": o["outcomes"],
                             **sel.counts(o["outcomes"], o["fuel_used"])}
                            for (u, n, _), o in zip(entries, outs)],
        }
        cache.write_text(json.dumps(doc) + "\n")
        fin = next(c for c in doc["checkpoints"] if c["name"] == "final")
        print(f"{run}: final test {fin['success']}/500; max over ckpts "
              f"{max(c['success'] for c in doc['checkpoints'] if c['name'] != 'best')}/500", flush=True)


# -------------------------------------------------------------------- report
def _ci(k: int, n: int) -> str:
    lo, hi = wilson_interval(k, n)
    return f"{k / n:.1%} ({lo:.1%}-{hi:.1%})"


def _agg(rates: list[float]) -> dict:
    r = np.asarray(rates)
    return {"mean": float(r.mean()), "median": float(np.median(r)), "min": float(r.min()),
            "max": float(r.max()), "n_ge_90": int((r >= 0.9).sum()), "n_ge_50": int((r >= 0.5).sum()),
            "n_zero": int((r == 0).sum()), "n": len(r)}


def cmd_report(args) -> None:
    out = Path(args.out)
    methods = [*sel.RULES, "final", "best_pt", "oracle"]
    per_run: dict[str, dict] = {}
    for run in _runs(args):
        s = json.loads((out / run / "selected.json").read_text())
        t = json.loads((out / run / "test_curve.json").read_text())
        tc = {c["name"]: c for c in t["checkpoints"]}
        entry = {"validation_final": s["final"]["rate"],
                 "validation_max": max(c["rate"] for c in s["validation"]["curve"]), "methods": {}}
        for r in sel.RULES:
            a = s["alternatives"][r]
            entry["methods"][r] = {"ckpt": a["name"], "update": a["update"], "val": a["rate"],
                                   "test": tc[a["name"]]["success"], "n": tc[a["name"]]["n"]}
        entry["methods"]["final"] = {"ckpt": "final", "update": s["final"]["update"], "val": s["final"]["rate"],
                                     "test": tc["final"]["success"], "n": tc["final"]["n"]}
        if "best" in tc:
            entry["methods"]["best_pt"] = {"ckpt": "best.pt", "update": tc["best"]["update"], "val": None,
                                           "test": tc["best"]["success"], "n": tc["best"]["n"]}
        cands = [c for c in t["checkpoints"] if c["name"] != "best"]
        o = max(cands, key=lambda c: (c["success"], c["update"]))
        entry["methods"]["oracle"] = {"ckpt": o["name"], "update": o["update"], "val": None,
                                      "test": o["success"], "n": o["n"]}
        entry["n_ckpts_ge_90_test"] = sum(c["success"] / c["n"] >= 0.9 for c in cands)
        entry["n_ckpts"] = len(cands)
        per_run[run] = entry

    groups: dict[str, list[str]] = {}
    for run in per_run:
        arm = run.rsplit("_s", 1)[0]
        groups.setdefault(arm, []).append(run)
    summary = {"per_run": per_run, "arms": {}}
    lines = []
    for arm, runs in groups.items():
        runs = sorted(runs, key=_natural)
        ms = [m for m in methods if all(m in per_run[r]["methods"] for r in runs)]
        lines += [f"### {arm}: held-out TEST success (500 episodes, seeds 50000-50499), 95% Wilson CI", "",
                  "| run | " + " | ".join(ms) + " |", "|---|" + "---|" * len(ms)]
        for r in runs:
            cells = []
            for m in ms:
                e = per_run[r]["methods"][m]
                cells.append(f"{_ci(e['test'], e['n'])} @{e['update']}")
            lines.append(f"| {r} | " + " | ".join(cells) + " |")
        summary["arms"][arm] = {m: _agg([per_run[r]["methods"][m]["test"] / per_run[r]["methods"][m]["n"]
                                         for r in runs]) for m in ms}
        lines += ["", f"Aggregate over {len(runs)} runs (unit = training seed)", "",
                  "| method | mean | median | min | max | runs >= 90% | runs >= 50% | runs at 0% |",
                  "|---|---|---|---|---|---|---|---|"]
        for m in ms:
            a = summary["arms"][arm][m]
            lines.append(f"| {m} | {a['mean']:.1%} | {a['median']:.1%} | {a['min']:.1%} | {a['max']:.1%} | "
                         f"{a['n_ge_90']}/{a['n']} | {a['n_ge_50']}/{a['n']} | {a['n_zero']}/{a['n']} |")
        lines.append("")
    Path(args.summary_out).write_text(json.dumps(summary, indent=1) + "\n")
    print("\n".join(lines))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("phase", choices=["validate", "rules", "test", "report"])
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--runs", nargs="+")
    ap.add_argument("--pattern", default="hl_v*_s*")
    ap.add_argument("--out", default="experiments/selection")
    ap.add_argument("--rule", default=sel.DEFAULT_RULE, choices=sorted(sel.RULES))
    ap.add_argument("--procs", type=int, default=3)
    ap.add_argument("--rules-out", default="experiments/selection/rules_validation.json")
    ap.add_argument("--summary-out", default="experiments/selection/summary.json")
    args = ap.parse_args(argv)
    {"validate": cmd_validate, "rules": cmd_rules, "test": cmd_test, "report": cmd_report}[args.phase](args)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Headline experiment: held-out evaluation + statistics for hl_v1_s0..9 / hl_v2_s0..9.

    uv run python scripts/headline_analysis.py eval  --out experiments/headline_eval.json
    uv run python scripts/headline_analysis.py stats --eval experiments/headline_eval.json \
        --out experiments/headline_stats.json

eval : FINAL checkpoint of every run (500 eps, held-out seeds 50000..50499, greedy) plus the
       last 5 saved checkpoints (200 eps, seeds 50000..50199), plus heuristic and random on the
       same 500 seeds. Seed disjointness from training / in-training-eval seeds is asserted.
stats: per-arm summaries and a v1-vs-v2 comparison on per-seed success (seed = unit, n=10/arm).
"""

from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from rocket_landing.baseline import HeuristicController
from rocket_landing.evaluate import (
    DEFAULT_SEED_START,
    RandomController,
    assert_heldout,
    evaluate_run,
    resolve_checkpoint,
    rollout,
    summarize,
    wilson_interval,
)

RESULTS = Path("experiments/results")
ARMS = {"v1": "reward v1", "v2": "reward v2"}
SEEDS = range(10)
LAST_N = 5
LAST_EPISODES = 200


def _run_job(args):
    import torch

    torch.set_num_threads(1)
    arm, seed = args
    name = f"hl_{arm}_s{seed}"
    run_dir = RESULTS / name
    final = evaluate_run(name, "final", 500, DEFAULT_SEED_START, False, RESULTS, write=False)
    cfg_file = json.loads((run_dir / "config.json").read_text())
    cfg = cfg_file["config"]
    assert_heldout(DEFAULT_SEED_START, LAST_EPISODES, int(cfg_file["seed"]), int(cfg["num_envs"]),
                   int(cfg["eval_seed_start"]), int(cfg["eval_episodes"]))
    from rocket_landing.policy import PolicyController, load_checkpoint

    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))[-LAST_N:]
    last = []
    for c in ckpts:
        model, norm, payload = load_checkpoint(c)
        s = summarize(rollout(lambda sd: PolicyController(model, norm, deterministic=True),
                              range(DEFAULT_SEED_START, DEFAULT_SEED_START + LAST_EPISODES),
                              None, cfg["reward_version"]))
        last.append({"ckpt": c.name, "update": payload.get("extra", {}).get("update"),
                     "success": s["success"], "crash": s["crash"],
                     "out_of_bounds": s["out_of_bounds"], "timeout": s["timeout"]})
    print(name, "final", final["policies"]["greedy"]["success"]["rate"],
          [round(x["success"]["rate"], 3) for x in last], flush=True)
    return {"run": name, "arm": arm, "seed": seed, "final": final, "last_ckpts": last}


def do_eval(out: str, procs: int) -> None:
    jobs = [(a, s) for a in ARMS for s in SEEDS]
    with Pool(procs) as p:
        runs = p.map(_run_job, jobs, chunksize=1)
    base = runs[0]["final"]["policies"]
    # heuristic/random use only (seed, env) and are identical across runs: verify, keep one copy
    for r in runs:
        assert r["final"]["policies"]["heuristic"]["success"]["count"] == base["heuristic"]["success"]["count"]
        assert r["final"]["policies"]["random"]["success"]["count"] == base["random"]["success"]["count"]
    Path(out).write_text(json.dumps({"baselines_500": {k: base[k] for k in ("heuristic", "random")},
                                     "runs": runs}, indent=1) + "\n")


# ------------------------------------------------------------------ stats
def perm_test(a, b, stat, n_boot=0):
    """Exact two-sided permutation test over all splits of the pooled per-seed values."""
    pooled = np.concatenate([a, b])
    n = len(a)
    obs = abs(stat(a) - stat(b))
    cnt = tot = 0
    for idx in itertools.combinations(range(len(pooled)), n):
        m = np.zeros(len(pooled), bool)
        m[list(idx)] = True
        tot += 1
        cnt += abs(stat(pooled[m]) - stat(pooled[~m])) >= obs - 1e-12
    return cnt / tot


def boot_ci(a, b, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = [rng.choice(a, len(a)).mean() - rng.choice(b, len(b)).mean() for _ in range(reps)]
    return [float(np.percentile(d, 2.5)), float(np.percentile(d, 97.5))]


def arm_stats(rates):
    r = np.asarray(rates)
    return {"mean": float(r.mean()), "min": float(r.min()), "max": float(r.max()),
            "median": float(np.median(r)), "std": float(r.std(ddof=1)),
            "n_ge_90": int((r >= 0.9).sum()), "n_gt_50": int((r > 0.5).sum()),
            "n_failed_lt_50": int((r < 0.5).sum()), "n_zero": int((r == 0).sum())}


def do_stats(eval_path: str, out: str) -> None:
    d = json.loads(Path(eval_path).read_text())
    res = {"arms": {}, "per_seed": {}}
    rates = {}
    for arm in ARMS:
        rs = sorted([r for r in d["runs"] if r["arm"] == arm], key=lambda r: r["seed"])
        rates[arm] = np.array([r["final"]["policies"]["greedy"]["success"]["rate"] for r in rs])
        res["arms"][arm] = arm_stats(rates[arm])
        # robustness: per-seed min/max over last 5 ckpts, and |final - mean(last5)|
        rob = []
        for r in rs:
            lr = [c["success"]["rate"] for c in r["last_ckpts"]]
            f = r["final"]["policies"]["greedy"]["success"]["rate"]
            rob.append({"seed": r["seed"], "final": f, "last5": lr, "last5_min": min(lr),
                        "last5_max": max(lr), "last5_mean": float(np.mean(lr)),
                        "final_within_last5_range": min(lr) - 1e-9 <= f <= max(lr) + 1e-9})
        res["per_seed"][arm] = rob
        res["arms"][arm]["last5_mean_of_means"] = float(np.mean([x["last5_mean"] for x in rob]))
        res["arms"][arm]["seeds_with_last5_spread_gt_0.2"] = int(
            sum(x["last5_max"] - x["last5_min"] > 0.2 for x in rob))
    a, b = rates["v1"], rates["v2"]
    res["comparison_v1_vs_v2"] = {
        "mean_diff_v2_minus_v1": float(b.mean() - a.mean()),
        "bootstrap95_ci_of_diff_v2_minus_v1": boot_ci(b, a),
        "perm_p_mean_diff": perm_test(a, b, np.mean),
        "perm_p_median_diff": perm_test(a, b, np.median),
        "note": "exact two-sided permutation tests on per-seed success rates (unit = training seed, n=10 vs 10)",
    }
    res["baselines_500"] = {k: {"success": v["success"]} for k, v in d["baselines_500"].items()}
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(json.dumps(res["arms"], indent=1))
    print(json.dumps(res["comparison_v1_vs_v2"], indent=1))


def do_plots(eval_path: str, out_dir: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    d = json.loads(Path(eval_path).read_text())
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    col = {"v1": "#0072B2", "v2": "#D55E00"}
    # 1) held-out final-checkpoint dot strip (+ last-5 checkpoints as small ticks, Wilson CI bars)
    fig, ax = plt.subplots(figsize=(8, 3.6), constrained_layout=True)
    for ai, arm in enumerate(ARMS):
        rs = sorted([r for r in d["runs"] if r["arm"] == arm], key=lambda r: r["seed"])
        for r in rs:
            g = r["final"]["policies"]["greedy"]["success"]
            x = ai + (r["seed"] - 4.5) * 0.065
            ax.plot([x, x], g["ci95"], color=col[arm], lw=1.2, alpha=0.7)
            ax.scatter([x], [g["rate"]], color=col[arm], s=34, zorder=3)
            ax.scatter([x] * len(r["last_ckpts"]), [c["success"]["rate"] for c in r["last_ckpts"]],
                       marker="_", s=40, color="#555555", alpha=0.7, zorder=2)
        m = np.mean([r["final"]["policies"]["greedy"]["success"]["rate"] for r in rs])
        ax.hlines(m, ai - 0.38, ai + 0.38, color=col[arm], lw=2)
    ax.axhline(d["baselines_500"]["heuristic"]["success"]["rate"], color="#009E73", ls="--", lw=1,
               label="heuristic baseline")
    ax.scatter([], [], marker="_", s=40, color="#555555", label="last 5 saved checkpoints (200 eps)")
    ax.set_xticks([0, 1]); ax.set_xticklabels(["reward v1", "reward v2"])
    ax.set_xlim(-0.5, 1.5); ax.set_ylim(-0.04, 1.06)
    ax.set_ylabel("held-out success (500 eps)")
    ax.set_title("Final checkpoint per seed (dot, 95% Wilson CI; line = arm mean)", fontsize=10)
    ax.legend(loc="center right", fontsize=7.5)
    fig.savefig(out / "headline_final_strip.png", dpi=110)
    plt.close(fig)
    # 2) per-seed training curves (in-training eval success), held-out final marked
    fig, axes = plt.subplots(2, 10, figsize=(16, 3.8), sharex=True, sharey=True, constrained_layout=True)
    for ai, arm in enumerate(ARMS):
        for sd in SEEDS:
            ax = axes[ai, sd]
            ev = [json.loads(l) for l in (RESULTS / f"hl_{arm}_s{sd}" / "eval.jsonl").read_text().splitlines() if l]
            ax.plot([e["global_step"] / 1e6 for e in ev], [e["success_rate"] for e in ev],
                    color=col[arm], lw=0.8)
            f = next(r for r in d["runs"] if r["arm"] == arm and r["seed"] == sd)
            ax.scatter([5.0], [f["final"]["policies"]["greedy"]["success"]["rate"]], color="k", s=14, zorder=3)
            ax.set_title(f"{arm} s{sd}", fontsize=8)
            ax.tick_params(labelsize=6)
    axes[0, 0].set_ylim(-0.05, 1.05)
    fig.supxlabel("env steps (M)", fontsize=8)
    fig.supylabel("in-training eval success (50 eps); black dot = held-out final", fontsize=8)
    fig.savefig(out / "headline_per_seed_curves.png", dpi=90)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval"); e.add_argument("--out", required=True); e.add_argument("--procs", type=int, default=4)
    s = sub.add_parser("stats"); s.add_argument("--eval", required=True); s.add_argument("--out", required=True)
    pl = sub.add_parser("plots"); pl.add_argument("--eval", required=True); pl.add_argument("--out-dir", default="docs/figures")
    a = ap.parse_args()
    if a.cmd == "eval":
        import torch
        torch.set_num_threads(1)
        do_eval(a.out, a.procs)
    elif a.cmd == "plots":
        do_plots(a.eval, a.out_dir)
    else:
        do_stats(a.eval, a.out)


if __name__ == "__main__":
    main()

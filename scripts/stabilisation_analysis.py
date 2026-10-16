"""Stabilisation experiment: held-out evaluation and paired statistics against hl_v2.

    # evaluate runs (final.pt + last-N saved checkpoints, heuristic/random on the same seeds)
    uv run python scripts/stabilisation_analysis.py eval --prefix st_b_ed50lr --seeds 0 1 2 3 4 5 6 7 8 9 \
        --episodes 500 --seed-start 50000 --out experiments/stab_eval.json
    # pilot: other seed range (never the confirmation range) and other prefix
    uv run python scripts/stabilisation_analysis.py eval --prefix stp_a_ed50 --seeds 6 7 8 \
        --episodes 300 --seed-start 60000 --out experiments/stab_pilot_a.json
    # paired comparison with hl_v2 (experiments/headline_eval.json), same seeds, same protocol
    uv run python scripts/stabilisation_analysis.py stats --eval experiments/stab_eval.json \
        --baseline experiments/headline_eval.json --out experiments/stab_stats.json

Seed discipline is enforced by ``rocket_landing.evaluate.assert_heldout`` (called inside
``evaluate_run`` and for the last-N check): the evaluation seeds never overlap training env seeds
(>= 1_000_000) or the in-training eval seeds (20000..20049).
"""

from __future__ import annotations

import argparse
import itertools
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from rocket_landing.evaluate import (
    assert_heldout,
    evaluate_run,
    rollout,
    summarize,
    wilson_interval,
)

RESULTS = Path("experiments/results")


def _run_job(args):
    import torch

    torch.set_num_threads(1)
    name, seed_start, episodes, last_n, last_episodes = args
    run_dir = RESULTS / name
    final = evaluate_run(name, "final", episodes, seed_start, True, RESULTS, write=False)
    cfg_file = json.loads((run_dir / "config.json").read_text())
    cfg = cfg_file["config"]
    assert_heldout(seed_start, last_episodes, int(cfg_file["seed"]), int(cfg["num_envs"]),
                   int(cfg["eval_seed_start"]), int(cfg["eval_episodes"]))
    from rocket_landing.policy import PolicyController, load_checkpoint

    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))[-last_n:]
    last = []
    for c in ckpts:
        model, norm, payload = load_checkpoint(c)
        s = summarize(rollout(lambda sd: PolicyController(model, norm, deterministic=True),
                              range(seed_start, seed_start + last_episodes), None,
                              cfg["reward_version"]))
        last.append({"ckpt": c.name, "update": payload.get("extra", {}).get("update"),
                     "success": s["success"], "crash": s["crash"],
                     "out_of_bounds": s["out_of_bounds"], "timeout": s["timeout"]})
    g = final["policies"]["greedy"]["success"]["rate"]
    st = final["policies"]["stochastic"]["success"]["rate"]
    print(name, f"final greedy {g:.3f} sampled {st:.3f}", [round(x["success"]["rate"], 3) for x in last],
          flush=True)
    return {"run": name, "seed": int(cfg_file["seed"]), "final": final, "last_ckpts": last}


def do_eval(prefix, seeds, episodes, seed_start, last_n, last_episodes, out, procs):
    jobs = [(f"{prefix}_s{s}", seed_start, episodes, last_n, last_episodes) for s in seeds]
    with Pool(procs) as p:
        runs = p.map(_run_job, jobs, chunksize=1)
    base = runs[0]["final"]["policies"]
    Path(out).write_text(json.dumps({
        "prefix": prefix, "seed_start": seed_start, "episodes": episodes,
        "last_n": last_n, "last_episodes": last_episodes,
        "baselines": {k: base[k] for k in ("heuristic", "random")}, "runs": runs}, indent=1) + "\n")


# ------------------------------------------------------------------ stats
def paired_signflip_p(d):
    """Exact two-sided sign-flip permutation test on paired differences (mean)."""
    d = np.asarray(d, float)
    obs = abs(d.mean())
    cnt = 0
    for signs in itertools.product((-1, 1), repeat=len(d)):
        cnt += abs((np.asarray(signs) * d).mean()) >= obs - 1e-12
    return cnt / 2 ** len(d)


def mcnemar_exact_p(b, c):
    """Two-sided exact binomial p for discordant counts b (A only) and c (B only)."""
    from math import comb

    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    p = sum(comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * p)


def boot_paired_ci(d, reps=20000, seed=0):
    rng = np.random.default_rng(seed)
    d = np.asarray(d, float)
    m = [rng.choice(d, len(d)).mean() for _ in range(reps)]
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


def arm(rates):
    r = np.asarray(rates)
    return {"mean": float(r.mean()), "median": float(np.median(r)), "min": float(r.min()),
            "max": float(r.max()), "n_ge_90": int((r >= 0.9).sum()), "n_gt_50": int((r > 0.5).sum()),
            "n_zero": int((r == 0).sum()), "n": int(len(r))}


def _rob(r):
    lr = [c["success"]["rate"] for c in r["last_ckpts"]]
    return {"last5": lr, "min": min(lr), "max": max(lr), "mean": float(np.mean(lr)),
            "spread_gt_0.2": max(lr) - min(lr) > 0.2}


def do_stats(eval_path, baseline_path, out, new_results=RESULTS, base_results=None, subsets=()):
    new = json.loads(Path(eval_path).read_text())
    old = json.loads(Path(baseline_path).read_text())
    nr = {r["seed"]: r for r in new["runs"]}
    orr = {r["seed"]: r for r in old["runs"] if r.get("arm", "v2") == "v2"}
    seeds = sorted(nr)
    assert set(seeds) <= set(orr), "baseline lacks some seeds"
    res = {"seeds": seeds, "per_seed": [], "pilot_seeds_note": "see docs/stabilisation.md"}
    for s in seeds:
        g = nr[s]["final"]["policies"]["greedy"]
        st = nr[s]["final"]["policies"]["stochastic"]
        og = orr[s]["final"]["policies"]["greedy"]
        res["per_seed"].append({
            "seed": s,
            "new": {"success": g["success"], "crash": g["crash"]["count"],
                    "oob": g["out_of_bounds"]["count"], "timeout": g["timeout"]["count"],
                    "sampled_success": st["success"], "robust": _rob(nr[s])},
            "hl_v2": {"success": og["success"], "crash": og["crash"]["count"],
                      "oob": og["out_of_bounds"]["count"], "timeout": og["timeout"]["count"],
                      "robust": _rob(orr[s])},
        })

    def summary(subset):
        a = np.array([p["new"]["success"]["rate"] for p in res["per_seed"] if p["seed"] in subset])
        b = np.array([p["hl_v2"]["success"]["rate"] for p in res["per_seed"] if p["seed"] in subset])
        d = a - b
        bn = int(((a >= 0.9) & (b < 0.9)).sum())  # new-only >=90%
        cn = int(((a < 0.9) & (b >= 0.9)).sum())  # hl_v2-only >=90%
        return {"seeds": sorted(subset), "new": arm(a), "hl_v2": arm(b),
                "mean_diff_new_minus_hl_v2": float(d.mean()),
                "paired_bootstrap95_ci_of_mean_diff": boot_paired_ci(d),
                "paired_signflip_p_mean_diff": paired_signflip_p(d),
                "ge90_new_only": bn, "ge90_hl_v2_only": cn,
                "mcnemar_exact_p_ge90": mcnemar_exact_p(bn, cn),
                "n_seeds_improved": int((d > 1e-9).sum()), "n_seeds_worse": int((d < -1e-9).sum()),
                "n_seeds_tied": int((np.abs(d) <= 1e-9).sum())}

    res["all"] = summary(set(seeds))
    res["subsets"] = {}
    for spec in subsets:  # "name=0,1,2"
        name, _, lst = spec.partition("=")
        res["subsets"][name] = summary({int(x) for x in lst.split(",")} & set(seeds))
    res["rob"] = {
        "new_final_within_last5_range": sum(
            min(p["new"]["robust"]["last5"]) - 1e-9 <= p["new"]["success"]["rate"]
            <= max(p["new"]["robust"]["last5"]) + 1e-9 for p in res["per_seed"]),
        "new_seeds_last5_spread_gt_0.2": sum(p["new"]["robust"]["spread_gt_0.2"] for p in res["per_seed"]),
        "hl_v2_seeds_last5_spread_gt_0.2": sum(p["hl_v2"]["robust"]["spread_gt_0.2"] for p in res["per_seed"]),
        "new_mean_of_last5_means": float(np.mean([p["new"]["robust"]["mean"] for p in res["per_seed"]])),
        "hl_v2_mean_of_last5_means": float(np.mean([p["hl_v2"]["robust"]["mean"] for p in res["per_seed"]])),
        "new_seeds_all_last5_ge_90": sum(min(p["new"]["robust"]["last5"]) >= 0.9 for p in res["per_seed"]),
        "hl_v2_seeds_all_last5_ge_90": sum(min(p["hl_v2"]["robust"]["last5"]) >= 0.9 for p in res["per_seed"]),
    }
    res["training_diag"] = {"new": {}, "hl_v2": {}}
    for sd in seeds:
        res["training_diag"]["new"][sd] = training_diag(new_results, nr[sd]["run"])
        if base_results:
            res["training_diag"]["hl_v2"][sd] = training_diag(base_results, orr[sd]["run"])
    res["baselines"] = {k: v["success"] for k, v in new["baselines"].items()}
    Path(out).write_text(json.dumps(res, indent=1) + "\n")
    print(markdown(res))
    print(json.dumps({k: res["all"][k] for k in res["all"] if k not in ("seeds",)}, indent=1))
    for name, sm in res["subsets"].items():
        print(name, json.dumps({k: sm[k] for k in ("seeds", "new", "hl_v2", "mean_diff_new_minus_hl_v2",
                                                  "ge90_new_only", "ge90_hl_v2_only")}))
    print(json.dumps(res["rob"], indent=1))


def do_pilot(files, out):
    """Merge per-config pilot eval files into one compact table and apply the selection rule.

    Rule (fixed before the pilot results were seen): choose the config with most pilot seeds whose
    final.pt has greedy validation success >= 90%; ties broken by the mean over seeds of
    (final success + mean of last-5 checkpoints) / 2.
    """
    cfgs: dict = {}
    for f in files:
        d = json.loads(Path(f).read_text())
        cfg = d["prefix"].removeprefix("stp_")
        for r in d["runs"]:
            g = r["final"]["policies"]["greedy"]["success"]
            st = r["final"]["policies"]["stochastic"]["success"]
            l5 = [c["success"]["rate"] for c in r["last_ckpts"]]
            cfgs.setdefault(cfg, {})[r["seed"]] = {
                "run": r["run"], "greedy": g, "sampled": st, "last5": l5,
                "val_seeds": [d["seed_start"], d["seed_start"] + d["episodes"] - 1]}
    table = {}
    for cfg, runs in cfgs.items():
        score = float(np.mean([(v["greedy"]["rate"] + np.mean(v["last5"])) / 2 for v in runs.values()]))
        table[cfg] = {"seeds": runs, "n_ge_90": sum(v["greedy"]["rate"] >= 0.9 for v in runs.values()),
                      "tiebreak_score": score}
    best = max(table, key=lambda c: (table[c]["n_ge_90"], table[c]["tiebreak_score"]))
    Path(out).write_text(json.dumps({"configs": table, "chosen": best}, indent=1) + "\n")
    for cfg, t in sorted(table.items()):
        cells = ", ".join(
            f"s{sd}: {100 * v['greedy']['rate']:.1f}% (sampled {100 * v['sampled']['rate']:.1f}%, "
            f"last5 {'/'.join(f'{100 * x:.0f}' for x in v['last5'])})" for sd, v in sorted(t["seeds"].items()))
        print(f"{cfg}: n>=90% = {t['n_ge_90']}/{len(t['seeds'])}, score {t['tiebreak_score']:.4f} | {cells}")
    print("chosen:", best)


def training_diag(results_dir, run, tail=10):
    """Training-curve diagnostics from metrics.jsonl / eval.jsonl of one run (no checkpoints needed)."""
    d = Path(results_dir) / run
    m = [json.loads(line) for line in (d / "metrics.jsonl").read_text().splitlines() if line]
    e = [json.loads(line) for line in (d / "eval.jsonl").read_text().splitlines() if line]
    last_m, last_e = m[-20:], e[-tail:]
    train_succ = [x["success_rate"] for x in last_m if x["success_rate"] is not None]
    out = {
        "final_entropy_mean_last20": float(np.mean([x["entropy"] for x in last_m])),
        "greedy_fuel_used_mean_last10_evals": float(np.mean([x["mean_fuel_used"] for x in last_e])),
        "greedy_steps_mean_last10_evals": float(np.mean([x["mean_steps"] for x in last_e])),
        "train_sampled_success_last20_updates": float(np.mean(train_succ)) if train_succ else None,
        "greedy_success_last10_evals": float(np.mean([x["success_rate"] for x in last_e])),
        "frac_evals_lt_0.8_last25": float(np.mean([x["success_rate"] < 0.8 for x in e[-25:]])),
    }
    if "stoch_success_rate" in last_e[-1]:
        out["sampled_eval_success_last10_evals"] = float(np.mean([x["stoch_success_rate"] for x in last_e]))
    return out


def markdown(res) -> str:
    def fmt(x, n_ep):
        lo, hi = x["ci95"]
        return f"{100 * x['rate']:.1f}% ({x['count']}/{n_ep}) {100 * lo:.1f}-{100 * hi:.1f}%"

    lines = ["| seed | new final (Wilson 95% CI) | crash/OOB/timeout | new last 5 ckpts % | "
             "hl_v2 final | hl_v2 last 5 % | sampled-action final |", "|---|---|---|---|---|---|---|"]
    for p in res["per_seed"]:
        n, o = p["new"], p["hl_v2"]
        ep_n = n["success"]["count"] + n["crash"] + n["oob"] + n["timeout"]
        ep_o = o["success"]["count"] + o["crash"] + o["oob"] + o["timeout"]
        l5 = " / ".join(f"{100 * x:.0f}" for x in n["robust"]["last5"])
        o5 = " / ".join(f"{100 * x:.0f}" for x in o["robust"]["last5"])
        lines.append(f"| {p['seed']} | {fmt(n['success'], ep_n)} | {n['crash']}/{n['oob']}/{n['timeout']} | {l5} | "
                     f"{fmt(o['success'], ep_o)} | {o5} | {100 * n['sampled_success']['rate']:.1f}% |")
    return "\n".join(lines)


def do_plots(eval_path, baseline_path, out_dir, rep_path=None):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    new = json.loads(Path(eval_path).read_text())
    old = json.loads(Path(baseline_path).read_text())
    groups = [("hl_v2\n(seeds 0-9)", [r for r in old["runs"] if r["arm"] == "v2"], "#D55E00"),
              ("st_c_ed30lr\n(seeds 0-9)", new["runs"], "#0072B2")]
    if rep_path:
        groups.append(("st_c_ed30lr\n(fresh seeds 10-19)", json.loads(Path(rep_path).read_text())["runs"], "#56B4E9"))
    fig, ax = plt.subplots(figsize=(3.3 * len(groups) + 1.5, 3.8), constrained_layout=True)
    for gi, (label, runs, col) in enumerate(groups):
        runs = sorted(runs, key=lambda r: r["seed"])
        for k, r in enumerate(runs):
            g = r["final"]["policies"]["greedy"]["success"]
            x = gi + (k - (len(runs) - 1) / 2) * 0.065
            ax.plot([x, x], g["ci95"], color=col, lw=1.2, alpha=0.7)
            ax.scatter([x], [g["rate"]], color=col, s=34, zorder=3)
            ax.scatter([x] * len(r["last_ckpts"]), [c["success"]["rate"] for c in r["last_ckpts"]],
                       marker="_", s=40, color="#555555", alpha=0.7, zorder=2)
        m = np.mean([r["final"]["policies"]["greedy"]["success"]["rate"] for r in runs])
        ax.hlines(m, gi - 0.38, gi + 0.38, color=col, lw=2)
    ax.axhline(new["baselines"]["heuristic"]["success"]["rate"], color="#009E73", ls="--", lw=1,
               label="heuristic baseline")
    ax.scatter([], [], marker="_", s=40, color="#555555", label="last 5 saved checkpoints (200 eps)")
    ax.set_xticks(range(len(groups)))
    ax.set_xticklabels([g[0] for g in groups], fontsize=8)
    ax.set_xlim(-0.5, len(groups) - 0.5)
    ax.set_ylim(-0.04, 1.06)
    ax.set_ylabel("held-out success (500 eps, seeds 50000-50499)")
    ax.set_title("final.pt per training seed (dot, 95% Wilson CI; line = group mean)", fontsize=10)
    ax.legend(loc="center left", bbox_to_anchor=(0.0, 0.5), fontsize=7.5)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "stabilisation_final_strip.png", dpi=110)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--prefix", required=True)
    e.add_argument("--seeds", type=int, nargs="+", default=list(range(10)))
    e.add_argument("--episodes", type=int, default=500)
    e.add_argument("--seed-start", type=int, default=50000)
    e.add_argument("--last-n", type=int, default=5)
    e.add_argument("--last-episodes", type=int, default=200)
    e.add_argument("--out", required=True)
    e.add_argument("--procs", type=int, default=2)
    s = sub.add_parser("stats")
    s.add_argument("--eval", required=True)
    s.add_argument("--baseline", default="experiments/headline_eval.json")
    s.add_argument("--out", required=True)
    s.add_argument("--subset", action="append", default=[], help="NAME=s1,s2,... extra paired summary")
    s.add_argument("--base-results", help="results dir holding the hl_v2 runs (for training diagnostics)")
    pl = sub.add_parser("plots")
    pl.add_argument("--eval", required=True)
    pl.add_argument("--baseline", default="experiments/headline_eval.json")
    pl.add_argument("--rep", help="optional replication eval json (fresh seeds)")
    pl.add_argument("--out-dir", default="docs/figures")
    pi = sub.add_parser("pilot")
    pi.add_argument("--files", nargs="+", required=True)
    pi.add_argument("--out", required=True)
    a = ap.parse_args()
    if a.cmd == "eval":
        import torch

        torch.set_num_threads(1)
        do_eval(a.prefix, a.seeds, a.episodes, a.seed_start, a.last_n, a.last_episodes, a.out, a.procs)
    elif a.cmd == "plots":
        do_plots(a.eval, a.baseline, a.out_dir, a.rep)
    elif a.cmd == "pilot":
        do_pilot(a.files, a.out)
    else:
        do_stats(a.eval, a.baseline, a.out, base_results=a.base_results, subsets=a.subset)


if __name__ == "__main__":
    main()

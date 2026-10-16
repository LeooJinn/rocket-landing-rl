"""Reward-v3 experiment: held-out evaluation and paired statistics.

    # evaluate final checkpoints (+ last 5 saved checkpoints) of runs; greedy AND sampled actions
    uv run python scripts/rv3_analysis.py eval --out experiments/rv3_pilot_eval.json \
        --runs rv3_pilot_v2_s10 rv3_pilot_v3a_s10 ... [--episodes 200] [--last-n 5] [--procs 2]
    # pilot table by variant (variant = run-name prefix before _s<seed>)
    uv run python scripts/rv3_analysis.py pilot --eval experiments/rv3_pilot_eval.json
    # confirmation: new arm vs baseline arm, paired by training seed
    uv run python scripts/rv3_analysis.py confirm --new experiments/rv3_confirm_eval.json \
        --base experiments/rv3_hlv2_eval.json --out experiments/rv3_confirm_stats.json

Held-out seeds are 50000.. (disjoint from training env seeds >= 1_000_000 and in-training
eval seeds 20000..20049; asserted via ``rocket_landing.evaluate.assert_heldout``). Outcome
metrics (success, fuel, steps, touchdown speed) do not depend on the reward version because
rewards never change the dynamics.
"""

from __future__ import annotations

import argparse
import itertools
import json
import re
from multiprocessing import Pool
from pathlib import Path

import numpy as np

from rocket_landing.evaluate import (
    DEFAULT_SEED_START,
    assert_heldout,
    rollout,
    wilson_interval,
)

RESULTS = Path("experiments/results")
HL_RESULTS = Path.home() / "projects/rocket-landing-rl-headline/experiments/results"


def find_run(name: str) -> Path:
    for base in (RESULTS, HL_RESULTS):
        if (base / name / "config.json").exists():
            return base / name
    raise FileNotFoundError(name)


def metrics(eps) -> dict:
    n = len(eps)
    out = {"n": n}
    for key in ("landed", "crashed", "out_of_bounds", "timeout"):
        k = sum(e.outcome == key for e in eps)
        lo, hi = wilson_interval(k, n)
        out[key] = {"count": k, "rate": k / n, "ci95": [lo, hi]}
    land = [e for e in eps if e.outcome == "landed"]
    td = [e for e in eps if e.landing_vy is not None]

    def mean(vals):
        vals = list(vals)
        return float(np.mean(vals)) if vals else None

    out["fuel_all"] = mean(e.fuel_used for e in eps)
    out["fuel_landed"] = mean(e.fuel_used for e in land)
    out["steps_all"] = mean(e.steps for e in eps)
    out["steps_landed"] = mean(e.steps for e in land)
    out["abs_vy_landed"] = mean(abs(e.landing_vy) for e in land)
    out["abs_vx_landed"] = mean(abs(e.landing_vx) for e in land)
    out["abs_vy_touchdown"] = mean(abs(e.landing_vy) for e in td)
    out["reward_own_version"] = mean(e.total_reward for e in eps)
    return out


def eval_job(args):
    import torch

    from rocket_landing.policy import PolicyController, load_checkpoint

    torch.set_num_threads(1)
    name, episodes, seed_start, last_n, last_eps = args
    run_dir = find_run(name)
    cf = json.loads((run_dir / "config.json").read_text())
    cfg, version = cf["config"], cf["config"]["reward_version"]
    for n_eps in (episodes, last_eps):
        assert_heldout(seed_start, n_eps, int(cf["seed"]), int(cfg["num_envs"]),
                       int(cfg["eval_seed_start"]), int(cfg["eval_episodes"]))
    model, norm, payload = load_checkpoint(run_dir / "final.pt")
    seeds = range(seed_start, seed_start + episodes)
    res = {
        "run": name, "train_seed": int(cf["seed"]), "reward_version": version,
        "episodes": episodes, "seed_start": seed_start,
        "greedy": metrics(rollout(lambda s: PolicyController(model, norm, deterministic=True),
                                  seeds, None, version)),
        "sampled": metrics(rollout(lambda s: PolicyController(model, norm, deterministic=False, seed=s),
                                   seeds, None, version)),
        "last_ckpts": [],
    }
    ckpts = sorted((run_dir / "checkpoints").glob("ckpt_*.pt"))[-last_n:] if last_n else []
    for c in ckpts:
        m, nm, pl = load_checkpoint(c)
        g = metrics(rollout(lambda s: PolicyController(m, nm, deterministic=True),
                            range(seed_start, seed_start + last_eps), None, version))
        res["last_ckpts"].append({"ckpt": c.name, "update": pl.get("extra", {}).get("update"),
                                  "success": g["landed"], "fuel_all": g["fuel_all"],
                                  "steps_all": g["steps_all"]})
    print(name, "greedy", round(res["greedy"]["landed"]["rate"], 3), "sampled",
          round(res["sampled"]["landed"]["rate"], 3),
          "last5", [round(c["success"]["rate"], 3) for c in res["last_ckpts"]], flush=True)
    return res


def do_eval(a) -> None:
    jobs = [(r, a.episodes, a.seed_start, a.last_n, a.last_episodes) for r in a.runs]
    with Pool(a.procs) as p:
        runs = p.map(eval_job, jobs, chunksize=1)
    Path(a.out).write_text(json.dumps({"runs": runs}, indent=1) + "\n")


def variant_of(run: str) -> str:
    return re.sub(r"_s\d+$", "", run)


def seed_of(run: str) -> int:
    return int(re.search(r"_s(\d+)$", run).group(1))


def do_pilot(a) -> None:
    runs = json.loads(Path(a.eval).read_text())["runs"]
    variants = sorted({variant_of(r["run"]) for r in runs})
    print("| variant | seed | greedy succ | sampled succ | fuel (all eps) | steps (all eps) | fuel (landed) | steps (landed) | touchdown abs vy | last-5 ckpt greedy |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    summ = {}
    for v in variants:
        rs = sorted([r for r in runs if variant_of(r["run"]) == v], key=lambda r: seed_of(r["run"]))
        for r in rs:
            g, s = r["greedy"], r["sampled"]
            f = lambda x, fmt="{:.3f}": "n/a" if x is None else fmt.format(x)  # noqa: E731
            print(f"| {v} | {seed_of(r['run'])} | {g['landed']['rate']:.3f} | {s['landed']['rate']:.3f} | "
                  f"{f(g['fuel_all'])} | {f(g['steps_all'], '{:.0f}')} | {f(g['fuel_landed'])} | "
                  f"{f(g['steps_landed'], '{:.0f}')} | {f(g['abs_vy_landed'], '{:.2f}')} | "
                  f"{[round(c['success']['rate'], 2) for c in r['last_ckpts']]} |")
        summ[v] = {
            "greedy_success_mean": float(np.mean([r["greedy"]["landed"]["rate"] for r in rs])),
            "sampled_success_mean": float(np.mean([r["sampled"]["landed"]["rate"] for r in rs])),
            "fuel_all_mean": float(np.mean([r["greedy"]["fuel_all"] for r in rs])),
            "steps_all_mean": float(np.mean([r["greedy"]["steps_all"] for r in rs])),
            "n_seeds": len(rs),
        }
    print()
    print(json.dumps(summ, indent=1))


# ------------------------------------------------------------ paired stats
def paired_perm_p(diff: np.ndarray) -> float:
    """Exact two-sided sign-flip test of mean(diff) = 0 (all 2^n sign patterns)."""
    d = np.asarray(diff, float)
    obs = abs(d.mean())
    cnt = tot = 0
    for signs in itertools.product((-1.0, 1.0), repeat=len(d)):
        tot += 1
        cnt += abs((np.array(signs) * d).mean()) >= obs - 1e-12
    return cnt / tot


def paired_boot_ci(diff: np.ndarray, reps: int = 20000, seed: int = 0) -> list[float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(diff, float)
    means = [rng.choice(d, len(d)).mean() for _ in range(reps)]
    return [float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))]


def _by_seed(runs):
    return {r["train_seed"]: r for r in runs}


def do_confirm(a) -> None:
    new = _by_seed(json.loads(Path(a.new).read_text())["runs"])
    base = _by_seed(json.loads(Path(a.base).read_text())["runs"])
    seeds = sorted(set(new) & set(base))
    assert seeds, "no common seeds"
    out = {"seeds": seeds, "n": len(seeds), "metrics": {}}

    def getter(policy, key):
        return lambda r: r[policy]["landed"]["rate"] if key == "success" else r[policy][key]

    specs = [
        ("greedy success", "greedy", "success"), ("sampled success", "sampled", "success"),
        ("greedy fuel used (all eps)", "greedy", "fuel_all"),
        ("greedy episode length (all eps)", "greedy", "steps_all"),
        ("greedy fuel used (landed eps)", "greedy", "fuel_landed"),
        ("greedy episode length (landed eps)", "greedy", "steps_landed"),
        ("greedy touchdown |vy| (landed eps)", "greedy", "abs_vy_landed"),
        ("sampled fuel used (all eps)", "sampled", "fuel_all"),
        ("sampled episode length (all eps)", "sampled", "steps_all"),
    ]
    for label, policy, key in specs:
        g = getter(policy, key)
        n_vals = np.array([g(new[s]) for s in seeds], float)
        b_vals = np.array([g(base[s]) for s in seeds], float)
        d = n_vals - b_vals
        out["metrics"][label] = {
            "new": n_vals.tolist(), "base": b_vals.tolist(),
            "new_mean": float(n_vals.mean()), "base_mean": float(b_vals.mean()),
            "mean_diff": float(d.mean()), "paired_boot95": paired_boot_ci(d),
            "sign_flip_p": paired_perm_p(d),
            "seeds_new_better": int((d > 0).sum()), "seeds_base_better": int((d < 0).sum()),
            "seeds_tied": int((d == 0).sum()),
        }
    for label, policy in (("greedy", "greedy"), ("sampled", "sampled")):
        for arm, runs in (("new", new), ("base", base)):
            rates = np.array([runs[s][policy]["landed"]["rate"] for s in seeds])
            out.setdefault("arm_summary", {})[f"{arm}_{label}"] = {
                "mean": float(rates.mean()), "min": float(rates.min()), "max": float(rates.max()),
                "median": float(np.median(rates)), "n_ge_90": int((rates >= 0.9).sum()),
                "n_lt_50": int((rates < 0.5).sum()), "n_zero": int((rates == 0).sum()),
            }
    for arm, runs in (("new", new), ("base", base)):
        lr = [[c["success"]["rate"] for c in runs[s]["last_ckpts"]] for s in seeds]
        out["arm_summary"][f"{arm}_last5"] = {
            "mean_of_means": float(np.mean([np.mean(x) for x in lr])) if all(lr) else None,
            "seeds_spread_gt_0.2": int(sum(max(x) - min(x) > 0.2 for x in lr if x)),
            "seeds_min_lt_0.5": int(sum(min(x) < 0.5 for x in lr if x)),
            "per_seed_min": [min(x) if x else None for x in lr],
        }
    Path(a.out).write_text(json.dumps(out, indent=1) + "\n")
    for k, v in out["metrics"].items():
        print(f"{k:40s} new {v['new_mean']:.3f} base {v['base_mean']:.3f} diff {v['mean_diff']:+.3f} "
              f"CI [{v['paired_boot95'][0]:+.3f},{v['paired_boot95'][1]:+.3f}] p={v['sign_flip_p']:.3f} "
              f"(better/worse/tied {v['seeds_new_better']}/{v['seeds_base_better']}/{v['seeds_tied']})")
    print(json.dumps(out["arm_summary"], indent=1))


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("eval")
    e.add_argument("--runs", nargs="+", required=True)
    e.add_argument("--out", required=True)
    e.add_argument("--episodes", type=int, default=200)
    e.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    e.add_argument("--last-n", type=int, default=0, help="also evaluate the last N saved checkpoints (greedy)")
    e.add_argument("--last-episodes", type=int, default=200)
    e.add_argument("--procs", type=int, default=2)
    p = sub.add_parser("pilot")
    p.add_argument("--eval", required=True)
    c = sub.add_parser("confirm")
    c.add_argument("--new", required=True)
    c.add_argument("--base", required=True)
    c.add_argument("--out", required=True)
    a = ap.parse_args()
    {"eval": do_eval, "pilot": do_pilot, "confirm": do_confirm}[a.cmd](a)


if __name__ == "__main__":
    main()

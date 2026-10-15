"""Held-out evaluation of trained policies against a heuristic and a random baseline.

Usage:
    uv run python -m rocket_landing.evaluate --run experiments/results/v2_s1 \
        --checkpoint final --episodes 500 [--seed-start 50000] [--stochastic]
    uv run python -m rocket_landing.evaluate --runs v2_s0 v2_s1 v2_s2 --checkpoint best

Seed discipline. Every policy (greedy, optional stochastic, heuristic, random)
is rolled out on the SAME reset seeds ``seed_start .. seed_start + episodes - 1``.
Those seeds are asserted (``assert_heldout``) to overlap neither

* the training-env reset seeds ``train.training_env_seed(seed, i)`` for
  ``i < num_envs`` (always >= ``TRAIN_SEED_BASE`` = 1_000_000), nor
* the periodic in-training eval seeds ``eval_seed_start .. + eval_episodes - 1``
  (default 20000..20049) that were used to *select* ``best.pt``.

Limit of the guarantee: training envs are reset with a seed only once and then
continue their own RNG stream, so the seed sets are provably disjoint but the
initial states drawn from the stream are only "held out" in the sense of a
fresh i.i.d. draw from the same initial-state distribution. This measures
generalisation to unseen initial conditions, not to a shifted distribution.
``best.pt`` is selected using the in-training eval seeds, so it is mildly
optimistic there but not on this held-out range.
"""

from __future__ import annotations

import argparse
import json
import math
import re
from pathlib import Path

import numpy as np

from rocket_landing.baseline import EpisodeResult, HeuristicController, run_episode
from rocket_landing.environment import EnvConfig, RocketLandingEnv
from rocket_landing.state import Outcome

TRAIN_SEED_BASE = 1_000_000  # mirrors rocket_landing.train.TRAIN_SEED_BASE (asserted in tests)
DEFAULT_SEED_START = 50_000
WILSON_Z = 1.959963984540054  # two-sided 95%
DEFAULT_RESULTS_DIR = Path("experiments/results")
POLICIES = ("greedy", "stochastic", "heuristic", "random")


# --------------------------------------------------------------------- stats
def wilson_interval(successes: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """Wilson score interval for a binomial proportion; (nan, nan) when n == 0."""
    if n <= 0:
        return (float("nan"), float("nan"))
    if not 0 <= successes <= n:
        raise ValueError(f"successes {successes} not in 0..{n}")
    p = successes / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def _mean_std(values: list[float]) -> dict:
    if not values:
        return {"mean": None, "std": None, "n": 0}
    arr = np.asarray(values, dtype=np.float64)
    std = float(arr.std(ddof=1)) if len(arr) > 1 else 0.0
    return {"mean": float(arr.mean()), "std": std, "n": len(arr)}


# ------------------------------------------------------------------ seeds
def train_seed_set(train_seed: int, num_envs: int) -> set[int]:
    """Reset seeds used by training env ``i`` (mirrors ``train.training_env_seed``)."""
    return {TRAIN_SEED_BASE + train_seed * 1000 + i for i in range(num_envs)}


def assert_heldout(
    seed_start: int,
    episodes: int,
    train_seed: int,
    num_envs: int,
    eval_seed_start: int,
    eval_episodes: int,
) -> dict:
    """Raise AssertionError unless the eval seed range is disjoint from training/selection seeds."""
    if episodes <= 0:
        raise AssertionError("episodes must be positive")
    held = range(seed_start, seed_start + episodes)
    train = train_seed_set(train_seed, num_envs)
    sel = range(eval_seed_start, eval_seed_start + eval_episodes)
    assert seed_start >= 0, "held-out seeds must be non-negative"
    assert all(s >= TRAIN_SEED_BASE for s in train), "unexpected training seed below TRAIN_SEED_BASE"
    assert held.stop <= TRAIN_SEED_BASE, (
        f"held-out range {held.start}..{held.stop - 1} reaches the training-seed region "
        f"(>= {TRAIN_SEED_BASE})"
    )
    assert not (train & set(held)), "held-out seeds overlap training env seeds"
    assert held.stop <= sel.start or sel.stop <= held.start, (
        f"held-out range {held.start}..{held.stop - 1} overlaps in-training eval range "
        f"{sel.start}..{sel.stop - 1}"
    )
    return {
        "heldout": [held.start, held.stop - 1],
        "train_env_seeds": [min(train), max(train)] if train else None,
        "in_training_eval": [sel.start, sel.stop - 1] if len(sel) else None,
    }


# -------------------------------------------------------------- controllers
class RandomController:
    """Uniform random discrete action; deterministic given ``seed``."""

    def __init__(self, seed: int = 0, n_actions: int = 4):
        self._rng = np.random.default_rng(seed)
        self.n_actions = n_actions

    def act(self, obs) -> int:
        return int(self._rng.integers(self.n_actions))


# ------------------------------------------------------------ rollouts
def summarize(eps: list[EpisodeResult]) -> dict:
    """Rates with Wilson CIs and mean/std diagnostics for a list of episodes."""
    n = len(eps)
    out: dict = {"n": n}
    for key, outcome in (
        ("success", Outcome.LANDED),
        ("crash", Outcome.CRASHED),
        ("out_of_bounds", Outcome.OUT_OF_BOUNDS),
        ("timeout", Outcome.TIMEOUT),
    ):
        k = sum(e.outcome == outcome.value for e in eps)
        lo, hi = wilson_interval(k, n)
        out[key] = {"count": k, "rate": k / n if n else float("nan"), "ci95": [lo, hi]}
    td = [e for e in eps if e.landing_vx is not None]
    out["touchdowns"] = len(td)
    out["touchdown_vx"] = _mean_std([e.landing_vx for e in td])
    out["touchdown_vy"] = _mean_std([e.landing_vy for e in td])
    out["touchdown_tilt_abs"] = _mean_std([abs(e.landing_angle) for e in td])
    out["fuel_used"] = _mean_std([e.fuel_used for e in eps])
    out["episode_length"] = _mean_std([float(e.steps) for e in eps])
    out["reward"] = _mean_std([e.total_reward for e in eps])
    return out


def rollout(
    make_controller,
    seeds: range,
    env_config: EnvConfig | None,
    reward_version: str,
) -> list[EpisodeResult]:
    """Run one episode per seed; ``make_controller(seed)`` builds a fresh controller each time."""
    env = RocketLandingEnv(env_config, reward_version=reward_version)
    return [run_episode(env, make_controller(s), s) for s in seeds]


def evaluate_policies(
    controllers: dict,
    seed_start: int,
    episodes: int,
    reward_version: str,
    env_config: EnvConfig | None = None,
) -> dict[str, dict]:
    """``controllers`` maps name -> factory(seed) -> controller. Same seeds for all."""
    seeds = range(seed_start, seed_start + episodes)
    return {
        name: summarize(rollout(factory, seeds, env_config, reward_version))
        for name, factory in controllers.items()
    }


# ------------------------------------------------------------- run loading
def resolve_run_dir(run: str | Path, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    p = Path(run)
    if p.is_dir():
        return p
    if (results_dir / p).is_dir():
        return results_dir / p
    raise FileNotFoundError(f"run directory not found: {run} (also tried {results_dir / p})")


def resolve_checkpoint(run_dir: Path, checkpoint: str) -> Path:
    if checkpoint in ("final", "best"):
        path = run_dir / f"{checkpoint}.pt"
    else:
        m = re.fullmatch(r"(?:ckpt_)?(\d+)(?:\.pt)?", checkpoint)
        if not m:
            raise ValueError(f"checkpoint must be final, best, or a ckpt id; got {checkpoint!r}")
        path = run_dir / "checkpoints" / f"ckpt_{int(m.group(1)):05d}.pt"
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def evaluate_run(
    run: str | Path,
    checkpoint: str = "final",
    episodes: int = 500,
    seed_start: int = DEFAULT_SEED_START,
    stochastic: bool = False,
    results_dir: Path = DEFAULT_RESULTS_DIR,
    write: bool = True,
) -> dict:
    """Evaluate one run's checkpoint + heuristic + random on identical held-out seeds."""
    from rocket_landing.policy import PolicyController, load_checkpoint

    run_dir = resolve_run_dir(run, results_dir)
    cfg_file = json.loads((run_dir / "config.json").read_text())
    cfg = cfg_file["config"]
    train_seed = int(cfg_file["seed"])
    reward_version = cfg["reward_version"]
    seed_info = assert_heldout(
        seed_start,
        episodes,
        train_seed,
        int(cfg["num_envs"]),
        int(cfg["eval_seed_start"]),
        int(cfg["eval_episodes"]),
    )
    ckpt_path = resolve_checkpoint(run_dir, checkpoint)
    model, obs_norm, payload = load_checkpoint(ckpt_path)

    controllers = {
        "greedy": lambda s: PolicyController(model, obs_norm, deterministic=True),
        "heuristic": lambda s: HeuristicController(),
        "random": lambda s: RandomController(seed=s),
    }
    if stochastic:
        controllers["stochastic"] = lambda s: PolicyController(
            model, obs_norm, deterministic=False, seed=s
        )
    results = evaluate_policies(controllers, seed_start, episodes, reward_version)
    result = {
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "checkpoint": checkpoint,
        "checkpoint_path": str(ckpt_path),
        "checkpoint_update": payload.get("extra", {}).get("update"),
        "train_seed": train_seed,
        "reward_version": reward_version,
        "episodes": episodes,
        "seed_start": seed_start,
        "seed_check": seed_info,
        "policies": results,
    }
    if write:
        (run_dir / "heldout_eval.json").write_text(json.dumps(result, indent=2) + "\n")
    return result


# ----------------------------------------------------------- aggregation
def _spread(values: list[float]) -> dict:
    return {"mean": float(np.mean(values)), "min": float(min(values)), "max": float(max(values))}


def aggregate(results: list[dict]) -> dict:
    """Mean/min/max across runs (seeds) of the headline rates; per-run values are kept.

    Failed runs (success rate 0) are counted in ``n_failed_seeds`` and included in
    every mean/min/max; nothing is dropped.
    """
    out: dict = {"n_runs": len(results), "policies": {}}
    names = [p for p in POLICIES if all(p in r["policies"] for r in results)]
    for name in names:
        rows = [r["policies"][name] for r in results]
        entry: dict = {
            "per_run": [
                {"run": r["run"], "success_rate": r["policies"][name]["success"]["rate"]}
                for r in results
            ],
            "n_failed_seeds": sum(row["success"]["count"] == 0 for row in rows),
        }
        for key in ("success", "crash", "out_of_bounds", "timeout"):
            entry[f"{key}_rate"] = _spread([row[key]["rate"] for row in rows])
        entry["mean_reward"] = _spread([row["reward"]["mean"] for row in rows])
        entry["mean_fuel_used"] = _spread([row["fuel_used"]["mean"] for row in rows])
        entry["mean_episode_length"] = _spread([row["episode_length"]["mean"] for row in rows])
        out["policies"][name] = entry
    return out


# ------------------------------------------------------------- formatting
def _pct(d: dict) -> str:
    lo, hi = d["ci95"]
    return f"{d['rate']:.1%} [{lo:.1%}, {hi:.1%}]"


def _ms(d: dict, fmt: str = "{:.2f}") -> str:
    if d["mean"] is None:
        return "n/a"
    return f"{fmt.format(d['mean'])} ± {fmt.format(d['std'])}"


def format_run(result: dict) -> str:
    head = (
        f"### {result['run']} ({result['checkpoint']}"
        + (f", update {result['checkpoint_update']}" if result.get("checkpoint_update") is not None else "")
        + f"; reward {result['reward_version']}; {result['episodes']} episodes; "
        f"seeds {result['seed_start']}..{result['seed_start'] + result['episodes'] - 1})"
    )
    cols = [p for p in POLICIES if p in result["policies"]]
    rows = [
        ("success (95% Wilson)", lambda s: _pct(s["success"])),
        ("crash", lambda s: _pct(s["crash"])),
        ("out of bounds", lambda s: _pct(s["out_of_bounds"])),
        ("timeout", lambda s: _pct(s["timeout"])),
        ("touchdown vx (m/s)", lambda s: _ms(s["touchdown_vx"])),
        ("touchdown vy (m/s)", lambda s: _ms(s["touchdown_vy"])),
        ("touchdown abs tilt (rad)", lambda s: _ms(s["touchdown_tilt_abs"], "{:.3f}")),
        ("fuel used", lambda s: _ms(s["fuel_used"], "{:.3f}")),
        ("episode length", lambda s: _ms(s["episode_length"], "{:.0f}")),
        (f"reward ({result['reward_version']})", lambda s: _ms(s["reward"], "{:.1f}")),
    ]
    lines = [head, "", "| metric | " + " | ".join(cols) + " |", "|---|" + "---|" * len(cols)]
    for label, fn in rows:
        lines.append(f"| {label} | " + " | ".join(fn(result["policies"][c]) for c in cols) + " |")
    return "\n".join(lines)


def format_aggregate(results: list[dict], agg: dict) -> str:
    lines = [f"## Aggregate over {agg['n_runs']} runs", ""]
    lines += [
        "### Per-seed held-out success rate (95% Wilson CI)",
        "",
        "| run | ckpt | greedy | heuristic | random |"
        + (" stochastic |" if "stochastic" in agg["policies"] else ""),
        "|---|---|---|---|---|" + ("---|" if "stochastic" in agg["policies"] else ""),
    ]
    for r in results:
        p = r["policies"]
        cells = [r["run"], r["checkpoint"], _pct(p["greedy"]["success"]),
                 _pct(p["heuristic"]["success"]), _pct(p["random"]["success"])]
        if "stochastic" in agg["policies"]:
            cells.append(_pct(p["stochastic"]["success"]))
        lines.append("| " + " | ".join(cells) + " |")
    lines += ["", "### Across seeds (mean / min / max; all runs included)", "",
              "| policy | success | crash | out of bounds | timeout | reward | failed seeds (0 successes) |",
              "|---|---|---|---|---|---|---|"]
    for name, e in agg["policies"].items():
        def f(k, fmt="{:.1%}"):
            s = e[k]
            return " / ".join(fmt.format(s[x]) for x in ("mean", "min", "max"))
        lines.append(
            f"| {name} | {f('success_rate')} | {f('crash_rate')} | {f('out_of_bounds_rate')} | "
            f"{f('timeout_rate')} | {f('mean_reward', '{:.1f}')} | "
            f"{e['n_failed_seeds']}/{agg['n_runs']} |"
        )
    return "\n".join(lines)


# --------------------------------------------------------------------- CLI
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--run", help="run directory (or name under --results-dir)")
    g.add_argument("--runs", nargs="+", help="several run dirs/names to aggregate")
    ap.add_argument("--checkpoint", default="final", help="final | best | <ckpt id, e.g. 500 or ckpt_00500>")
    ap.add_argument("--episodes", type=int, default=500)
    ap.add_argument("--seed-start", type=int, default=DEFAULT_SEED_START)
    ap.add_argument("--stochastic", action="store_true", help="also evaluate the sampling policy")
    ap.add_argument("--results-dir", type=Path, default=DEFAULT_RESULTS_DIR)
    ap.add_argument("--aggregate-out", type=Path, help="write the aggregate JSON here (--runs)")
    ap.add_argument("--no-write", action="store_true", help="do not write heldout_eval.json")
    args = ap.parse_args(argv)

    import torch

    torch.set_num_threads(1)
    runs = [args.run] if args.run else args.runs
    results = []
    for run in runs:
        res = evaluate_run(
            run, args.checkpoint, args.episodes, args.seed_start, args.stochastic,
            args.results_dir, write=not args.no_write,
        )
        results.append(res)
        print(format_run(res), "\n", flush=True)
    if len(results) > 1:
        agg = aggregate(results)
        print(format_aggregate(results, agg))
        if args.aggregate_out:
            args.aggregate_out.parent.mkdir(parents=True, exist_ok=True)
            args.aggregate_out.write_text(
                json.dumps({"aggregate": agg, "runs": results}, indent=2) + "\n"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

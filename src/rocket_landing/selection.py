"""Validation-based checkpoint selection for PPO runs whose greedy success swings between checkpoints.

A training run saves a checkpoint every ``checkpoint_every`` updates plus ``final.pt``. Greedy
success of consecutive checkpoints can jump between ~0% and ~100%, so "ship final.pt" is a lottery.
This module evaluates EVERY saved checkpoint greedily on a VALIDATION seed range, picks one with a
rule that is fixed before the held-out TEST range is ever touched, writes ``selected.json`` and
exports the winner as an ordinary loadable checkpoint.

Seed discipline (all asserted by ``assert_validation_disjoint``). Four reset-seed ranges exist:

* training:   ``1_000_000 + 1000*seed + i`` (never used by evaluation)
* in-training eval (picks ``best.pt``): ``eval_seed_start .. + eval_episodes - 1`` (20000..20049)
* TEST (reported numbers):  50000..50499   -- never used for selection
* VALIDATION (this module): 70000..70199   -- used for selection only, never reported as a result
* also reserved: 0..99 (baseline / earlier experiment test seeds), 1000..1999 (heuristic gain tuning)
  and 60000..60299 (pilot config comparison of the stabilisation runs)

Rules (``RULES``), each a pure function ``scores -> index`` over scores sorted by update:

* ``best_val``          highest validation success; ties -> latest checkpoint.
* ``latest_near_best``  among checkpoints whose Wilson 95% interval overlaps the best checkpoint's
                        (not distinguishable from the best), the LATEST one.
* ``stable3``           best rolling mean of validation success over 3 consecutive checkpoints
                        (ties -> latest window); inside that window the highest-scoring checkpoint
                        (ties -> latest). A lone spike with bad neighbours cannot win.
* ``max_margin``        among the near-best checkpoints (as above) the one with the largest fuel
                        margin, the 10th percentile over validation episodes of fuel left at the end
                        (collapsed greedy policies run out of fuel just before touchdown, so a large
                        margin means the policy is far from that cliff). Ties -> latest.

A saved checkpoint is a fixed deterministic policy, so its validation success is an unbiased estimate
of its test success. Selection costs compute (one extra evaluation per checkpoint) and a small
winner's-curse bias on the *validation* numbers; it cannot repair a run in which no checkpoint is good.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path

import numpy as np

from rocket_landing.evaluate import (
    assert_heldout,
    resolve_run_dir,
    rollout,
    wilson_interval,
)

# ----------------------------------------------------------------- seed ranges
VALIDATION_SEED_START = 70_000
VALIDATION_EPISODES = 200
TEST_SEED_START = 50_000
TEST_EPISODES = 500
BASELINE_TEST_SEEDS = (0, 100)  # heuristic baseline + earlier experiments' test seeds (half-open)
BASELINE_TUNING_SEEDS = (1_000, 2_000)  # heuristic gain tuning "seeds 1000+" (docs/environment.md); reserved
PILOT_SELECTION_SEEDS = (60_000, 60_300)  # used by the stabiliser's pilot runs to compare configs; reserved
SCHEMA_VERSION = 1

OUTCOME_CODES = {"landed": "L", "crashed": "C", "out_of_bounds": "O", "timeout": "T"}


def _overlaps(a: tuple[int, int], b: tuple[int, int]) -> bool:
    """Half-open ranges [a0, a1) and [b0, b1) intersect."""
    return a[0] < b[1] and b[0] < a[1]


def assert_validation_disjoint(
    seed_start: int,
    episodes: int,
    train_seed: int,
    num_envs: int,
    eval_seed_start: int,
    eval_episodes: int,
    test_seed_start: int = TEST_SEED_START,
    test_episodes: int = TEST_EPISODES,
) -> dict:
    """Raise AssertionError unless the validation range avoids every other seed use.

    Checks (in addition to ``evaluate.assert_heldout``: training-env seeds, in-training eval seeds,
    below ``TRAIN_SEED_BASE``): the held-out TEST range, the baseline test seeds 0..99, the
    heuristic-tuning seeds 1000..1999 and the pilot-selection seeds 60000..60299.
    """
    info = assert_heldout(seed_start, episodes, train_seed, num_envs, eval_seed_start, eval_episodes)
    val = (seed_start, seed_start + episodes)
    test = (test_seed_start, test_seed_start + test_episodes)
    assert not _overlaps(val, test), (
        f"validation range {val[0]}..{val[1] - 1} overlaps the TEST range {test[0]}..{test[1] - 1}"
    )
    assert not _overlaps(val, BASELINE_TEST_SEEDS), "validation range overlaps baseline test seeds 0..99"
    assert not _overlaps(val, BASELINE_TUNING_SEEDS), (
        "validation range overlaps heuristic-tuning seeds 1000..1999"
    )
    assert not _overlaps(val, PILOT_SELECTION_SEEDS), (
        "validation range overlaps the stabiliser's pilot-selection seeds 60000..60299"
    )
    info["validation"] = [val[0], val[1] - 1]
    info["test"] = [test[0], test[1] - 1]
    return info


# ----------------------------------------------------------------------- scores
@dataclass(frozen=True)
class Score:
    """Validation result of one checkpoint (``update`` orders the training timeline)."""

    name: str
    update: int
    successes: int
    n: int
    margin: float = 0.0  # 10th percentile over episodes of fuel left at episode end (1 - fuel_used)

    @property
    def rate(self) -> float:
        return self.successes / self.n if self.n else float("nan")

    @property
    def ci95(self) -> tuple[float, float]:
        return wilson_interval(self.successes, self.n)


def _sorted(scores: Sequence[Score]) -> list[Score]:
    if not scores:
        raise ValueError("no checkpoints to select from")
    return sorted(scores, key=lambda s: (s.update, s.name))


def _frac(s: Score) -> Fraction:
    return Fraction(s.successes, s.n)


# ------------------------------------------------------------------------ rules
def select_best_val(scores: Sequence[Score]) -> int:
    """Index (into ``scores`` sorted by update) of the highest validation success; ties -> latest."""
    s = _sorted(scores)
    return max(range(len(s)), key=lambda i: (_frac(s[i]), i))


def select_latest_near_best(scores: Sequence[Score]) -> int:
    """Latest checkpoint whose Wilson 95% interval overlaps that of the best checkpoint.

    "Near-best" = upper bound >= the best checkpoint's lower bound, i.e. not distinguishable from the
    best at the validation sample size. Among those, the later the checkpoint, the better (it has seen
    more training and ties are otherwise arbitrary).
    """
    s = _sorted(scores)
    best = s[select_best_val(s)]
    floor = best.ci95[0]
    cand = [i for i, c in enumerate(s) if c.ci95[1] >= floor - 1e-12]
    return max(cand)


def select_stable(scores: Sequence[Score], window: int = 3) -> int:
    """Best rolling mean over ``window`` consecutive checkpoints; best checkpoint inside it.

    Window score is the exact pooled success rate (sum successes / sum n). Ties between windows go
    to the later window, ties inside the window to the later checkpoint. With fewer checkpoints than
    ``window`` the window is the whole list.
    """
    s = _sorted(scores)
    w = max(1, min(window, len(s)))
    starts = range(len(s) - w + 1)
    best_start = max(starts, key=lambda a: (Fraction(sum(c.successes for c in s[a : a + w]),
                                                     sum(c.n for c in s[a : a + w])), a))
    inside = range(best_start, best_start + w)
    return max(inside, key=lambda i: (_frac(s[i]), i))


def select_max_margin(scores: Sequence[Score]) -> int:
    """Among the near-best checkpoints (Wilson interval overlaps the best), the largest fuel margin."""
    s = _sorted(scores)
    best = s[select_best_val(s)]
    floor = best.ci95[0]
    cand = [i for i, c in enumerate(s) if c.ci95[1] >= floor - 1e-12]
    return max(cand, key=lambda i: (s[i].margin, i))


RULES: dict[str, Callable[[Sequence[Score]], int]] = {
    "best_val": select_best_val,
    "latest_near_best": select_latest_near_best,
    "stable3": select_stable,
    "max_margin": select_max_margin,
}
# Default rule. Fixed after the validation phase and BEFORE any checkpoint was evaluated on the TEST
# range: the validation-only split-half check (experiments/selection/rules_validation.json) could not
# separate best_val from stable3 (68.2% vs 68.2%), so the tie was broken on the design argument that a
# chosen checkpoint should have good neighbours (see docs/checkpoint-selection.md).
DEFAULT_RULE = "stable3"


def apply_rule(scores: Sequence[Score], rule: str) -> Score:
    if rule not in RULES:
        raise ValueError(f"unknown rule {rule!r}; expected one of {sorted(RULES)}")
    s = _sorted(scores)
    return s[RULES[rule](s)]


# ---------------------------------------------------------------- checkpoints
def list_checkpoints(run_dir: Path) -> list[tuple[int, str, Path]]:
    """``(update, name, path)`` for every ``checkpoints/ckpt_*.pt`` and ``final.pt``, by update.

    ``best.pt`` is NOT a candidate (it was already picked on the in-training eval seeds); compare to
    it separately. ``final.pt``'s update is read from its payload.
    """
    out: list[tuple[int, str, Path]] = []
    for p in sorted((run_dir / "checkpoints").glob("ckpt_*.pt")):
        m = re.fullmatch(r"ckpt_(\d+)\.pt", p.name)
        if m:
            out.append((int(m.group(1)), p.stem, p))
    final = run_dir / "final.pt"
    if final.exists():
        out.append((checkpoint_update(final), "final", final))
    if not out:
        raise FileNotFoundError(f"no checkpoints under {run_dir}")
    return sorted(out, key=lambda t: (t[0], t[1]))


def checkpoint_update(path: Path) -> int:
    import torch

    payload = torch.load(path, map_location="cpu", weights_only=True)
    return int(payload.get("extra", {}).get("update", -1))


# ------------------------------------------------------------------ evaluation
def outcomes_string(eps) -> str:
    """One letter per episode: L(anded) C(rashed) O(ut of bounds) T(imeout), in seed order."""
    return "".join(OUTCOME_CODES[e.outcome] for e in eps)


def fuel_margin(fuel_used: Sequence[float]) -> float:
    """10th percentile over episodes of fuel left at the end of the episode (tank is 1.0)."""
    if not len(fuel_used):
        return 0.0
    return float(np.percentile(1.0 - np.asarray(fuel_used, dtype=np.float64), 10))


def counts(outcomes: str, fuel_used: Sequence[float] | None = None) -> dict:
    n = len(outcomes)
    out = {"n": n}
    for key, code in (("success", "L"), ("crash", "C"), ("out_of_bounds", "O"), ("timeout", "T")):
        k = outcomes.count(code)
        out[key] = k
    out["rate"] = out["success"] / n if n else float("nan")
    out["ci95"] = list(wilson_interval(out["success"], n))
    if fuel_used is not None and len(fuel_used):
        out["fuel_used_mean"] = float(np.mean(fuel_used))
        out["fuel_left_p10"] = fuel_margin(fuel_used)
    return out


def evaluate_checkpoint(
    path: Path | str, seed_start: int, episodes: int, reward_version: str
) -> dict:
    """Greedy rollout of one checkpoint on ``seed_start .. + episodes - 1``.

    Returns ``{"outcomes": outcomes_string, "fuel_used": [per-episode fuel burned]}``.
    """
    from rocket_landing.policy import PolicyController, load_checkpoint

    model, obs_norm, _ = load_checkpoint(path)
    eps = rollout(
        lambda s: PolicyController(model, obs_norm, deterministic=True),
        range(seed_start, seed_start + episodes),
        None,
        reward_version,
    )
    return {"outcomes": outcomes_string(eps), "fuel_used": [round(float(e.fuel_used), 4) for e in eps]}


def _worker(job: tuple[str, int, int, str]) -> dict:
    import torch

    torch.set_num_threads(1)
    path, seed_start, episodes, reward_version = job
    return evaluate_checkpoint(path, seed_start, episodes, reward_version)


def evaluate_many(
    paths: Sequence[Path], seed_start: int, episodes: int, reward_version: str, procs: int = 3
) -> list[dict]:
    """``evaluate_checkpoint`` result per path, in order. ``procs`` is the number of worker processes (<= 3 advised)."""
    jobs = [(str(p), seed_start, episodes, reward_version) for p in paths]
    if procs <= 1 or len(jobs) <= 1:
        return [_worker(j) for j in jobs]
    from multiprocessing import get_context

    with get_context("spawn").Pool(procs) as pool:
        return pool.map(_worker, jobs, chunksize=1)


def run_config(run_dir: Path) -> dict:
    cfg_file = json.loads((run_dir / "config.json").read_text())
    cfg = cfg_file["config"]
    return {
        "train_seed": int(cfg_file["seed"]),
        "num_envs": int(cfg["num_envs"]),
        "eval_seed_start": int(cfg["eval_seed_start"]),
        "eval_episodes": int(cfg["eval_episodes"]),
        "reward_version": cfg["reward_version"],
    }


def build_validation_curve(
    run: str | Path,
    seed_start: int = VALIDATION_SEED_START,
    episodes: int = VALIDATION_EPISODES,
    procs: int = 3,
    results_dir: Path = Path("experiments/results"),
    cache: Path | None = None,
) -> dict:
    """Evaluate every checkpoint of ``run`` on the validation range (reusing ``cache`` if complete)."""
    run_dir = resolve_run_dir(run, results_dir)
    rc = run_config(run_dir)
    seed_check = assert_validation_disjoint(
        seed_start, episodes, rc["train_seed"], rc["num_envs"], rc["eval_seed_start"], rc["eval_episodes"]
    )
    ckpts = list_checkpoints(run_dir)
    if cache is not None and cache.exists():
        old = json.loads(cache.read_text())
        same = (
            old.get("seed_start") == seed_start
            and old.get("episodes") == episodes
            and [c["name"] for c in old["checkpoints"]] == [c[1] for c in ckpts]
            and [c["update"] for c in old["checkpoints"]] == [c[0] for c in ckpts]
        )
        if same:
            return old
    outs = evaluate_many([c[2] for c in ckpts], seed_start, episodes, rc["reward_version"], procs)
    curve = {
        "schema_version": SCHEMA_VERSION,
        "run": run_dir.name,
        "run_dir": str(run_dir),
        "reward_version": rc["reward_version"],
        "train_seed": rc["train_seed"],
        "seed_start": seed_start,
        "episodes": episodes,
        "seed_check": seed_check,
        "checkpoints": [
            {"name": name, "update": upd, "path": str(path), "outcomes": o["outcomes"],
             "fuel_used": o["fuel_used"], **counts(o["outcomes"], o["fuel_used"])}
            for (upd, name, path), o in zip(ckpts, outs)
        ],
    }
    if cache is not None:
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_text(json.dumps(curve) + "\n")
    return curve


def scores_from_curve(curve: dict, episode_slice: slice | None = None) -> list[Score]:
    """``Score`` per checkpoint; ``episode_slice`` restricts to a subset of the validation episodes."""
    out = []
    for c in curve["checkpoints"]:
        o = c["outcomes"] if episode_slice is None else c["outcomes"][episode_slice]
        f = c.get("fuel_used", [])
        f = f if episode_slice is None else f[episode_slice]
        out.append(Score(c["name"], int(c["update"]), o.count("L"), len(o), fuel_margin(f)))
    return out


# ---------------------------------------------------------------- selection
def _score_json(s: Score) -> dict:
    lo, hi = s.ci95
    return {"name": s.name, "update": s.update, "successes": s.successes, "n": s.n,
            "rate": s.rate, "ci95": [lo, hi], "fuel_left_p10": s.margin}


def select(curve: dict, rule: str = DEFAULT_RULE) -> dict:
    """Apply ``rule`` to a validation curve; returns the ``selected.json`` document (no file IO)."""
    scores = _sorted(scores_from_curve(curve))
    chosen = apply_rule(scores, rule)
    by_name = {c["name"]: c for c in curve["checkpoints"]}
    final = next((s for s in scores if s.name == "final"), scores[-1])
    return {
        "schema_version": SCHEMA_VERSION,
        "run": curve["run"],
        "run_dir": curve["run_dir"],
        "rule": rule,
        "rule_params": {"window": 3} if rule == "stable3" else {},
        "chosen": {
            **_score_json(chosen),
            "path": by_name[chosen.name]["path"],
            "fuel_used_mean": by_name[chosen.name].get("fuel_used_mean"),
            "crash": by_name[chosen.name]["crash"],
            "out_of_bounds": by_name[chosen.name]["out_of_bounds"],
            "timeout": by_name[chosen.name]["timeout"],
        },
        "final": _score_json(final),
        "alternatives": {r: _score_json(apply_rule(scores, r)) for r in RULES},
        "validation": {
            "seed_start": curve["seed_start"],
            "seed_end": curve["seed_start"] + curve["episodes"] - 1,
            "episodes": curve["episodes"],
            "seed_check": curve["seed_check"],
            "n_checkpoints": len(scores),
            "curve": [_score_json(s) for s in scores],
        },
    }


SELECTED_KEYS = {"schema_version", "run", "run_dir", "rule", "rule_params", "chosen", "final",
                 "alternatives", "validation"}


def export_checkpoint(src: Path | str, dst: Path | str, selection: dict) -> Path:
    """Copy ``src`` to ``dst`` as an ordinary checkpoint, recording the selection under ``extra``."""
    import torch

    payload = torch.load(src, map_location="cpu", weights_only=True)
    extra = dict(payload.get("extra", {}))
    ch = selection["chosen"]
    extra["selection"] = {
        "rule": selection["rule"],
        "source_checkpoint": ch["name"],
        "source_update": ch["update"],
        "validation_successes": ch["successes"],
        "validation_episodes": ch["n"],
        "validation_seeds": [selection["validation"]["seed_start"], selection["validation"]["seed_end"]],
    }
    payload["extra"] = extra
    dst = Path(dst)
    dst.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, dst)
    return dst


def select_run(
    run: str | Path,
    out_dir: Path,
    rule: str = DEFAULT_RULE,
    seed_start: int = VALIDATION_SEED_START,
    episodes: int = VALIDATION_EPISODES,
    procs: int = 3,
    results_dir: Path = Path("experiments/results"),
) -> dict:
    """Full pipeline for one run: validate all checkpoints, select, write ``selected.json`` + ``selected.pt``.

    ``out_dir`` must not be inside the run directory's worktree results (the source run is read-only
    by convention); it receives ``validation_curve.json``, ``selected.json`` and ``selected.pt``.
    """
    run_dir = resolve_run_dir(run, results_dir)
    out_dir = Path(out_dir)
    if out_dir.resolve() == run_dir.resolve():
        raise ValueError("out_dir must differ from the run directory (runs are read-only)")
    out_dir.mkdir(parents=True, exist_ok=True)
    curve = build_validation_curve(run, seed_start, episodes, procs, results_dir,
                                   cache=out_dir / "validation_curve.json")
    sel = select(curve, rule)
    export = export_checkpoint(sel["chosen"]["path"], out_dir / "selected.pt", sel)
    sel["exported_checkpoint"] = export.name
    sel["exported_sha256"] = hashlib.sha256(export.read_bytes()).hexdigest()
    (out_dir / "selected.json").write_text(json.dumps(sel, indent=2) + "\n")
    return sel


# ------------------------------------------------- validation-only rule check
def split_half_score(curve: dict, rule: str) -> dict:
    """Selection-noise check using ONLY validation episodes.

    Select on the first half of the validation episodes, score the chosen checkpoint on the second
    half (independent episodes), then swap the halves. Returns both held-in success rates and the mean
    held-out one; a rule that chases a lucky spike scores lower on the held-out half.
    """
    n = curve["episodes"]
    h = n // 2
    halves = (slice(0, h), slice(h, n))
    sel_rates, out_rates = [], []
    for a, b in (halves, halves[::-1]):
        chosen = apply_rule(scores_from_curve(curve, a), rule)
        other = {s.name: s for s in scores_from_curve(curve, b)}[chosen.name]
        sel_rates.append(chosen.rate)
        out_rates.append(other.rate)
    return {"selected_half": sum(sel_rates) / 2, "other_half": sum(out_rates) / 2}

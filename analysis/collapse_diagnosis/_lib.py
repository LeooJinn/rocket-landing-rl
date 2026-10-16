"""Shared helpers for the collapse-diagnosis analysis scripts.

The 20 headline runs (``hl_v1_s0..s9``, ``hl_v2_s0..s9``) are not committed (checkpoints are
git-ignored). Point ``RESULTS_DIR`` at a directory that contains them, either by editing the
constant below or by setting the environment variable ``COLLAPSE_RESULTS_DIR``, e.g.

    COLLAPSE_RESULTS_DIR=~/projects/rocket-landing-rl-headline/experiments/results \
        uv run python analysis/collapse_diagnosis/fuel_vs_outcome.py

Scripts that only read ``metrics.jsonl`` / ``eval.jsonl`` need no checkpoints. Scripts that roll
out checkpoints need ``checkpoints/ckpt_*.pt`` / ``final.pt`` and ``torch`` (run with ``uv run``).

Rollouts use a numpy re-implementation of the actor/critic (float64, same weights as the torch
model) and the repo's ``RocketLandingEnv``. Reset seeds start at ``HELDOUT_START`` = 50000, the
same held-out seeds as the headline evaluation, so these numbers are diagnostic and must not be
used to select checkpoints.
"""

from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

sys.dont_write_bytecode = True
REPO_ROOT = Path(__file__).resolve().parents[2]
RESULTS_DIR = Path(
    os.environ.get("COLLAPSE_RESULTS_DIR", REPO_ROOT / "experiments" / "results")
).expanduser()
RAW_DIR = Path(__file__).resolve().parent / "raw"
HELDOUT_START = 50000

V1_SEEDS = list(range(10))
V2_SEEDS = list(range(10))
ALL_RUNS = [f"hl_v1_s{s}" for s in V1_SEEDS] + [f"hl_v2_s{s}" for s in V2_SEEDS]
# v2 seeds that learned to land at some point (v2 s6 never did).
V2_LEARNED = ["hl_v2_s0", "hl_v2_s1", "hl_v2_s2", "hl_v2_s3", "hl_v2_s4", "hl_v2_s5",
              "hl_v2_s7", "hl_v2_s8", "hl_v2_s9"]
# v1 seeds with a stable-ish learned policy at the end.
V1_LEARNED = ["hl_v1_s1", "hl_v1_s2", "hl_v1_s3"]


def load(run: str) -> tuple[list[dict], list[dict]]:
    """Return (metrics, evals) records of a run (one metrics line per update, one eval per 10)."""
    d = RESULTS_DIR / run
    if not (d / "metrics.jsonl").exists():
        raise SystemExit(
            f"{d}/metrics.jsonl not found. Set COLLAPSE_RESULTS_DIR to the directory holding "
            "the hl_* runs (see analysis/collapse_diagnosis/README.md)."
        )
    metrics = [json.loads(line) for line in open(d / "metrics.jsonl")]
    evals = [json.loads(line) for line in open(d / "eval.jsonl")]
    return metrics, evals


def col(metrics: list[dict], key: str) -> np.ndarray:
    return np.array([np.nan if r.get(key) is None else r[key] for r in metrics], dtype=float)


def ckpt_path(run: str, ck) -> Path:
    """``ck`` is an update number (multiple of 50) or the string 'final'."""
    d = RESULTS_DIR / run
    p = d / "final.pt" if ck == "final" else d / "checkpoints" / f"ckpt_{int(ck):05d}.pt"
    if not p.exists():
        raise SystemExit(f"{p} not found (checkpoints are git-ignored; see the README).")
    return p


def parse_ck(s: str):
    return "final" if s == "final" else int(s)


def parse_spec(spec: str) -> tuple[str, object]:
    """'hl_v2_s8@1200' -> ('hl_v2_s8', 1200); 'hl_v2_s3@final' -> ('hl_v2_s3', 'final')."""
    run, ck = spec.split("@")
    return run, parse_ck(ck)


class NumpyPolicy:
    """numpy forward pass of a checkpoint's actor and critic (tanh MLPs, float64)."""

    def __init__(self, path):
        from rocket_landing.policy import load_checkpoint  # needs torch

        model, norm, payload = load_checkpoint(path)
        sd = {k: v.double().numpy() for k, v in model.state_dict().items()}
        self.actor = [(sd[f"actor.{i}.weight"], sd[f"actor.{i}.bias"]) for i in (0, 2, 4)]
        self.critic = [(sd[f"critic.{i}.weight"], sd[f"critic.{i}.bias"]) for i in (0, 2, 4)]
        self.norm = norm
        self.update = payload["extra"].get("update")
        self.reward_version = payload["config"]["reward_version"]

    @classmethod
    def from_spec(cls, spec: str) -> "NumpyPolicy":
        run, ck = parse_spec(spec)
        return cls(ckpt_path(run, ck))

    @staticmethod
    def _mlp(layers, z):
        for i, (w, b) in enumerate(layers):
            z = z @ w.T + b
            if i < len(layers) - 1:
                z = np.tanh(z)
        return z

    def logits(self, obs):
        return self._mlp(self.actor, self.norm.normalize(obs).astype(np.float64))

    def value(self, obs):
        return self._mlp(self.critic, self.norm.normalize(obs).astype(np.float64))[..., 0]

    def probs(self, obs, temperature: float = 1.0):
        z = self.logits(obs) / temperature
        z = z - z.max(-1, keepdims=True)
        p = np.exp(z)
        return p / p.sum(-1, keepdims=True)


def make_env(pol: NumpyPolicy):
    from rocket_landing.environment import RocketLandingEnv

    return RocketLandingEnv(reward_version=pol.reward_version)


def rollout(pol: NumpyPolicy, seed: int, mode: str = "greedy", rng=None, record: bool = False):
    """One episode. mode 'greedy' (argmax) or 'sample' (T=1, ``rng.choice``).

    Returns (final info dict, total reward, trajectory [(obs, action, probs)] if ``record``).
    """
    env = make_env(pol)
    obs, _ = env.reset(seed=seed)
    traj, total, done = [], 0.0, False
    while not done:
        p = pol.probs(obs)
        a = int(np.argmax(pol.logits(obs))) if mode == "greedy" else int(rng.choice(4, p=p))
        if record:
            traj.append((obs.copy(), a, p.copy()))
        obs, r, term, trunc, info = env.step(a)
        total += r
        done = term or trunc
    return info, total, traj

"""Offline training/evaluation plots read from run directories.

A run directory (see ``train.py``) holds ``config.json``, ``metrics.jsonl``
(one record per PPO update, training-time statistics), ``eval.jsonl`` (one
record per periodic deterministic evaluation on fixed seeds) and, once the
run has finished, ``summary.json``.

Everything here tolerates runs that are still being written: a truncated last
line is skipped, a missing ``summary.json`` marks the run ``in_progress``.
Requires the optional ``plots`` extra (matplotlib).

Terminology used on figures:
    final       eval success rate at the LAST evaluation written so far.
    best-so-far maximum eval success rate over all evaluations; selecting it
                uses the same eval episodes it reports, so it is optimistic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable, Sequence

import numpy as np

OUTCOME_COLORS = {
    "success_rate": "#009E73",
    "crash_rate": "#D55E00",
    "out_of_bounds_rate": "#0072B2",
}
OUTCOME_LABELS = {
    "success_rate": "landed",
    "crash_rate": "crashed",
    "out_of_bounds_rate": "out of bounds",
}
GROUP_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#555555"]


def _plt():
    import matplotlib.pyplot as plt  # imported lazily: matplotlib is an optional extra

    return plt


# --------------------------------------------------------------------------- loading


def load_jsonl(path: str | Path) -> list[dict]:
    """Parse a JSONL file, skipping blank, partial or malformed lines.

    A missing file gives ``[]``. Only dict records are kept.
    """
    p = Path(path)
    if not p.exists():
        return []
    out: list[dict] = []
    with open(p, encoding="utf-8", errors="replace") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(rec, dict):
                out.append(rec)
    return out


def _load_json(path: Path) -> dict | None:
    try:
        with open(path) as f:
            data = json.load(f)
    except (OSError, json.JSONDecodeError):
        return None
    return data if isinstance(data, dict) else None


def flatten_config(cfg: dict) -> dict:
    """Flatten one nesting level: ``{"ppo": {"gamma": 1}}`` -> ``{"ppo.gamma": 1}``."""
    flat: dict = {}
    for k, v in cfg.items():
        if isinstance(v, dict):
            for k2, v2 in v.items():
                flat[f"{k}.{k2}"] = v2
        else:
            flat[k] = v
    return flat


@dataclass
class Run:
    name: str
    path: Path
    config: dict = field(default_factory=dict)  # flattened training config
    seed: int | None = None
    metrics: list[dict] = field(default_factory=list)
    evals: list[dict] = field(default_factory=list)
    summary: dict | None = None

    @property
    def in_progress(self) -> bool:
        return self.summary is None

    @property
    def total_timesteps(self) -> int | None:
        return self.config.get("total_timesteps")

    def series(self, records: str, key: str) -> tuple[np.ndarray, np.ndarray]:
        """(global_step, value) arrays for ``metrics`` or ``evals``; None values dropped."""
        recs = self.metrics if records == "metrics" else self.evals
        xs, ys = [], []
        for r in recs:
            if r.get("global_step") is None or r.get(key) is None:
                continue
            xs.append(r["global_step"])
            ys.append(r[key])
        return np.asarray(xs, dtype=float), np.asarray(ys, dtype=float)

    @property
    def final_eval(self) -> dict | None:
        return self.evals[-1] if self.evals else None

    @property
    def final_success(self) -> float | None:
        e = self.final_eval
        return None if e is None else e.get("success_rate")

    @property
    def best_success(self) -> float | None:
        vals = [e["success_rate"] for e in self.evals if e.get("success_rate") is not None]
        return max(vals) if vals else None


def load_run(path: str | Path) -> Run:
    p = Path(path)
    raw = _load_json(p / "config.json") or {}
    cfg = raw.get("config", {}) if isinstance(raw.get("config"), dict) else {}
    return Run(
        name=p.name,
        path=p,
        config=flatten_config(cfg),
        seed=raw.get("seed"),
        metrics=load_jsonl(p / "metrics.jsonl"),
        evals=load_jsonl(p / "eval.jsonl"),
        summary=_load_json(p / "summary.json"),
    )


def load_runs(root: str | Path, pattern: str = "*") -> list[Run]:
    """Load every run directory under ``root`` matching ``pattern`` (needs a config.json
    or a metrics/eval file)."""
    runs = []
    for d in sorted(Path(root).glob(pattern)):
        if d.is_dir() and any((d / f).exists() for f in ("config.json", "metrics.jsonl", "eval.jsonl")):
            runs.append(load_run(d))
    return runs


# -------------------------------------------------------------------------- grouping


def group_runs(
    runs: Iterable[Run], keys: Sequence[str] = ("reward_version",)
) -> dict[str, list[Run]]:
    """Group runs by the values of the given (flattened) config keys.

    The label looks like ``reward_version=v2`` (several keys joined by ``, ``);
    groups come out sorted by label and runs sorted by seed.
    """
    groups: dict[str, list[Run]] = {}
    for r in runs:
        label = ", ".join(f"{k}={r.config.get(k, '?')}" for k in keys)
        groups.setdefault(label, []).append(r)
    return {
        k: sorted(v, key=lambda r: (r.seed is None, r.seed if r.seed is not None else 0, r.name))
        for k, v in sorted(groups.items())
    }


def common_mean(
    runs: Sequence[Run], records: str, key: str
) -> tuple[np.ndarray, np.ndarray]:
    """Mean of ``key`` across runs at the global steps every run has reached.

    Only steps present in ALL runs are used, so a run still in progress truncates
    the mean instead of biasing it with fewer seeds.
    """
    series = [r.series(records, key) for r in runs]
    series = [s for s in series if len(s[0])]
    if not series or len(series) != len(runs):
        return np.array([]), np.array([])
    common = set(series[0][0].tolist())
    for xs, _ in series[1:]:
        common &= set(xs.tolist())
    steps = np.asarray(sorted(common))
    if not len(steps):
        return steps, steps
    rows = [np.asarray([dict(zip(xs.tolist(), ys.tolist()))[s] for s in steps]) for xs, ys in series]
    return steps, np.mean(rows, axis=0)


# ------------------------------------------------------------------------- plotting


def _style(plt) -> None:
    plt.rcParams.update(
        {
            "figure.dpi": 100,
            "savefig.dpi": 110,
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "axes.titlesize": 9.5,
            "axes.titleweight": "bold",
            "legend.frameon": False,
            "legend.fontsize": 8,
        }
    )


def _steps_fmt(ax) -> None:
    from matplotlib.ticker import FuncFormatter

    ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v / 1e6:g}M"))
    ax.set_xlabel("env steps")


def _smooth(y: np.ndarray, k: int) -> np.ndarray:
    if k <= 1 or len(y) < 3:
        return y
    k = min(k, len(y))
    pad = np.concatenate([np.full(k - 1, y[0]), y])
    return np.convolve(pad, np.ones(k) / k, mode="valid")


def plot_run(run: Run, out_path: str | Path, smooth: int = 5) -> Path:
    """2x2 figure: eval outcome rates, eval mean reward, entropy, value loss."""
    plt = _plt()
    _style(plt)
    fig, axes = plt.subplots(2, 2, figsize=(9, 6), constrained_layout=True)
    (ax_out, ax_rew), (ax_ent, ax_vl) = axes

    for key, color in OUTCOME_COLORS.items():
        x, y = run.series("evals", key)
        if len(x):
            ax_out.plot(x, y, color=color, lw=1.3, label=OUTCOME_LABELS[key])
    ax_out.set_ylim(-0.02, 1.02)
    ax_out.set_ylabel("fraction of eval episodes")
    ax_out.set_title("Evaluation outcomes (deterministic policy)")
    if ax_out.lines:
        ax_out.legend(loc="center right", frameon=True, framealpha=0.9)

    x, y = run.series("evals", "mean_reward")
    if len(x):
        ax_rew.plot(x, y, color="#333333", lw=1.3)
    ax_rew.set_ylabel("mean episode reward (raw)")
    ax_rew.set_title("Evaluation reward")

    x, y = run.series("metrics", "entropy")
    if len(x):
        ax_ent.plot(x, y, color="#CC79A7", lw=1.2)
    ax_ent.set_ylabel("policy entropy (nats)")
    ax_ent.set_title("Entropy (training)")

    x, y = run.series("metrics", "value_loss")
    if len(x):
        ax_vl.plot(x, y, color="#999999", lw=0.6, alpha=0.7)
        ax_vl.plot(x, _smooth(y, smooth * 4), color="#0072B2", lw=1.3)
        if (y > 0).all():
            ax_vl.set_yscale("log")
    ax_vl.set_ylabel("value loss")
    ax_vl.set_title("Value loss (training)")

    for ax in axes.flat:
        _steps_fmt(ax)

    status = "in progress" if run.in_progress else "finished"
    fin, best = run.final_success, run.best_success
    sub = ""
    if fin is not None:
        sub = f"  |  final eval success {fin:.0%}, best-so-far {best:.0%}"
    fig.suptitle(f"{run.name} ({status}){sub}", fontsize=10.5)
    return _save(fig, out_path, plt)


def _save(fig, out_path: str | Path, plt) -> Path:
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out)
    plt.close(fig)
    return out


def plot_comparison(
    groups: dict[str, list[Run]],
    out_path: str | Path,
    title: str | None = None,
) -> Path:
    """Compare config groups: per-seed thin lines + mean, and a final-success dot strip.

    Left/middle: eval success rate and eval mean reward vs env steps. Thin lines are
    individual seeds, the thick line the mean over seeds (only at steps all seeds
    have reached). Right: for each seed, a filled dot = FINAL eval success, a hollow
    dot = best-so-far (max over its evals). Every seed is drawn. Runs still in
    progress get a diamond marker.
    """
    plt = _plt()
    _style(plt)
    fig, (ax_s, ax_r, ax_d) = plt.subplots(
        1, 3, figsize=(13, 4.4), constrained_layout=True,
        gridspec_kw={"width_ratios": [1.5, 1.5, 1]},
    )
    rng = np.random.default_rng(0)  # deterministic jitter
    labels = list(groups)
    for gi, label in enumerate(labels):
        runs = groups[label]
        color = GROUP_COLORS[gi % len(GROUP_COLORS)]
        for ax, key in ((ax_s, "success_rate"), (ax_r, "mean_reward")):
            for r in runs:
                x, y = r.series("evals", key)
                if len(x):
                    ax.plot(x, y, color=color, lw=0.7, alpha=0.35)
            mx, my = common_mean(runs, "evals", key)
            if len(mx):
                ax.plot(mx, my, color=color, lw=2.2, label=f"{label} (mean of {len(runs)})")

        for r in runs:
            fin, best = r.final_success, r.best_success
            if fin is None:
                continue
            xj = gi + rng.uniform(-0.18, 0.18)
            marker = "D" if r.in_progress else "o"
            ax_d.scatter([xj], [best], s=38, marker=marker, facecolors="none",
                         edgecolors=color, linewidths=1.2, zorder=2)
            ax_d.scatter([xj], [fin], s=38, marker=marker, color=color, zorder=3)
        fins = [r.final_success for r in runs if r.final_success is not None]
        if fins:
            ax_d.hlines(np.mean(fins), gi - 0.3, gi + 0.3, color=color, lw=2.2, zorder=1)

    ax_s.set_ylim(-0.02, 1.02)
    ax_s.set_ylabel("eval success rate")
    ax_s.set_title("Eval success vs env steps (thin: seeds)")
    ax_s.legend(loc="upper left")
    ax_r.set_ylabel("eval mean reward (raw)")
    ax_r.set_title("Eval mean reward vs env steps")
    for ax in (ax_s, ax_r):
        _steps_fmt(ax)

    ax_d.set_xticks(range(len(labels)))
    ax_d.set_xticklabels([l.replace(", ", "\n") for l in labels], fontsize=7.5)
    ax_d.set_xlim(-0.6, len(labels) - 0.4)
    ax_d.set_ylim(-0.04, 1.04)
    ax_d.set_ylabel("eval success rate")
    ax_d.set_title("Each seed: final vs best-so-far")
    ax_d.scatter([], [], color="#555555", s=38, label="final (last eval)")
    ax_d.scatter([], [], facecolors="none", edgecolors="#555555", s=38, label="best-so-far (max over evals)")
    ax_d.plot([], [], color="#555555", lw=2.2, label="mean of final")
    if any(r.in_progress for rs in groups.values() for r in rs):
        ax_d.scatter([], [], marker="D", color="#555555", s=30, label="run in progress")
    ax_d.legend(loc="center", fontsize=7, frameon=True, framealpha=0.9)
    if title:
        fig.suptitle(title, fontsize=10.5)
    return _save(fig, out_path, plt)


def final_table(groups: dict[str, list[Run]]) -> list[dict]:
    """Per-seed rows (group, run, seed, final, best, steps, in_progress) for text reports."""
    rows = []
    for label, runs in groups.items():
        for r in runs:
            e = r.final_eval or {}
            rows.append(
                {
                    "group": label,
                    "run": r.name,
                    "seed": r.seed,
                    "final_success": r.final_success,
                    "best_success": r.best_success,
                    "final_step": e.get("global_step"),
                    "in_progress": r.in_progress,
                }
            )
    return rows

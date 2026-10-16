"""Round-2 summary: clean-seed subset, pooled v3a-vs-v2 check over both seed sets, and the dot-strip figure.

Inputs are the held-out evaluation JSONs (500 episodes, seeds 50000-50499, final.pt, greedy):
  experiments/r2_{control,schedule,v3a,combo}.json   fresh training seeds 10-19 (this round)
  experiments/headline_eval.json                      hl_v2 (control recipe), training seeds 0-9
  experiments/rv3_confirm_eval.json                   v3a, training seeds 0-9

    uv run python scripts/round2_summary.py            # writes experiments/r2_summary.json, docs/figures/round2_arms.png

Pre-registered analysis (docs/stabilisation.md, "Round 2") is in experiments/r2_stats.json
(`stabilisation_analysis.py arms`). This script adds two analyses that were NOT pre-registered and are labelled
as such in the write-up: (1) the seeds not used to choose between reward variants (13-19), because the v3a-vs-v3b
pilot used seeds 10-12; (2) the pooled paired comparison of v3a against the control recipe over all 20 seeds.
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from stabilisation_analysis import boot_paired_ci, mcnemar_exact_p, paired_signflip_p  # noqa: E402

EXP = Path("experiments")
ARMS = {"control": "r2_control.json", "schedule": "r2_schedule.json", "v3a": "r2_v3a.json", "combo": "r2_combo.json"}
LABEL = {"control": "v2 control", "schedule": "entropy schedule", "v3a": "reward v3a", "combo": "schedule + v3a"}
COLOR = {"control": "#7a869a", "schedule": "#e0b457", "v3a": "#3ddc97", "combo": "#8f9bff"}


def seed_of(name: str) -> int:
    return int(re.search(r"s(\d+)$", name).group(1))


def per_seed_greedy(path: Path) -> dict[int, float]:
    out = {}
    for r in json.loads(path.read_text())["runs"]:
        g = r["final"]["policies"]["greedy"] if "final" in r else r["greedy"]
        s = g["success"] if "success" in g else g["landed"]
        out[r.get("train_seed", seed_of(r["run"]))] = float(s["rate"])
    return out


def paired(a: np.ndarray, b: np.ndarray) -> dict:
    d = a - b
    return {"n": len(d), "mean_a": float(a.mean()), "mean_b": float(b.mean()), "mean_diff": float(d.mean()),
            "bootstrap95_ci": boot_paired_ci(d), "signflip_p": paired_signflip_p(d),
            "n_higher": int((d > 1e-9).sum()), "n_lower": int((d < -1e-9).sum()),
            "n_ge90_a": int((a >= 0.9).sum()), "n_ge90_b": int((b >= 0.9).sum()),
            "mcnemar_p_ge90": mcnemar_exact_p(int(((a >= .9) & (b < .9)).sum()), int(((a < .9) & (b >= .9)).sum()))}


def main() -> None:
    r2 = {k: per_seed_greedy(EXP / f) for k, f in ARMS.items()}
    seeds = list(range(10, 20))
    res: dict = {"seeds_10_19": {k: [r2[k][s] for s in seeds] for k in ARMS}}

    clean = list(range(13, 20))  # seeds not used for the v3a-vs-v3b pilot (seeds 10-12)
    res["clean_seeds_13_19"] = {}
    for k in ARMS:
        v = np.array([r2[k][s] for s in clean])
        res["clean_seeds_13_19"][k] = {"mean": float(v.mean()), "n_ge90": int((v >= 0.9).sum()), "n": len(v),
                                       "per_seed": v.tolist()}
    res["clean_seeds_13_19"]["v3a_vs_control"] = paired(np.array([r2["v3a"][s] for s in clean]),
                                                        np.array([r2["control"][s] for s in clean]))

    v3_old = {}
    for r in json.loads((EXP / "rv3_confirm_eval.json").read_text())["runs"]:
        v3_old[r["train_seed"]] = float(r["greedy"]["landed"]["rate"])
    old_seeds = list(range(10))
    hl_v2 = {}
    for r in json.loads((EXP / "headline_eval.json").read_text())["runs"]:
        if r["arm"] == "v2":
            hl_v2[r["seed"]] = float(r["final"]["policies"]["greedy"]["success"]["rate"])
    a = np.array([v3_old[s] for s in old_seeds] + [r2["v3a"][s] for s in seeds])
    b = np.array([hl_v2[s] for s in old_seeds] + [r2["control"][s] for s in seeds])
    res["pooled_20_seeds_v3a_vs_control"] = paired(a, b)
    res["pooled_20_seeds_v3a_vs_control"]["v3a_per_seed"] = a.tolist()
    res["pooled_20_seeds_v3a_vs_control"]["control_per_seed"] = b.tolist()
    res["seeds_0_9"] = {"v3a": [v3_old[s] for s in old_seeds], "control_hl_v2": [hl_v2[s] for s in old_seeds]}
    Path(EXP / "r2_summary.json").write_text(json.dumps(res, indent=1) + "\n")

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9, 4.2))
    order = ["control", "schedule", "v3a", "combo"]
    rng = np.random.default_rng(0)
    for i, k in enumerate(order):
        y = np.array([r2[k][s] for s in seeds]) * 100
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(y)), y, s=46, color=COLOR[k], edgecolor="black", linewidth=.5,
                   zorder=3)
        ax.hlines(y.mean(), i - 0.3, i + 0.3, color="black", linewidth=2, zorder=4)
        ax.text(i, 104, f"{(y >= 90).sum()}/10 at 90%+\nmean {y.mean():.0f}%", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(range(4), [LABEL[k] for k in order])
    ax.set_ylim(-5, 118)
    ax.set_yticks([0, 25, 50, 75, 100])
    ax.set_ylabel("held-out landing success, final checkpoint (%)")
    ax.set_title("Fresh training seeds 10-19: 500 unseen starts per seed (black bar = mean)", fontsize=10)
    ax.grid(axis="y", alpha=.25)
    ax.spines[["top", "right"]].set_visible(False)
    fig.tight_layout()
    Path("docs/figures").mkdir(parents=True, exist_ok=True)
    fig.savefig("docs/figures/round2_arms.png", dpi=130)
    print(json.dumps({k: v for k, v in res.items() if k != "seeds_10_19"}, indent=1)[:2500])


if __name__ == "__main__":
    main()

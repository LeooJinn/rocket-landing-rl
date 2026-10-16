# Documentation index

Start with the [project README](../README.md) for the summary and quickstart. This directory holds the
detail behind it. Every number in the README comes from one of the documents or data files listed here.

## Read in this order

| Document | What it covers |
|---|---|
| [environment.md](environment.md) | The simulator: observation and action spaces, physics, fuel, initial states, landing thresholds, termination, determinism, and the heuristic baseline controller with its 100-episode result. |
| [reward-design.md](reward-design.md) | Reward v1 and v2 term by term, why shaping cannot be farmed by hovering, the known risks of each, and the invariants the tests check. |
| [evaluation.md](evaluation.md) | The held-out evaluation protocol (seed discipline, Wilson intervals, greedy vs stochastic, baselines) and its results for the earlier running-normaliser runs. |
| [experiments.md](experiments.md) | Chronological log: first PPO policy, training curve, seed reliability, what was tried, the free-fall and hover failure modes, reward v1 vs v2 on six seeds, run families and figures, limitations. |
| [normalisation-experiment.md](normalisation-experiment.md) | Running vs fixed observation scales on six seeds (27.6% to 95.0% mean held-out success), and the `clip_vloss` ablation. |
| [headline-results.md](headline-results.md) | Current headline: reward v1 vs v2, fixed-scale normalisation, 10 seeds each, 500 held-out episodes per final checkpoint, statistics and limitations. |
| [viewer.md](viewer.md) | The viewer backend API (`scripts/serve_viewer.py`), following a live training run, and the static (serverless) demo export. |

## Data behind the documents

| File | Used by |
|---|---|
| [heldout_final.json](heldout_final.json), [heldout_best.json](heldout_best.json) | Raw results of [evaluation.md](evaluation.md): `final.pt` (greedy and stochastic) and `best.pt` (greedy) of nine earlier runs. |
| [../experiments/headline_eval.json](../experiments/headline_eval.json) | Raw per-run evaluation of the 20 headline runs ([headline-results.md](headline-results.md)). |
| [../experiments/headline_stats.json](../experiments/headline_stats.json) | Aggregates, bootstrap and permutation statistics of the headline comparison. |
| [../experiments/norm_experiment/norm_eval_A.json](../experiments/norm_experiment/norm_eval_A.json), [norm_eval_BC.json](../experiments/norm_experiment/norm_eval_BC.json) | Raw results of [normalisation-experiment.md](normalisation-experiment.md). |
| [../experiments/configs/](../experiments/configs/) | Every training config referenced in the documents (`smoke.json` is a 3-second sanity config). |

## Figures

| File | Shows | Made by |
|---|---|---|
| [figures/headline_final_strip.png](figures/headline_final_strip.png) | Held-out final-checkpoint success per seed, v1 vs v2, with the last five checkpoints as grey ticks. | `scripts/headline_analysis.py plots` |
| [figures/headline_per_seed_curves.png](figures/headline_per_seed_curves.png) | In-training evaluation success of each of the 20 headline runs. | `scripts/headline_analysis.py plots` |
| [figures/headline_v1_vs_v2.png](figures/headline_v1_vs_v2.png) | Headline comparison curves, mean over seeds, final vs best-so-far. | `scripts/plot_training.py` |
| [figures/v1_vs_v2.png](figures/v1_vs_v2.png) | Earlier six-seed v1 vs v2 comparison (running normaliser). | `scripts/plot_training.py` |
| [figures/run_v2_s0.png](figures/run_v2_s0.png) | A run that reaches 100% evaluation success and collapses back to 0% (running normaliser). | `scripts/plot_training.py` |
| [figures/run_v2_s1.png](figures/run_v2_s1.png) | A run that ends with a good policy (running normaliser). | `scripts/plot_training.py` |
| [demo.gif](demo.gif) | Storyboard preview (7 frames) of the browser viewer in static-demo mode. A smoother recording is planned. | screen capture |

## Conventions

- Success means a landing inside every safe limit on the pad ([environment.md](environment.md)).
- "Final" means the last checkpoint (`final.pt`); "best-so-far" or `best.pt` is selected on the same
  evaluation episodes it is reported on and is optimistic. Headline numbers use final checkpoints on
  held-out seeds (50000 and up) that training never saw.
- The unit of replication is the training seed, not the episode. Wilson intervals describe evaluation
  noise of one policy, not seed-to-seed variation.
- Run directories (`experiments/results/*`) are git-ignored except two small reference runs; the raw
  evaluation JSON files above are committed instead.

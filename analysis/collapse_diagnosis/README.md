# Collapse diagnosis: scripts and raw outputs

Behind [docs/collapse-diagnosis.md](../../docs/collapse-diagnosis.md). Produced by an analysis agent from the 20
headline runs (`hl_v1_s0..s9`, `hl_v2_s0..s9`). Nothing here trains a model. Rollouts use reset seeds 50000 and
up (the headline's held-out seeds), so they are **diagnostic only**, not a checkpoint-selection set.

## What you need

- The 20 run directories (`metrics.jsonl`, `eval.jsonl`, `checkpoints/ckpt_*.pt`, `final.pt`). They are
  git-ignored in this repository (only `hl_v2_s2/final.pt` is committed), so point the scripts at a directory that
  has them, either by editing `RESULTS_DIR` at the top of [`_lib.py`](_lib.py) or with the environment variable
  `COLLAPSE_RESULTS_DIR`:

  ```bash
  export COLLAPSE_RESULTS_DIR=/path/to/experiments/results     # contains hl_v1_s0/ ... hl_v2_s9/
  ```
- `uv run` (torch is needed by the rollout scripts; the numpy forward pass in [`_lib.py`](_lib.py) re-implements
  the actor and critic and uses the repository's `RocketLandingEnv`).
- Scripts that only read `metrics.jsonl` / `eval.jsonl` (the first group below) need no checkpoints and
  run in seconds.

`bash analysis/collapse_diagnosis/run_all.sh fast` regenerates the metrics-only outputs in `raw/`;
`... run_all.sh full` also does every checkpoint rollout. The full set is slow (the greedy-vs-sampled sweep alone
is 10 runs x 25 checkpoints x 200 episodes; it took roughly half an hour on a busy 10-core machine).

## Re-running the four headline claims

Run from the repository root. The expected output is what is committed in `raw/`. The committed files were
regenerated from this directory's scripts against the 20 headline runs, and they match the numbers of the first
analysis exactly (the scripts are deterministic: fixed seeds for the sampler, fixed reset seeds).

1. **Failures are fuel exhaustion.**
   - `uv run python analysis/collapse_diagnosis/fuel_vs_outcome.py` (metrics only). Expect
     `upd>=600 crash>=0.9 evals: fuel_used>=0.99: 558 / 601` and greedy success by fuel-use bin falling from
     94-95% (fuel used 0.4-0.7) to 3% (0.99 and above).
   - `uv run python analysis/collapse_diagnosis/final_failure_modes.py` (final checkpoints). Expect fuel to run out
     in 20 of 20 episodes at step 779-788 for v1 s0, s4-s9 and v2 s6, and at step 690 and 4.2 m altitude for
     v2 s8.
2. **The collapse is greedy-only.**
   - `uv run python analysis/collapse_diagnosis/collapse_window.py hl_v2_s8 1000 1220` (metrics only). Expect
     training success 1.00 in updates 1140-1220 while the greedy eval column is 0.00 at 1140 and 1160-1220.
   - `uv run python analysis/collapse_diagnosis/greedy_vs_sampled.py hl_v2_s8` (about a minute on an idle machine). Expect checkpoint 1200 at
     greedy 0.00 / sampled 0.99 and `final` at 0.00 / 0.97 (100 episodes).
3. **Greedy and sampled are different controllers.**
   `uv run python analysis/collapse_diagnosis/temperature_sweep.py hl_v2_s8@1200 hl_v2_s8@final hl_v2_s7@1100 --tag demo`.
   Expect `hl_v2_s8@1200` landed 0 [1.00] at T=0 and 98 [0.92] at T=1; fuel use at T=1 below fuel use at T=0 on
   every checkpoint.
4. **The argmax is nearly tied.**
   `uv run python analysis/collapse_diagnosis/argmax_margins.py`. Expect 58-64% of steps with a top-1 minus top-2
   probability margin below 0.1 (v2 s1 final: 48%) and, between checkpoints 50 updates apart, a changed greedy
   action in 33-72% of visited states.

## Script index

| Script | Needs checkpoints | Output in `raw/` | Section of the write-up |
|---|---|---|---|
| `eval_curves.py` | no | `eval_curves.txt` | in-training eval traces, first 90% eval per run |
| `training_vs_greedy.py` | no | `training_vs_greedy.txt` | entropy, training vs greedy success, training fuel use |
| `collapse_window.py RUN FIRST LAST [STEP]` | no | `window_*.txt` | 2, 6 |
| `fuel_vs_outcome.py` | no | `fuel_vs_outcome.txt` | 3 |
| `persistence.py` | no | `persistence.txt` | 6, 7 |
| `entropy_fuel_drift.py` | no | `entropy_fuel_drift.txt` | 6 |
| `final_failure_modes.py` | yes | `final_failure_modes.txt` | 3 |
| `fuel_race_trace.py` | yes | `fuel_race_trace.txt` | 3 |
| `hover_rollout.py` | yes | `hover_rollout_hl_v2_s6.txt` | 3 |
| `greedy_vs_sampled.py`, then `summarise_greedy_vs_sampled.py` (reads `raw/`) | yes | `greedy_vs_sampled*.{txt,json}` | 4 |
| `temperature_sweep.py` | yes | `temperature_sweep_t1.{txt,json}` | 4 |
| `altitude_profile.py` | yes | `altitude_profile.txt` | 4 |
| `argmax_margins.py` | yes | `argmax_margins.txt` | 5 |
| `discounted_returns.py` | yes | `discounted_returns.txt` | 6 |
| `advantage_scale.py` | yes | `advantage_scale.txt` | 6 |

Shared helpers (paths, numpy policy, rollout) are in `_lib.py`. Terminology: "collapsed" evaluation means greedy
success below 20% on the 50-episode in-training evaluation; "fuel used" is the fraction of the initial tank
burned by the end of the episode.

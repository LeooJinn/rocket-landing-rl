# Held-out evaluation (milestone 7)

Code: `src/rocket_landing/evaluate.py`; tests: `tests/test_evaluate.py`.

```
uv run python -m rocket_landing.evaluate --run <run_dir|name> --checkpoint final|best|<ckpt id> \
    --episodes 500 [--seed-start 50000] [--stochastic]
uv run python -m rocket_landing.evaluate --runs a b c --checkpoint best --aggregate-out out.json
```

`--results-dir` (default `experiments/results`) resolves bare run names. Each run writes
`<run>/heldout_eval.json` (unless `--no-write`); `--runs` also prints a per-seed table and a
mean/min/max-across-seeds table, and failed seeds (0 successes) are counted, never dropped.

## Seed discipline

* Training env `i` of run seed `s` is reset once with `1_000_000 + s*1000 + i`, then continues its own RNG stream.
* The in-training eval (used to pick `best.pt`) uses `eval_seed_start .. + eval_episodes - 1` (default 20000..20049).
* Held-out default: `50000 .. 50499`. `assert_heldout` checks, from each run's `config.json`, that this range
  is disjoint from that run's training seeds and in-training eval seeds, and that it ends below
  `TRAIN_SEED_BASE`, so it is disjoint from every run's training seeds. Evaluation aborts otherwise.
* Limit: the seed sets are provably disjoint; the initial states are fresh i.i.d. draws from the same
  distribution as training (no distribution shift is tested). `best.pt` is selected on the 20000 range, so
  its in-training numbers are optimistic; the held-out numbers are not.

All policies (greedy, optional stochastic, heuristic baseline, uniform-random) use the same seeds. Stochastic
and random controllers are re-seeded per episode from the episode seed, so results are exactly reproducible.
Reward is the run's own reward version. Std is the sample std (ddof 1); touchdown stats cover episodes that
reached the ground (crashed or landed); success CIs are 95% Wilson intervals.

## Results (500 episodes, seeds 50000..50499, CPU)

Raw JSON: `docs/heldout_final.json` (final.pt, with stochastic) and `docs/heldout_best.json` (best.pt, greedy).
Success rate, greedy policy, with 95% Wilson CI (heuristic is 99.6% [98.6, 99.9] and random 0.0% [0.0, 0.8] on every row):

| run | reward | final.pt greedy | final.pt stochastic | best.pt greedy |
|---|---|---|---|---|
| v2_s0 | v2 | 0.2% [0.0, 1.1] | 90.2% [87.3, 92.5] | 99.8% [98.9, 100] |
| v2_s1 | v2 | 87.0% [83.8, 89.7] | 95.2% [93.0, 96.8] | 97.4% [95.6, 98.5] |
| v2_s2 | v2 | 0.0% [0.0, 0.8] | 78.6% [74.8, 82.0] | 86.2% [82.9, 88.9] |
| v2_s3 | v2 | 0.0% [0.0, 0.8] | 0.0% [0.0, 0.8] | 0.0% [0.0, 0.8] |
| v2_s4 | v2 | 81.6% [78.0, 84.8] | 84.8% [81.4, 87.7] | 99.2% [98.0, 99.7] |
| v2_s5 | v2 | 0.0% [0.0, 0.8] | 0.0% [0.0, 0.8] | 0.0% [0.0, 0.8] |
| r2_A_s1 | v1 | 100.0% [99.2, 100] | 99.0% [97.7, 99.6] | 100.0% [99.2, 100] |
| r2_A_s0 | v1 | 0.0% [0.0, 0.8] | 0.0% [0.0, 0.8] | 25.0% [21.4, 29.0] |
| sweep_g999e03_s1 | v1 | 98.2% [96.6, 99.1] | 86.4% [83.1, 89.1] | 99.8% [98.9, 100] |

Across the 9 runs (mean / min / max success): final greedy 40.8% / 0.0% / 100.0% (4 of 9 runs never land);
final stochastic 59.4% / 0.0% / 99.0% (3 of 9 never land); best greedy 67.5% / 0.0% / 100.0% (2 of 9 never land).
Heuristic 99.6%, random 0.0%, identical across rows because the seeds are identical.

Observations (from these numbers only):

* Seed instability is large: with the same config, v2 final.pt ranges from 0% to 87% greedy success.
* Failure at the end of training is often a greedy-policy collapse, not an absent skill: v2_s0 and v2_s2
  have ~0% greedy but 90% / 79% when sampling, and their best.pt landed 99.8% / 86%. Final checkpoints
  are therefore a poor model-selection default; best.pt (selected on 50 other seeds) held up on 500 fresh seeds.
* v2_s3, v2_s5 and r2_A_s0 (final) are plain failures; their best.pt were early checkpoints (updates 50 / 10 / 570).
* The heuristic baseline (99.6%) is stronger than all but a few trained runs; only r2_A_s1 (100%), v2_s0/s4 best,
  and sweep_g999e03_s1 best match it.
* v1 vs v2 is not a fair comparison here (different run counts and configs); no claim is made about which is better.

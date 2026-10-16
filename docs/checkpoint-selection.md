# Checkpoint selection on a validation seed range

## Summary

Greedy success of consecutive checkpoints of one PPO run swings between ~0% and ~100% (headline runs:
`docs/headline-results.md`), so shipping `final.pt` is a lottery. A saved checkpoint is a *fixed* policy, so
its success on fresh episodes can be measured and the best one kept. `scripts/select_checkpoint.py`
(`src/rocket_landing/selection.py`) evaluates every saved checkpoint greedily on **validation seeds
70000..70199**, applies a selection rule, writes `selected.json` and exports the winner as an ordinary
checkpoint. Everything below is measured on held-out reset seeds the selection never saw.

Numbers (20 existing runs `hl_v1_s0-9`, `hl_v2_s0-9`; success = landed, greedy, 500 episodes, mean over the 10
training seeds of an arm, in brackets the number of seeds with >= 90%). Pool A = the 25 saved checkpoints of a run
(`checkpoints/ckpt_*.pt` every 50 updates + `final.pt`); pool B = pool A + `best.pt`. Rule `stable3` in both.

Held-out test range 50000..50499 (the pre-committed analysis; pool A only):

| arm | final.pt | A: stable3 | best.pt | oracle: best saved checkpoint, chosen on test |
|---|---|---|---|---|
| v2 | 77.0% (7/10) | **90.1% (9/10)** | 90.1% (9/10) | 90.2% (9/10) |
| v1 | 30.0% (3/10) | **45.5% (3/10)** | 71.0% (5/10) | 45.5% (3/10) |

Fresh range 80000..80499 (used once, after every choice was fixed; adds pool B, a post-hoc variant, see Protocol):

| arm | final.pt | A: stable3 | B: stable3 | best.pt |
|---|---|---|---|---|
| v2 | 77.2% (7/10) | 90.1% (9/10) | **90.1% (9/10)** | 90.1% (9/10) |
| v1 | 30.0% (3/10) | 46.3% (3/10) | **71.4% (5/10)** | 71.3% (5/10) |

What this does and does not show, in order of importance:

1. **Validation selection beats `final.pt`, because `final.pt` is a bad default.** v2: +13.1 pp mean, and 9/10
   instead of 7/10 seeds at >= 90%. The gain comes from two seeds (v2 s4: 70.6% -> 100%, v2 s8: 0% -> 99.8%); the other
   seeds are unchanged within 2 pp (v2 s6 0% -> 1.6%). With 10 seeds the paired bootstrap CI of the mean gain is +0.1 to +33.2 pp, i.e.
   suggestive, not conclusive. (v1 is the same story with a smaller gain: 3 seeds improve, none gets worse.)
2. **The "cost of not peeking" is ~0.** The oracle (best saved checkpoint chosen on the test seeds themselves) is
   90.2% for v2 and 45.5% for v1, equal to the validation-selected value to within 0.1 pp. Validation success predicts
   test success of the same checkpoint almost exactly (193 checkpoints evaluated on both: correlation 0.9989, mean
   absolute gap 1.0 pp, max 8.5 pp).
3. **Selection does not fix an unstable training process.** It can only pick a checkpoint that exists. Six of the 20
   runs have no checkpoint above 70% on any seed range (v1 s0, s4, s6, s7, s9 and v2 s6); selection returns the least
   bad one and says so (validation 0-65%).
4. **The existing `best.pt` is as good as the validated selection, and better than selecting among the 50-update
   checkpoints in v1.** In v1 the good policies live in windows shorter than the checkpoint spacing (50 updates), so
   no saved checkpoint is good (v1 s5: best saved checkpoint 46%, `best.pt` 100%). Adding `best.pt` (picked every
   10 updates on the in-training eval seeds) to the candidate pool fixes that. This was added AFTER seeing the test table
   and is therefore confirmed on a second, fresh seed range (see "Protocol").
5. Practical recommendation: keep checkpoints at the in-training-eval cadence (every 10 updates) or at least keep
   `best.pt`; run the selection with `--include-best`; read the validation number in `selected.json` as the
   deployability check (chosen validation >= 90%: 14/20 runs, all >= 97.8% on the fresh test range; the other 6 are
   <= 68.8%).

## Seed ranges

All reset seeds are integers passed to `env.reset(seed=...)`. Ranges are asserted disjoint in code
(`selection.assert_validation_disjoint`, `selection.assert_test_range_fresh`, on top of
`evaluate.assert_heldout`), and the assertions are unit-tested (`tests/test_selection.py`).

| use | seeds | notes |
|---|---|---|
| training envs | `1_000_000 + 1000*seed + i` | never used by evaluation |
| in-training eval (chooses `best.pt`) | 20000..20049 | 50 episodes, every 10 updates |
| **validation** (this document: selection only) | **70000..70199** | 200 episodes per checkpoint; never reported as a result |
| **test** (reported) | **50000..50499** | 500 episodes, the headline test range; read only after selections were written |
| **fresh test 2** (post-hoc variant) | 80000..80499 | 500 episodes, used once, after all selections were fixed |
| reserved | 0..99, 1000..1999, 60000..60299 | baseline test, heuristic tuning, stabilisation pilot comparison |

Validation, test and test 2 are pairwise disjoint, and disjoint from every training and in-training-eval range.
The validation range was moved from the suggested 60000..60199 to 70000..70199 because the stabilisation runs use
60000..60299 to compare configurations.

## Method

**Candidates.** Every `checkpoints/ckpt_*.pt` (every 50 updates, 24 files) plus `final.pt` (update 1220): 25 per run.
Variant B (`--include-best`) adds `best.pt` as a candidate (skipped when it is the same update as a saved
checkpoint). Each candidate is rolled out greedily on the 200 validation seeds (`evaluate.rollout`, the same code
path as the test evaluation); the per-episode outcomes and fuel used are stored in `validation_curve.json`.

**Rules** (`selection.RULES`; each is a pure function of the validation curve, ordered by update):

| rule | definition |
|---|---|
| `best_val` | highest validation success; ties -> latest checkpoint |
| `latest_near_best` | among checkpoints whose Wilson 95% interval overlaps the best checkpoint's (not distinguishable from it), the latest |
| `stable3` (default) | best rolling mean of validation success over 3 consecutive candidates (ties -> latest window); the highest-scoring candidate inside that window (ties -> latest). A lone spike with bad neighbours cannot win against a plateau |
| `max_margin` | among the `latest_near_best` set, the candidate with the largest fuel margin (10th percentile over validation episodes of fuel left at the end). Motivated by the finding that collapsed greedy policies run out of fuel just before touchdown |

**Outputs** (`--out-dir`, never the run directory): `validation_curve.json`, `selected.json` (rule, pool, chosen
candidate with its validation successes/Wilson interval/fuel margin, final.pt's validation, the choice of every rule,
the full validation curve, the seed check, the exported file's sha256) and `selected.pt` (the chosen checkpoint
with the selection recorded under `extra["selection"]`; loads with `PolicyController.from_checkpoint`).

```
uv run python scripts/select_checkpoint.py --run <run dir> --out-dir <dir> --include-best [--rule stable3] [--procs 3]
```

**Cost.** 25 (26 with `best.pt`) x 200 = 5000-5200 rollouts per run; for `hl_v2_s8` (the only run timed) that was 180 CPU-seconds
(104 s wall on 3 processes under a loaded machine), against a mean of 6.5 minutes of training per run; the in-training
eval already spends a comparable 6100 episodes per run (122 evals x 50). Cached curves make adding `best.pt` or another rule free. There is no inference cost: the deployed
policy is the same 64x64 network.

## Protocol: what was fixed when (no test leakage in the primary result)

1. Rules and `DEFAULT_RULE = "stable3"` were committed (`4dc924e`) together with the validation curves of all 20 runs and
   BEFORE any checkpoint was rolled out on seeds 50000..50499. The only evidence available for the default was a
   validation-only split-half check (`experiments/selection/rules_validation.json`: select on 100 validation
   episodes, score on the other 100 and swap). It could not separate the rules: `best_val` 68.2%, `stable3` 68.2%,
   `latest_near_best` 67.5%, `max_margin` 67.3%, `final.pt` 53.3% on the other half, and the optimism (selecting
   half minus other half) is 0.0 pp for all because both halves pick the same checkpoint in nearly every run. So
   `stable3` was kept as the default on the design argument (a chosen checkpoint should have good neighbours), not
   because it measured better. It did not measure better, see below.
2. The saved-checkpoint pool was then evaluated on test 50000..50499 (500 episodes, every checkpoint with at least one
   validation success, plus every chosen checkpoint, `final.pt` and `best.pt`). Checkpoints with 0/200 validation
   successes were skipped to save compute; this was verified afterwards by rolling out all 307 skipped checkpoints on
   100 test episodes: at most 1/100 succeeded on any of them (`experiments/selection/prune_check.json`).
   The `final.pt` and last-5-checkpoint counts reproduce `experiments/headline_eval.json` exactly (77 counts compared,
   0 mismatches).
3. **The test table showed that `best.pt` is far better than every saved checkpoint in five v1 runs.** Variant B (add
   `best.pt` to the pool) is a design change made after seeing that, so it is NOT evaluated on the range that
   motivated it. Its validation curves and choices were written and committed (`8285fa0`) and only then rolled out
   on the fresh range 80000..80499; the sha256 of the `selected.json` files used is stored in each `test2_curve.json`.

## Results: saved-checkpoint pool on the held-out test range (seeds 50000-50499)

Each cell: test success, 95% Wilson CI over the 500 episodes (evaluation noise only, not seed-to-seed variation),
`@` = update of the chosen checkpoint. `final.pt` is update 1220. Columns 3-6 choose by validation only.

**hl_v2**

| seed | final.pt | best.pt (in-training eval) | **stable3** | best_val | latest_near_best | max_margin | oracle (best saved ckpt on test) |
|---|---|---|---|---|---|---|---|
| 0 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1000 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1000 | 100.0% [99.2-100.0] @1220 |
| 1 | 99.8% [98.9-100.0] @1220 | 100.0% [99.2-100.0] @880 | 99.8% [98.9-100.0] @1100 | 99.8% [98.9-100.0] @1220 | 99.8% [98.9-100.0] @1220 | 99.8% [98.9-100.0] @1100 | 100.0% [99.2-100.0] @1200 |
| 2 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1050 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 99.2% [98.0-99.7] @600 | 100.0% [99.2-100.0] @1220 |
| 3 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @830 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1200 | 100.0% [99.2-100.0] @1220 |
| 4 | 70.6% [66.5-74.4] @1220 | 100.0% [99.2-100.0] @1080 | 100.0% [99.2-100.0] @1100 | 100.0% [99.2-100.0] @1100 | 100.0% [99.2-100.0] @1100 | 100.0% [99.2-100.0] @1000 | 100.0% [99.2-100.0] @1100 |
| 5 | 99.8% [98.9-100.0] @1220 | 100.0% [99.2-100.0] @910 | 100.0% [99.2-100.0] @950 | 99.8% [98.9-100.0] @1220 | 99.8% [98.9-100.0] @1220 | 100.0% [99.2-100.0] @950 | 100.0% [99.2-100.0] @950 |
| 6 | 0.0% [0.0-0.8] @1220 | 1.6% [0.8-3.1] @100 | 1.6% [0.8-3.1] @100 | 1.6% [0.8-3.1] @100 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 1.6% [0.8-3.1] @100 |
| 7 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @930 | 100.0% [99.2-100.0] @800 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1050 | 100.0% [99.2-100.0] @1220 |
| 8 | 0.0% [0.0-0.8] @1220 | 100.0% [99.2-100.0] @870 | 99.8% [98.9-100.0] @1150 | 99.8% [98.9-100.0] @1150 | 99.8% [98.9-100.0] @1150 | 100.0% [99.2-100.0] @1050 | 100.0% [99.2-100.0] @1100 |
| 9 | 100.0% [99.2-100.0] @1220 | 99.4% [98.3-99.8] @1090 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 99.2% [98.0-99.7] @1100 | 100.0% [99.2-100.0] @1220 |

| method | mean | median | min | max | seeds >= 90% | seeds >= 50% | seeds at 0% |
|---|---|---|---|---|---|---|---|
| final.pt | 77.0% | 99.9% | 0.0% | 100.0% | 7/10 | 8/10 | 2/10 |
| best.pt (in-training eval) | 90.1% | 100.0% | 1.6% | 100.0% | 9/10 | 9/10 | 0/10 |
| **stable3** | 90.1% | 100.0% | 1.6% | 100.0% | 9/10 | 9/10 | 0/10 |
| best_val | 90.1% | 100.0% | 1.6% | 100.0% | 9/10 | 9/10 | 0/10 |
| latest_near_best | 89.9% | 100.0% | 0.0% | 100.0% | 9/10 | 9/10 | 1/10 |
| max_margin | 89.8% | 100.0% | 0.0% | 100.0% | 9/10 | 9/10 | 1/10 |
| oracle (best saved ckpt on test) | 90.2% | 100.0% | 1.6% | 100.0% | 9/10 | 9/10 | 0/10 |

| paired vs final.pt | mean diff (pp) | bootstrap 95% CI over seeds | seeds better | worse | equal (+-1 pp) |
|---|---|---|---|---|---|
| best.pt (in-training eval) | +13.1 | +0.0 to +33.3 | 3 | 0 | 7 |
| **stable3** | +13.1 | +0.1 to +33.2 | 3 | 0 | 7 |
| best_val | +13.1 | +0.0 to +33.2 | 3 | 0 | 7 |
| latest_near_best | +12.9 | +0.0 to +32.9 | 2 | 0 | 8 |
| max_margin | +12.8 | -0.3 to +32.9 | 2 | 0 | 8 |
| oracle (best saved ckpt on test) | +13.1 | +0.1 to +33.3 | 3 | 0 | 7 |

**hl_v1**

| seed | final.pt | best.pt (in-training eval) | **stable3** | best_val | latest_near_best | max_margin | oracle (best saved ckpt on test) |
|---|---|---|---|---|---|---|---|
| 0 | 0.0% [0.0-0.8] @1220 | 63.4% [59.1-67.5] @1050 | 63.4% [59.1-67.5] @1050 | 63.4% [59.1-67.5] @1050 | 63.4% [59.1-67.5] @1050 | 63.4% [59.1-67.5] @1050 | 63.4% [59.1-67.5] @1050 |
| 1 | 100.0% [99.2-100.0] @1220 | 98.6% [97.1-99.3] @850 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 98.6% [97.1-99.3] @850 | 100.0% [99.2-100.0] @1220 |
| 2 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1130 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @950 | 100.0% [99.2-100.0] @1220 |
| 3 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @760 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 99.2% [98.0-99.7] @800 | 100.0% [99.2-100.0] @1220 |
| 4 | 0.0% [0.0-0.8] @1220 | 1.4% [0.7-2.9] @190 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @50 | 0.0% [0.0-0.8] @1220 |
| 5 | 0.0% [0.0-0.8] @1220 | 100.0% [99.2-100.0] @620 | 46.0% [41.7-50.4] @650 | 46.0% [41.7-50.4] @650 | 46.0% [41.7-50.4] @650 | 46.0% [41.7-50.4] @650 | 46.0% [41.7-50.4] @650 |
| 6 | 0.0% [0.0-0.8] @1220 | 40.8% [36.6-45.2] @380 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 |
| 7 | 0.0% [0.0-0.8] @1220 | 42.0% [37.8-46.4] @240 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @50 | 0.0% [0.0-0.8] @1220 |
| 8 | 0.0% [0.0-0.8] @1220 | 97.2% [95.4-98.3] @590 | 45.0% [40.7-49.4] @600 | 45.0% [40.7-49.4] @600 | 36.0% [31.9-40.3] @650 | 36.0% [31.9-40.3] @650 | 45.0% [40.7-49.4] @600 |
| 9 | 0.0% [0.0-0.8] @1220 | 66.4% [62.1-70.4] @370 | 0.6% [0.2-1.7] @450 | 0.6% [0.2-1.7] @450 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @50 | 0.6% [0.2-1.7] @450 |

| method | mean | median | min | max | seeds >= 90% | seeds >= 50% | seeds at 0% |
|---|---|---|---|---|---|---|---|
| final.pt | 30.0% | 0.0% | 0.0% | 100.0% | 3/10 | 3/10 | 7/10 |
| best.pt (in-training eval) | 71.0% | 81.8% | 1.4% | 100.0% | 5/10 | 7/10 | 0/10 |
| **stable3** | 45.5% | 45.5% | 0.0% | 100.0% | 3/10 | 4/10 | 3/10 |
| best_val | 45.5% | 45.5% | 0.0% | 100.0% | 3/10 | 4/10 | 3/10 |
| latest_near_best | 44.5% | 41.0% | 0.0% | 100.0% | 3/10 | 4/10 | 4/10 |
| max_margin | 44.3% | 41.0% | 0.0% | 100.0% | 3/10 | 4/10 | 4/10 |
| oracle (best saved ckpt on test) | 45.5% | 45.5% | 0.0% | 100.0% | 3/10 | 4/10 | 3/10 |

| paired vs final.pt | mean diff (pp) | bootstrap 95% CI over seeds | seeds better | worse | equal (+-1 pp) |
|---|---|---|---|---|---|
| best.pt (in-training eval) | +41.0 | +18.1 to +64.8 | 7 | 1 | 2 |
| **stable3** | +15.5 | +0.2 to +31.0 | 3 | 0 | 7 |
| best_val | +15.5 | +0.2 to +31.0 | 3 | 0 | 7 |
| latest_near_best | +14.5 | +0.0 to +29.9 | 3 | 0 | 7 |
| max_margin | +14.3 | -0.1 to +29.9 | 3 | 1 | 6 |
| oracle (best saved ckpt on test) | +15.5 | +0.2 to +31.0 | 3 | 0 | 7 |

Reading the tables:

- v2: all four rules end at 89.8-90.1% mean; the differences are one or two seeds (v2 s6, where nothing is good, and
  1-pp rounding of the chosen checkpoint). `max_margin` picks earlier checkpoints that are slightly worse on test in
  4 runs (v1 s1 98.6%, v1 s3 99.2%, v2 s2 99.2%, v2 s9 99.2%, against 100%), no gain. The rules are indistinguishable here; `stable3` is the default because
  it was pre-committed and costs nothing, not because it won.
- Cost of not peeking: oracle minus `stable3` is 0.1 pp (v2) and 0.0 pp (v1) in the mean; per seed it is at most
  0.2 pp (v2 s1, s8: 100% vs 99.8%) apart from v2 s6 (1.6% vs 1.6%). The oracle is itself a max over noisy
  estimates, so it is optimistic by construction.
- Validation vs test of the chosen checkpoint, over the 4 rules x 20 runs: mean val - test = +0.3 to +0.4 pp (the
  slight positive sign is the expected winner's curse), largest gap 8.5 pp (v1 s5 @650: 54.5% validation, 46.0% test:
  about 2 standard errors of the difference, consistent with sampling noise plus selection among 25 candidates). No chosen checkpoint is a validation spike that collapses on test.

## Selected checkpoints that are still bad, and why

Six runs stay below 70% on test even with `best.pt` in the pool; two more are bad with saved checkpoints only.

| run | chosen (variant B) | test 80000-80499 | why |
|---|---|---|---|
| v1 s4 | `final` (all 25 at 0/200 validation) | 0.0% | never learned; in-training eval peaks at 4% |
| v2 s6 | `ckpt_00100` (6/200) | 1.6% | never learned (best.pt also 1.6%); the chosen candidate burns the whole tank in the 98% of episodes that crash (fuel margin 0.00) |
| v1 s0 | `ckpt_01050` (61.5% validation) | 62.8% | partial policy: 37% of episodes crash WITH fuel left (fuel margin 0.23), a different failure than the fuel-exhaustion collapse; in-training eval peaks at 0.66 and the best test result of any candidate is 63.4% |
| v1 s6, v1 s7 | `best` (43.0%, 38.5% validation) | 43.0%, 41.8% | partial policies from a single in-training eval peak (0.42, 0.38); fuel left at the end, so not the fuel-out collapse |
| v1 s9 | `best` (65.0% validation) | 68.8% | partial policy (in-training peak 0.86 once, update 370); saved checkpoints are at 0-1% |
| v1 s5, v1 s8 | saved checkpoints only: `ckpt_00650` (46%) and `ckpt_00600` (45%); with `best.pt`: 100% and 97.8% | see tables | the good policy exists only in a window shorter than 50 updates (s5: updates 620-640 in the in-training eval; s8: isolated single evals at 540, 590, 630). The 50-update grid misses it. Not a fragile argmax: `best.pt` validates at 200/200 and 192/200 and holds on test |

Over the 160 (run, rule, pool) choices on the fresh range, validation and test success of the chosen checkpoint differ
by at most 5.4 pp, and the only choices where validation exceeds test by more than 5 pp are the four rules'
identical pick in v1 s5 (5.3 pp). So the failure mode of a validation spike that collapses on test (a fragile argmax) did
not occur in this data. The remaining failures are
(a) no good policy was ever produced, or (b) the only good policies fall between saved checkpoints.

Validation also works as a deployability check without touching test: for variant B `stable3`, 14 of 20 chosen
checkpoints validate at >= 90% and all 14 score >= 97.8% on the fresh test range; the other 6 validate at 0-65% and
score 0-68.8%. (The 90% threshold is a descriptive cut, chosen after the fact; the gap between 68.8% and 97.8% is what
matters.)

## Results: adding `best.pt` to the pool, on the FRESH range (seeds 80000-80499)

A = pool of saved checkpoints (same choices as above), B = saved checkpoints + `best.pt`. All choices were written
before this range was rolled out.

**hl_v2**

| seed | final.pt | best.pt | A: stable3 (saved ckpts) | **B: stable3** (saved + best.pt) | B: best_val | B: latest_near_best | B: max_margin |
|---|---|---|---|---|---|---|---|
| 0 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1000 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1000 |
| 1 | 100.0% [99.2-100.0] @1220 | 99.8% [98.9-100.0] @880 | 100.0% [99.2-100.0] @1100 | 100.0% [99.2-100.0] @1100 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1100 |
| 2 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1050 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 99.0% [97.7-99.6] @600 |
| 3 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @830 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1200 |
| 4 | 72.4% [68.3-76.1] @1220 | 100.0% [99.2-100.0] @1080 | 100.0% [99.2-100.0] @1100 | 100.0% [99.2-100.0] @1100 | 100.0% [99.2-100.0] @1100 | 100.0% [99.2-100.0] @1100 | 100.0% [99.2-100.0] @1080 |
| 5 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @910 | 100.0% [99.2-100.0] @950 | 100.0% [99.2-100.0] @950 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @950 |
| 6 | 0.0% [0.0-0.8] @1220 | 1.6% [0.8-3.1] @100 | 1.6% [0.8-3.1] @100 | 1.6% [0.8-3.1] @100 | 1.6% [0.8-3.1] @100 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 |
| 7 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @930 | 100.0% [99.2-100.0] @800 | 100.0% [99.2-100.0] @800 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @930 |
| 8 | 0.0% [0.0-0.8] @1220 | 100.0% [99.2-100.0] @870 | 99.6% [98.6-99.9] @1150 | 99.6% [98.6-99.9] @1150 | 99.6% [98.6-99.9] @1150 | 99.6% [98.6-99.9] @1150 | 100.0% [99.2-100.0] @870 |
| 9 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1090 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1050 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1090 |

| method | mean | median | min | max | seeds >= 90% | seeds >= 50% | seeds at 0% |
|---|---|---|---|---|---|---|---|
| final.pt | 77.2% | 100.0% | 0.0% | 100.0% | 7/10 | 8/10 | 2/10 |
| best.pt | 90.1% | 100.0% | 1.6% | 100.0% | 9/10 | 9/10 | 0/10 |
| A:best_val | 90.1% | 100.0% | 1.6% | 100.0% | 9/10 | 9/10 | 0/10 |
| A:latest_near_best | 90.0% | 100.0% | 0.0% | 100.0% | 9/10 | 9/10 | 1/10 |
| A: stable3 (saved ckpts) | 90.1% | 100.0% | 1.6% | 100.0% | 9/10 | 9/10 | 0/10 |
| A:max_margin | 89.9% | 100.0% | 0.0% | 100.0% | 9/10 | 9/10 | 1/10 |
| B: best_val | 90.1% | 100.0% | 1.6% | 100.0% | 9/10 | 9/10 | 0/10 |
| B: latest_near_best | 90.0% | 100.0% | 0.0% | 100.0% | 9/10 | 9/10 | 1/10 |
| **B: stable3** (saved + best.pt) | 90.1% | 100.0% | 1.6% | 100.0% | 9/10 | 9/10 | 0/10 |
| B: max_margin | 89.9% | 100.0% | 0.0% | 100.0% | 9/10 | 9/10 | 1/10 |

**hl_v1**

| seed | final.pt | best.pt | A: stable3 (saved ckpts) | **B: stable3** (saved + best.pt) | B: best_val | B: latest_near_best | B: max_margin |
|---|---|---|---|---|---|---|---|
| 0 | 0.0% [0.0-0.8] @1220 | 62.8% [58.5-66.9] @1050 | 62.8% [58.5-66.9] @1050 | 62.8% [58.5-66.9] @1050 | 62.8% [58.5-66.9] @1050 | 62.8% [58.5-66.9] @1050 | 62.8% [58.5-66.9] @1050 |
| 1 | 100.0% [99.2-100.0] @1220 | 98.6% [97.1-99.3] @850 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 98.6% [97.1-99.3] @850 |
| 2 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1130 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @950 |
| 3 | 100.0% [99.2-100.0] @1220 | 99.8% [98.9-100.0] @760 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 100.0% [99.2-100.0] @1220 | 98.8% [97.4-99.4] @800 |
| 4 | 0.0% [0.0-0.8] @1220 | 0.8% [0.3-2.0] @190 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @1220 | 0.0% [0.0-0.8] @50 |
| 5 | 0.0% [0.0-0.8] @1220 | 100.0% [99.2-100.0] @620 | 49.2% [44.8-53.6] @650 | 100.0% [99.2-100.0] @620 | 100.0% [99.2-100.0] @620 | 100.0% [99.2-100.0] @620 | 100.0% [99.2-100.0] @620 |
| 6 | 0.0% [0.0-0.8] @1220 | 43.0% [38.7-47.4] @380 | 0.0% [0.0-0.8] @1220 | 43.0% [38.7-47.4] @380 | 43.0% [38.7-47.4] @380 | 43.0% [38.7-47.4] @380 | 43.0% [38.7-47.4] @380 |
| 7 | 0.0% [0.0-0.8] @1220 | 41.8% [37.6-46.2] @240 | 0.0% [0.0-0.8] @1220 | 41.8% [37.6-46.2] @240 | 41.8% [37.6-46.2] @240 | 41.8% [37.6-46.2] @240 | 41.8% [37.6-46.2] @240 |
| 8 | 0.0% [0.0-0.8] @1220 | 97.8% [96.1-98.8] @590 | 49.4% [45.0-53.8] @600 | 97.8% [96.1-98.8] @590 | 97.8% [96.1-98.8] @590 | 97.8% [96.1-98.8] @590 | 97.8% [96.1-98.8] @590 |
| 9 | 0.0% [0.0-0.8] @1220 | 68.8% [64.6-72.7] @370 | 1.4% [0.7-2.9] @450 | 68.8% [64.6-72.7] @370 | 68.8% [64.6-72.7] @370 | 68.8% [64.6-72.7] @370 | 68.8% [64.6-72.7] @370 |

| method | mean | median | min | max | seeds >= 90% | seeds >= 50% | seeds at 0% |
|---|---|---|---|---|---|---|---|
| final.pt | 30.0% | 0.0% | 0.0% | 100.0% | 3/10 | 3/10 | 7/10 |
| best.pt | 71.3% | 83.3% | 0.8% | 100.0% | 5/10 | 7/10 | 0/10 |
| A:best_val | 46.3% | 49.3% | 0.0% | 100.0% | 3/10 | 4/10 | 3/10 |
| A:latest_near_best | 45.1% | 44.3% | 0.0% | 100.0% | 3/10 | 4/10 | 4/10 |
| A: stable3 (saved ckpts) | 46.3% | 49.3% | 0.0% | 100.0% | 3/10 | 4/10 | 3/10 |
| A:max_margin | 44.9% | 44.3% | 0.0% | 100.0% | 3/10 | 4/10 | 4/10 |
| B: best_val | 71.4% | 83.3% | 0.0% | 100.0% | 5/10 | 7/10 | 1/10 |
| B: latest_near_best | 71.4% | 83.3% | 0.0% | 100.0% | 5/10 | 7/10 | 1/10 |
| **B: stable3** (saved + best.pt) | 71.4% | 83.3% | 0.0% | 100.0% | 5/10 | 7/10 | 1/10 |
| B: max_margin | 71.2% | 83.3% | 0.0% | 100.0% | 5/10 | 7/10 | 1/10 |

- v2 is unchanged by adding `best.pt` (the saved checkpoints already contain a good one in 9/10 runs); v1 goes from
  46.3% to 71.4% mean, seeds >= 90% from 3 to 5 (s5, s8 recovered), and 3 more seeds (s6, s7, s9) move from 0-1% to
  43-69% partial policies.
- B-selection and `best.pt` alone are within 0.1 pp in the mean and within 1.4 pp in every run, in both directions
  (largest: v1 s1, `best.pt` 98.6% vs 100% for the validated pick; v2 s8, `best.pt` 100% vs 99.6%). Re-validating
  `best.pt` on 200 independent seeds guards against `best.pt`'s 50-episode selection noise and tells you whether the
  chosen policy is deployable, but it did not change the aggregate here. The honest summary is "use `best.pt` or a validated checkpoint,
  never `final.pt`", and the validation step is what turns that into a measured, auditable number.

## Caveats

- **Selection only helps if a good checkpoint exists, and exists on the candidate grid.** It does not fix an
  unstable training process: runs that never learned stay at 0-69%; the v2 instability is hidden, not removed: after a v2 run first reaches >= 90% validation success, 13% of its later
  saved checkpoints (14/106, 9 runs) are below 20% and 25% are below 90%. Fixing the process is a training
  problem (entropy/LR schedules), not a selection problem.
- **The candidate grid matters more than the rule.** Greedy success flips between checkpoints only 10 updates apart, so
  checkpoints behave like nearly independent draws. A 50-update grid samples one in five of the 10-update in-training
  evals; none of the good in-training evals of v1 s5 (3) and v1 s8 (3) falls on a multiple of 50.
- **The validation number is an unbiased estimate for a checkpoint chosen WITHOUT looking at it; for the chosen one it is
  slightly optimistic** (winner's curse over 25-26 candidates sharing the same 200 episodes). Measured: +0.3 to
  +0.4 pp on 20 runs. Report test (or a second validation set) for the chosen policy, never the validation number.
- **Cost:** about 45% extra compute on top of the training run for 5000 validation rollouts per run (measured on one run;
  comparable to what the in-training eval already spends), zero inference cost. Storing checkpoints every 10 updates is ~5 MB per run.
- **Distribution:** validation, test and training episodes share one initial-state distribution; nothing here tests
  a shifted distribution, dynamics noise or different gravity. A policy with 5% fuel left at touchdown (e.g. v2 s8 @1150)
  validates and tests at ~100% but may be brittle under perturbation; `max_margin` is a hedge for that but it was only
  tested on i.i.d. episodes, where it gives up 0.2-1.4 pp for no measured gain.
- **Statistics:** n = 10 training seeds per arm and 2 arms; the per-seed Wilson intervals describe episode sampling
  noise, not seed variation. Gains are driven by 2-3 seeds per arm. The v1 vs v2 comparison is untouched by this
  work (each arm is selected independently).
- **Not independent of the headline run set:** the 20 runs were the ones whose swings motivated this work; the
  variant-B finding was designed after seeing their test table and is confirmed on a fresh range, but on the same
  checkpoints. A new set of runs (the stabilised configs) is the real out-of-sample check.
- **Oracle** = best of the evaluated saved checkpoints on the same test episodes; it is optimistic by construction.
  Checkpoints with 0/200 validation successes were not rolled out on test (max 1/100 each when checked).
- Greedy policy only; one training run per seed; reward v2 and v1 with fixed normalisation only.

## Reproduction

```
R=experiments/results   # directory holding the 20 hl_* runs (read only)
uv run python scripts/selection_experiment.py validate --results-dir $R --out experiments/selection --procs 3
uv run python scripts/selection_experiment.py rules    --results-dir $R --out experiments/selection
uv run python scripts/selection_experiment.py test     --results-dir $R --out experiments/selection --prune-zero-val
uv run python scripts/selection_experiment.py report   --results-dir $R --out experiments/selection
uv run python scripts/selection_experiment.py verify-prune --results-dir $R --out experiments/selection --procs 2
# variant B (copy the validation curves in first so only best.pt is evaluated)
mkdir -p experiments/selection_b; for d in experiments/selection/hl_*; do mkdir -p experiments/selection_b/$(basename $d); cp $d/validation_curve.json experiments/selection_b/$(basename $d)/; done
uv run python scripts/selection_experiment.py validate --include-best --results-dir $R --out experiments/selection_b
uv run python scripts/selection_experiment.py test2 --results-dir $R --out experiments/selection --out-b experiments/selection_b
uv run python scripts/selection_experiment.py report2 --results-dir $R --out experiments/selection --out-b experiments/selection_b
```

Committed results: `experiments/selection/<run>/{validation_curve,selected,test_curve}.json`,
`experiments/selection/{rules_validation,prune_check,summary}.json`, `experiments/selection_b/<run>/{validation_curve,
selected,test2_curve}.json`, `experiments/selection_b/summary_test2.json`. The exported `selected.pt` files and the run
checkpoints are not committed (`*.pt` is ignored). Tests: `tests/test_selection.py` (rule logic on synthetic curves,
disjointness assertions, `selected.json` schema, export loads via `PolicyController`, process-pool equivalence,
incremental cache, CLI).

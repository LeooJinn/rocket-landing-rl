# Browser viewer backend

`scripts/serve_viewer.py` serves `frontend/` at `/` and a JSON API over an
experiment results directory. Standard library only (plus the project's own
code; rolling out checkpoints needs the `torch` extra).

```
uv sync --extra torch
uv run python scripts/serve_viewer.py --results-dir ~/projects/rocket-landing-rl/experiments/results --port 8000
```

It binds to 127.0.0.1 only. Open http://127.0.0.1:8000/.

## API

- `GET /api/runs` - list of runs, newest-modified first: `name, reward_version,
  seed, updates, total_timesteps, global_step, finished` (summary.json exists),
  `last_eval` (success/crash/out_of_bounds/timeout rates, mean_reward, update) and `mtime`.
- `GET /api/runs/<name>/metrics` - `{train, eval, config}` (rows of
  metrics.jsonl / eval.jsonl, and config.json or null).
- `GET /api/runs/<name>/checkpoints` - `[{id, update, mtime}]` with ids
  `latest` (newest `ckpt_XXXXX` by update), `best`, `final`, `ckpt_00150`, ...
- `GET /api/runs/<name>/episode?seed=20000&checkpoint=latest` - a greedy
  (argmax) rollout as a trajectory. 404 `{"error": ...}` if no checkpoint exists yet.
- `GET /api/baseline/episode?seed=N` - rollout of the heuristic controller
  (`meta.checkpoint = "baseline"`).

Trajectory: `meta` (run, checkpoint, update, seed, reward_version, dt, world,
limits taken from `EnvConfig`), `outcome`, `total_reward` and `frames`
(`t, x, y, vx, vy, angle, av, fuel, action, reward, cum_reward`). Frame 0 is
the initial state with action -1 and reward 0. Rewards are the raw environment
rewards (not the training `reward_scale`).

Errors are JSON `{"error": ...}`: 400 for a bad seed or checkpoint id, 404 for
an unknown run or missing checkpoint. Run names are matched against the actual
directory listing, so path traversal is rejected; static files are confined to `frontend/`.

## Following a live training run

Nothing needs to be told about a training run: every request re-reads
`metrics.jsonl`, `eval.jsonl`, `config.json` and the checkpoint directory, and
a partially written last jsonl line is skipped. The frontend should poll
`/api/runs/<name>/metrics` and `/checkpoints`; `finished` flips to true when
training writes `summary.json`. Requesting `checkpoint=latest` picks the newest
`ckpt_XXXXX.pt` at that moment, so each new periodic checkpoint shows the
current policy flying.

Rolled-out episodes are cached in memory keyed by (run, checkpoint path, file
mtime, seed), so a checkpoint that is rewritten is re-rolled; the cache is not
bounded and resets on restart. Reward version is read from the run's `config.json` (default `v1`).

## Static demo (no server, e.g. GitHub Pages)

`frontend/demo/` holds a pre-computed recording of one finished run, so the
viewer works as a plain static site. Re-export (the one command):

```
uv run python scripts/export_static_demo.py ~/projects/rocket-landing-rl-norm/experiments/results/exp_fixed_s0
```

Options: `--out DIR` (default `frontend/demo`), `--seeds 10 --first-seed 20000`,
`--fractions 0.1 0.4` (earlier checkpoints, as a share of the final update;
`final` is always exported). The export wipes and rewrites the output dir,
is deterministic, and is about 3.2 MB for 10 seeds x 3 checkpoints + baseline
(it compresses well under gzip). Files mirror the API: `runs.json`,
`metrics_<run>.json` (train rows thinned to ~150), `checkpoints_<run>.json`,
`episode_<run>__<ckpt>__<seed>.json`, `baseline_<seed>.json`, plus `manifest.json`.

Serve it with any static server from `frontend/`, for example
`cd frontend && python3 -m http.server`. All URLs are relative, so a sub-path
works. The page uses the static data when opened with `?static=1`, or
automatically when `/api/runs` is unreachable and `demo/manifest.json` exists
(`?static=0` disables the fallback). In static mode the UI is labelled as a
recorded run, the seed picker is limited to the exported seeds, the checkpoint
list to the exported checkpoints, and "Replay training progress" plays the
exported checkpoints in order (early crashes to landing). The live API mode is unchanged.

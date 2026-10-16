# Browser viewer backend

`scripts/serve_viewer.py` serves `frontend/` at `/` and a JSON API over an
experiment results directory. Standard library only (plus the project's own
code; rolling out checkpoints needs the `torch` extra).

```
uv sync --extra torch
uv run python scripts/serve_viewer.py --results-dir experiments/results --port 8000
```

It binds to 127.0.0.1 only. Open http://127.0.0.1:8000/. `experiments/results/` in this repository
contains two runs: `ppo_v1_tuned_s1` (complete) and `hl_v2_s2` (final checkpoint only), so the viewer
lists them and can roll out their `final` (and, for the first, `best`) checkpoints. Point `--results-dir`
at any directory of runs written by `rocket_landing.train` to browse other runs.

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
viewer works as a plain static site. The committed recording was exported from run `exp_fixed_s0` (reward v2, fixed-scale normalisation,
seed 0; that run directory is not in this repository). Re-export from any run directory with:

```
uv run python scripts/export_static_demo.py RUN_DIR [--out DIR]
```

Without `--out` this overwrites `frontend/demo/`. For example, `RUN_DIR=experiments/results/hl_v2_s2`
(final checkpoint only) exports 8 files, 0.20 MB, for 2 seeds.

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

## Publishing the static demo to GitHub Pages

`frontend/` is a self-contained static site: with no `/api` it falls back to the recorded demo in
`frontend/demo/`, using relative URLs only, so it works under the `/rocket-landing-rl/` sub-path that Pages uses.
The workflow `.github/workflows/pages.yml` deploys `frontend/` on every push to `main` that touches it (or by
hand from the Actions tab).

One-time setup (a repository setting, so it is deliberate): Settings -> Pages -> Build and deployment ->
Source: **GitHub Actions**. From the command line:

```
gh api -X POST repos/<owner>/<repo>/pages -f build_type=workflow
```

The site is then served at `https://<owner>.github.io/<repo>/`. To check it locally the way Pages serves it, put
`frontend/` under a sub-path and serve the parent: `mkdir site && ln -s ../frontend site/rocket-landing-rl &&
python3 -m http.server --directory site`, then open `http://localhost:8000/rocket-landing-rl/`.

Refresh the demo data from a better run with `scripts/export_static_demo.py` (see above) and re-render the README
animation with `scripts/render_demo_gif.py`.

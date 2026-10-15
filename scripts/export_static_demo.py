"""Export a trained run as static JSON for the serverless viewer (frontend/demo/).

Usage:
    uv run python scripts/export_static_demo.py RUN_DIR [--out frontend/demo]
        [--seeds 10] [--first-seed 20000] [--fractions 0.1 0.4]

The files mirror the /api contract of scripts/serve_viewer.py (same payloads,
produced by the same functions):

    manifest.json                       what was exported (seeds, checkpoints)
    runs.json                           GET /api/runs
    metrics_<run>.json                  GET /api/runs/<run>/metrics (train rows thinned)
    checkpoints_<run>.json              GET /api/runs/<run>/checkpoints (exported ones only)
    episode_<run>__<ckpt>__<seed>.json  GET /api/runs/<run>/episode (greedy)
    baseline_<seed>.json                GET /api/baseline/episode

Checkpoints exported: the checkpoints nearest to each --fractions share of the
final update, plus ``final``. Output is deterministic (sorted keys, floats
rounded to 3 decimals, no mtimes).
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
_spec = importlib.util.spec_from_file_location("serve_viewer", ROOT / "scripts" / "serve_viewer.py")
sv = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(sv)

TRAIN_MAX_ROWS = 150
DECIMALS = 3


def _round(obj):
    if isinstance(obj, float):
        return round(obj, DECIMALS)
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def _dump(path: Path, obj) -> int:
    text = json.dumps(_round(obj), sort_keys=True, separators=(",", ":"))
    path.write_text(text + "\n")
    return len(text) + 1


def thin(rows: list[dict], max_rows: int) -> list[dict]:
    """Keep every k-th row plus the last one so that at most ~max_rows remain."""
    if len(rows) <= max_rows:
        return rows
    k = -(-len(rows) // max_rows)
    out = rows[::k]
    if out[-1] is not rows[-1]:
        out.append(rows[-1])
    return out


def pick_checkpoints(checkpoints: list[dict], fractions: list[float]) -> list[str]:
    """Ids (ascending update): per-fraction nearest periodic checkpoints, then final."""
    periodic = [c for c in checkpoints if re.fullmatch(r"ckpt_\d+", c["id"])]
    final = next((c for c in checkpoints if c["id"] == "final"), None)
    last_update = (final or {}).get("update") or (max(c["update"] for c in periodic) if periodic else None)
    chosen: dict[str, int] = {}
    if periodic and last_update:
        for fr in fractions:
            target = fr * last_update
            best = min(periodic, key=lambda c: (abs(c["update"] - target), c["update"]))
            chosen[best["id"]] = best["update"]
    ids = sorted(chosen, key=chosen.get)
    if final is not None:
        ids.append("final")
    elif periodic:  # no final.pt: use the newest periodic checkpoint
        newest = max(periodic, key=lambda c: c["update"])["id"]
        if newest not in ids:
            ids.append(newest)
    return ids


def export(run_dir: Path, out: Path, n_seeds: int = 10, first_seed: int = 20000,
           fractions: list[float] | None = None) -> dict:
    run_dir = Path(run_dir).expanduser().resolve()
    results, name = run_dir.parent, run_dir.name
    fractions = [0.1, 0.4] if fractions is None else fractions
    seeds = list(range(first_seed, first_seed + n_seeds))

    runs = [r for r in sv.list_runs(results) if r["name"] == name]
    if not runs:
        raise SystemExit(f"{run_dir} is not a run directory")
    run = dict(runs[0], mtime=0, finished=True)
    metrics = sv.run_metrics(results, name)
    metrics["train"] = thin(metrics["train"], TRAIN_MAX_ROWS)
    all_ckpts = sv.run_checkpoints(results, name)
    ids = pick_checkpoints(all_ckpts, fractions)
    if not ids:
        raise SystemExit("run has no checkpoints to export")
    by_id = {c["id"]: c for c in all_ckpts}
    checkpoints = [{"id": i, "update": by_id[i]["update"], "mtime": 0} for i in ids]

    if out.exists():
        shutil.rmtree(out)
    out.mkdir(parents=True)
    sizes = {}
    sizes["runs.json"] = _dump(out / "runs.json", [run])
    sizes[f"metrics_{name}.json"] = _dump(out / f"metrics_{name}.json", metrics)
    sizes[f"checkpoints_{name}.json"] = _dump(out / f"checkpoints_{name}.json", checkpoints)
    for ck in ids:
        for seed in seeds:
            ep = sv.run_episode(results, name, seed, ck)
            fn = f"episode_{name}__{ck}__{seed}.json"
            sizes[fn] = _dump(out / fn, ep)
    for seed in seeds:
        fn = f"baseline_{seed}.json"
        sizes[fn] = _dump(out / fn, sv.baseline_episode(seed))
    manifest = {
        "kind": "recorded-training-run",
        "run": name,
        "seeds": seeds,
        "checkpoints": checkpoints,
        "train_rows_total": len(sv.read_jsonl(run_dir / "metrics.jsonl")),
        "train_rows_exported": len(metrics["train"]),
        "files": len(sizes) + 1,
    }
    sizes["manifest.json"] = _dump(out / "manifest.json", manifest)
    manifest["total_bytes"] = sum(sizes.values())
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("run_dir")
    ap.add_argument("--out", default=str(ROOT / "frontend" / "demo"))
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--first-seed", type=int, default=20000)
    ap.add_argument("--fractions", type=float, nargs="*", default=[0.1, 0.4])
    a = ap.parse_args()
    m = export(Path(a.run_dir), Path(a.out), a.seeds, a.first_seed, a.fractions)
    print(f"exported {m['files']} files, {m['total_bytes'] / 1e6:.2f} MB -> {a.out}")
    print("checkpoints:", ", ".join(f"{c['id']}(upd {c['update']})" for c in m["checkpoints"]))


if __name__ == "__main__":
    sys.exit(main())

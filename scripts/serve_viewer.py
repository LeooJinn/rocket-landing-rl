"""Serve the browser viewer (frontend/) and a JSON API over experiment results.

Usage: uv run python scripts/serve_viewer.py --results-dir DIR [--port 8000]
Binds 127.0.0.1 only. Run files are re-read on every request so live training
runs can be followed.
"""

from __future__ import annotations

import argparse
import json
import mimetypes
import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

ROOT = Path(__file__).resolve().parent.parent
FRONTEND = ROOT / "frontend"
DEFAULT_SEED = 20000
CKPT_RE = re.compile(r"^ckpt_(\d+)\.pt$")
CKPT_ID_RE = re.compile(r"^ckpt_\d+$")

_cache: dict[tuple, dict] = {}
_cache_lock = threading.Lock()
_policy_cache: dict[tuple, object] = {}


class ApiError(Exception):
    def __init__(self, status: int, message: str):
        super().__init__(message)
        self.status = status
        self.message = message


# ------------------------------------------------------------------ file utils
def read_jsonl(path: Path) -> list[dict]:
    """Parse a jsonl file, skipping blank / partially written lines."""
    rows: list[dict] = []
    try:
        text = path.read_text()
    except OSError:
        return rows
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(row, dict):
            rows.append(row)
    return rows


def read_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def run_dir(results_dir: Path, name: str) -> Path:
    """Resolve a run name against the actual directory listing (no traversal)."""
    results_dir = Path(results_dir)
    if name and "/" not in name and "\\" not in name and "\0" not in name:
        for child in results_dir.iterdir() if results_dir.is_dir() else []:
            if child.name == name and child.is_dir():
                return child
    raise ApiError(404, f"unknown run {name!r}")


def parse_seed(value: str | None) -> int:
    if value is None or value == "":
        return DEFAULT_SEED
    if not re.fullmatch(r"-?\d{1,12}", value):
        raise ApiError(400, "seed must be an integer")
    seed = int(value)
    if seed < 0:
        raise ApiError(400, "seed must be non-negative")
    return seed


# ------------------------------------------------------------------------ API
def _mtime(p: Path) -> float:
    try:
        return p.stat().st_mtime
    except OSError:
        return 0.0


def _cfg_inner(config) -> dict:
    if isinstance(config, dict):
        inner = config.get("config")
        return inner if isinstance(inner, dict) else config
    return {}


def list_runs(results_dir: Path) -> list[dict]:
    out = []
    results_dir = Path(results_dir)
    if not results_dir.is_dir():
        return out
    for d in results_dir.iterdir():
        if not d.is_dir():
            continue
        config = read_json(d / "config.json")
        inner = _cfg_inner(config)
        train = read_jsonl(d / "metrics.jsonl")
        evals = read_jsonl(d / "eval.jsonl")
        last = train[-1] if train else {}
        le = None
        if evals:
            e = evals[-1]
            le = {
                k: e.get(k)
                for k in (
                    "success_rate",
                    "crash_rate",
                    "out_of_bounds_rate",
                    "timeout_rate",
                    "mean_reward",
                    "update",
                )
            }
        mt = max(
            [_mtime(d)]
            + [_mtime(d / f) for f in ("metrics.jsonl", "eval.jsonl", "summary.json")]
        )
        out.append(
            {
                "name": d.name,
                "reward_version": inner.get("reward_version"),
                "seed": config.get("seed") if isinstance(config, dict) else None,
                "updates": int(last.get("update", 0)),
                "total_timesteps": inner.get("total_timesteps"),
                "global_step": int(last.get("global_step", 0)),
                "finished": (d / "summary.json").exists(),
                "last_eval": le,
                "mtime": mt,
            }
        )
    out.sort(key=lambda r: r["mtime"], reverse=True)
    return out


def run_metrics(results_dir: Path, name: str) -> dict:
    d = run_dir(results_dir, name)
    return {
        "train": read_jsonl(d / "metrics.jsonl"),
        "eval": read_jsonl(d / "eval.jsonl"),
        "config": read_json(d / "config.json"),
    }


def run_checkpoints(results_dir: Path, name: str) -> list[dict]:
    d = run_dir(results_dir, name)
    items: list[dict] = []
    ck = d / "checkpoints"
    if ck.is_dir():
        for f in ck.iterdir():
            m = CKPT_RE.match(f.name)
            if m:
                items.append(
                    {"id": f.stem, "update": int(m.group(1)), "mtime": _mtime(f)}
                )
    items.sort(key=lambda i: i["update"])
    out: list[dict] = []
    if items:
        newest = items[-1]
        out.append({"id": "latest", "update": newest["update"], "mtime": newest["mtime"]})
    for cid in ("best", "final"):
        f = d / f"{cid}.pt"
        if f.is_file():
            out.append({"id": cid, "update": _ckpt_update(f), "mtime": _mtime(f)})
    return out + items


_update_cache: dict[tuple, int | None] = {}


def _ckpt_update(path: Path) -> int | None:
    key = (str(path), _mtime(path))
    if key not in _update_cache:
        try:
            import torch

            payload = torch.load(path, map_location="cpu", weights_only=True)
            u = (payload.get("extra") or {}).get("update")
            _update_cache[key] = int(u) if u is not None else None
        except Exception:
            _update_cache[key] = None
    return _update_cache[key]


def resolve_checkpoint(d: Path, ckpt_id: str) -> tuple[Path, int | None, str]:
    """Return (path, update, resolved_id). Raises ApiError(404)."""
    if ckpt_id == "latest":
        ck = d / "checkpoints"
        best: tuple[int, Path] | None = None
        if ck.is_dir():
            for f in ck.iterdir():
                m = CKPT_RE.match(f.name)
                if m and (best is None or int(m.group(1)) > best[0]):
                    best = (int(m.group(1)), f)
        if best is None:
            raise ApiError(404, "no checkpoint exists yet for this run")
        return best[1], best[0], "latest"
    if ckpt_id in ("best", "final"):
        f = d / f"{ckpt_id}.pt"
        if not f.is_file():
            raise ApiError(404, f"checkpoint {ckpt_id!r} does not exist")
        return f, _ckpt_update(f), ckpt_id
    if CKPT_ID_RE.match(ckpt_id):
        f = d / "checkpoints" / f"{ckpt_id}.pt"
        if not f.is_file():
            raise ApiError(404, f"checkpoint {ckpt_id!r} does not exist")
        return f, int(ckpt_id.split("_")[1]), ckpt_id
    raise ApiError(400, "invalid checkpoint id")


def run_episode(results_dir: Path, name: str, seed: int, ckpt_id: str) -> dict:
    from rocket_landing.recording import load_policy_from_checkpoint, record_episode

    d = run_dir(results_dir, name)
    path, update, resolved = resolve_checkpoint(d, ckpt_id)
    key = (name, str(path), _mtime(path), seed)
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    inner = _cfg_inner(read_json(d / "config.json"))
    reward_version = inner.get("reward_version") or "v1"
    try:
        controller = load_policy_from_checkpoint(path)
        traj = record_episode(
            controller,
            seed,
            reward_version=reward_version,
            run=name,
            checkpoint=ckpt_id,
            update=update,
        )
    except ApiError:
        raise
    except Exception as exc:  # e.g. checkpoint being written
        raise ApiError(500, f"could not roll out checkpoint: {exc}")
    with _cache_lock:
        _cache[key] = traj
    return traj


def baseline_episode(seed: int) -> dict:
    from rocket_landing.recording import record_baseline_episode

    key = ("__baseline__", "", 0.0, seed)
    with _cache_lock:
        hit = _cache.get(key)
    if hit is not None:
        return hit
    traj = record_baseline_episode(seed)
    with _cache_lock:
        _cache[key] = traj
    return traj


def route(results_dir: Path, path: str, query: dict[str, list[str]]):
    """Dispatch an /api path to a JSON-able result. Raises ApiError."""
    parts = [unquote(p) for p in path.strip("/").split("/")]
    q = lambda k, default=None: (query.get(k) or [default])[0]  # noqa: E731
    if parts == ["api", "runs"]:
        return list_runs(results_dir)
    if parts == ["api", "baseline", "episode"]:
        return baseline_episode(parse_seed(q("seed")))
    if len(parts) == 4 and parts[:2] == ["api", "runs"]:
        name, what = parts[2], parts[3]
        if what == "metrics":
            return run_metrics(results_dir, name)
        if what == "checkpoints":
            return run_checkpoints(results_dir, name)
        if what == "episode":
            run_dir(results_dir, name)  # 404 before validating params
            return run_episode(
                results_dir, name, parse_seed(q("seed")), q("checkpoint", "latest")
            )
    raise ApiError(404, "not found")


# --------------------------------------------------------------------- server
def make_handler(results_dir: Path, frontend: Path = FRONTEND):
    frontend = Path(frontend)

    class Handler(BaseHTTPRequestHandler):
        def _send(self, status: int, body: bytes, ctype: str) -> None:
            self.send_response(status)
            self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(len(body)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(body)

        def _json(self, status: int, obj) -> None:
            self._send(status, json.dumps(obj).encode(), "application/json")

        def do_GET(self) -> None:  # noqa: N802
            url = urlparse(self.path)
            if url.path.startswith("/api/"):
                try:
                    self._json(200, route(results_dir, url.path, parse_qs(url.query)))
                except ApiError as e:
                    self._json(e.status, {"error": e.message})
                except Exception as e:  # pragma: no cover
                    self._json(500, {"error": str(e)})
                return
            self._static(url.path)

        def _static(self, path: str) -> None:
            rel = unquote(path).lstrip("/") or "index.html"
            base = frontend.resolve()
            target = (base / rel).resolve()
            if target.is_dir():
                target = target / "index.html"
            if base != target and base not in target.parents or not target.is_file():
                self._json(404, {"error": "not found"})
                return
            ctype = mimetypes.guess_type(target.name)[0] or "application/octet-stream"
            self._send(200, target.read_bytes(), ctype)

        def log_message(self, fmt, *args) -> None:
            pass

    return Handler


def make_server(results_dir, port: int = 8000, frontend: Path = FRONTEND) -> ThreadingHTTPServer:
    return ThreadingHTTPServer(
        ("127.0.0.1", port), make_handler(Path(results_dir).expanduser(), frontend)
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--results-dir", default="experiments/results")
    ap.add_argument("--port", type=int, default=8000)
    args = ap.parse_args()
    server = make_server(args.results_dir, args.port)
    print(f"Viewer at http://127.0.0.1:{args.port}/  (results: {args.results_dir})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()

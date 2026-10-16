from pathlib import Path

import pytest

pytest.importorskip("PIL", reason="needs Pillow (plots extra)")

ROOT = Path(__file__).resolve().parents[1]


def test_render_demo_gif_from_committed_demo_data(tmp_path, monkeypatch):
    demo = ROOT / "frontend" / "demo"
    if not (demo / "manifest.json").exists():
        pytest.skip("static demo data not present")
    import importlib.util
    import sys

    spec = importlib.util.spec_from_file_location("render_demo_gif", ROOT / "scripts" / "render_demo_gif.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    out = tmp_path / "demo.gif"
    monkeypatch.setattr(sys, "argv", ["render_demo_gif.py", "--demo-dir", str(demo), "--out", str(out),
                                      "--step", "60", "--hold", "3", "--panel-w", "160", "--panel-h", "180"])
    mod.main()
    from PIL import Image

    im = Image.open(out)
    assert im.n_frames > 3
    assert im.size[0] > 400

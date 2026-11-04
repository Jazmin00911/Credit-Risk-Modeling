"""Smoke test of the deploy path: the app must start without pre-built artifacts and render a decision."""
import os
import shutil
import tempfile
from pathlib import Path

os.environ.setdefault("MPLBACKEND", "Agg")

from streamlit.testing.v1 import AppTest  # noqa: E402

APP = Path(__file__).resolve().parents[1] / "app" / "streamlit_app.py"


def test_app_cold_start_and_decision():
    tmp = Path(tempfile.gettempdir())
    for d in ("creditrisk_artifacts", "creditrisk_reports"):
        shutil.rmtree(tmp / d, ignore_errors=True)  # force the first-launch training path
    at = AppTest.from_file(str(APP), default_timeout=300).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any("Probability of default" in m.label for m in at.metric)

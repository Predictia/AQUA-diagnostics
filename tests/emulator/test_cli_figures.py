"""The emulator CLIs must save figures for every diagnostic."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import pytest

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from aqua.diagnostics.emulator import _cli  # noqa: E402

pytestmark = pytest.mark.emulator


def test_every_emulator_cli_has_figures():
    package = Path(_cli.__file__).parent
    names = {path.stem.removeprefix("cli_") for path in package.glob("*/cli_*.py")}
    assert names == set(_cli.FIGURES)


def test_save_figures_draws_and_closes_each_figure(monkeypatch):
    calls = []

    class StubPlotter:
        @classmethod
        def from_diagnostic(cls, diagnostic):
            return cls()

        def _figure(self, data, **kwargs):
            calls.append((data, kwargs))
            return plt.figure(), None

        plot_spatial = plot_temporal = _figure

    monkeypatch.setitem(
        _cli.FIGURES, "rmse", (StubPlotter, _cli._spatial_temporal("RMSE"))
    )
    results = {"spatial": {"2t": "map", "zg_50000": "map"}, "temporal": {"2t": "series"}}
    _cli.save_figures(
        "rmse", object(), results, {}, save_format=["png", "pdf"], dpi=100
    )

    assert len(calls) == 3
    assert calls[1][1]["extra_keys"] == {"var": "zg_50000"}
    assert calls[0][1]["format"] == ["png", "pdf"]
    assert all(kwargs["save"] for _, kwargs in calls)
    assert plt.get_fignums() == []

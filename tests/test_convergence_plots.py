"""Tests für Debug-Konvergenz-Plots und Konvergenz-Historie."""

from __future__ import annotations

import os

import numpy as np
import pytest

from mat_sim.md import (
    RampConfig,
    _CombinedEquilibriumMonitor,
    _EquilibriumStop,
    save_convergence_plot,
)


# ── Hilfs-Fakes (wie in test_combined_equilibrium.py) ─────────────────────
class _FakeDyn:
    def __init__(self) -> None:
        self._temp = 300.0

    def get_temperature(self) -> float:
        return self._temp


class _FakeAtoms:
    def __init__(self, n_atoms: int = 4) -> None:
        self._positions = np.zeros((n_atoms, 3))

    def get_positions(self) -> np.ndarray:
        return self._positions.copy()


def _fast_cfg(**overrides) -> RampConfig:
    defaults = dict(
        early_stop_min_steps=5,
        early_stop_window=5,
        early_stop_rel_std=0.05,
        msd_sample_interval=1,
        pos_convergence_min_samples=6,
        pos_convergence_window_mult=2,
        pos_convergence_min_window=3,
        pos_convergence_threshold=0.01,
    )
    defaults.update(overrides)
    return RampConfig(**defaults)


# ── 1. Historie wird aufgezeichnet ─────────────────────────────────────────
def test_history_recorded():
    """Nach MD-Schritten enthält history Einträge für jeden Schritt."""
    cfg = _fast_cfg()
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    # 10 Schritte mit stabilen Positionen → sollte konvergieren und stoppen
    for step in range(1, 30):
        monitor._dyn._temp = 300.0
        p = np.zeros((4, 3))
        p[:, 0] = 1.0 + 0.001 * np.sin(step)
        monitor._atoms._positions = p
        try:
            monitor()
        except _EquilibriumStop:
            break

    hist = monitor.history
    assert len(hist["steps"]) > 0
    assert len(hist["temp_rel_std"]) == len(hist["steps"])
    assert len(hist["pos_rms"]) == len(hist["steps"])
    assert len(hist["temp_converged"]) == len(hist["steps"])
    assert len(hist["pos_converged"]) == len(hist["steps"])

    # temp_rel_std sollte endliche Werte haben (nach min_steps)
    valid = [v for v in hist["temp_rel_std"] if not np.isnan(v)]
    assert len(valid) > 0
    assert all(v >= 0 for v in valid)


def test_history_cleared_on_reset():
    """Nach reset() ist die Historie leer."""
    cfg = _fast_cfg()
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    for step in range(1, 15):
        monitor._dyn._temp = 300.0
        monitor._atoms._positions = np.zeros((4, 3))
        try:
            monitor()
        except _EquilibriumStop:
            break

    assert len(monitor.history["steps"]) > 0
    monitor.reset()
    assert len(monitor.history["steps"]) == 0


# ── 2. Plot-Funktion speichert PNG ────────────────────────────────────────
def test_save_convergence_plot_creates_png(tmp_path):
    """save_convergence_plot erzeugt eine PNG-Datei."""
    history = {
        "steps": list(range(5, 50)),
        "temp_rel_std": [0.1, 0.08, 0.06, 0.04, 0.03, 0.02, 0.02, 0.02,
                         0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                         0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                         0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                         0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                         0.02, 0.02, 0.02, 0.02, 0.02, 0.02, 0.02,
                         0.02, 0.02],
        "pos_rms": [float("nan"), float("nan"), 0.1, 0.05, 0.03,
                    0.02, 0.015, 0.01, 0.008, 0.007, 0.006, 0.005,
                    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
                    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
                    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
                    0.005, 0.005, 0.005, 0.005, 0.005, 0.005, 0.005,
                    0.005, 0.005, 0.005, 0.005, 0.005],
        "temp_converged": [False] * 4 + [True] * 41,
        "pos_converged": [False] * 6 + [True] * 39,
    }

    filepath = save_convergence_plot(
        history=history,
        temperature=340.0,
        formula="VO2",
        material_id="mp-18248",
        output_dir=str(tmp_path),
        early_stop_rel_std=0.02,
        pos_convergence_threshold=0.01,
        stopped_at=45,
    )

    assert os.path.exists(filepath)
    assert filepath.endswith(".png")
    assert "mp-18248" in filepath
    assert "VO2" in filepath
    assert "T340" in filepath


def test_save_convergence_plot_nan_heavy(tmp_path):
    """Plot funktioniert auch mit vielen NaN-Werten (frühe Schritte)."""
    history = {
        "steps": [1, 2, 3, 4, 5],
        "temp_rel_std": [float("nan"), float("nan"), 0.1, 0.05, 0.02],
        "pos_rms": [float("nan")] * 5,
        "temp_converged": [False, False, False, False, True],
        "pos_converged": [False] * 5,
    }

    filepath = save_convergence_plot(
        history=history,
        temperature=0.0,
        formula="TiO2",
        material_id="mp-390",
        output_dir=str(tmp_path),
        early_stop_rel_std=0.02,
        pos_convergence_threshold=0.01,
        stopped_at=None,
    )

    assert os.path.exists(filepath)
    assert "T0" in filepath


def test_save_convergence_plot_creates_subdirectory(tmp_path):
    """Plot wird in Unterordner material_id_formula/ gespeichert."""
    history = {
        "steps": [1, 2],
        "temp_rel_std": [0.1, 0.02],
        "pos_rms": [0.1, 0.005],
        "temp_converged": [False, True],
        "pos_converged": [False, True],
    }

    filepath = save_convergence_plot(
        history=history,
        temperature=100.0,
        formula="V2O3",
        material_id="mp-510",
        output_dir=str(tmp_path),
        early_stop_rel_std=0.02,
        pos_convergence_threshold=0.01,
        stopped_at=2,
    )

    # Datei sollte in tmp_path/mp-510_V2O3/ liegen
    assert os.path.exists(filepath)
    rel = os.path.relpath(filepath, str(tmp_path))
    assert rel.startswith("mp-510_V2O3" + os.sep)

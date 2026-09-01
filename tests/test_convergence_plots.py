"""Tests für Debug-Konvergenz-Plots und Konvergenz-Historie."""

from __future__ import annotations

import json
import os
import math

import numpy as np

from mat_sim.md import (
    RampConfig,
    _CombinedEquilibriumMonitor,
    _EquilibriumStop,
    save_convergence_plot,
    save_raw_data_json,
)


# ── Hilfs-Fakes (wie in test_combined_equilibrium.py) ─────────────────────
class _FakeDyn:
    def __init__(self) -> None:
        self._temp = 300.0


class _FakeAtoms:
    def __init__(self, n_atoms: int = 4) -> None:
        self._positions = np.zeros((n_atoms, 3))
        self._temp = 300.0
        self._volume = 1000.0

    def get_positions(self) -> np.ndarray:
        return self._positions.copy()

    def get_temperature(self) -> float:
        return self._temp

    def get_volume(self) -> float:
        return self._volume


def _fast_cfg(**overrides) -> RampConfig:
    defaults = dict(
        early_stop_min_steps=5,
        early_stop_window=5,
        early_stop_rel_std=0.05,
        msd_sample_interval=1,
        pos_convergence_min_samples=6,
        pos_convergence_window_mult=2,
        pos_convergence_min_window=3,
        pos_convergence_threshold=0.5,
        pos_convergence_rel_std=0.10,
        pos_convergence_eval_window=10,
        pos_convergence_persistence=3,
        vol_convergence_rel_std=0.02,
        vol_convergence_eval_window=10,
        vol_convergence_persistence=3,
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
        monitor._atoms._temp = 300.0
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
    assert len(hist["pos_rel_std"]) == len(hist["steps"])
    assert len(hist["temp_converged"]) == len(hist["steps"])
    assert len(hist["pos_converged"]) == len(hist["steps"])
    assert len(hist["vol_rel_std"]) == len(hist["steps"])
    assert len(hist["vol_converged"]) == len(hist["steps"])

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
        monitor._atoms._temp = 300.0
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
        "temp_rel_std": [
            0.1,
            0.08,
            0.06,
            0.04,
            0.03,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
            0.02,
        ],
        "pos_rms": [
            float("nan"),
            float("nan"),
            0.1,
            0.05,
            0.03,
            0.02,
            0.015,
            0.01,
            0.008,
            0.007,
            0.006,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
            0.005,
        ],
        "pos_rel_std": [float("nan")] * 2
        + [
            0.2,
            0.15,
            0.1,
            0.08,
            0.06,
            0.05,
            0.04,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
            0.03,
        ],
        "temp_converged": [False] * 4 + [True] * 41,
        "pos_converged": [False] * 6 + [True] * 39,
        "vol_rel_std": [float("nan")] * 10 + [0.01] * 35,
        "vol_converged": [False] * 13 + [True] * 32,
    }
    raw_data = {
        "steps": list(range(1, 50)),
        "temperatures": [340.0 + 0.1 * math.sin(s) for s in range(1, 50)],
        "mean_pos": [[1.0 + 0.001 * s, 0.0, 0.0] for s in range(1, 50)],
        "volumes": [1000.0 + 0.5 * math.sin(s) for s in range(1, 50)],
    }

    filepath = save_convergence_plot(
        history=history,
        raw_data=raw_data,
        temperature=340.0,
        formula="VO2",
        material_id="mp-18248",
        output_dir=str(tmp_path),
        early_stop_rel_std=0.02,
        pos_convergence_rel_std=0.01,
        vol_convergence_rel_std=0.02,
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
        "pos_rel_std": [float("nan")] * 5,
        "temp_converged": [False, False, False, False, True],
        "pos_converged": [False] * 5,
        "vol_rel_std": [float("nan")] * 5,
        "vol_converged": [False] * 5,
    }
    raw_data = {
        "steps": [1, 2, 3, 4, 5],
        "temperatures": [float("nan"), float("nan"), 300.0, 300.0, 300.0],
        "mean_pos": [
            [float("nan")] * 3,
            [float("nan")] * 3,
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ],
        "volumes": [float("nan"), float("nan"), 1000.0, 1000.0, 1000.0],
    }

    filepath = save_convergence_plot(
        history=history,
        raw_data=raw_data,
        temperature=0.0,
        formula="TiO2",
        material_id="mp-390",
        output_dir=str(tmp_path),
        early_stop_rel_std=0.02,
        pos_convergence_rel_std=0.01,
        vol_convergence_rel_std=0.02,
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
        "pos_rel_std": [float("nan"), 0.05],
        "temp_converged": [False, True],
        "pos_converged": [False, True],
        "vol_rel_std": [float("nan"), 0.01],
        "vol_converged": [False, True],
    }
    raw_data = {
        "steps": [1, 2],
        "temperatures": [100.0, 100.0],
        "mean_pos": [[0.5, 0.0, 0.0], [0.5, 0.0, 0.0]],
        "volumes": [1000.0, 1000.0],
    }

    filepath = save_convergence_plot(
        history=history,
        raw_data=raw_data,
        temperature=100.0,
        formula="V2O3",
        material_id="mp-510",
        output_dir=str(tmp_path),
        early_stop_rel_std=0.02,
        pos_convergence_rel_std=0.01,
        vol_convergence_rel_std=0.02,
        stopped_at=2,
    )

    # Datei sollte in tmp_path/mp-510_V2O3/ liegen
    assert os.path.exists(filepath)
    rel = os.path.relpath(filepath, str(tmp_path))
    assert rel.startswith("mp-510_V2O3" + os.sep)


# ── 3. raw_data-Property ──────────────────────────────────────────────────
def test_raw_data_recorded():
    """Nach MD-Schritten enthält raw_data Einträge für jeden Schritt."""
    cfg = _fast_cfg()
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    for step in range(1, 30):
        monitor._atoms._temp = 300.0
        p = np.zeros((4, 3))
        p[:, 0] = 1.0 + 0.001 * np.sin(step)
        monitor._atoms._positions = p
        try:
            monitor()
        except _EquilibriumStop:
            break

    raw = monitor.raw_data
    assert len(raw["steps"]) > 0
    assert len(raw["temperatures"]) == len(raw["steps"])
    assert len(raw["mean_pos"]) == len(raw["steps"])
    assert len(raw["volumes"]) == len(raw["steps"])
    # Jeder mean_pos-Eintrag hat 3 Koordinaten
    assert all(len(pos) == 3 for pos in raw["mean_pos"])
    # Temperaturen sollten endliche Werte haben
    valid_t = [t for t in raw["temperatures"] if not np.isnan(t)]
    assert len(valid_t) > 0


def test_raw_data_cleared_on_reset():
    """Nach reset() ist raw_data leer."""
    cfg = _fast_cfg()
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    for step in range(1, 15):
        monitor._atoms._temp = 300.0
        monitor._atoms._positions = np.zeros((4, 3))
        try:
            monitor()
        except _EquilibriumStop:
            break

    assert len(monitor.raw_data["steps"]) > 0
    monitor.reset()
    assert len(monitor.raw_data["steps"]) == 0
    assert len(monitor.raw_data["temperatures"]) == 0
    assert len(monitor.raw_data["mean_pos"]) == 0
    assert len(monitor.raw_data["volumes"]) == 0


# ── 4. save_raw_data_json ─────────────────────────────────────────────────
def test_save_raw_data_json_creates_file(tmp_path):
    """save_raw_data_json erzeugt eine gültige JSON-Datei."""
    history = {
        "steps": [1, 2, 3],
        "temp_rel_std": [float("nan"), 0.1, 0.02],
        "pos_rms": [float("nan"), 0.05, 0.005],
        "pos_rel_std": [float("nan"), 0.15, 0.03],
        "temp_converged": [False, False, True],
        "pos_converged": [False, False, True],
        "vol_rel_std": [float("nan"), 0.02, 0.01],
        "vol_converged": [False, False, True],
    }
    raw_data = {
        "steps": [1, 2, 3],
        "temperatures": [float("nan"), 300.0, 300.0],
        "mean_pos": [[float("nan"), 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
        "volumes": [float("nan"), 1000.0, 1000.0],
    }

    filepath = save_raw_data_json(
        raw_data=raw_data,
        history=history,
        temperature=300.0,
        formula="VO2",
        material_id="mp-18248",
        output_dir=str(tmp_path),
        stopped_at=3,
    )

    assert os.path.exists(filepath)
    assert filepath.endswith(".json")
    assert "mp-18248" in filepath
    assert "T300" in filepath

    with open(filepath, encoding="utf-8") as f:
        payload = json.load(f)

    assert payload["material_id"] == "mp-18248"
    assert payload["formula"] == "VO2"
    assert payload["temperature_K"] == 300.0
    assert payload["stopped_at"] == 3
    assert payload["raw"]["steps"] == [1, 2, 3]
    # NaN sollte zu null konvertiert sein
    assert payload["raw"]["temperatures"][0] is None
    assert payload["raw"]["temperatures"][1] == 300.0
    assert payload["raw"]["mean_pos"][0][0] is None
    assert payload["raw"]["mean_pos"][1][0] == 1.0
    assert payload["history"]["temp_rel_std"][0] is None
    assert payload["history"]["temp_rel_std"][1] == 0.1


def test_save_raw_data_json_stopped_at_none(tmp_path):
    """JSON funktioniert auch ohne Early-Stop."""
    raw_data = {
        "steps": [1],
        "temperatures": [300.0],
        "mean_pos": [[0.0, 0.0, 0.0]],
        "volumes": [1000.0],
    }
    history = {
        "steps": [1],
        "temp_rel_std": [0.1],
        "pos_rms": [0.05],
        "pos_rel_std": [float("nan")],
        "temp_converged": [False],
        "pos_converged": [False],
        "vol_rel_std": [float("nan")],
        "vol_converged": [False],
    }

    filepath = save_raw_data_json(
        raw_data=raw_data,
        history=history,
        temperature=100.0,
        formula="TiO2",
        material_id="mp-390",
        output_dir=str(tmp_path),
        stopped_at=None,
    )

    assert os.path.exists(filepath)
    with open(filepath, encoding="utf-8") as f:
        payload = json.load(f)
    assert payload["stopped_at"] is None

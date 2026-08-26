"""Tests für Speicherung und Rekonstruktion der vollständigen Trajektorie."""

from __future__ import annotations

import os
import tempfile

import numpy as np
import pytest
from ase import Atoms

from mat_sim.acquisition import MPEntry
from mat_sim.metrics import StepMetrics, TrajectoryResult
from mat_sim.storage import (
    init_db,
    load_result,
    reconstruct_atoms_at_step,
    store_result,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    os.unlink(path)
    init_db(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)
        for ext in ("-wal", "-shm"):
            p = path + ext
            if os.path.exists(p):
                os.unlink(p)


def _make_entry() -> MPEntry:
    """MPEntry mit einer einfachen pymatgen-Struktur (O2 kubisch)."""
    from pymatgen.core import Lattice, Structure

    struct = Structure(
        Lattice.cubic(5.0),
        ["O", "O"],
        [[0, 0, 0], [0.5, 0.5, 0.5]],
    )
    return MPEntry(material_id="mp-test", formula_pretty="O2", structure=struct)


def _make_result(n_steps: int = 3, n_atoms: int = 2) -> TrajectoryResult:
    """TrajectoryResult mit Positions-/Zell-History für n_steps."""
    result = TrajectoryResult()
    for i in range(n_steps):
        m = StepMetrics(
            temperature=float(i * 10),
            volume=125.0 + i,
            energy=-10.0 - i,
            msd=0.01 * i,
            rdf_r=np.linspace(0, 6, 50),
            rdf_g=np.random.rand(50),
            ql=0.76 + 0.001 * i,
            positions=np.random.rand(n_atoms, 3) * 5.0,
            cell=np.eye(3) * (5.0 + 0.01 * i),
        )
        result.add(m)
    result.t_switch = 10.0
    result.status = "converged"
    return result


# ── Tests: Roundtrip store → load ───────────────────────────────────────────

def test_store_load_positions_history(tmp_db: str) -> None:
    """positions_history wird gespeichert und korrekt zurückgeladen."""
    conn = init_db(tmp_db)
    entry = _make_entry()
    result = _make_result(n_steps=3, n_atoms=2)

    store_result(conn, entry, result)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")

    assert loaded.positions_history is not None
    assert len(loaded.positions_history) == 3
    for orig, stored_arr in zip(result.positions_history, loaded.positions_history):
        np.testing.assert_array_almost_equal(orig, stored_arr)


def test_store_load_cell_history(tmp_db: str) -> None:
    """cell_history wird gespeichert und korrekt zurückgeladen."""
    conn = init_db(tmp_db)
    entry = _make_entry()
    result = _make_result(n_steps=4, n_atoms=2)

    store_result(conn, entry, result)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")

    assert loaded.cell_history is not None
    assert len(loaded.cell_history) == 4
    for orig, stored_arr in zip(result.cell_history, loaded.cell_history):
        np.testing.assert_array_almost_equal(orig, stored_arr)


def test_store_load_symbols(tmp_db: str) -> None:
    """symbols werden aus entry.structure extrahiert und korrekt gespeichert."""
    conn = init_db(tmp_db)
    entry = _make_entry()
    result = _make_result(n_steps=2, n_atoms=2)

    store_result(conn, entry, result)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")

    assert loaded.symbols is not None
    assert loaded.symbols == ["O", "O"]


def test_store_load_full_roundtrip(tmp_db: str) -> None:
    """Alle Trajektorie-Felder zusammen speichern und laden."""
    conn = init_db(tmp_db)
    entry = _make_entry()
    result = _make_result(n_steps=5, n_atoms=2)
    result.t_switch = 30.0
    result.t_decay = 40.0
    result.status = "converged"

    store_result(conn, entry, result)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")

    # Skalar-Metriken
    assert loaded.temperatures == result.temperatures
    assert loaded.volumes == result.volumes
    assert loaded.msd_values == result.msd_values
    assert loaded.ql_values == result.ql_values
    assert loaded.t_switch == 30.0
    assert loaded.t_decay == 40.0
    assert loaded.status == "converged"

    # Trajektorie
    assert loaded.positions_history is not None
    assert loaded.cell_history is not None
    assert loaded.symbols == ["O", "O"]
    assert len(loaded.positions_history) == 5
    assert len(loaded.cell_history) == 5


def test_store_result_no_trajectory(tmp_db: str) -> None:
    """Ergebnis ohne positions_history speichern → Felder sind None."""
    conn = init_db(tmp_db)
    entry = _make_entry()
    result = TrajectoryResult()
    result.temperatures = [0.0, 10.0]
    result.msd_values = [0.0, 0.01]
    result.volumes = [125.0, 126.0]
    result.ql_values = [0.76, 0.76]
    result.energies = [-10.0, -10.1]
    result.status = "converged"

    store_result(conn, entry, result)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")

    assert loaded.positions_history is None
    assert loaded.cell_history is None
    # symbols werden trotzdem gespeichert (aus entry.structure)
    assert loaded.symbols == ["O", "O"]


# ── Tests: reconstruct_atoms_at_step ────────────────────────────────────────

def test_reconstruct_atoms_at_step(tmp_db: str) -> None:
    """reconstruct_atoms_at_step liefert korrektes Atoms-Objekt."""
    conn = init_db(tmp_db)
    entry = _make_entry()
    result = _make_result(n_steps=3, n_atoms=2)

    store_result(conn, entry, result)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")
    atoms = reconstruct_atoms_at_step(loaded, step=1)

    assert isinstance(atoms, Atoms)
    assert len(atoms) == 2
    assert list(atoms.get_chemical_symbols()) == ["O", "O"]
    np.testing.assert_array_almost_equal(
        atoms.get_positions(), result.positions_history[1]
    )
    np.testing.assert_array_almost_equal(
        atoms.get_cell(), result.cell_history[1]
    )
    expected_vol = float(np.linalg.det(result.cell_history[1]))
    assert atoms.get_volume() == pytest.approx(expected_vol, rel=1e-4)


def test_reconstruct_atoms_step_zero(tmp_db: str) -> None:
    """Schritt 0 (Startkonfiguration) rekonstruieren."""
    conn = init_db(tmp_db)
    entry = _make_entry()
    result = _make_result(n_steps=3, n_atoms=2)

    store_result(conn, entry, result)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")
    atoms = reconstruct_atoms_at_step(loaded, step=0)

    np.testing.assert_array_almost_equal(
        atoms.get_positions(), result.positions_history[0]
    )


def test_reconstruct_atoms_last_step(tmp_db: str) -> None:
    """Letzter Schritt rekonstruieren."""
    conn = init_db(tmp_db)
    entry = _make_entry()
    result = _make_result(n_steps=5, n_atoms=2)

    store_result(conn, entry, result)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")
    atoms = reconstruct_atoms_at_step(loaded, step=4)

    np.testing.assert_array_almost_equal(
        atoms.get_positions(), result.positions_history[4]
    )


def test_reconstruct_atoms_out_of_range(tmp_db: str) -> None:
    """Schritt außerhalb des gültigen Bereichs → ValueError."""
    conn = init_db(tmp_db)
    entry = _make_entry()
    result = _make_result(n_steps=3, n_atoms=2)

    store_result(conn, entry, result)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")

    with pytest.raises(ValueError, match="außerhalb"):
        reconstruct_atoms_at_step(loaded, step=10)

    with pytest.raises(ValueError, match="außerhalb"):
        reconstruct_atoms_at_step(loaded, step=-1)


def test_reconstruct_atoms_no_trajectory(tmp_db: str) -> None:
    """Rekonstruktion ohne gespeicherte Trajektorie → ValueError."""
    conn = init_db(tmp_db)
    entry = _make_entry()
    result = TrajectoryResult()
    result.temperatures = [0.0]
    result.msd_values = [0.0]
    result.volumes = [125.0]
    result.ql_values = [0.76]
    result.energies = [-10.0]
    result.status = "converged"

    store_result(conn, entry, result)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")

    with pytest.raises(ValueError, match="Keine Trajektorie"):
        reconstruct_atoms_at_step(loaded, step=0)


# ── Tests: Overwrite / Insert-or-Replace ────────────────────────────────────

def test_store_overwrites_trajectory(tmp_db: str) -> None:
    """Erneutes Speichern überschreibt die alte Trajektorie."""
    conn = init_db(tmp_db)
    entry = _make_entry()

    result1 = _make_result(n_steps=3, n_atoms=2)
    store_result(conn, entry, result1)

    result2 = _make_result(n_steps=5, n_atoms=2)
    store_result(conn, entry, result2)
    conn.close()

    loaded = load_result(tmp_db, "mp-test")

    assert loaded.positions_history is not None
    assert len(loaded.positions_history) == 5  # zweite Speicherung gewinnt

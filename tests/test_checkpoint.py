"""Tests für Checkpoint-Serialisierung und Resume-Logik."""

from __future__ import annotations

import json
import os
import tempfile

import numpy as np
import pytest

from mat_sim.acquisition import MPEntry
from mat_sim.metrics import StepMetrics, TrajectoryResult
from mat_sim.pipeline import _deserialize_result, _serialize_result
from mat_sim.storage import (
    claim_next_structure,
    delete_checkpoint,
    has_checkpoint,
    ingest_structures,
    init_db,
    load_checkpoint,
    save_checkpoint,
)

# ── Fixtures ────────────────────────────────────────────────────────────────

@pytest.fixture
def tmp_db() -> str:
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        path = f.name
    os.unlink(path)  # remove so init_db creates fresh
    init_db(path)
    yield path
    if os.path.exists(path):
        os.unlink(path)
        for ext in ("-wal", "-shm"):
            p = path + ext
            if os.path.exists(p):
                os.unlink(p)


def _make_result(n_steps: int = 5) -> TrajectoryResult:
    """Erstelle ein TrajectoryResult mit n_steps Einträgen."""
    result = TrajectoryResult()
    for i in range(n_steps):
        m = StepMetrics(
            temperature=float(i * 10),
            volume=100.0 + i,
            energy=-10.0 - i,
            msd=0.01 * i,
            rdf_r=np.linspace(0, 6, 100),
            rdf_g=np.random.rand(100),
            ql=0.76 + 0.001 * i,
            positions=np.random.rand(4, 3),
            cell=np.eye(3) * (5.0 + 0.01 * i),
        )
        result.add(m)
    return result


# ── Tests: Serialisierung ───────────────────────────────────────────────────

def test_serialize_deserialize_roundtrip() -> None:
    """Serialisierung → Deserialisierung ergibt identisches TrajectoryResult."""
    original = _make_result(5)
    original.t_switch = 200.0
    original.t_decay = 500.0
    original.status = "converged"

    json_str = _serialize_result(original)
    restored = _deserialize_result(json_str)

    assert restored.temperatures == original.temperatures
    assert restored.volumes == original.volumes
    assert restored.energies == original.energies
    assert restored.msd_values == original.msd_values
    assert restored.ql_values == original.ql_values
    assert restored.t_switch == 200.0
    assert restored.t_decay == 500.0
    assert restored.status == "converged"

    # RDF history
    assert len(restored.rdf_history) == len(original.rdf_history)
    for (r1, g1), (r2, g2) in zip(restored.rdf_history, original.rdf_history):
        np.testing.assert_array_almost_equal(r1, r2)
        np.testing.assert_array_almost_equal(g1, g2)

    # Positions / Cell history
    assert len(restored.positions_history) == len(original.positions_history)
    for p1, p2 in zip(restored.positions_history, original.positions_history):
        np.testing.assert_array_almost_equal(p1, p2)

    assert len(restored.cell_history) == len(original.cell_history)
    for c1, c2 in zip(restored.cell_history, original.cell_history):
        np.testing.assert_array_almost_equal(c1, c2)


def test_serialize_empty_result() -> None:
    """Leeres TrajectoryResult serialisieren/deserialisieren."""
    result = TrajectoryResult()
    json_str = _serialize_result(result)
    restored = _deserialize_result(json_str)

    assert restored.temperatures == []
    assert restored.volumes == []
    assert restored.rdf_history == []
    assert restored.positions_history == []
    assert restored.cell_history == []
    assert restored.t_switch is None


def test_deserialize_partial_json() -> None:
    """Deserialisierung mit fehlenden Feldern sollte sicher funktionieren."""
    partial = json.dumps({"temperatures": [0.0, 10.0], "status": "timed_out"})
    result = _deserialize_result(partial)
    assert result.temperatures == [0.0, 10.0]
    assert result.status == "timed_out"
    assert result.rdf_history == []
    assert result.volumes == []


# ── Tests: Checkpoint storage ───────────────────────────────────────────────

def test_save_load_delete_checkpoint(tmp_db: str) -> None:
    """Checkpoint speichern, laden, löschen."""
    positions = np.random.rand(4, 3)
    cell = np.eye(3) * 5.0
    metrics = _serialize_result(_make_result(3))

    # Speichern
    save_checkpoint(tmp_db, "mp-test", 2, 20.0, positions, cell, metrics)
    assert has_checkpoint(tmp_db, "mp-test")

    # Laden
    cp = load_checkpoint(tmp_db, "mp-test")
    assert cp is not None
    assert cp.step_index == 2
    assert cp.temperature == 20.0
    np.testing.assert_array_almost_equal(cp.positions, positions)
    np.testing.assert_array_almost_equal(cp.cell, cell)
    assert "temperatures" in cp.metrics

    # Löschen
    delete_checkpoint(tmp_db, "mp-test")
    assert not has_checkpoint(tmp_db, "mp-test")
    assert load_checkpoint(tmp_db, "mp-test") is None


def test_save_checkpoint_overwrite(tmp_db: str) -> None:
    """Checkpoints für dasselbe Material werden überschrieben (Insert-or-Replace)."""
    positions = np.zeros((4, 3))
    cell = np.eye(3) * 5.0
    metrics = _serialize_result(_make_result(1))

    save_checkpoint(tmp_db, "mp-overwrite", 0, 0.0, positions, cell, metrics)
    save_checkpoint(tmp_db, "mp-overwrite", 5, 50.0, positions, cell, metrics)

    cp = load_checkpoint(tmp_db, "mp-overwrite")
    assert cp is not None
    assert cp.step_index == 5
    assert cp.temperature == 50.0


def test_load_checkpoint_nonexistent(tmp_db: str) -> None:
    """Laden eines nicht existierenden Checkpoints gibt None zurück."""
    assert load_checkpoint(tmp_db, "mp-nonexistent") is None
    assert not has_checkpoint(tmp_db, "mp-nonexistent")


# ── Tests: Checkpoint-Priorisierung in claim_next_structure ─────────────────

def test_claim_prioritizes_checkpoint(tmp_db: str) -> None:
    """Struktur mit Checkpoint wird vor Structure ohne Checkpoint geclaimt."""
    # Zwei Strukturen ingesten
    from pymatgen.core import Lattice, Structure

    struct = Structure(Lattice.cubic(5.0), ["O", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    entries = [
        MPEntry(material_id="mp-A", formula_pretty="O2", structure=struct),
        MPEntry(material_id="mp-B", formula_pretty="O2", structure=struct),
    ]
    ingest_structures(tmp_db, entries, "O")

    # Checkpoint für mp-B speichern (zweite in alphabetischer Reihenfolge)
    positions = np.zeros((2, 3))
    cell = np.eye(3) * 5.0
    metrics = _serialize_result(_make_result(2))
    save_checkpoint(tmp_db, "mp-B", 1, 10.0, positions, cell, metrics)

    # claim_next_structure sollte mp-B zuerst zurückgeben
    claimed = claim_next_structure(tmp_db, "worker-1")
    assert claimed is not None
    assert claimed.material_id == "mp-B"

    # Danach mp-A
    claimed2 = claim_next_structure(tmp_db, "worker-2")
    assert claimed2 is not None
    assert claimed2.material_id == "mp-A"


def test_claim_without_checkpoint(tmp_db: str) -> None:
    """Ohne Checkpoints wird alphabetisch geclaimt."""
    from pymatgen.core import Lattice, Structure

    struct = Structure(Lattice.cubic(5.0), ["O", "O"], [[0, 0, 0], [0.5, 0.5, 0.5]])
    entries = [
        MPEntry(material_id="mp-Z", formula_pretty="O2", structure=struct),
        MPEntry(material_id="mp-A", formula_pretty="O2", structure=struct),
    ]
    ingest_structures(tmp_db, entries, "O")

    claimed = claim_next_structure(tmp_db, "worker-1")
    assert claimed is not None
    assert claimed.material_id == "mp-A"  # alphabetisch zuerst

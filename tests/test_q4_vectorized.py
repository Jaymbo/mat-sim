"""Tests für die vektorisierte compute_steinhardt_q4-Implementierung.

Die Tests verifizieren, dass die vektorisierte Version (mit bincount)
dieselben Ergebnisse liefert wie eine unabhängige Referenz-Implementierung
mit expliziter Python-Schleife.
"""

from __future__ import annotations

import numpy as np
import pytest
from ase import Atoms
from ase.spacegroup import crystal

from mat_sim.metrics import compute_steinhardt_q4


def _reference_q4(atoms: Atoms, cutoff: float = 3.5) -> float:
    """Unabhängige Referenz-Implementierung (explizite Schleife)."""
    from ase.neighborlist import neighbor_list
    from scipy.special import sph_harm_y

    i, _j, _d, D = neighbor_list("ijdD", atoms, cutoff)
    if i.size == 0:
        return 0.0

    n_atoms = len(atoms)
    q4_sq = 0.0

    for atom_idx in range(n_atoms):
        mask = i == atom_idx
        if not np.any(mask):
            continue
        dv = D[mask]
        norms = np.linalg.norm(dv, axis=1)
        norms[norms == 0] = 1.0
        theta = np.arccos(dv[:, 2] / norms)
        phi = np.arctan2(dv[:, 1], dv[:, 0])
        qm = np.zeros(9, dtype=complex)
        for m_idx, m in enumerate(range(-4, 5)):
            qm[m_idx] = np.mean(sph_harm_y(4, m, theta, phi))
        q4_sq += np.sum(np.abs(qm) ** 2)

    return float(np.sqrt(4 * np.pi / 9 * q4_sq / n_atoms))


# ── Fixtures ────────────────────────────────────────────────────────────────

def _make_cubic_crystal() -> Atoms:
    """Einfacher kubischer Kristall (NaCl-Struktur, 8 Atome)."""
    return crystal(
        ["Na", "Cl"],
        [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=225,
        cellpar=[5.64, 5.64, 5.64, 90, 90, 90],
    )


def _make_tetragonal() -> Atoms:
    """Tetragonale Verzerrung der NaCl-Struktur."""
    return crystal(
        ["Na", "Cl"],
        [(0, 0, 0), (0.5, 0.5, 0.5)],
        spacegroup=123,
        cellpar=[5.0, 5.0, 6.5, 90, 90, 90],
    )


def _make_random_structure(n: int = 12) -> Atoms:
    """Zufällige Struktur ohne Symmetrie."""
    rng = np.random.default_rng(seed=42)
    positions = rng.random((n, 3)) * 6.0
    return Atoms(
        symbols=["O"] * n,
        positions=positions,
        cell=np.eye(3) * 6.0,
        pbc=True,
    )


# ── Tests: Übereinstimmung mit Referenz ─────────────────────────────────────

def test_q4_cubic_matches_reference() -> None:
    """Kubischer Kristall: vektorisiert == Referenz."""
    atoms = _make_cubic_crystal()
    q4_vec = compute_steinhardt_q4(atoms, cutoff=3.5)
    q4_ref = _reference_q4(atoms, cutoff=3.5)
    assert q4_vec == pytest.approx(q4_ref, rel=1e-10)


def test_q4_tetragonal_matches_reference() -> None:
    """Tetragonale Verzerrung: vektorisiert == Referenz."""
    atoms = _make_tetragonal()
    q4_vec = compute_steinhardt_q4(atoms, cutoff=3.5)
    q4_ref = _reference_q4(atoms, cutoff=3.5)
    assert q4_vec == pytest.approx(q4_ref, rel=1e-10)


def test_q4_random_matches_reference() -> None:
    """Zufällige Struktur: vektorisiert == Referenz."""
    atoms = _make_random_structure(12)
    q4_vec = compute_steinhardt_q4(atoms, cutoff=3.5)
    q4_ref = _reference_q4(atoms, cutoff=3.5)
    assert q4_vec == pytest.approx(q4_ref, rel=1e-10)


def test_q4_different_cutoffs() -> None:
    """Verschiedene Cutoffs: vektorisiert == Referenz."""
    atoms = _make_cubic_crystal()
    for cutoff in [2.5, 3.0, 4.0, 5.0]:
        q4_vec = compute_steinhardt_q4(atoms, cutoff=cutoff)
        q4_ref = _reference_q4(atoms, cutoff=cutoff)
        assert q4_vec == pytest.approx(q4_ref, rel=1e-10)


# ── Tests: Spezielle Fälle ──────────────────────────────────────────────────

def test_q4_no_neighbors() -> None:
    """Isolierte Atome (großer Cutoff-abstand) → Q4 = 0."""
    atoms = Atoms(
        symbols=["O", "O"],
        positions=[[0, 0, 0], [10, 10, 10]],
        cell=np.eye(3) * 20.0,
        pbc=True,
    )
    # Mit cutoff=3.5 gibt es keine Nachbarn in dieser Zelle
    q4 = compute_steinhardt_q4(atoms, cutoff=3.5)
    assert q4 == pytest.approx(0.0, abs=1e-12)


def test_q4_single_atom() -> None:
    """Einzelnes Atom → Q4 = 0 (keine Nachbarn)."""
    atoms = Atoms(
        symbols=["O"],
        positions=[[0, 0, 0]],
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    q4 = compute_steinhardt_q4(atoms, cutoff=3.5)
    assert q4 == pytest.approx(0.0, abs=1e-12)


def test_q4_is_non_negative() -> None:
    """Q4 ist immer nicht-negativ."""
    for atoms in [_make_cubic_crystal(), _make_tetragonal(), _make_random_structure()]:
        q4 = compute_steinhardt_q4(atoms, cutoff=3.5)
        assert q4 >= 0.0


def test_q4_is_finite() -> None:
    """Q4 ist endlich (kein NaN/Inf)."""
    for atoms in [_make_cubic_crystal(), _make_tetragonal(), _make_random_structure()]:
        q4 = compute_steinhardt_q4(atoms, cutoff=3.5)
        assert np.isfinite(q4)


# ── Tests: Determinismus ────────────────────────────────────────────────────

def test_q4_deterministic() -> None:
    """Zweimalige Berechnung gibt dasselbe Ergebnis."""
    atoms = _make_random_structure(10)
    q4_1 = compute_steinhardt_q4(atoms, cutoff=3.5)
    q4_2 = compute_steinhardt_q4(atoms, cutoff=3.5)
    assert q4_1 == pytest.approx(q4_2, rel=1e-12)

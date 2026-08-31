"""Tests für make_supercell_atoms()."""

from __future__ import annotations

import numpy as np
from ase import Atoms
from ase.build import bulk

from mat_sim.acquisition import make_supercell_atoms


# ── Fixtures ────────────────────────────────────────────────────────────────

def _simple_cell(n_atoms: int = 4) -> Atoms:
    """Eine kleine kubische Zelle mit n_atoms Atomen."""
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)
    #diamond hat 8 Atome in kubischer Zelle
    if n_atoms < len(atoms):
        # Nehme nur die ersten n_atoms (nur für Testzwecke)
        atoms = atoms[:n_atoms]
    return atoms


# ── Tests ───────────────────────────────────────────────────────────────────

def test_already_large_enough() -> None:
    """Zelle mit >= min_atoms Atomen → unverändert zurück."""
    atoms = bulk("Si", "diamond", a=5.43, cubic=True)  # 8 Atome
    # make_supercell(3×3×3) → 216 Atome
    from ase.build import make_supercell
    big = make_supercell(atoms, 3 * np.eye(3, dtype=int))

    result = make_supercell_atoms(big, min_atoms=100)
    assert result is big  # identisches Objekt, keine Kopie


def test_small_cell_expanded() -> None:
    """Primitive Zelle mit < min_atoms → Supercell mit >= min_atoms."""
    atoms = bulk("Si", "diamond", a=5.43)  # 2 Atome (primitive)
    assert len(atoms) == 2

    result = make_supercell_atoms(atoms, min_atoms=100)
    assert len(result) >= 100


def test_factor_calculation() -> None:
    """6-Atom-Zelle, min_atoms=100 → Faktor 3 (6*27=162 ≥ 100)."""
    # Erstelle eine 6-Atom-Zelle
    atoms = Atoms(
        symbols=["Si"] * 6,
        positions=np.random.rand(6, 3) * 5.0,
        cell=np.eye(3) * 5.0,
        pbc=True,
    )
    result = make_supercell_atoms(atoms, min_atoms=100)
    # 6 * 3³ = 162 ≥ 100
    assert len(result) == 6 * 27  # 162


def test_factor_capped_at_4() -> None:
    """2-Atom-Zelle, min_atoms=1000 → Faktor max 4 (2*64=128 < 1000, aber capped)."""
    atoms = bulk("Si", "diamond", a=5.43)  # 2 Atome
    result = make_supercell_atoms(atoms, min_atoms=1000)
    # Faktor capped at 4 → 2 * 4³ = 128 (nicht 1000)
    assert len(result) == 2 * 64  # 128


def test_min_atoms_zero_no_expansion() -> None:
    """min_atoms=0 → keine Supercell (Zelle unverändert)."""
    atoms = bulk("Si", "diamond", a=5.43)  # 2 Atome
    result = make_supercell_atoms(atoms, min_atoms=0)
    assert len(result) == 2


def test_preserves_chemical_symbols() -> None:
    """Supercell behält die chemischen Symbole bei."""
    atoms = bulk("NaCl", "rocksalt", a=5.64)  # 2 Atome (1 Na, 1 Cl)
    result = make_supercell_atoms(atoms, min_atoms=100)
    symbols = result.get_chemical_symbols()
    assert symbols.count("Na") == symbols.count("Cl")
    assert len(symbols) >= 100


def test_preserves_pbc() -> None:
    """Supercell behält PBC=True."""
    atoms = bulk("Si", "diamond", a=5.43)
    result = make_supercell_atoms(atoms, min_atoms=100)
    assert all(result.get_pbc())


def test_cell_scaled_correctly() -> None:
    """Zellvektoren der Supercell sind Vielfache der ursprünglichen."""
    atoms = bulk("Si", "diamond", a=5.43)
    original_cell = atoms.get_cell()
    result = make_supercell_atoms(atoms, min_atoms=100)
    result_cell = result.get_cell()
    # Bei isotroper Vergrößerung mit Faktor f:
    # result_cell[i] = f * original_cell[i]
    # Wir vergleichen die Längen
    orig_lengths = np.linalg.norm(original_cell, axis=1)
    result_lengths = np.linalg.norm(result_cell, axis=1)
    ratios = result_lengths / orig_lengths
    # Alle Verhältnisse sollten gleich sein (isotrop)
    assert np.allclose(ratios, ratios[0])

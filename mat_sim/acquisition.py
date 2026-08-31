"""Datenakquisition: Materials Project → ASE-Atoms.

Kernfunktionen
--------------
- ``query_mp_structures``  – Chemische Filter an MP, Rückgabe pymatgen-Strukturen
- ``pmg_to_ase``           – Typsichere Konvertierung pymatgen.Structure → ase.Atoms
- ``build_structure_batch`` – Komfort-Funktion: Query + Konvertierung in einem Aufruf

Suchstrategie
-------------
- ``energy_above_hull`` bis 0.1 eV/Atom (inkl. metastabile Phasen)
- GNoME / theoretische Strukturen explizit eingeschlossen (kein Filter
  auf ``theoretical`` oder ``is_stable``)
- Ternäre Erweiterung: Bei Eingabe von ``V-O`` werden zusätzlich alle
  Verbindungen V-O-X gefunden, die ein drittes Element enthalten.
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

from ase import Atoms
from ase.build import niggli_reduce, make_supercell

logger = logging.getLogger(__name__)

# Alle Elemente, die als drittes Dotierungs-Element in Frage kommen
# (kompakt — ohne Edelgase, ohne rein radioaktive)
_ALL_ELEMENTS: list[str] = [
    "H", "Li", "Be", "B", "C", "N", "O", "F", "Na", "Mg", "Al", "Si", "P", "S",
    "Cl", "K", "Ca", "Sc", "Ti", "V", "Cr", "Mn", "Fe", "Co", "Ni", "Cu", "Zn",
    "Ga", "Ge", "As", "Se", "Br", "Rb", "Sr", "Y", "Zr", "Nb", "Mo", "Tc", "Ru",
    "Rh", "Pd", "Ag", "Cd", "In", "Sn", "Sb", "Te", "I", "Cs", "Ba", "La", "Ce",
    "Pr", "Nd", "Sm", "Eu", "Gd", "Tb", "Dy", "Ho", "Er", "Tm", "Yb", "Lu", "Hf",
    "Ta", "W", "Re", "Os", "Ir", "Pt", "Au", "Hg", "Tl", "Pb", "Bi",
]


# ── Hilfs-Datentyp ──────────────────────────────────────────────────────────
@dataclass(frozen=True)
class MPEntry:
    """Container für eine einzelne MP-Struktur nebst Metadaten."""

    material_id: str
    formula_pretty: str
    structure: Structure


# ── MP-Query ────────────────────────────────────────────────────────────────
def query_mp_structures(
    chemsys: str,
    api_key: str | None = None,
    max_results: int = 2000,
    e_hull_max: float = 0.1,
    include_ternary: bool = True,
) -> list[MPEntry]:
    """Strukturen aus dem Materials Project abfragen.

    Parameter
    ---------
    chemsys
        Chemisches System, z. B. ``"V-O"``, ``"Ti-O"``, ``"Sr-Ti-O"``.
    api_key
        Materials-Project-API-Key.  Wenn *None*, wird die Umgebungsvariable
        ``MP_API_KEY`` verwendet.
    max_results
        Maximale Anzahl zurückgegebener Einträge (Default 2000).
    e_hull_max
        Maximal zulässige Energie über der Hull in eV/Atom (Default 0.1).
        ``0.0`` = nur stabile Phasen; ``0.1`` = inkl. metastabile Phasen.
    include_ternary
        Wenn *True* (Default), werden bei binären Systemen wie ``"V-O"``
        zusätzlich ternäre Verbindungen ``V-O-X`` gesucht (drittes Element
        als Dotierung).

    Returns
    -------
    list[MPEntry]
        Sortierte Liste der gefundenen Strukturen.
    """
    api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Kein MP-API-Key gefunden.  Setze die Umgebungsvariable MP_API_KEY "
            "oder übergebe api_key explizit."
        )

    elements = chemsys.split("-")
    fields = ["material_id", "formula_pretty", "structure"]

    # ── 1. Exakte chemsys-Abfrage (binär oder ternär) ──────────────────
    all_docs: list = []
    seen_ids: set[str] = set()

    with MPRester(api_key) as rester:
        # Exaktes chemsys
        logger.info(
            "MP-Query: chemsys=%s, e_hull ≤ %.3f eV/atom, max=%d",
            chemsys, e_hull_max, max_results,
        )
        docs = rester.materials.summary.search(
            chemsys=chemsys,
            energy_above_hull=(0, e_hull_max),
            fields=fields,
            num_sites=(2, 50),
            num_elements=(len(elements), len(elements)),
        )
        for doc in docs:
            if doc.material_id not in seen_ids:
                all_docs.append(doc)
                seen_ids.add(doc.material_id)

        # ── 2. Ternäre Erweiterung bei binären Systemen ────────────────
        if include_ternary and len(elements) == 2:
            el_a, el_b = elements
            logger.info(
                "Ternäre Erweiterung: suche %s-%s-X für alle Elemente X …",
                el_a, el_b,
            )
            for el_x in _ALL_ELEMENTS:
                if el_x in (el_a, el_b):
                    continue
                ternary_chemsys = "-".join(sorted([el_a, el_b, el_x]))
                try:
                    docs = rester.materials.summary.search(
                        chemsys=ternary_chemsys,
                        energy_above_hull=(0, e_hull_max),
                        fields=fields,
                        num_sites=(2, 50),
                        num_elements=(3, 3),
                    )
                    for doc in docs:
                        if doc.material_id not in seen_ids:
                            all_docs.append(doc)
                            seen_ids.add(doc.material_id)
                except Exception as exc:  # noqa: BLE001
                    logger.debug("Ternary %s fehlgeschlagen: %s", ternary_chemsys, exc)

    logger.info(
        "Query beendet: %d eindeutige Strukturen für %s",
        len(all_docs), chemsys,
    )

    entries: list[MPEntry] = []
    for doc in all_docs[:max_results]:
        struct: Structure = doc.structure
        # Primitive Standard-Zelle für effizientere MD
        try:
            spa = SpacegroupAnalyzer(struct)
            struct = spa.get_primitive_standard_structure()
        except Exception:  # noqa: BLE001
            pass
        entries.append(
            MPEntry(
                material_id=doc.material_id,
                formula_pretty=doc.formula_pretty,
                structure=struct,
            )
        )

    return entries


# ── Konvertierung pymatgen → ASE ───────────────────────────────────────────
def pmg_to_ase(structure: Structure) -> Atoms:
    """``pymatgen.Structure`` → ``ase.Atoms`` konvertieren.

    Übernimmt Gittervektoren, skalierte Koordinaten und Element-Symbole.
    Wendet anschließend eine Niggli-Reduktion an, um eine untere
    Dreiecksmatrix als Zelle zu erhalten (erforderlich für ASE-NPT-MD).
    """
    lattice = structure.lattice
    cell = lattice.matrix  # (3, 3) ndarray, Zeilen = Vektoren

    symbols = [str(site.specie.symbol) for site in structure]
    positions = structure.frac_coords  # skalierte Koordinaten

    atoms = Atoms(
        symbols=symbols,
        scaled_positions=positions,
        cell=cell,
        pbc=True,
    )

    # Niggli-Reduktion → untere Dreiecksmatrix (Voraussetzung für NPT-MD)
    niggli_reduce(atoms)

    return atoms


def make_supercell_atoms(atoms: Atoms, min_atoms: int = 100) -> Atoms:
    """Supercell aus einer primitiven Zelle erstellen.

    Vergrößert die Zelle so, dass mindestens ``min_atoms`` Atome
    enthalten sind.  Die Vergrößerung erfolgt isotrop (gleicher Faktor
    in alle drei Richtungen), um Verzerrungen zu vermeiden.

    Phasenübergänge in MD benötigen ausreichend große Zellen, damit
    die neue Phase genügend Freiheitsgrade hat.  Primitive Zellen mit
    < 50 Atomen sind für Phasenübergangssimulationen ungeeignet.

    Parameters
    ----------
    atoms
        Primitive (oder konventionelle) Zelle als ASE-Atoms.
    min_atoms
        Mindestanzahl Atome in der Supercell (Default: 100).

    Returns
    -------
    Atoms
        Supercell mit ≥ ``min_atoms`` Atomen.
    """
    n = len(atoms)
    if n >= min_atoms:
        return atoms

    # Isotroper Faktor: kleinste ganze Zahl, die min_atoms erreicht
    factor = 1
    while n * factor**3 < min_atoms:
        factor += 1

    # Bei sehr kleinen Zellen kann factor groß werden → auf max 4 begrenzen
    factor = min(factor, 4)

    size = factor * np.eye(3, dtype=int)
    supercell = make_supercell(atoms, size)

    logger.info(
        "Supercell: %d → %d Atome (Faktor %d×%d×%d)",
        n, len(supercell), factor, factor, factor,
    )
    return supercell


# ── Komfort-Funktion ────────────────────────────────────────────────────────
def build_structure_batch(
    chemsys: str | Iterable[str],
    api_key: str | None = None,
    max_results_per_sys: int = 2000,
    e_hull_max: float = 0.1,
    include_ternary: bool = True,
) -> list[tuple[MPEntry, Atoms]]:
    """MP-Query + ASE-Konvertierung für ein oder mehrere chemische Systeme.

    Returns
    -------
    list[tuple[MPEntry, Atoms]]
        Liste von ``(MPEntry, ase.Atoms)``-Paaren.
    """
    systems = [chemsys] if isinstance(chemsys, str) else list(chemsys)
    batch: list[tuple[MPEntry, Atoms]] = []

    for sys in systems:
        try:
            entries = query_mp_structures(
                chemsys=sys,
                api_key=api_key,
                max_results=max_results_per_sys,
                e_hull_max=e_hull_max,
                include_ternary=include_ternary,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[WARN] Query für {sys!r} fehlgeschlagen: {exc}")
            continue

        for entry in entries:
            try:
                atoms = pmg_to_ase(entry.structure)
            except Exception as exc:  # noqa: BLE001
                print(
                    f"[WARN] Konvertierung von {entry.material_id} "
                    f"({entry.formula_pretty}) fehlgeschlagen: {exc}"
                )
                continue
            batch.append((entry, atoms))

    return batch

"""Datenakquisition: Materials Project → ASE-Atoms.

Kernfunktionen
--------------
- ``query_mp_structures``  – Chemische Filter an MP, Rückgabe pymatgen-Strukturen
- ``pmg_to_ase``           – Typsichere Konvertierung pymatgen.Structure → ase.Atoms
- ``build_structure_batch`` – Komfort-Funktion: Query + Konvertierung in einem Aufruf
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable

from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from mp_api.client import MPRester

from ase import Atoms
from ase.build import bulk, niggli_reduce


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
    max_results: int = 50,
    stable_only: bool = True,
) -> list[MPEntry]:
    """Strukturen aus dem Materials Project abfragen.

    Parameters
    ----------
    chemsys
        Chemisches System, z. B. ``"V-O"``, ``"Ti-O"``, ``"Sr-Ti-O"``.
    api_key
        Materials-Project-API-Key.  Wenn *None*, wird die Umgebungsvariable
        ``MP_API_KEY`` verwendet.
    max_results
        Maximale Anzahl zurückgegebener Einträge.
    stable_only
        Wenn *True*, werden nur thermodynamisch stabile Phasen (energy_above_hull ≈ 0)
        abgefragt.

    Returns
    -------
    list[MPEntry]
        Sortierte Liste der gefundenen Strukturen.
    """
    api_key = api_key or os.environ.get("MP_API_KEY")
    if not api_key:
        # Fallback: .env laden, falls run.py nicht der Einstiegspunkt war
        from dotenv import load_dotenv
        load_dotenv()
        api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError(
            "Kein MP-API-Key gefunden.  Setze die Umgebungsvariable MP_API_KEY "
            "oder übergebe api_key explizit."
        )

    criteria: dict[str, object] = {
        "chemsys": chemsys,
    }
    if stable_only:
        criteria["energy_above_hull"] = (0, 0)  # nur stabile Phasen

    fields = ["material_id", "formula_pretty", "structure"]

    with MPRester(api_key) as rester:
        docs = rester.materials.summary.search(
            **criteria,
            fields=fields,
            num_sites=(2, 50),
        )

    entries: list[MPEntry] = []
    for doc in docs[:max_results]:
        struct: Structure = doc.structure
        # Primitive Standard-Zelle für effizientere MD
        try:
            spa = SpacegroupAnalyzer(struct)
            struct = spa.get_primitive_standard_structure()
        except Exception:  # noqa: BLE001 – Symmetrie-Erkennung kann scheitern
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


# ── Komfort-Funktion ────────────────────────────────────────────────────────
def build_structure_batch(
    chemsys: str | Iterable[str],
    api_key: str | None = None,
    max_results_per_sys: int = 50,
    stable_only: bool = True,
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
                stable_only=stable_only,
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

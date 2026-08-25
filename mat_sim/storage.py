"""Persistierung der Ergebnisse in einer SQLite-Datenbank.

Tabellen
--------
``structures`` – Warteschlange (Queue) der heruntergeladenen Strukturen:
    material_id, formula, chemsys, structure_json, status, claimed_by, ...
    Status: 'pending' → 'processing' → 'done' / 'error'

``materials`` – Ergebnisse der MD-Simulationen:
    material_id, formula, t_switch, t_decay, rdf_before (JSON), rdf_after (JSON),
    temperature_ramp (JSON), msd_curve (JSON)

Workflow
--------
1. **Ingest**: ``ingest_structures()`` lädt Strukturen von MP → ``structures``
   (status='pending').  Einmal pro chemisches System.
2. **Process**: ``claim_next_structure()`` holt atomar die nächste pending-
   Struktur (status → 'processing').  Nach Simulation → 'done'.
3. **Recovery**: ``reset_stale()`` setzt abgestürzte 'processing'-Einträge
   nach Timeout zurück auf 'pending'.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .acquisition import MPEntry
from .metrics import TrajectoryResult

logger = logging.getLogger(__name__)


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Datenbank initialisieren (Tabellen anlegen, falls nicht vorhanden)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
    conn.execute("PRAGMA journal_mode=WAL")  # bessere Parallelität
    conn.execute("PRAGMA busy_timeout=30000")  # 30 s auf Lock warten
    conn.execute("""
        CREATE TABLE IF NOT EXISTS structures (
            material_id     TEXT PRIMARY KEY,
            formula         TEXT NOT NULL,
            chemsys         TEXT NOT NULL,
            structure_json  TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'pending',
            claimed_by      TEXT,
            claimed_at      TIMESTAMP,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE INDEX IF NOT EXISTS idx_structures_status
        ON structures(status)
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS materials (
            material_id     TEXT PRIMARY KEY,
            formula         TEXT NOT NULL,
            status          TEXT NOT NULL DEFAULT 'converged',
            t_switch        REAL,
            t_decay         REAL,
            rdf_before_json TEXT,
            rdf_after_json  TEXT,
            temperatures    TEXT,
            msd_values      TEXT,
            ql_values       TEXT,
            volumes         TEXT,
            energies        TEXT,
            structure_before_json TEXT,
            structure_after_json  TEXT,
            rdf_history_json      TEXT,
            cooling_score   REAL,
            heating_score   REAL,
            total_score     REAL,
            contrast_score  REAL,
            optical_evaluated TIMESTAMP,
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # Schema-Migration: Spalten nachträglich hinzufügen, falls DB schon existiert
    _migrate_columns(conn, "materials", {
        "structure_before_json": "TEXT",
        "structure_after_json": "TEXT",
        "rdf_history_json": "TEXT",
        "cooling_score": "REAL",
        "heating_score": "REAL",
        "total_score": "REAL",
        "contrast_score": "REAL",
        "optical_evaluated": "TIMESTAMP",
    })

    conn.commit()
    return conn


def _migrate_columns(conn: sqlite3.Connection, table: str, columns: dict[str, str]) -> None:
    """Spalten nachträglich hinzufügen, falls sie noch nicht existieren (Schema-Migration)."""
    existing = {row[1] for row in conn.execute(f"PRAGMA table_info({table})").fetchall()}
    for col_name, col_type in columns.items():
        if col_name not in existing:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {col_name} {col_type}")
            logger.info("Schema-Migration: %s.%s (%s) hinzugefügt", table, col_name, col_type)


def _rdf_to_json(rdf: tuple[np.ndarray, np.ndarray]) -> str:
    r, g = rdf
    return json.dumps({"r": r.tolist(), "g": g.tolist()})


def _rdf_history_to_json(history: list[tuple[np.ndarray, np.ndarray]]) -> str:
    """Vollständige RDF-History als JSON serialisieren."""
    return json.dumps([
        {"r": r.tolist(), "g": g.tolist()} for r, g in history
    ])


def _rdf_history_from_json(raw: str | None) -> list[tuple[np.ndarray, np.ndarray]] | None:
    """JSON → Liste von (r, g)-Tupeln deserialisieren."""
    if raw is None:
        return None
    obj = json.loads(raw)
    return [(np.array(item["r"]), np.array(item["g"])) for item in obj]


def _atoms_to_json(atoms) -> str:
    """ASE-Atoms-Objekt als JSON serialisieren (Positionen, Zelle, Symbole)."""
    return json.dumps({
        "symbols": list(atoms.get_chemical_symbols()),
        "positions": atoms.get_positions().tolist(),
        "cell": atoms.get_cell().tolist(),
        "pbc": list(atoms.get_pbc()),
        "volume": atoms.get_volume(),
    })


def _atoms_from_json(raw: str | None):
    """JSON → ASE-Atoms-Objekt deserialisieren."""
    from ase import Atoms

    if raw is None:
        return None
    obj = json.loads(raw)
    return Atoms(
        symbols=obj["symbols"],
        positions=obj["positions"],
        cell=obj["cell"],
        pbc=obj["pbc"],
    )


def store_result(
    conn: sqlite3.Connection,
    entry: MPEntry,
    result: TrajectoryResult,
    snapshots: dict | None = None,
) -> None:
    """Ein einzelnes Ergebnis in die DB schreiben (Insert-or-Replace).

    Parameters
    ----------
    snapshots
        ``{"before": ((r, g), atoms), "after": ((r, g), atoms)}`` oder
        ``{"before": (r, g), "after": (r, g)}`` (altes Format, nur RDF).
    """
    snapshots = snapshots or {}

    rdf_before = None
    rdf_after = None
    atoms_before = None
    atoms_after = None

    for key in ("before", "after"):
        val = snapshots.get(key)
        if val is None:
            continue

        # Format: ((r, g), atoms) — RDF + Struktur
        if (
            isinstance(val, tuple)
            and len(val) == 2
            and isinstance(val[0], tuple)
            and len(val[0]) == 2
            and hasattr(val[1], "get_positions")
        ):
            rdf_tuple = (np.asarray(val[0][0]), np.asarray(val[0][1]))
            atoms_json = _atoms_to_json(val[1])
        # Format: (r, g) — nur RDF (altes Format)
        elif isinstance(val, tuple) and len(val) == 2:
            rdf_tuple = (np.asarray(val[0]), np.asarray(val[1]))
            atoms_json = None
        else:
            continue

        rdf_json = _rdf_to_json(rdf_tuple)
        if key == "before":
            rdf_before = rdf_json
            atoms_before = atoms_json
        else:
            rdf_after = rdf_json
            atoms_after = atoms_json

    rdf_history_json = _rdf_history_to_json(result.rdf_history) if result.rdf_history else None

    conn.execute(
        """
        INSERT OR REPLACE INTO materials
            (material_id, formula, status, t_switch, t_decay,
             rdf_before_json, rdf_after_json,
             temperatures, msd_values, ql_values, volumes, energies,
             structure_before_json, structure_after_json, rdf_history_json)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            entry.material_id,
            entry.formula_pretty,
            result.status,
            result.t_switch,
            result.t_decay,
            rdf_before,
            rdf_after,
            json.dumps(result.temperatures),
            json.dumps(result.msd_values),
            json.dumps(result.ql_values),
            json.dumps(result.volumes),
            json.dumps(result.energies),
            atoms_before,
            atoms_after,
            rdf_history_json,
        ),
    )
    conn.commit()


def store_optical_scores(
    db_path: str | Path,
    material_id: str,
    cooling_score: float,
    heating_score: float,
    total_score: float,
    contrast_score: float | None = None,
) -> None:
    """Optische Scores für ein Material in der DB persistieren.

    Parameters
    ----------
    contrast_score
        Optionaler Switching-Kontrast-Score (0–100).  Wird ignoriert,
        wenn *None* (Abwärtskompatibilität).
    """
    conn = init_db(db_path)
    try:
        if contrast_score is not None:
            conn.execute(
                """
                UPDATE materials
                SET cooling_score = ?, heating_score = ?, total_score = ?,
                    contrast_score = ?, optical_evaluated = CURRENT_TIMESTAMP
                WHERE material_id = ?
                """,
                (cooling_score, heating_score, total_score,
                 contrast_score, material_id),
            )
        else:
            conn.execute(
                """
                UPDATE materials
                SET cooling_score = ?, heating_score = ?, total_score = ?,
                    optical_evaluated = CURRENT_TIMESTAMP
                WHERE material_id = ?
                """,
                (cooling_score, heating_score, total_score, material_id),
            )
        conn.commit()
    finally:
        conn.close()


def list_switching_materials(db_path: str | Path) -> list[str]:
    """Material-IDs aller Materialien mit detektiertem T_switch zurückgeben."""
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT material_id FROM materials "
            "WHERE t_switch IS NOT NULL AND status != 'diverged' "
            "ORDER BY material_id"
        ).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def list_unevaluated_materials(db_path: str | Path, only_switching: bool = True) -> list[str]:
    """Material-IDs ohne gespeicherte optische Scores zurückgeben.

    Parameters
    ----------
    only_switching
        Wenn *True*, nur Materialien mit T_switch berücksichtigen.
    """
    conn = init_db(db_path)
    try:
        query = (
            "SELECT material_id FROM materials "
            "WHERE total_score IS NULL AND status != 'diverged'"
        )
        if only_switching:
            query += " AND t_switch IS NOT NULL"
        query += " ORDER BY material_id"
        rows = conn.execute(query).fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


def update_t_switch(db_path: str | Path, material_id: str, t_switch: float | None) -> None:
    """T_switch-Wert in der DB aktualisieren (für --recheck-switch)."""
    conn = init_db(db_path)
    try:
        conn.execute(
            "UPDATE materials SET t_switch = ? WHERE material_id = ?",
            (t_switch, material_id),
        )
        conn.commit()
    finally:
        conn.close()


def store_batch(
    db_path: str | Path,
    results: Sequence[tuple[MPEntry, TrajectoryResult, dict]],
) -> None:
    """Mehrere Ergebnisse persistieren."""
    conn = init_db(db_path)
    try:
        for entry, result, snapshots in results:
            store_result(conn, entry, result, snapshots)
    finally:
        conn.close()


# ── Auslesen ────────────────────────────────────────────────────────────────
@dataclass
class StoredMaterial:
    """Datensatz eines gespeicherten Materials (aus DB geladen)."""

    material_id: str
    formula: str
    status: str
    t_switch: float | None
    t_decay: float | None
    rdf_before: tuple[np.ndarray, np.ndarray] | None
    rdf_after: tuple[np.ndarray, np.ndarray] | None
    temperatures: list[float]
    msd_values: list[float]
    ql_values: list[float]
    volumes: list[float]
    energies: list[float]
    # Neue Felder (können None sein bei alten DB-Einträgen)
    structure_before: object | None = None  # ase.Atoms oder None
    structure_after: object | None = None
    rdf_history: list | None = None  # list[tuple[np.ndarray, np.ndarray]] oder None
    cooling_score: float | None = None
    heating_score: float | None = None
    total_score: float | None = None
    contrast_score: float | None = None
    optical_evaluated: str | None = None


def _rdf_from_json(raw: str | None) -> tuple[np.ndarray, np.ndarray] | None:
    if raw is None:
        return None
    obj = json.loads(raw)
    return np.array(obj["r"]), np.array(obj["g"])


def load_result(db_path: str | Path, material_id: str) -> StoredMaterial:
    """Ein einzelnes Material aus der SQLite-Datenbank laden.

    Raises
    ------
    KeyError
        Wenn ``material_id`` nicht in der Datenbank gefunden wird.
    """
    conn = init_db(db_path)
    try:
        row = conn.execute(
            "SELECT material_id, formula, status, t_switch, t_decay, "
            "rdf_before_json, rdf_after_json, "
            "temperatures, msd_values, ql_values, volumes, energies, "
            "structure_before_json, structure_after_json, rdf_history_json, "
            "cooling_score, heating_score, total_score, contrast_score, "
            "optical_evaluated "
            "FROM materials WHERE material_id = ?",
            (material_id,),
        ).fetchone()
    finally:
        conn.close()

    if row is None:
        raise KeyError(f"Material-ID {material_id!r} nicht in DB gefunden.")

    return StoredMaterial(
        material_id=row[0],
        formula=row[1],
        status=row[2],
        t_switch=row[3],
        t_decay=row[4],
        rdf_before=_rdf_from_json(row[5]),
        rdf_after=_rdf_from_json(row[6]),
        temperatures=json.loads(row[7]),
        msd_values=json.loads(row[8]),
        ql_values=json.loads(row[9]),
        volumes=json.loads(row[10]),
        energies=json.loads(row[11]) if row[11] else [],
        structure_before=_atoms_from_json(row[12]),
        structure_after=_atoms_from_json(row[13]),
        rdf_history=_rdf_history_from_json(row[14]),
        cooling_score=row[15],
        heating_score=row[16],
        total_score=row[17],
        contrast_score=row[18],
        optical_evaluated=row[19],
    )


def list_material_ids(db_path: str | Path) -> list[str]:
    """Alle gespeicherten Material-IDs zurückgeben (für Übersicht/Autovervollständigung)."""
    conn = init_db(db_path)
    try:
        rows = conn.execute("SELECT material_id FROM materials ORDER BY material_id").fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]


# ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══
#  STRUCTURES QUEUE
# ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══ ══


@dataclass
class QueuedStructure:
    """Ein Eintrag aus der structures-Queue."""

    material_id: str
    formula: str
    chemsys: str
    structure_json: str


def _structure_to_json(structure) -> str:
    """pymatgen.Structure → JSON-String serialisieren."""
    from pymatgen.core import Structure

    if isinstance(structure, Structure):
        return structure.to_json()
    # Fallback: bereits JSON-String oder dict
    if isinstance(structure, str):
        return structure
    return json.dumps(structure)


def _structure_from_json(raw: str):
    """JSON-String → pymatgen.Structure deserialisieren."""
    from pymatgen.core import Structure

    return Structure.from_str(raw, fmt="json")


def ingest_structures(
    db_path: str | Path,
    entries: list[MPEntry],
    chemsys: str,
) -> int:
    """Strukturen in die Queue-Tabelle einfügen (Idempotent, Insert-or-Ignore).

    Returns
    -------
    int
        Anzahl tatsächlich neu eingefügter Strukturen.
    """
    conn = init_db(db_path)
    inserted = 0
    try:
        for entry in entries:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO structures
                    (material_id, formula, chemsys, structure_json, status)
                VALUES (?, ?, ?, ?, 'pending')
                """,
                (
                    entry.material_id,
                    entry.formula_pretty,
                    chemsys,
                    _structure_to_json(entry.structure),
                ),
            )
            inserted += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return inserted


def claim_next_structure(
    db_path: str | Path,
    worker_id: str,
) -> QueuedStructure | None:
    """Atomar die nächste pending-Struktur claimen (status → 'processing').

    Verwendet eine SQLite-Transaktion, um sicherzustellen, dass keine zwei
    Worker dieselbe Struktur bekommen.

    Returns
    -------
    QueuedStructure | None
        Die geclaimte Struktur oder *None*, wenn keine pending mehr vorhanden.
    """
    conn = init_db(db_path)
    try:
        conn.execute("BEGIN IMMEDIATE")
        row = conn.execute(
            """
            SELECT material_id, formula, chemsys, structure_json
            FROM structures
            WHERE status = 'pending'
            ORDER BY material_id
            LIMIT 1
            """
        ).fetchone()

        if row is None:
            conn.execute("ROLLBACK")
            return None

        material_id = row[0]
        conn.execute(
            """
            UPDATE structures
            SET status = 'processing', claimed_by = ?, claimed_at = ?
            WHERE material_id = ? AND status = 'pending'
            """,
            (worker_id, time.strftime("%Y-%m-%d %H:%M:%S"), material_id),
        )
        conn.execute("COMMIT")
        return QueuedStructure(
            material_id=row[0],
            formula=row[1],
            chemsys=row[2],
            structure_json=row[3],
        )
    except Exception:
        conn.execute("ROLLBACK")
        raise
    finally:
        conn.close()


def mark_structure_done(db_path: str | Path, material_id: str) -> None:
    """Struktur nach erfolgreicher Simulation als 'done' markieren."""
    conn = init_db(db_path)
    try:
        conn.execute(
            "UPDATE structures SET status = 'done' WHERE material_id = ?",
            (material_id,),
        )
        conn.commit()
    finally:
        conn.close()


def mark_structure_error(db_path: str | Path, material_id: str) -> None:
    """Struktur nach fehlgeschlagener Simulation als 'error' markieren."""
    conn = init_db(db_path)
    try:
        conn.execute(
            "UPDATE structures SET status = 'error' WHERE material_id = ?",
            (material_id,),
        )
        conn.commit()
    finally:
        conn.close()


def reset_stale(
    db_path: str | Path,
    stale_minutes: int = 30,
) -> int:
    """'processing'-Einträge, die zu lange laufen, zurück auf 'pending' setzen.

    Returns
    -------
    int
        Anzahl zurückgesetzter Einträge.
    """
    conn = init_db(db_path)
    try:
        cur = conn.execute(
            """
            UPDATE structures
            SET status = 'pending', claimed_by = NULL, claimed_at = NULL
            WHERE status = 'processing'
              AND claimed_at IS NOT NULL
              AND (
                CAST(strftime('%s', 'now') AS INTEGER)
                - CAST(strftime('%s', claimed_at) AS INTEGER)
              ) > ?
            """,
            (stale_minutes * 60,),
        )
        conn.commit()
        return cur.rowcount
    finally:
        conn.close()


def queue_stats(db_path: str | Path) -> dict[str, int]:
    """Queue-Statistiken zurückgeben: counts per status."""
    conn = init_db(db_path)
    try:
        rows = conn.execute(
            "SELECT status, COUNT(*) FROM structures GROUP BY status"
        ).fetchall()
    finally:
        conn.close()
    stats = {r[0]: r[1] for r in rows}
    stats.setdefault("pending", 0)
    stats.setdefault("processing", 0)
    stats.setdefault("done", 0)
    stats.setdefault("error", 0)
    stats["total"] = sum(stats[s] for s in ("pending", "processing", "done", "error"))
    return stats

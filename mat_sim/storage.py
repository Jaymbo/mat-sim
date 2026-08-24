"""Persistierung der Ergebnisse in einer SQLite-Datenbank.

Tabelle ``materials`` speichert pro Kristall:
  material_id, formula, t_switch, t_decay, rdf_before (JSON), rdf_after (JSON),
  temperature_ramp (JSON), msd_curve (JSON)
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np

from .acquisition import MPEntry
from .metrics import TrajectoryResult


def init_db(db_path: str | Path) -> sqlite3.Connection:
    """Datenbank initialisieren (Tabelle anlegen, falls nicht vorhanden)."""
    db_path = Path(db_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(db_path))
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
            created_at      TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.commit()
    return conn


def _rdf_to_json(rdf: tuple[np.ndarray, np.ndarray]) -> str:
    r, g = rdf
    return json.dumps({"r": r.tolist(), "g": g.tolist()})


def store_result(
    conn: sqlite3.Connection,
    entry: MPEntry,
    result: TrajectoryResult,
    snapshots: dict[str, tuple[np.ndarray, np.ndarray]] | None = None,
) -> None:
    """Ein einzelnes Ergebnis in die DB schreiben (Insert-or-Replace)."""
    snapshots = snapshots or {}
    rdf_before = _rdf_to_json(snapshots["before"]) if "before" in snapshots else None
    rdf_after = _rdf_to_json(snapshots["after"]) if "after" in snapshots else None

    conn.execute(
        """
        INSERT OR REPLACE INTO materials
            (material_id, formula, status, t_switch, t_decay,
             rdf_before_json, rdf_after_json,
             temperatures, msd_values, ql_values, volumes, energies)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
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
        ),
    )
    conn.commit()


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
            "temperatures, msd_values, ql_values, volumes, energies "
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
        energies=json.loads(row[11]),
    )


def list_material_ids(db_path: str | Path) -> list[str]:
    """Alle gespeicherten Material-IDs zurückgeben (für Übersicht/Autovervollständigung)."""
    conn = init_db(db_path)
    try:
        rows = conn.execute("SELECT material_id FROM materials ORDER BY material_id").fetchall()
    finally:
        conn.close()
    return [r[0] for r in rows]

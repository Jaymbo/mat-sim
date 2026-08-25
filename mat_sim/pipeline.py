"""End-to-End-Orchestrierung der Screening-Pipeline.

Verbindet Akquisition → Calculator → MD-Rampe → Metriken → Speicherung.

Zwei-Phasen-Workflow
---------------------
1. **Ingest** (``ingest_structures``):  Einmalig Strukturen von Materials
   Project herunterladen und in der SQLite-DB (Tabelle ``structures``)
   speichern.  Status='pending'.
2. **Process** (``run_pipeline``):  Calculator initialisieren, dann in einer
   Schleife atomar die nächste pending-Struktur aus der DB claimen,
   simulieren, Ergebnis speichern, als 'done' markieren.

   Mehrere Worker (SLURM-Jobs) können parallel laufen: Jeder claimt
   exklusiv eine Struktur dank SQLite-Transaktionen.

SLURM Time-Out-Handling:
  Die Pipeline überwacht die verstrichene Laufzeit.  Sobald die
  konfigurierte Dauer (``duration_min``) minus 2 Minuten Puffer
  erreicht ist, wird die Schleife sauber abgebrochen, die DB
  geschlossen und Exit-Code 88 zurückgegeben (Signal für
  Auto-Rescheduling).  Ein kompletter Durchlauf endet mit Code 0.
"""

from __future__ import annotations

import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Literal

from ase import Atoms
from ase.calculators.calculator import Calculator

from .acquisition import MPEntry, query_mp_structures, pmg_to_ase
from .calculator import get_calculator
from .md import ThermalRamp, RampConfig
from .metrics import TrajectoryResult
from .storage import (
    init_db,
    store_result,
    ingest_structures,
    claim_next_structure,
    mark_structure_done,
    mark_structure_error,
    reset_stale,
    queue_stats,
    _structure_from_json,
)

logger = logging.getLogger(__name__)

# Puffer in Sekunden vor dem harten Time-Out (für sauberen Abschluss)
_TIMEOUT_BUFFER_S = 2 * 60  # 2 Minuten


@dataclass
class PipelineConfig:
    """Gesamtkonfiguration der Pipeline."""

    chemsys_list: list[str]
    api_key: str | None = None
    max_results_per_sys: int = 2000
    e_hull_max: float = 0.1  # eV/atom — inkl. metastabile Phasen
    include_ternary: bool = True
    mlip_backend: Literal["mace", "chgnet"] = "mace"
    device: str = "cpu"
    ramp: RampConfig = None  # type: ignore[assignment]  # → Default in __post_init__
    db_path: str = "results.db"
    duration_min: int = 25  # SLURM Time-Out in Minuten
    stale_minutes: int = 30  # processing-Einträge älter als → reset

    def __post_init__(self) -> None:
        if self.ramp is None:
            self.ramp = RampConfig()


# ── Phase 1: Ingest ─────────────────────────────────────────────────────────

def ingest_phase(cfg: PipelineConfig) -> int:
    """Strukturen von Materials Project herunterladen und in DB speichern.

    Idempotent: Bereits vorhandene material_ids werden übersprungen.

    Returns
    -------
    int
        Gesamtzahl neu eingefügter Strukturen.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    total_inserted = 0
    for chemsys in cfg.chemsys_list:
        logger.info("=== Ingest: %s ===", chemsys)
        entries = query_mp_structures(
            chemsys=chemsys,
            api_key=cfg.api_key,
            max_results=cfg.max_results_per_sys,
            e_hull_max=cfg.e_hull_max,
            include_ternary=cfg.include_ternary,
        )
        inserted = ingest_structures(cfg.db_path, entries, chemsys)
        total_inserted += inserted
        logger.info(
            "%s: %d neu eingefügt (%d total in Queue)",
            chemsys,
            inserted,
            queue_stats(cfg.db_path)["total"],
        )

    stats = queue_stats(cfg.db_path)
    logger.info(
        "Ingest beendet. Queue: %d pending, %d total",
        stats["pending"],
        stats["total"],
    )
    print(
        f"[INGEST] {total_inserted} Strukturen neu eingefügt. "
        f"Queue: {stats['pending']} pending, {stats['total']} total.",
        flush=True,
    )
    return total_inserted


# ── Phase 2: Process ────────────────────────────────────────────────────────

def run_pipeline(cfg: PipelineConfig) -> int:
    """MD-Simulationen für alle pending-Strukturen in der DB ausführen.

    Die Schleife claimt atomar eine Struktur, simuliert sie, speichert das
    Ergebnis und markiert sie als 'done'.  Bei Time-Out → Exit 88.

    Returns
    -------
    int
        Exit-Code: ``0`` wenn keine pending mehr, ``88`` bei Time-Out.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.perf_counter()
    max_runtime_s = cfg.duration_min * 60
    worker_id = f"{socket.gethostname()}-{os.getpid()}"

    def _time_remaining() -> float:
        return max_runtime_s - (time.perf_counter() - t_start)

    # Stale-Reset: abgestürzte Jobs von vorherigen Läufen freigeben
    n_reset = reset_stale(cfg.db_path, stale_minutes=cfg.stale_minutes)
    if n_reset:
        logger.info("Reset: %d stale 'processing'-Einträge → pending", n_reset)

    # Calculator initialisieren (einmalig pro Job)
    logger.info("=== Calculator initialisieren (%s, device=%s) ===",
                cfg.mlip_backend, cfg.device)
    calc = get_calculator(backend=cfg.mlip_backend, device=cfg.device)

    # DB-Verbindung für Ergebnisse
    conn = init_db(cfg.db_path)

    processed = 0
    timed_out = False

    while True:
        # Time-Out-Prüfung
        if _time_remaining() < _TIMEOUT_BUFFER_S:
            logger.warning(
                "Time-Out: nur noch %.0f s bis zum Limit (%d min).",
                _time_remaining(),
                cfg.duration_min,
            )
            timed_out = True
            break

        # Nächste Struktur claimen
        queued = claim_next_structure(cfg.db_path, worker_id)
        if queued is None:
            logger.info("Keine pending-Strukturen mehr — Pipeline fertig.")
            break

        logger.info(
            "[%d] %s (%s) — verbleibend: %.1f min",
            processed + 1,
            queued.material_id,
            queued.formula,
            _time_remaining() / 60.0,
        )

        # Struktur deserialisieren → ASE-Atoms
        try:
            structure = _structure_from_json(queued.structure_json)
            entry = MPEntry(
                material_id=queued.material_id,
                formula_pretty=queued.formula,
                structure=structure,
            )
            atoms = pmg_to_ase(structure)
        except Exception as exc:
            logger.error("Deserialisierung/Conversion %s fehlgeschlagen: %s",
                         queued.material_id, exc)
            mark_structure_error(cfg.db_path, queued.material_id)
            continue

        # MD-Simulation
        _process_single(entry, atoms, calc, cfg.ramp, conn)
        mark_structure_done(cfg.db_path, queued.material_id)
        processed += 1

        stats = queue_stats(cfg.db_path)
        logger.info(
            "%s done. Queue: %d pending, %d done, %d error",
            queued.material_id,
            stats["pending"],
            stats["done"],
            stats["error"],
        )

    conn.close()

    stats = queue_stats(cfg.db_path)
    print(
        f"[PIPELINE] {processed} Strukturen simuliert. "
        f"Queue: {stats['pending']} pending, {stats['done']} done, "
        f"{stats['error']} error, {stats['total']} total.",
        flush=True,
    )

    if timed_out:
        logger.warning("=== Pipeline durch Time-Out abgebrochen (Exit 88) ===")
        print("[TIME-OUT] Pipeline wurde nach Ablauf der Zeit abgebrochen. "
              "Exit-Code 88 für SLURM Auto-Rescheduling.", flush=True)
        return 88

    logger.info("=== Pipeline beendet. Ergebnisse in %s ===", cfg.db_path)
    return 0


def _process_single(
    entry: MPEntry,
    atoms: Atoms,
    calc: Calculator,
    ramp_cfg: RampConfig,
    conn,
) -> None:
    """Eine einzelne Struktur durch die Rampen-Engine schicken."""
    atoms.calc = calc  # Calculator anhängen (wird von ThermalRamp verlangt)

    try:
        ramp = ThermalRamp(atoms, config=ramp_cfg)
        result: TrajectoryResult = ramp.run()
        snapshots = ramp.snapshots_around_t_switch(result)
    except Exception as exc:  # noqa: BLE001
        logger.error("Fehler bei %s: %s", entry.material_id, exc)
        result = TrajectoryResult(status="diverged")
        snapshots = {}

    store_result(conn, entry, result, snapshots)
    logger.info(
        "%s gespeichert: status=%s, T_switch=%s, T_decay=%s",
        entry.material_id,
        result.status,
        result.t_switch,
        result.t_decay,
    )

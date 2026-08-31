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

import json
import logging
import os
import socket
import time
from dataclasses import dataclass
from typing import Literal

import numpy as np
from ase import Atoms
from ase.calculators.calculator import Calculator

from .acquisition import MPEntry, make_supercell_atoms, pmg_to_ase, query_mp_structures
from .calculator import get_calculator
from .md import RampConfig, ThermalRamp
from .metrics import TrajectoryResult
from .storage import (
    _structure_from_json,
    claim_next_structure,
    delete_checkpoint,
    ingest_structures,
    init_db,
    load_checkpoint,
    mark_structure_done,
    mark_structure_error,
    queue_stats,
    requeue_structure,
    reset_stale,
    save_checkpoint,
    store_result,
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
    supercell_min_atoms: int = 50  # Mindestatomzahl für MD-Tauglichkeit

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

        # Supercell erstellen (falls primitive Zelle zu klein für Phasenübergänge)
        atoms = make_supercell_atoms(atoms, min_atoms=cfg.supercell_min_atoms)

        # MD-Simulation
        deadline = time.perf_counter() + _time_remaining() - _TIMEOUT_BUFFER_S
        completed = _process_single(
            entry, atoms, calc, cfg.ramp, conn,
            db_path=cfg.db_path,
            material_id=queued.material_id,
            deadline=deadline,
        )
        if completed:
            mark_structure_done(cfg.db_path, queued.material_id)
            processed += 1
        else:
            # Time-Out: Struktur zurück auf 'pending' für Resume im nächsten Job
            requeue_structure(cfg.db_path, queued.material_id)
            logger.info(
                "%s timed_out → zurück auf 'pending' (Checkpoint vorhanden).",
                queued.material_id,
            )

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


def _serialize_result(result: TrajectoryResult) -> str:
    """TrajectoryResult als JSON-String für Checkpoint serialisieren."""
    return json.dumps({
        "temperatures": result.temperatures,
        "volumes": result.volumes,
        "energies": result.energies,
        "msd_values": result.msd_values,
        "ql_values": result.ql_values,
        "rdf_history": [
            {"r": r.tolist(), "g": g.tolist()}
            for r, g in result.rdf_history
        ],
        "positions_history": [
            p.tolist() for p in result.positions_history
        ],
        "cell_history": [
            c.tolist() for c in result.cell_history
        ],
        "t_switch": result.t_switch,
        "t_decay": result.t_decay,
        "status": result.status,
    })


def _deserialize_result(metrics_json: str) -> TrajectoryResult:
    """JSON-String → TrajectoryResult (für Resume)."""
    obj = json.loads(metrics_json)
    result = TrajectoryResult(
        temperatures=obj.get("temperatures", []),
        volumes=obj.get("volumes", []),
        energies=obj.get("energies", []),
        msd_values=obj.get("msd_values", []),
        ql_values=obj.get("ql_values", []),
        t_switch=obj.get("t_switch"),
        t_decay=obj.get("t_decay"),
        status=obj.get("status", "converged"),
    )
    result.rdf_history = [
        (np.array(item["r"]), np.array(item["g"]))
        for item in obj.get("rdf_history", [])
    ]
    result.positions_history = [
        np.array(p) for p in obj.get("positions_history", [])
    ]
    result.cell_history = [
        np.array(c) for c in obj.get("cell_history", [])
    ]
    return result


def _process_single(
    entry: MPEntry,
    atoms: Atoms,
    calc: Calculator,
    ramp_cfg: RampConfig,
    conn,
    db_path: str,
    material_id: str,
    deadline: float | None = None,
) -> bool:
    """Eine einzelne Struktur durch die Rampen-Engine schicken.

    Unterstützt Checkpoint/Resume: lädt vorhandenen Checkpoint, stellt
    Atompositionen/Zelle/Metriken wieder her und übergibt eine Deadline
    für sauberes Time-Out-Handling.

    Returns
    -------
    bool
        *True* wenn die Simulation vollständig abgeschlossen wurde,
        *False* bei Time-Out (Checkpoint wurde gespeichert).
    """
    atoms.calc = calc

    # ── Checkpoint laden (falls vorhanden → Resume) ──────────────────
    cp = load_checkpoint(db_path, material_id)
    resume_step = 0
    initial_result: TrajectoryResult | None = None

    if cp is not None:
        logger.info(
            "Checkpoint gefunden: Schritt %d, T=%.1f K — Resume.",
            cp.step_index, cp.temperature,
        )
        atoms.set_positions(cp.positions)
        atoms.set_cell(cp.cell)
        resume_step = cp.step_index
        initial_result = _deserialize_result(cp.metrics)

    # ── Checkpoint-Callback ──────────────────────────────────────────
    def _checkpoint_cb(step_index: int, temperature: float,
                       result: TrajectoryResult) -> None:
        metrics_json = _serialize_result(result)
        save_checkpoint(
            db_path,
            material_id,
            step_index,
            temperature,
            atoms.get_positions(),
            atoms.get_cell(),
            metrics_json,
        )

    # ── MD-Simulation ────────────────────────────────────────────────
    try:
        ramp = ThermalRamp(atoms, config=ramp_cfg)
        result: TrajectoryResult = ramp.run(
            deadline=deadline,
            checkpoint_cb=_checkpoint_cb,
            resume_step=resume_step,
            initial_result=initial_result,
        )
        snapshots = ramp.snapshots_around_t_switch(result)
    except Exception as exc:  # noqa: BLE001
        logger.error("Fehler bei %s: %s", entry.material_id, exc)
        result = TrajectoryResult(status="diverged")
        snapshots = {}

    # ── Bei Time-Out: nicht in materials speichern, nur Checkpoint behalten ──
    if result.status == "timed_out":
        logger.info(
            "%s timed_out — Checkpoint gespeichert, kein Ergebnis-Eintrag.",
            entry.material_id,
        )
        return False

    # ── Ergebnis speichern ───────────────────────────────────────────
    store_result(conn, entry, result, snapshots)
    logger.info(
        "%s gespeichert: status=%s, T_switch=%s, T_decay=%s",
        entry.material_id,
        result.status,
        result.t_switch,
        result.t_decay,
    )

    # ── Checkpoint aufräumen (Simulation vollständig abgeschlossen) ──
    delete_checkpoint(db_path, material_id)
    logger.info("Checkpoint für %s gelöscht (Simulation abgeschlossen).",
                 material_id)
    return True

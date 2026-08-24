"""End-to-End-Orchestrierung der Screening-Pipeline.

Verbindet Akquisition → Calculator → MD-Rampe → Metriken → Speicherung.

SLURM Time-Out-Handling:
  Die Pipeline überwacht die verstrichene Laufzeit.  Sobald die
  konfigurierte Dauer (``duration_min``) minus 2 Minuten Puffer
  erreicht ist, wird die Schleife sauber abgebrochen, die DB
  geschlossen und Exit-Code 88 zurückgegeben (Signal für
  Auto-Rescheduling).  Ein kompletter Durchlauf endet mit Code 0.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Literal

from ase import Atoms
from ase.calculators.calculator import Calculator

from .acquisition import MPEntry, build_structure_batch
from .calculator import get_calculator
from .md import ThermalRamp, RampConfig
from .metrics import TrajectoryResult
from .storage import init_db, store_result

logger = logging.getLogger(__name__)

# Puffer in Sekunden vor dem harten Time-Out (für sauberen Abschluss)
_TIMEOUT_BUFFER_S = 2 * 60  # 2 Minuten


@dataclass
class PipelineConfig:
    """Gesamtkonfiguration der Pipeline."""

    chemsys_list: list[str]
    api_key: str | None = None
    max_results_per_sys: int = 50
    stable_only: bool = True
    mlip_backend: Literal["mace", "chgnet"] = "mace"
    device: str = "cpu"
    ramp: RampConfig = None  # type: ignore[assignment]  # → Default in __post_init__
    db_path: str = "results.db"
    duration_min: int = 25  # SLURM Time-Out in Minuten

    def __post_init__(self) -> None:
        if self.ramp is None:
            self.ramp = RampConfig()


def run_pipeline(cfg: PipelineConfig) -> int:
    """Komplette Pipeline ausführen.

    Returns
    -------
    int
        Exit-Code: ``0`` bei vollständigem Durchlauf, ``88`` bei Time-Out.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    t_start = time.perf_counter()
    max_runtime_s = cfg.duration_min * 60

    def _time_remaining() -> float:
        return max_runtime_s - (time.perf_counter() - t_start)

    # 1 – Akquisition
    logger.info("=== Schritt 1: Datenakquisition ===")
    batch = build_structure_batch(
        chemsys=cfg.chemsys_list,
        api_key=cfg.api_key,
        max_results_per_sys=cfg.max_results_per_sys,
        stable_only=cfg.stable_only,
    )
    logger.info("%d Strukturen geladen.", len(batch))

    # 2 – Calculator
    logger.info("=== Schritt 2: Calculator initialisieren (%s) ===", cfg.mlip_backend)
    calc = get_calculator(backend=cfg.mlip_backend, device=cfg.device)

    # 3 – DB
    conn = init_db(cfg.db_path)

    # 4 – MD-Rampen
    logger.info("=== Schritt 3: MD-Rampen (Time-Out nach %d min) ===", cfg.duration_min)
    timed_out = False

    for idx, (entry, atoms) in enumerate(batch, start=1):
        # Time-Out-Prüfung VOR dem Start eines neuen Materials
        if _time_remaining() < _TIMEOUT_BUFFER_S:
            logger.warning(
                "Time-Out: nur noch %.0f s bis zum Limit (%d min). "
                "Breche vor Material %d/%d ab.",
                _time_remaining(),
                cfg.duration_min,
                idx,
                len(batch),
            )
            timed_out = True
            break

        logger.info(
            "[%d/%d] %s (%s) — verbleibend: %.1f min",
            idx,
            len(batch),
            entry.material_id,
            entry.formula_pretty,
            _time_remaining() / 60.0,
        )
        _process_single(entry, atoms, calc, cfg.ramp, conn)

    conn.close()

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

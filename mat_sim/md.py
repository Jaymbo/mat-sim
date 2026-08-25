"""Thermische Rampe: NPT-MD mit Nosé-Hoover-Thermostat.

Klasse ``ThermalRamp`` kapselt die komplette Schleife:
  1. Geometrieoptimierung bei T ≈ 0 K (BFGS, fmax-Schwellwert)
  2. NPT-MD initialisieren (Melchionna-NPT, konservative Parameter)
  3. Temperatur in Schritten erhöhen, pro Schritt thermalisieren
  4. Pro Schritt Metriken sammeln (RDF, MSD, Q4)
  5. T_switch & T_decay bestimmen
  6. Divergenz-Erkennung (Kräfte > Schwellwert → kontrollierter Abbruch)
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass

import numpy as np
from ase import Atoms
from ase.md.melchionna import MelchionnaNPT as NPT
from ase.optimize import LBFGS
from ase.units import fs, GPa

from .metrics import (
    TrajectoryResult,
    StepMetrics,
    compute_msd,
    compute_rdf,
    compute_steinhardt_q4,
    detect_t_decay,
    detect_t_switch,
    nearest_neighbor_distance,
)

logger = logging.getLogger(__name__)


# ── Early-Stop Signal ───────────────────────────────────────────────────────
class _EquilibriumStop(Exception):
    """Wird ausgelöst, wenn das thermische Gleichgewicht frühzeitig erreicht ist."""


# ── Equilibrium Monitor (ASE-Callback) ──────────────────────────────────────
class _EquilibriumMonitor:
    """Rotierendes Fenster über die kinetische Temperatur zur Gleichgewichtserkennung.

    Wird als Callback an das ASE-Dynamics-Objekt angehängt.  Überwacht die
    relative Standardabweichung der Temperatur innerhalb eines gleitenden
    Fensters.  Sobald diese unter einen Schwellenwert fällt, wird
    ``_EquilibriumStop`` ausgelöst, um ``dyn.run()`` kontrolliert abzubrechen.

    Der Monitor wird **einmal** an das Dynamics-Objekt angehängt und vor
    jeder Temperaturstufe mit :meth:`reset` zurückgesetzt.
    """

    def __init__(self, dyn, cfg: RampConfig) -> None:
        self._dyn = dyn
        self._min_steps = cfg.early_stop_min_steps
        self._window_size = cfg.early_stop_window
        self._rel_std_threshold = cfg.early_stop_rel_std
        self._temperatures: list[float] = []
        self._step_count = 0
        self._stopped_at: int | None = None

    # ── API ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Fenster und Zähler für eine neue Temperaturstufe zurücksetzen."""
        self._temperatures.clear()
        self._step_count = 0
        self._stopped_at = None

    @property
    def stopped_at(self) -> int | None:
        """Schritt-Index, bei dem das Gleichgewicht erkannt wurde, oder *None*."""
        return self._stopped_at

    # ── ASE-Callback-Signatur ───────────────────────────────────────────
    def __call__(self) -> None:
        self._step_count += 1

        # Aktuelle kinetische Temperatur aus dem Dynamics-Objekt auslesen
        try:
            temp = self._dyn.get_temperature()
        except Exception:
            return

        self._temperatures.append(float(temp))

        # Fenster begrenzen
        if len(self._temperatures) > self._window_size:
            self._temperatures = self._temperatures[-self._window_size:]

        # Prüfung erst nach Mindestschritten
        if self._step_count < self._min_steps:
            return

        if len(self._temperatures) < self._window_size:
            return

        window = np.array(self._temperatures)
        mean_t = np.mean(window)
        if mean_t <= 1e-6:
            return

        rel_std = float(np.std(window) / mean_t)

        if rel_std < self._rel_std_threshold:
            self._stopped_at = self._step_count
            raise _EquilibriumStop


# ── Konfiguration ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RampConfig:
    """Parameter für die Temperaturrampe."""

    t_start: float = 0.0            # K
    t_max: float = 600.0            # K
    delta_t: float = 10.0           # K pro Schritt
    thermalization_steps: int = 100     # Zeitschritte pro T-Stufe (CPU-Screening)
    time_step: float = 0.5          # fs  (konservativ für Oxide + MLIP)
    pressure: float = 1.0e-4        # GPa (≈ 1 atm)
    temperature_time_constant: float = 200.0 * fs   # τ_T (träge → stabil)
    pressure_time_constant: float = 2000.0 * fs     # τ_P (sehr träge)
    bulk_modulus: float = 100.0     # GPa (typisch für Übergangsmetall-Oxide)
    lindemann_fraction: float = 0.12
    rdf_r_max: float = 6.0
    rdf_n_bins: int = 100
    log_interval: int = 50
    # Geometrieoptimierung
    fmax: float = 0.05              # eV/Å – Schwellwert für BFGS
    opt_max_steps: int = 200        # Max. Optimierungs-Schritte
    # Divergenz-Erkennung
    force_threshold: float = 100.0  # eV/Å – Kräfte darüber → Divergenz
    # Early Stopping (thermisches Gleichgewicht)
    early_stop_min_steps: int = 20      # Mindestschritte, bevor Abbruch geprüft wird
    early_stop_window: int = 10         # Grösse des rotierenden Fensters
    early_stop_rel_std: float = 0.02    # rel. Std-Abw.-Schwelle (2 %)


# ── Rampen-Engine ───────────────────────────────────────────────────────────
class ThermalRamp:
    """Führt eine temperaturabhängige NPT-MD-Simulation durch.

    Parameters
    ----------
    atoms
        Zu untersuchende Struktur (inkl. Calculator).
    config
        Rampen-Parameter.
    """

    def __init__(
        self,
        atoms: Atoms,
        config: RampConfig | None = None,
    ) -> None:
        if atoms.calc is None:
            raise ValueError("Atoms-Objekt benötigt einen Calculator (atoms.calc).")
        self.atoms = atoms
        self.cfg = config or RampConfig()

    # ── öffentliche API ─────────────────────────────────────────────────────
    def run(self) -> TrajectoryResult:
        """Komplette Temperaturrampe ausführen und Ergebnis zurückgeben."""
        cfg = self.cfg
        result = TrajectoryResult()

        logger.info(
            "Starte Rampe: %s | N=%d",
            self.atoms.get_chemical_formula(),
            len(self.atoms),
        )

        # --- Schritt 1: Geometrieoptimierung bei T ≈ 0 K ----------------
        logger.info("Geometrieoptimierung (BFGS, fmax=%.3f eV/Å) …", cfg.fmax)
        try:
            self._optimize_geometry()
        except Exception as exc:
            logger.error("Geometrieoptimierung fehlgeschlagen: %s", exc)
            result.status = "diverged"
            return result

        # Referenzpositionen nach Optimierung speichern
        initial_positions = self.atoms.get_positions().copy()
        nn_dist = nearest_neighbor_distance(self.atoms)
        logger.info(
            "Optimiert | d_NN=%.3f Å | V=%.1f Å³",
            nn_dist,
            self.atoms.get_volume(),
        )

        # --- Schritt 2: NPT-MD initialisieren ----------------------------
        dyn = NPT(
            self.atoms,
            timestep=cfg.time_step * fs,
            temperature_K=max(cfg.t_start, 1e-3),
            externalstress=cfg.pressure * GPa,
            ttime=cfg.temperature_time_constant,
            # pfactor = τ_P² × B  (B = Bulk-Modulus in ASE-Einheiten)
            pfactor=(cfg.pressure_time_constant**2) * (cfg.bulk_modulus * GPa),
            loginterval=cfg.log_interval,
        )

        # Early-Stopping-Monitor einmalig anhängen
        monitor = _EquilibriumMonitor(dyn, cfg)
        dyn.attach(monitor, interval=1)

        temperatures = np.arange(
            cfg.t_start, cfg.t_max + cfg.delta_t, cfg.delta_t
        )

        # --- Schritt 3: Temperaturschleife --------------------------------
        n_steps_total = len(temperatures)
        for i, T in enumerate(temperatures):
            T_sim = max(T, 1e-3)  # 0 K vermeiden
            dyn.set_temperature(temperature_K=T_sim)

            # Divergenz-Check VOR dem MD-Schritt
            if self._check_divergence():
                logger.error("Divergenz bei T=%.1f K – Abbruch.", T)
                result.status = "diverged"
                break

            print(f"-> Starte MD-Simulation für T = {T:.1f} K ... "
                  f"(Schritt {i + 1}/{n_steps_total})", flush=True)

            # Thermalisieren (mit Profiling + Early Stopping)
            monitor.reset()
            t0 = time.perf_counter()
            early_stopped = False
            try:
                dyn.run(cfg.thermalization_steps)
            except _EquilibriumStop:
                early_stopped = True
            except Exception as exc:
                logger.error("MD-Divergenz bei T=%.1f K: %s", T, exc)
                result.status = "diverged"
                break
            elapsed = time.perf_counter() - t0

            if early_stopped:
                stopped_at = monitor.stopped_at or 0
                print(f"   [INFO] Thermisches Gleichgewicht vorzeitig erreicht "
                      f"bei Schritt {stopped_at}. Springe zur nächsten Temperatur.",
                      flush=True)

            # Divergenz-Check NACH dem MD-Schritt
            if self._check_divergence():
                logger.error("Divergenz nach T=%.1f K – Abbruch.", T)
                result.status = "diverged"
                break

            # Metriken sammeln
            metrics = self._collect_metrics(T, initial_positions)
            result.add(metrics)

            print(f"   [ERFOLG] T = {T:.1f} K nach {elapsed:.2f} Sekunden beendet. "
                  f"(MSD: {metrics.msd:.4f})", flush=True)

            # Frühzeitiger Abbruch bei Zerfall (Lindemann)
            if (
                result.t_decay is None
                and metrics.msd > (cfg.lindemann_fraction * nn_dist) ** 2
            ):
                result.t_decay = T
                result.status = "decayed"
                logger.warning("Zerfall detektiert bei T=%.1f K – Abbruch.", T)
                print(f"   [WARNUNG] Zerfall detektiert bei T = {T:.1f} K – Abbruch.",
                      flush=True)
                break

        # --- Post-Processing ---------------------------------------------
        if result.t_switch is None and len(result.rdf_history) >= 2:
            result.t_switch = detect_t_switch(
                result.rdf_history,
                result.temperatures,
            )
        if result.t_decay is None and len(result.msd_values) >= 1:
            result.t_decay = detect_t_decay(
                result.msd_values,
                result.temperatures,
                nn_distance=nn_dist,
                lindemann_fraction=cfg.lindemann_fraction,
            )

        logger.info(
            "Rampe beendet: status=%s, T_switch=%s, T_decay=%s",
            result.status,
            result.t_switch,
            result.t_decay,
        )
        return result

    # ── interne Hilfsfunktionen ────────────────────────────────────────────
    def _optimize_geometry(self) -> None:
        """Struktur bei T ≈ 0 K mit LBFGS relaxieren (nutzt atoms.calc)."""
        opt = LBFGS(self.atoms, logfile=None)
        opt.run(fmax=self.cfg.fmax, steps=self.cfg.opt_max_steps)

    def _check_divergence(self) -> bool:
        """Prüfen, ob die Kräfte auf den Atomen divergiert sind.

        Returns
        -------
        bool
            *True*, wenn die maximale Kraftkomponente den Schwellwert
            überschreitet (→ MD ist divergiert).
        """
        try:
            forces = self.atoms.get_forces()
        except Exception:
            return True
        max_force = float(np.max(np.linalg.norm(forces, axis=1)))
        return max_force > self.cfg.force_threshold

    def _collect_metrics(
        self,
        temperature: float,
        initial_positions: np.ndarray,
    ) -> StepMetrics:
        """Alle Metriken für den aktuellen Snapshot erfassen."""
        atoms = self.atoms
        r, g = compute_rdf(
            atoms,
            r_max=self.cfg.rdf_r_max,
            n_bins=self.cfg.rdf_n_bins,
        )
        msd = compute_msd(initial_positions, atoms.get_positions())
        ql = compute_steinhardt_q4(atoms)

        return StepMetrics(
            temperature=temperature,
            volume=atoms.get_volume(),
            energy=atoms.get_potential_energy(),
            msd=msd,
            rdf_r=r,
            rdf_g=g,
            ql=ql,
            positions=atoms.get_positions().copy(),
            cell=atoms.get_cell().copy(),
        )

    # ── Snapshot-Extraktion für spektrale Analyse ──────────────────────────
    def snapshots_around_t_switch(
        self,
        result: TrajectoryResult,
        window: int = 1,
    ) -> dict[str, tuple]:
        """RDF- **und** Struktur-Snapshots kurz vor und nach T_switch extrahieren.

        Returns
        -------
        dict
            ``{"before": ((r, g), atoms), "after": ((r, g), atoms)}``
            oder leeres Dict, falls kein T_switch gefunden wurde.
            ``atoms`` ist ein ``ase.Atoms``-Objekt, das aus den gespeicherten
            Positionen/Zellen rekonstruiert wurde.
        """
        if result.t_switch is None:
            return {}

        temps = np.array(result.temperatures)
        idx = int(np.argmin(np.abs(temps - result.t_switch)))
        lo = max(idx - window, 0)
        hi = min(idx + window, len(result.rdf_history) - 1)

        symbols = self.atoms.get_chemical_symbols()
        pbc = self.atoms.get_pbc()

        def _make_atoms(i: int) -> Atoms:
            return Atoms(
                symbols=symbols,
                positions=result.positions_history[i],
                cell=result.cell_history[i],
                pbc=pbc,
            )

        return {
            "before": (result.rdf_history[lo], _make_atoms(lo)),
            "after": (result.rdf_history[hi], _make_atoms(hi)),
        }

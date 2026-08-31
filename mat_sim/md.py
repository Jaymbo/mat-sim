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
from ase.units import GPa, fs

from .metrics import (
    StepMetrics,
    TrajectoryResult,
    compute_msd,
    compute_rdf,
    compute_steinhardt_q4,
    compute_vibrational_msd,
    detect_t_decay,
    detect_t_switch,
    detect_t_switch_from_msd,
    nearest_neighbor_distance,
)

logger = logging.getLogger(__name__)


# ── Early-Stop Signal ───────────────────────────────────────────────────────
class _EquilibriumStop(Exception):
    """Wird ausgelöst, wenn das thermische Gleichgewicht frühzeitig erreicht ist."""


# ── Kombinierter Gleichgewichts-Monitor (ASE-Callback) ─────────────────────
class _CombinedEquilibriumMonitor:
    """Kombinierter Gleichgewichts-Monitor: Temperatur + Rolling-Mean Positionen.

    Stoppt die MD nur, wenn **beide** Bedingungen erfüllt sind:

    1. **Thermisches Gleichgewicht** – rel. Std-Abw. der Temperatur im
       Rolling-Window < ``early_stop_rel_std``.
    2. **Strukturelles Gleichgewicht** – Rolling-Mean der Positionen
       konvergiert: aufeinanderfolgende Rolling-Means unterscheiden sich um
       weniger als ``pos_convergence_threshold``.

    Der Rolling-Mean passt sich automatisch an strukturelle Umordnungen an:
    nach einem Phasenwechsel pendelt sich der Mean auf die neue Position ein,
    und erst wenn er dort konvergiert, wird gestoppt.  Die alten Positionen
    werden aus dem Fenster hinausgeschoben und verfälschen nicht das Ergebnis.

    Die Fenstergröße für den Rolling-Mean wird adaptiv aus der
    Schwingungsperiode bestimmt: Autokorrelation der Positionen → Periode →
    Fenster = ``pos_convergence_window_mult`` × Periode.

    Die vibrational MSD wird aus dem konvergierten Fenster berechnet
    (drift-frei, ohne alte Positionen vor einem strukturellen Shift).

    Der Monitor wird **einmal** an das Dynamics-Objekt angehängt und vor
    jeder Temperaturstufe mit :meth:`reset` zurückgesetzt.
    """

    def __init__(self, dyn, atoms: Atoms, cfg: RampConfig) -> None:
        self._dyn = dyn
        self._atoms = atoms

        # Temperatur-Parameter
        self._temp_min_steps = cfg.early_stop_min_steps
        self._temp_window = cfg.early_stop_window
        self._temp_rel_std = cfg.early_stop_rel_std

        # Positions-Parameter
        self._pos_sample_interval = max(cfg.msd_sample_interval, 1)
        self._pos_min_samples = cfg.pos_convergence_min_samples
        self._pos_window_mult = cfg.pos_convergence_window_mult
        self._pos_min_window = cfg.pos_convergence_min_window
        self._pos_threshold = cfg.pos_convergence_threshold
        self._pos_persistence = max(cfg.pos_convergence_persistence, 1)

        # Zustand
        self._step_count = 0
        self._temperatures: list[float] = []
        self._position_samples: list[np.ndarray] = []
        self._temp_converged = False
        self._pos_converged = False
        self._pos_converged_count = 0   # aufeinanderfolgende Schritte unter Schwelle
        self._stopped_at: int | None = None
        self._converged_window_size: int | None = None
        self._period_samples: int | None = None

        # Debug-Historie (pro T-Stufe)
        self._history_steps: list[int] = []
        self._history_temp_rel_std: list[float] = []
        self._history_pos_rms: list[float] = []
        self._history_temp_converged: list[bool] = []
        self._history_pos_converged: list[bool] = []

    # ── API ────────────────────────────────────────────────────────────
    def reset(self) -> None:
        """Zustand für eine neue Temperaturstufe zurücksetzen."""
        self._step_count = 0
        self._temperatures.clear()
        self._position_samples.clear()
        self._temp_converged = False
        self._pos_converged = False
        self._pos_converged_count = 0
        self._stopped_at = None
        self._converged_window_size = None
        self._period_samples = None
        self._history_steps.clear()
        self._history_temp_rel_std.clear()
        self._history_pos_rms.clear()
        self._history_temp_converged.clear()
        self._history_pos_converged.clear()

    @property
    def stopped_at(self) -> int | None:
        """MD-Schritt, bei dem das Gleichgewicht erkannt wurde, oder *None*."""
        return self._stopped_at

    @property
    def converged_samples(self) -> np.ndarray | None:
        """Positionssamples aus dem konvergierten Fenster (für drift-freie MSD).

        Gibt die letzten ``converged_window_size`` Samples zurück.
        Fallback (keine Konvergenz): alle Samples.
        """
        if not self._position_samples:
            return None
        if self._converged_window_size is not None:
            n = min(self._converged_window_size, len(self._position_samples))
            return np.array(self._position_samples[-n:])
        return np.array(self._position_samples)

    @property
    def samples(self) -> np.ndarray | None:
        """Alle gesammelten Positionssamples (Kompatibilität)."""
        if not self._position_samples:
            return None
        return np.array(self._position_samples)

    @property
    def history(self) -> dict[str, list]:
        """Konvergenz-Historie der aktuellen T-Stufe für Debug-Plots.

        Returns
        -------
        dict
            Schlüssel: ``steps``, ``temp_rel_std``, ``pos_rms``,
            ``temp_converged``, ``pos_converged``.
        """
        return {
            "steps": list(self._history_steps),
            "temp_rel_std": list(self._history_temp_rel_std),
            "pos_rms": list(self._history_pos_rms),
            "temp_converged": list(self._history_temp_converged),
            "pos_converged": list(self._history_pos_converged),
        }

    # ── ASE-Callback-Signatur ───────────────────────────────────────────
    def __call__(self) -> None:
        self._step_count += 1

        # Temperatur sammeln (aus Kinetic-Energie der Atome, nicht dyn)
        try:
            temp = self._atoms.get_temperature()
            self._temperatures.append(float(temp))
            if len(self._temperatures) > self._temp_window:
                self._temperatures = self._temperatures[-self._temp_window:]
        except Exception:
            pass

        # Positionen sammeln (alle pos_sample_interval Schritte)
        if self._step_count % self._pos_sample_interval == 0:
            self._position_samples.append(self._atoms.get_positions().copy())

        # ── Konvergenz-Prüfung (erst nach Mindestschritten) ──
        past_min_steps = self._step_count >= self._temp_min_steps

        # ── 1. Temperatur-Konvergenz ──
        temp_rel_std_val = float("nan")
        if past_min_steps and len(self._temperatures) >= self._temp_window:
            window = np.array(self._temperatures)
            mean_t = np.mean(window)
            if mean_t < 1.0:
                # Bei T < 1 K ist die Temperaturkonvergenz trivial erfüllt.
                self._temp_converged = True
                # 1e-10 statt 0.0: im log-Plot sichtbar (log(0) = -∞)
                temp_rel_std_val = 1e-10
            elif mean_t > 1e-6:
                temp_rel_std_val = float(np.std(window) / mean_t)
                if temp_rel_std_val < self._temp_rel_std:
                    self._temp_converged = True

        # ── 2. Positions-Konvergenz (mit Persistenz) ──
        pos_rms_val = float("nan")
        if past_min_steps and len(self._position_samples) >= self._pos_min_samples:
            pos_rms_val = self._compute_pos_rms()
            if not self._pos_converged:
                if np.isnan(pos_rms_val):
                    self._pos_converged_count = 0
                elif pos_rms_val < self._pos_threshold:
                    self._pos_converged_count += 1
                    if self._pos_converged_count >= self._pos_persistence:
                        self._pos_converged = True
                        # Fenstergröße für konvergierte Samples speichern
                        n = len(self._position_samples)
                        window = max(
                            self._pos_window_mult * (self._period_samples or 5),
                            self._pos_min_window,
                        )
                        window = min(window, n // 2)
                        self._converged_window_size = window
                else:
                    self._pos_converged_count = 0

        # ── Debug-Historie aufzeichnen (jeder Schritt, nicht erst ab min_steps) ──
        self._history_steps.append(self._step_count)
        self._history_temp_rel_std.append(temp_rel_std_val)
        self._history_pos_rms.append(pos_rms_val)
        self._history_temp_converged.append(self._temp_converged)
        self._history_pos_converged.append(self._pos_converged)

        # ── 3. Beide konvergiert → Stop ──
        if self._temp_converged and self._pos_converged:
            self._stopped_at = self._step_count
            raise _EquilibriumStop

    # ── Positions-Konvergenz ────────────────────────────────────────────
    def _compute_pos_rms(self) -> float:
        """RMS-Verschiebung zwischen zwei aufeinanderfolgenden Rolling-Means.

        Gibt ``nan`` zurück, wenn nicht genügend Samples für zwei Fenster
        vorhanden sind.
        """
        samples = np.array(self._position_samples)
        n = len(samples)

        # Schwingungsperiode schätzen (einmalig pro T-Stufe)
        if self._period_samples is None:
            self._period_samples = self._estimate_oscillation_period(samples)

        window = max(
            self._pos_window_mult * self._period_samples,
            self._pos_min_window,
        )
        window = min(window, n // 2)

        if window < 2 or n < 2 * window:
            return float("nan")

        mean_recent = np.mean(samples[-window:], axis=0)
        mean_prev = np.mean(samples[-2 * window:-window], axis=0)
        disp = mean_recent - mean_prev
        return float(np.sqrt(np.mean(np.sum(disp**2, axis=1))))

    @staticmethod
    def _estimate_oscillation_period(samples: np.ndarray) -> int:
        """Schwingungsperiode (in Samples) aus Autokorrelation schätzen.

        Verwendet die mittlere Position aller Atome als Signal und sucht
        den ersten Nulldurchgang der normierten Autokorrelation.
        Periode = 2 × Nulldurchgang.

        Fallback bei zu wenigen Samples oder keinem Nulldurchgang: 5.
        """
        n = len(samples)
        if n < 4:
            return 5

        # Kollektives Signal: mittlere Position pro Sample
        signal = np.mean(samples, axis=1)  # (n, 3)
        centered = signal - np.mean(signal, axis=0)

        # Normierte Autokorrelation (gemittelt über x, y, z)
        norm = float(np.sum(centered**2))
        if norm < 1e-12:
            return 5

        for lag in range(1, n):
            acf = float(np.sum(centered[:-lag] * centered[lag:]) / norm)
            if acf <= 0:
                return max(2 * lag, 2)

        # Kein Nulldurchgang gefunden → Fallback
        return 5


# ── Konfiguration ───────────────────────────────────────────────────────────
@dataclass(frozen=True)
class RampConfig:
    """Parameter für die Temperaturrampe."""

    t_start: float = 0.0            # K
    t_max: float = 1200.0           # K  (erhöht: deckt Übergänge bis ~1000 K ab)
    delta_t: float = 20.0           # K pro Schritt (20K → halbe Schrittzahl, 10K für feine Auflösung)
    thermalization_steps: int = 1000    # Zeitschritte pro T-Stufe (1 ps bei 1 fs/Step)
    time_step: float = 1.0          # fs  (Standard für Oxid-MD)
    pressure: float = 1.0e-4        # GPa (≈ 1 atm)
    temperature_time_constant: float = 200.0 * fs   # τ_T (träge → stabil)
    pressure_time_constant: float = 2000.0 * fs     # τ_P (sehr träge)
    bulk_modulus: float = 100.0     # GPa (typisch für Übergangsmetall-Oxide)
    lindemann_fraction: float = 0.12
    rdf_r_max: float = 6.0
    rdf_n_bins: int = 100
    log_interval: int = 50
    checkpoint_interval: int = 5    # alle N Temperaturschritte einen Checkpoint speichern
    # Geometrieoptimierung
    fmax: float = 0.05              # eV/Å – Schwellwert für BFGS
    opt_max_steps: int = 200        # Max. Optimierungs-Schritte
    # Divergenz-Erkennung
    force_threshold: float = 100.0  # eV/Å – Kräfte darüber → Divergenz
    # Early Stopping (thermisches Gleichgewicht)
    early_stop_min_steps: int = 100     # Mindestschritte, bevor Abbruch geprüft wird
    early_stop_window: int = 100        # Grösse des rotierenden Fensters
    early_stop_rel_std: float = 0.05    # rel. Std-Abw.-Schwelle (5 %, NPT-tauglich)
    # MSD-Sampling (vibrational MSD um Gleichgewichtsposition, nicht 0 K)
    msd_sample_interval: int = 10       # alle N MD-Schritte Positionen sampeln
    # Positions-Konvergenz (Rolling-Mean)
    pos_convergence_min_samples: int = 20   # Mindestsamples vor Prüfung (20 × sample_interval = 200 MD-Schritte)
    pos_convergence_window_mult: int = 3    # Fenster = mult × Schwingungsperiode
    pos_convergence_min_window: int = 5     # Mindestfenstergröße
    pos_convergence_threshold: float = 0.01 # RMS-Verschiebung < threshold → konvergiert (Å)
    pos_convergence_persistence: int = 5    # N aufeinanderfolgende Schritte unter Schwelle nötig
    # Persistenz für Zerfalls-Erkennung (Lindemann): MSD muss für N
    # aufeinanderfolgende Schritte über dem Schwellwert bleiben, bevor
    # abgebrochen wird.  Filtert transiente Spikes.
    decay_persistence: int = 2
    # Debug: Konvergenz-Plots pro T-Schritt speichern (headless)
    debug_convergence_plots: bool = False
    debug_plot_dir: str = "debug_convergence"


# ── Debug: Konvergenz-Plot speichern ────────────────────────────────────────
def save_convergence_plot(
    history: dict[str, list],
    temperature: float,
    formula: str,
    material_id: str,
    output_dir: str,
    early_stop_rel_std: float,
    pos_convergence_threshold: float,
    stopped_at: int | None,
) -> str:
    """Konvergenz-Verlauf als PNG speichern (headless, matplotlib Agg-Backend).

    Erzeugt zwei Subplots:
      1. Temperatur rel. Std-Abw. über MD-Schritten mit Schwellwert-Linie
      2. Positions-RMS-Verschiebung über MD-Schritten mit Schwellwert-Linie

    Beide zeigen grüne Markierung wenn konvergiert, rote Linie für
    Early-Stop-Punkt.

    Parameters
    ----------
    history
        Konvergenz-Historie aus ``monitor.history``.
    temperature
        Temperatur der aktuellen Stufe (K) — für Dateinamen und Titel.
    formula
        Chemische Formel — für Ordner- und Dateinamen.
    material_id
        MP-ID — für Ordnername.
    output_dir
        Basis-Verzeichnis für Debug-Plots.
    early_stop_rel_std
        Schwellwert für Temperatur-Konvergenz (für horizontale Linie).
    pos_convergence_threshold
        Schwellwert für Positions-Konvergenz (für horizontale Linie).
    stopped_at
        MD-Schritt, bei dem gestoppt wurde, oder *None*.

    Returns
    -------
    str
        Pfad zur gespeicherten PNG-Datei.
    """
    import matplotlib
    matplotlib.use("Agg")  # headless
    import matplotlib.pyplot as plt
    from pathlib import Path

    # Ordner: output_dir/material_id_formula/
    safe_formula = formula.replace(" ", "_")
    subdir = Path(output_dir) / f"{material_id}_{safe_formula}"
    subdir.mkdir(parents=True, exist_ok=True)

    filename = subdir / f"convergence_T{temperature:.0f}K.png"
    filepath = str(filename)

    steps = np.array(history["steps"], dtype=float)
    temp_rel_std = np.array(history["temp_rel_std"], dtype=float)
    pos_rms = np.array(history["pos_rms"], dtype=float)
    temp_conv = np.array(history["temp_converged"], dtype=bool)
    pos_conv = np.array(history["pos_converged"], dtype=bool)

    # 0-Werte auf kleine positive Zahl clippen (für log-Skala: log(0) = -∞)
    _EPS = 1e-12
    temp_rel_std = np.where(temp_rel_std == 0, _EPS, temp_rel_std)
    pos_rms = np.where(pos_rms == 0, _EPS, pos_rms)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # ── Subplot 1: Temperatur rel. Std ──
    valid_temp = ~np.isnan(temp_rel_std)
    if np.any(valid_temp):
        ax1.plot(steps[valid_temp], temp_rel_std[valid_temp],
                 "b-", linewidth=0.8, label="rel. Std (Temperatur)")
    ax1.axhline(y=early_stop_rel_std, color="r", linestyle="--",
                linewidth=0.8, label=f"Schwelle ({early_stop_rel_std:.2f})")
    # Konvergenz-Zeitpunkt markieren
    temp_conv_idx = np.where(temp_conv)[0]
    if len(temp_conv_idx) > 0:
        first_conv = temp_conv_idx[0]
        ax1.axvline(x=steps[first_conv], color="g", linestyle=":",
                    linewidth=1.0, label=f"konvergiert (Schritt {int(steps[first_conv])})")
    ax1.set_ylabel("rel. Std-Abw. (Temperatur)")
    ax1.set_title(f"{formula} ({material_id}) — T = {temperature:.1f} K")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.set_yscale("log")
    ax1.grid(True, alpha=0.3)

    # ── Subplot 2: Positions-RMS ──
    valid_pos = ~np.isnan(pos_rms)
    if np.any(valid_pos):
        ax2.plot(steps[valid_pos], pos_rms[valid_pos],
                 "b-", linewidth=0.8, label="RMS (Rolling-Mean Positionen)")
    ax2.axhline(y=pos_convergence_threshold, color="r", linestyle="--",
                linewidth=0.8, label=f"Schwelle ({pos_convergence_threshold:.3f} Å)")
    pos_conv_idx = np.where(pos_conv)[0]
    if len(pos_conv_idx) > 0:
        first_conv = pos_conv_idx[0]
        ax2.axvline(x=steps[first_conv], color="g", linestyle=":",
                    linewidth=1.0, label=f"konvergiert (Schritt {int(steps[first_conv])})")
    # Early-Stop markieren
    if stopped_at is not None:
        ax2.axvline(x=stopped_at, color="orange", linestyle="-",
                    linewidth=1.5, label=f"Stop (Schritt {stopped_at})")
    ax2.set_xlabel("MD-Schritt")
    ax2.set_ylabel("RMS-Verschiebung (Å)")
    ax2.legend(loc="upper right", fontsize=8)
    ax2.set_yscale("log")
    ax2.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(filepath, dpi=150)
    plt.close(fig)
    return filepath


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
    def run(
        self,
        deadline: float | None = None,
        checkpoint_cb=None,
        resume_step: int = 0,
        initial_result: TrajectoryResult | None = None,
        material_id: str = "",
    ) -> TrajectoryResult:
        """Komplette Temperaturrampe ausführen und Ergebnis zurückgeben.

        Parameters
        ----------
        deadline
            Absolute Zeit in Sekunden (``time.perf_counter()``), ab der die
            Rampe sauber nach dem nächsten Temperaturschritt abbricht.
            Der aktuelle Fortschritt wird über ``checkpoint_cb`` gespeichert.
        checkpoint_cb
            Callback ``cb(step_index, temperature, result)``, der nach jedem
            abgeschlossenen Temperaturschritt aufgerufen wird, um einen
            Checkpoint zu speichern.
        resume_step
            Index des Schritts, ab dem die Rampe fortgesetzt wird (0 = neu).
            Wird mit ``resume_positions`` / ``resume_cell`` kombiniert.
        initial_result
            Vorausgefülltes ``TrajectoryResult`` (für Resume: enthält die
            Metriken aller bisherigen Schritte).  Wenn *None*, wird ein
            neues leeres Objekt erstellt.
        material_id
            MP-ID der Struktur — für Debug-Plot-Ordnername.
        """
        cfg = self.cfg
        result = initial_result or TrajectoryResult()

        logger.info(
            "Starte Rampe: %s | N=%d | resume_step=%d",
            self.atoms.get_chemical_formula(),
            len(self.atoms),
            resume_step,
        )

        # --- Schritt 1: Geometrieoptimierung ODER Resume ----------------
        if resume_step > 0:
            logger.info("Resume ab Schritt %d — überspringe Optimierung.", resume_step)
            initial_positions = self.atoms.get_positions().copy()
        else:
            logger.info("Geometrieoptimierung (BFGS, fmax=%.3f eV/Å) …", cfg.fmax)
            try:
                self._optimize_geometry()
            except Exception as exc:
                logger.error("Geometrieoptimierung fehlgeschlagen: %s", exc)
                result.status = "diverged"
                return result
            initial_positions = self.atoms.get_positions().copy()

        nn_dist = nearest_neighbor_distance(self.atoms)
        logger.info(
            "Optimiert | d_NN=%.3f Å | V=%.1f Å³",
            nn_dist,
            self.atoms.get_volume(),
        )

        # --- Schritt 2: NPT-MD initialisieren ----------------------------
        start_temp = max(cfg.t_start, 1e-3)
        dyn = NPT(
            self.atoms,
            timestep=cfg.time_step * fs,
            temperature_K=start_temp,
            externalstress=cfg.pressure * GPa,
            ttime=cfg.temperature_time_constant,
            # pfactor = τ_P² × B  (B = Bulk-Modulus in ASE-Einheiten)
            pfactor=(cfg.pressure_time_constant**2) * (cfg.bulk_modulus * GPa),
            loginterval=cfg.log_interval,
        )

        # Kombinierter Gleichgewichts-Monitor (Temperatur + Positionen)
        monitor = _CombinedEquilibriumMonitor(dyn, self.atoms, cfg)
        dyn.attach(monitor, interval=1)

        temperatures = np.arange(
            cfg.t_start, cfg.t_max + cfg.delta_t, cfg.delta_t
        )

        # Beim Resume: ab resume_step+1 weitermachen
        if resume_step > 0:
            temperatures = temperatures[resume_step + 1:]

        # --- Schritt 3: Temperaturschleife --------------------------------
        n_steps_total = len(temperatures)
        timed_out = False
        decay_consecutive = 0        # aufeinanderfolgende Schritte über Lindemann-Schwelle
        decay_temp_candidate: float | None = None  # T des ersten Überschreitens

        for i, T in enumerate(temperatures):
            global_step = resume_step + 1 + i
            T_sim = max(T, 1e-3)  # 0 K vermeiden
            dyn.set_temperature(temperature_K=T_sim)

            # Divergenz-Check VOR dem MD-Schritt
            if self._check_divergence():
                logger.error("Divergenz bei T=%.1f K – Abbruch.", T)
                result.status = "diverged"
                break

            print(f"-> Starte MD-Simulation für T = {T:.1f} K ... "
                  f"(Schritt {global_step}/{resume_step + n_steps_total})", flush=True)

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
                print(f"   [INFO] Gleichgewicht (Temp+Pos) vorzeitig erreicht "
                      f"bei Schritt {stopped_at}. Springe zur nächsten Temperatur.",
                      flush=True)

            # Divergenz-Check NACH dem MD-Schritt
            if self._check_divergence():
                logger.error("Divergenz nach T=%.1f K – Abbruch.", T)
                result.status = "diverged"
                break

            # Metriken sammeln — konvergierte Samples bevorzugen (drift-frei)
            metrics = self._collect_metrics(T, initial_positions, monitor.converged_samples)
            result.add(metrics)

            print(f"   [ERFOLG] T = {T:.1f} K nach {elapsed:.2f} Sekunden beendet. "
                  f"(MSD: {metrics.msd:.4f})", flush=True)

            # ── Debug: Konvergenz-Plot speichern ──
            if cfg.debug_convergence_plots:
                try:
                    formula = self.atoms.get_chemical_formula()
                    plot_path = save_convergence_plot(
                        history=monitor.history,
                        temperature=T,
                        formula=formula,
                        material_id=material_id,
                        output_dir=cfg.debug_plot_dir,
                        early_stop_rel_std=cfg.early_stop_rel_std,
                        pos_convergence_threshold=cfg.pos_convergence_threshold,
                        stopped_at=monitor.stopped_at,
                    )
                    logger.info("Debug-Plot gespeichert: %s", plot_path)
                except Exception as exc:
                    logger.warning("Debug-Plot fehlgeschlagen: %s", exc)

            # ── Checkpoint nur alle N Schritte (und immer vor Abbruch) ───
            is_interval_step = global_step % cfg.checkpoint_interval == 0
            if checkpoint_cb is not None and is_interval_step:
                checkpoint_cb(global_step, T, result)

            # ── Deadline-Check: sauber abbrechen nach vollem Schritt ─────
            if deadline is not None and time.perf_counter() > deadline:
                logger.warning(
                    "Deadline erreicht nach T=%.1f K (Schritt %d) — breche ab.",
                    T, global_step,
                )
                # Finaler Checkpoint vor Abbruch (auch wenn nicht Intervall-Schritt)
                if checkpoint_cb is not None and not is_interval_step:
                    checkpoint_cb(global_step, T, result)
                result.status = "timed_out"
                timed_out = True
                break

            # Frühzeitiger Abbruch bei Zerfall (Lindemann) mit Persistenz
            lindemann_threshold = (cfg.lindemann_fraction * nn_dist) ** 2
            if result.t_decay is None:
                if metrics.msd > lindemann_threshold:
                    if decay_consecutive == 0:
                        decay_temp_candidate = T
                    decay_consecutive += 1
                    if decay_consecutive >= cfg.decay_persistence:
                        result.t_decay = decay_temp_candidate
                        result.status = "decayed"
                        if checkpoint_cb is not None and not is_interval_step:
                            checkpoint_cb(global_step, T, result)
                        logger.warning(
                            "Zerfall detektiert bei T=%.1f K "
                            "(nach %d konsekutiven Schritten) – Abbruch.",
                            decay_temp_candidate, decay_consecutive,
                        )
                        print(
                            f"   [WARNUNG] Zerfall detektiert bei T = "
                            f"{decay_temp_candidate:.1f} K "
                            f"(nach {decay_consecutive} konsekutiven Schritten) "
                            f"– Abbruch.",
                            flush=True,
                        )
                        break
                else:
                    # MSD wieder unter Schwelle → Reset
                    decay_consecutive = 0
                    decay_temp_candidate = None

        # --- Post-Processing (nur wenn nicht timed_out) -----------------
        if not timed_out:
            # T_switch: primär RDF-basiert, dann MSD-basiert als Fallback
            if result.t_switch is None and len(result.rdf_history) >= 2:
                result.t_switch = detect_t_switch(
                    result.rdf_history,
                    result.temperatures,
                    volumes=result.volumes,
                    min_temperature=100.0,
                )
            if result.t_switch is None and len(result.msd_values) >= 1:
                result.t_switch = detect_t_switch_from_msd(
                    result.msd_values,
                    result.temperatures,
                    min_absolute_msd=0.001,
                    min_temperature=100.0,
                )
            # T_decay: nur wenn nicht schon inline detektiert
            if result.t_decay is None and len(result.msd_values) >= 1:
                result.t_decay = detect_t_decay(
                    result.msd_values,
                    result.temperatures,
                    nn_distance=nn_dist,
                    lindemann_fraction=cfg.lindemann_fraction,
                    min_persistence=cfg.decay_persistence,
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
        positions_samples: np.ndarray | None = None,
    ) -> StepMetrics:
        """Alle Metriken für den aktuellen Snapshot erfassen.

        Parameters
        ----------
        positions_samples
            Gesammelte Positionssnapshots aus der Thermalisierungsphase
            (Form ``(n_samples, N, 3)``).  Wenn vorhanden, wird die
            vibrational MSD (Schwingungsamplitude um die Gleichgewichtsposition)
            berechnet.  Andernfalls wird auf die alte drift-behaftete MSD
            gegen die 0 K-Positionen zurückgegriffen.
        """
        atoms = self.atoms
        r, g = compute_rdf(
            atoms,
            r_max=self.cfg.rdf_r_max,
            n_bins=self.cfg.rdf_n_bins,
        )
        # Vibrational MSD (um Gleichgewichtsposition) bevorzugen,
        # falls Positionssamples vorhanden sind.
        if positions_samples is not None and positions_samples.shape[0] >= 2:
            msd = compute_vibrational_msd(positions_samples)
        else:
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

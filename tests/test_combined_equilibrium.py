"""Tests für den kombinierten Gleichgewichts-Monitor (_CombinedEquilibriumMonitor).

Testet:
- Temperatur-Konvergenz alleine reicht nicht zum Stop
- Positions-Konvergenz alleine reicht nicht zum Stop
- Beide zusammen → Stop
- Rolling-Mean findet neue Gleichgewichtsposition nach strukturellem Shift
- Vibrational MSD aus konvergiertem Fenster (drift-frei)
- Schwingungsperioden-Schätzung via Autokorrelation
- max_steps als Safety-Cap greift
"""

from __future__ import annotations

import numpy as np

from mat_sim.md import (
    RampConfig,
    _CombinedEquilibriumMonitor,
    _EquilibriumStop,
)


# ── Hilfs-Fakes ────────────────────────────────────────────────────────────
class _FakeDyn:
    """Minimaler ASE-Dynamics-Stub (wird für set_temperature gebraucht)."""

    def __init__(self) -> None:
        self._temp = 300.0

    def set_temperature(self, temperature_K: float) -> None:
        self._temp = temperature_K


class _FakeAtoms:
    """Minimaler Atoms-Stub mit setzbaren Positionen und Temperatur."""

    def __init__(self, n_atoms: int = 4) -> None:
        self._positions = np.zeros((n_atoms, 3))
        self._temp = 300.0
        self._volume = 1000.0

    def get_positions(self) -> np.ndarray:
        return self._positions.copy()

    def set_positions(self, positions: np.ndarray) -> None:
        self._positions = positions

    def get_temperature(self) -> float:
        return self._temp

    def get_volume(self) -> float:
        return self._volume


def _run_monitor(
    monitor: _CombinedEquilibriumMonitor,
    n_steps: int,
    temp_fn,
    pos_fn,
    vol_fn=None,
) -> tuple[bool, int | None]:
    """Führt den Monitor für n_steps Schritte aus.

    Parameters
    ----------
    vol_fn
        Funktion ``(step) -> float`` für das Volumen.  Default: konstant 1000.
    """
    stopped = False
    for step in range(1, n_steps + 1):
        monitor._atoms._temp = temp_fn(step)
        monitor._atoms._positions = pos_fn(step)
        if vol_fn is not None:
            monitor._atoms._volume = vol_fn(step)
        try:
            monitor()
        except _EquilibriumStop:
            stopped = True
            return stopped, monitor.stopped_at
    return stopped, monitor.stopped_at


# ── Konfigurationen ────────────────────────────────────────────────────────
def _fast_cfg(**overrides) -> RampConfig:
    """Config mit kleinen Fenstern für schnelle Tests."""
    defaults = dict(
        early_stop_min_steps=5,
        early_stop_window=5,
        early_stop_rel_std=0.05,
        msd_sample_interval=1,
        pos_convergence_min_samples=6,
        pos_convergence_window_mult=2,
        pos_convergence_min_window=3,
        pos_convergence_threshold=0.5,
        pos_convergence_rel_std=0.10,
        pos_convergence_eval_window=10,
        pos_convergence_persistence=3,
        vol_convergence_rel_std=0.02,
        vol_convergence_eval_window=10,
        vol_convergence_persistence=3,
    )
    defaults.update(overrides)
    return RampConfig(**defaults)


# ── 1. Nur Temperatur konvergiert → kein Stop ─────────────────────────────
def test_temp_converged_pos_not_stops():
    """Temperatur ist stabil, aber Positionen verschieben sich → kein Stop.

    Die Atome verschieben sich **unterschiedlich** (keine reine Translation),
    so dass die interne Struktur tatsächlich wandelt.
    """
    cfg = _fast_cfg()
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    # Konstante Temperatur, aber unterschiedlich driftende Positionen
    def temp_fn(step):
        return 300.0

    def pos_fn(step):
        p = np.zeros((4, 3))
        # Jedes Atom driftet mit unterschiedlicher Rate → interne Struktur ändert sich
        p[0, 0] = step * 0.1
        p[1, 0] = step * 0.2
        p[2, 0] = step * 0.05
        p[3, 0] = step * 0.15
        return p

    stopped, _ = _run_monitor(monitor, 200, temp_fn, pos_fn)
    assert not stopped, "Sollte nicht stoppen — Positionen noch in Bewegung"


# ── 2. Nur Positionen konvergiert → kein Stop ─────────────────────────────
def test_pos_converged_temp_not_stops():
    """Positionen stabil, aber Temperatur schwankt stark → kein Stop."""
    cfg = _fast_cfg()
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    def temp_fn(step):
        # Große Schwankungen: 200–400 K
        return 300.0 + 100.0 * np.sin(step * 0.5)

    def pos_fn(step):
        # Konstante Positionen nach Schritt 3
        p = np.zeros((4, 3))
        if step > 3:
            p[:, 0] = 1.0
        else:
            p[:, 0] = 0.0
        return p

    stopped, _ = _run_monitor(monitor, 200, temp_fn, pos_fn)
    assert not stopped, "Sollte nicht stoppen — Temperatur nicht konvergiert"


# ── 3. Beide konvergiert → Stop ───────────────────────────────────────────
def test_both_converged_stops():
    """Temperatur stabil UND Positionen stabil → Stop."""
    cfg = _fast_cfg()
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    def temp_fn(step):
        # Nach Schritt 5: stabil bei 300K mit kleinem Rauschen
        if step < 5:
            return 300.0 + 50.0 * np.exp(-step)
        return 300.0 + 0.5 * np.sin(step * 0.1)

    def pos_fn(step):
        # Nach Schritt 3: konstante Positionen mit kleinem Rauschen
        p = np.zeros((4, 3))
        p[:, 0] = 1.0
        if step > 3:
            p[:, 0] = 1.0 + 0.001 * np.sin(step * 0.3)
        else:
            p[:, 0] = 0.0 + 0.1 * step
        return p

    stopped, stopped_at = _run_monitor(monitor, 200, temp_fn, pos_fn)
    assert stopped, "Sollte stoppen — beide Bedingungen erfüllt"
    assert stopped_at is not None
    assert stopped_at >= 5, "Mindestschritte müssen erreicht sein"


# ── 4. Rolling-Mean findet neues Gleichgewicht nach Shift ─────────────────
def test_finds_new_equilibrium_after_shift():
    """Positionen verschieben sich (Phasenwechsel), pendeln dann ein → Stop.

    Szenario: Start bei Position x=0, dann allmählicher Shift zu x=1,
    dann Stabilisierung bei x=1. Der Rolling-Mean soll konvergieren,
    sobald die neue Position stabil ist.

    Die Atome verschieben sich **unterschiedlich** (nicht-uniform),
    so dass die interne Struktur tatsächlich wandelt.
    """
    cfg = _fast_cfg(pos_convergence_min_samples=6, pos_convergence_min_window=3)
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    # Basis-Struktur: 4 Atome in einem Quadrat
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    # Ziel-Struktur: Atome verschieben sich unterschiedlich (intern)
    target = np.array([[0.5, 0, 0], [1.2, 0, 0], [0, 1.3, 0], [1.1, 1.1, 0]])

    def temp_fn(step):
        return 340.0  # konstant

    def pos_fn(step):
        p = base.copy()
        # Phase 1 (Schritte 1-5): allmählicher interner Shift
        if step <= 5:
            frac = step / 5.0
            p = base + frac * (target - base)
        # Phase 2 (Schritte 6-15): weiterer Shift
        elif step <= 15:
            frac = (step - 5) / 10.0
            p = base + (0.5 + 0.5 * frac) * (target - base)
        # Phase 3 (Schritte 16+): stabil bei target
        else:
            p = target.copy()
            p[:, 0] += 0.001 * np.sin(step)
        return p

    stopped, stopped_at = _run_monitor(monitor, 100, temp_fn, pos_fn)
    assert stopped, "Sollte stoppen, nachdem neues Gleichgewicht gefunden"
    assert stopped_at is not None
    # Stop sollte erst nach der Stabilisierung (Schritt 15+) passieren
    assert stopped_at >= 15, (
        f"Stop darf nicht während des Shifts passieren (stopped_at={stopped_at})"
    )


# ── 5. konvergiertes Fenster liefert drift-freie Samples ──────────────────
def test_converged_samples_after_shift():
    """converged_samples enthält nur Samples aus dem neuen Gleichgewicht."""
    cfg = _fast_cfg(pos_convergence_min_samples=6, pos_convergence_min_window=3)
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)
    target = np.array([[0.5, 0, 0], [1.2, 0, 0], [0, 1.3, 0], [1.1, 1.1, 0]])

    def temp_fn(step):
        return 300.0

    def pos_fn(step):
        p = base.copy()
        if step <= 5:
            frac = step / 5.0
            p = base + frac * (target - base)
        elif step <= 15:
            frac = (step - 5) / 10.0
            p = base + (0.5 + 0.5 * frac) * (target - base)
        else:
            p = target.copy()
            p[:, 0] += 0.001 * np.sin(step)
        return p

    stopped, _ = _run_monitor(monitor, 100, temp_fn, pos_fn)
    assert stopped

    conv = monitor.converged_samples
    assert conv is not None
    assert conv.shape[0] > 0
    # Alle konvergierten Samples sollten nahe der neuen Position (target) sein
    for i in range(conv.shape[1]):
        assert np.all(np.abs(conv[:, i, :] - target[i]) < 0.2), (
            "Konvergierte Samples sollten die neue Position widerspiegeln"
        )


# ── 6. Max-Steps als Safety-Cap ───────────────────────────────────────────
def test_max_steps_no_convergence():
    """Keine Konvergenz → läuft bis max_steps (hier: n_steps der Schleife)."""
    cfg = _fast_cfg()
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    def temp_fn(step):
        return 300.0 + 100.0 * np.sin(step * 0.3)  # nie konvergiert

    def pos_fn(step):
        p = np.zeros((4, 3))
        # Jedes Atom driftet unterschiedlich → interne Struktur ändert sich
        p[0, 0] = step * 0.5
        p[1, 0] = step * 0.3
        p[2, 0] = step * 0.7
        p[3, 0] = step * 0.4
        return p

    stopped, _ = _run_monitor(monitor, 50, temp_fn, pos_fn)
    assert not stopped, "Sollte nicht stoppen — nichts konvergiert"


# ── 7. Schwingungsperioden-Schätzung ──────────────────────────────────────
def test_estimate_oscillation_period_sinus():
    """Autokorrelation erkennt Periode eines Sinussignals."""
    n = 50
    t = np.arange(n)
    period_true = 10  # Samples
    signal = np.sin(2 * np.pi * t / period_true)  # (n,)
    # Als (n, N=1, 3) Format
    samples = np.zeros((n, 1, 3))
    samples[:, 0, 0] = signal

    period_est = _CombinedEquilibriumMonitor._estimate_oscillation_period(samples)
    # Periode = 2 × Nulldurchgang → sollte ~period_true/2 = 5 sein
    # (Nulldurchgang bei period_true/2 = 5, dann 2×5 = 10)
    assert period_est >= 2
    # Sollte in der richtigen Größenordnung liegen
    assert 3 <= period_est <= 12


def test_estimate_oscillation_period_constant():
    """Konstante Positionen → Fallback (norm ≈ 0)."""
    samples = np.ones((10, 4, 3))
    period_est = _CombinedEquilibriumMonitor._estimate_oscillation_period(samples)
    assert period_est == 100  # Fallback (100 MD-Schritte = 100 fs)


def test_estimate_oscillation_period_too_short():
    """Zu wenige Samples → Fallback."""
    samples = np.random.rand(3, 4, 3)
    period_est = _CombinedEquilibriumMonitor._estimate_oscillation_period(samples)
    assert period_est == 100  # Fallback (100 MD-Schritte = 100 fs)


# ── 8. Reset funktioniert ─────────────────────────────────────────────────
def test_reset_clears_state():
    """Nach reset() sind alle Zustände geleert."""
    cfg = _fast_cfg()
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    # Ein paar Schritte laufen (kann vorzeitig stoppen)
    for step in range(1, 20):
        monitor._atoms._temp = 300.0
        monitor._atoms._positions = np.zeros((4, 3))
        try:
            monitor()
        except _EquilibriumStop:
            break

    assert monitor._step_count > 0
    assert len(monitor._position_samples) > 0

    monitor.reset()

    assert monitor._step_count == 0
    assert len(monitor._position_samples) == 0
    assert monitor._temp_converged is False
    assert monitor._pos_converged is False
    assert monitor._vol_converged is False
    assert monitor._stopped_at is None
    assert monitor._converged_window_size is None
    assert monitor._period_samples is None
    assert len(monitor._volumes) == 0


# ── 9. Tieftemperatur: Temperaturkonvergenz trivial ───────────────────────
def test_low_temp_trivial_convergence():
    """Bei T < 1 K ist Temperaturkonvergenz trivial → nur Positionen zählen."""
    cfg = _fast_cfg()
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    def temp_fn(step):
        return 0.5  # < 1 K

    def pos_fn(step):
        p = np.zeros((4, 3))
        if step > 3:
            p[:, 0] = 0.5 + 0.001 * np.sin(step)
        else:
            p[:, 0] = 0.0
        return p

    stopped, _ = _run_monitor(monitor, 200, temp_fn, pos_fn)
    assert stopped, "Bei T<1K soll nur Positions-Konvergenz entscheiden"


# ── 10. samples-Property (Kompatibilität) ─────────────────────────────────
def test_samples_property_returns_all():
    """samples gibt alle gesammelten Samples zurück (nicht nur konvergierte)."""
    cfg = _fast_cfg(msd_sample_interval=2)
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    for step in range(1, 30):
        monitor._atoms._temp = 300.0
        # Driftende Positionen (nicht-uniform) → keine Konvergenz → kein Stop
        p = np.zeros((4, 3))
        p[0, 0] = step * 0.1
        p[1, 0] = step * 0.2
        p[2, 0] = step * 0.05
        p[3, 0] = step * 0.15
        monitor._atoms._positions = p
        try:
            monitor()
        except _EquilibriumStop:
            break

    all_samples = monitor.samples
    assert all_samples is not None
    # sample_interval=2 → 29 Schritte → 14 Samples (kein Stop wegen Drift)
    assert all_samples.shape[0] == 14
    assert all_samples.shape[1] == 4
    assert all_samples.shape[2] == 3


# ── 11. Persistenz: transienter RMS-Dip triggert keinen Stop ──────────────
def test_persistence_prevents_transient_stop():
    """Ein einzelner RMS-Wert unter Schwelle reicht nicht zum Stop.

    Szenario: Positionen driften konstant (RMS hoch), aber für 2 Schritte
    springt RMS kurz unter die Schwelle, dann wieder hoch.  Mit
    persistence=3 sollte das NICHT als konvergiert gelten.
    """
    cfg = _fast_cfg(pos_convergence_persistence=3)
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    def temp_fn(step):
        return 300.0

    def pos_fn(step):
        p = np.zeros((4, 3))
        # Konstanter nicht-uniformer Drift, außer Schritt 8-9: kurz eingefroren
        if step in (8, 9):
            # Gleiche interne Struktur wie Schritt 7 → pos_rms ≈ 0
            p[0, 0] = 0.7 * 0.1
            p[1, 0] = 0.7 * 0.2
            p[2, 0] = 0.7 * 0.05
            p[3, 0] = 0.7 * 0.15
        else:
            # Jedes Atom driftet unterschiedlich
            p[0, 0] = 0.1 * step
            p[1, 0] = 0.2 * step
            p[2, 0] = 0.05 * step
            p[3, 0] = 0.15 * step
        return p

    stopped, _ = _run_monitor(monitor, 50, temp_fn, pos_fn)
    assert not stopped, "Transienter Dip darf nicht als Konvergenz zählen"


# ── 12. Persistenz: dauerhaft unter Schwelle → Stop ───────────────────────
def test_persistence_sustained_convergence_stops():
    """N aufeinanderfolgende Schritte unter Schwelle → Stop.

    Szenario: Positionen pendeln sich nach Schritt 8 ein.  Mit
    persistence=3 sollte der Stop bei Schritt 8+3=11 passieren.
    """
    cfg = _fast_cfg(
        pos_convergence_persistence=3,
        pos_convergence_min_samples=5,
        pos_convergence_min_window=3,
    )
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    def temp_fn(step):
        return 300.0

    def pos_fn(step):
        p = np.zeros((4, 3))
        if step <= 7:
            p[:, 0] = 0.1 * step  # Drift
        else:
            p[:, 0] = 1.0 + 0.0005 * np.sin(step)  # stabil
        return p

    stopped, stopped_at = _run_monitor(monitor, 100, temp_fn, pos_fn)
    assert stopped, "Sollte stoppen nach sustained convergence"
    assert stopped_at is not None
    # Stop erst nach persistence Schritten unter Schwelle
    assert stopped_at >= 10, f"Stop zu früh (stopped_at={stopped_at})"


# ── 13. Nicht-sticky: eine Bedingung fällt weg → kein Stop ────────────────
def test_non_sticky_temp_falls_away_no_stop():
    """Temperatur konvergiert, Positionen konvergieren später — aber Temperatur
    fällt vorher wieder weg → kein Stop.

    Szenario: Temperatur stabil in Schritten 10–20 (konvergiert),
    Positionen stabil ab Schritt 15.  Aber bei Schritt 25 fliegt die
    Temperatur wieder hoch.  Da beide **im selben Schritt** konvergiert
    sein müssen, sollte kein Stop passieren wenn die Temperatur weg ist.
    """
    cfg = _fast_cfg(
        early_stop_min_steps=5,
        early_stop_window=5,
        pos_convergence_min_samples=6,
        pos_convergence_min_window=3,
        pos_convergence_persistence=3,
    )
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    def temp_fn(step):
        # Stabil 300K in Schritten 5–20, dann große Schwankung ab 21
        if step <= 20:
            return 300.0 + 0.1 * np.sin(step)
        else:
            return 300.0 + 50.0 * np.sin(step * 0.8)

    def pos_fn(step):
        # Stabil ab Schritt 8
        p = np.zeros((4, 3))
        if step <= 7:
            p[:, 0] = 0.1 * step
        else:
            p[:, 0] = 1.0 + 0.0005 * np.sin(step)
        return p

    stopped, stopped_at = _run_monitor(monitor, 60, temp_fn, pos_fn)
    # Positionen sind stabil ab Schritt 8, Temperatur stabil bis 20.
    # Beide sind gleichzeitig konvergiert in Schritten ~11–20.
    # Ein Stop in diesem Fenster ist korrekt.
    # ABER: wenn der Stop in 11–20 nicht passiert (z.B. wegen Persistenz),
    # und die Temperatur ab 21 weg ist, darf er danach nicht mehr stoppen.
    if stopped:
        assert stopped_at is not None
        # Stop muss im Fenster passiert sein, wo beide konvergiert waren
        assert stopped_at <= 20, (
            f"Stop bei {stopped_at} — Temperatur war aber nur bis 20 konvergiert"
        )


# ── 14. Nicht-sticky: Positionen fallen weg → kein Stop ───────────────────
def test_non_sticky_pos_falls_away_no_stop():
    """Positionen konvergieren, Temperatur konvergiert später — aber Positionen
    fallen vorher wieder weg → kein Stop.

    Szenario: Positionen stabil in Schritten 8–20, Temperatur stabil ab 15.
    Ab Schritt 21 driftet die Position wieder.  Stop nur wenn beide im
    selben Schritt konvergiert sind.
    """
    cfg = _fast_cfg(
        early_stop_min_steps=5,
        early_stop_window=5,
        pos_convergence_min_samples=6,
        pos_convergence_min_window=3,
        pos_convergence_persistence=3,
    )
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    def temp_fn(step):
        # Stabil ab Schritt 10
        if step <= 9:
            return 300.0 + 50.0 * np.exp(-step)
        return 300.0 + 0.1 * np.sin(step)

    def pos_fn(step):
        # Stabil in Schritten 8–20, dann Drift ab 21
        p = np.zeros((4, 3))
        if 8 <= step <= 20:
            p[:, 0] = 1.0 + 0.0005 * np.sin(step)
        elif step > 20:
            p[:, 0] = 1.0 + 0.5 * (
                step - 20
            )  # Drift (schnell genug für pos_rms > ceiling)
        else:
            p[:, 0] = 0.1 * step
        return p

    stopped, stopped_at = _run_monitor(monitor, 60, temp_fn, pos_fn)
    if stopped:
        assert stopped_at is not None
        # Stop muss passieren, bevor Positionen wegdriften (≤ 20)
        assert stopped_at <= 20, (
            f"Stop bei {stopped_at} — Positionen waren aber nur bis 20 konvergiert"
        )


# ── 15. Volumen nicht konvergiert → kein Stop ─────────────────────────────
def test_volume_not_converged_prevents_stop():
    """Temp + Pos konvergiert, aber Volumen oszilliert → kein Stop.

    Szenario: Konstante Temperatur, konstante Positionen, aber das
    Volumen oszilliert mit großer Amplitude (Barostat-Artefakt).
    Der Stop darf nicht passieren, solange das Volumen nicht konvergiert ist.
    """
    cfg = _fast_cfg(
        vol_convergence_rel_std=0.01,  # strenge Schwelle
        vol_convergence_eval_window=10,
        vol_convergence_persistence=3,
    )
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    def temp_fn(step):
        return 300.0

    def pos_fn(step):
        p = np.zeros((4, 3))
        p[:, 0] = 1.0 + 0.001 * np.sin(step * 0.3)
        return p

    def vol_fn(step):
        # ±10% Oszillation → rel. Std >> 1%
        return 1000.0 + 100.0 * np.sin(step * 0.5)

    stopped, _ = _run_monitor(monitor, 200, temp_fn, pos_fn, vol_fn)
    assert not stopped, "Sollte nicht stoppen — Volumen noch in Oszillation"


# ── 16. Alle drei konvergiert → Stop ──────────────────────────────────────
def test_all_three_converged_stops():
    """Temp + Pos + Vol alle konvergiert → Stop.

    Szenario: Konstante Temperatur, konstante Positionen, konstantes Volumen.
    Alle drei Bedingungen erfüllt → Stop nach eval_window + persistence.
    """
    cfg = _fast_cfg(
        vol_convergence_rel_std=0.02,
        vol_convergence_eval_window=10,
        vol_convergence_persistence=3,
    )
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    def temp_fn(step):
        return 300.0

    def pos_fn(step):
        p = np.zeros((4, 3))
        p[:, 0] = 1.0 + 0.001 * np.sin(step * 0.3)
        return p

    def vol_fn(step):
        # Konstantes Volumen mit minimalem Rauschen (< 2% rel. Std)
        return 1000.0 + 0.5 * np.sin(step)

    stopped, stopped_at = _run_monitor(monitor, 200, temp_fn, pos_fn, vol_fn)
    assert stopped, "Sollte stoppen — alle drei Bedingungen erfüllt"
    assert stopped_at is not None
    # Vol braucht mindestens eval_window(10) + persistence(3) - 1 = 12 Schritte
    assert stopped_at >= 12, f"Stop zu früh (stopped_at={stopped_at})"


# ── 17. COM-Drift wird aus pos_rms entfernt → Stop ────────────────────────
def test_com_drift_ignored_for_convergence():
    """Alle Atome verschieben sich gemeinsam (COM-Drift), interne Struktur
    bleibt konstant → pos_rms sollte ~0 sein → Stop.

    Ohne COM-Entfernung würde der Rolling-Mean den Drift erfassen und
    pos_rms wäre groß → kein Stop.  Mit COM-Entfernung sollte die
    Konvergenz erkannt werden.
    """
    cfg = _fast_cfg(
        pos_convergence_min_samples=6,
        pos_convergence_min_window=3,
        pos_convergence_persistence=3,
    )
    dyn = _FakeDyn()
    atoms = _FakeAtoms(n_atoms=4)
    monitor = _CombinedEquilibriumMonitor(dyn, atoms, cfg)

    # Basis-Positionen: 4 Atome in einem Quadrat
    base = np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0], [1, 1, 0]], dtype=float)

    def temp_fn(step):
        return 300.0

    def pos_fn(step):
        # Alle Atome verschieben sich gemeinsam in x-Richtung (COM-Drift)
        # nach Schritt 3, aber die interne Struktur bleibt konstant.
        p = base.copy()
        if step > 3:
            p[:, 0] += 0.5 * (step - 3)  # COM-Drift
        return p

    stopped, _ = _run_monitor(monitor, 100, temp_fn, pos_fn)
    assert stopped, (
        "COM-Drift sollte ignoriert werden — interne Struktur ist konstant"
    )

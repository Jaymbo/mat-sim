"""Tests für detect_t_switch_from_msd()."""

from __future__ import annotations

import numpy as np

from mat_sim.metrics import detect_t_switch_from_msd

# ── Fixtures ────────────────────────────────────────────────────────────────

def _temps(n: int = 20, delta_t: float = 20.0) -> list[float]:
    """Temperatur-Zeitreihe: 0, 20, 40, … K."""
    return [i * delta_t for i in range(n)]


def _smooth_msd(n: int = 20, base: float = 0.01, drift: float = 0.0005) -> list[float]:
    """Glatte MSD-Kurve ohne Phasenwechsel (langsam ansteigend)."""
    return [base + drift * i for i in range(n)]


# ── Tests: Phasenwechsel detektieren ────────────────────────────────────────

def test_step_up_detected() -> None:
    """MSD springt auf ein 3× höheres Niveau und bleibt dort → T_switch."""
    msd = _smooth_msd(n=20, base=0.01, drift=0.0005)
    # Sprung bei Schritt 10 (T=200): von ~0.015 auf 0.05
    for i in range(10, 20):
        msd[i] = 0.05 + 0.001 * (i - 10)
    temps = _temps(len(msd))

    t = detect_t_switch_from_msd(msd, temps)
    assert t is not None
    assert t == 200.0


def test_step_down_detected() -> None:
    """MSD springt auf ein niedrigeres Niveau → T_switch (Abwärts-Sprung)."""
    msd = _smooth_msd(n=20, base=0.01, drift=0.0005)
    # Sprung bei Schritt 10: von ~0.015 auf 0.003 (≈ 0.2× der alten Basis)
    for i in range(10, 20):
        msd[i] = 0.003 + 0.0001 * (i - 10)
    temps = _temps(len(msd))

    t = detect_t_switch_from_msd(msd, temps)
    assert t is not None
    assert t == 200.0


# ── Tests: Kein Phasenwechsel ───────────────────────────────────────────────

def test_smooth_no_switch() -> None:
    """Glatte MSD-Kurve ohne Sprung → None."""
    msd = _smooth_msd(n=30)
    temps = _temps(len(msd))

    t = detect_t_switch_from_msd(msd, temps)
    assert t is None


def test_transient_spike_filtered() -> None:
    """Ein einzelner MSD-Spike, der sofort zurückkehrt → kein T_switch."""
    msd = _smooth_msd(n=20, base=0.01, drift=0.0005)
    # Spike bei Schritt 10: nur 1 Schritt hoch, dann zurück
    msd[10] = 0.05
    temps = _temps(len(msd))

    t = detect_t_switch_from_msd(msd, temps, min_persistence=3)
    assert t is None


def test_short_persistence_filtered() -> None:
    """Sprung hält nur 2 Schritte an (min_persistence=3) → verworfen."""
    msd = _smooth_msd(n=20, base=0.01, drift=0.0005)
    msd[10] = 0.05
    msd[11] = 0.05
    # ab Schritt 12 zurück zur smooth ramp
    temps = _temps(len(msd))

    t = detect_t_switch_from_msd(msd, temps, min_persistence=3)
    assert t is None


def test_gradual_increase_no_switch() -> None:
    """Langsame graduelle MSD-Zunahme (kein Sprung) → None."""
    # MSD steigt langsam von 0.01 auf 0.03 über 20 Schritte
    msd = list(np.linspace(0.01, 0.03, 20))
    temps = _temps(len(msd))

    t = detect_t_switch_from_msd(msd, temps)
    assert t is None


# ── Tests: Persistenz ──────────────────────────────────────────────────────

def test_persistent_shift_detected() -> None:
    """Sprung + Persistenz für 3 Schritte → erkannt."""
    msd = _smooth_msd(n=20, base=0.01, drift=0.0005)
    for i in range(10, 20):
        msd[i] = 0.04 + 0.001 * (i - 10)
    temps = _temps(len(msd))

    t = detect_t_switch_from_msd(msd, temps, min_persistence=3)
    assert t is not None
    assert t == 200.0


# ── Tests: Randfälle ────────────────────────────────────────────────────────

def test_too_few_steps() -> None:
    """Weniger als baseline_window+1+min_persistence Schritte → None."""
    msd = [0.01] * 5  # 5 < 3+1+3=7
    temps = _temps(len(msd))

    t = detect_t_switch_from_msd(msd, temps)
    assert t is None


def test_first_possible_step() -> None:
    """Sprung am frühestmöglichen Schritt (idx=baseline_window=3)."""
    msd = _smooth_msd(n=15, base=0.01, drift=0.0005)
    # Sprung bei idx=3 (T=60): MSD von ~0.0115 auf 0.04
    for i in range(3, 15):
        msd[i] = 0.04 + 0.001 * (i - 3)
    temps = _temps(len(msd))

    t = detect_t_switch_from_msd(msd, temps, baseline_window=3, min_persistence=3)
    assert t is not None
    assert t == 60.0


def test_custom_shift_factor() -> None:
    """Kleinerer shift_factor (1.5) detektiert auch kleinere Sprünge."""
    msd = _smooth_msd(n=20, base=0.01, drift=0.0005)
    # Sprung von 0.015 auf 0.025 (≈1.67×, knapp über 1.5×)
    for i in range(10, 20):
        msd[i] = 0.025 + 0.0005 * (i - 10)
    temps = _temps(len(msd))

    # Default shift_factor=2.0 → 0.025/0.015 ≈ 1.67 < 2.0 → None
    assert detect_t_switch_from_msd(msd, temps) is None

    # Mit shift_factor=1.5 → 1.67 > 1.5 → erkannt
    t = detect_t_switch_from_msd(msd, temps, shift_factor=1.5)
    assert t is not None
    assert t == 200.0


def test_zero_baseline_skipped() -> None:
    """Pre-Baseline mit MSD=0 wird übersprungen (keine Division durch Null)."""
    msd = [0.0] * 4 + [0.001, 0.001, 0.001, 0.04, 0.04, 0.04, 0.04]
    temps = _temps(len(msd))

    # Sollte nicht abstürzen; idx=3 hat pre=0 → übersprungen
    # idx=4: pre=mean(0,0,0.001)=0.00033 → 0.001/0.00033≈3 → könnte matchen
    # Aber step_diff zwischen 0.001 und 0.001 = 0 → nicht sudden
    # idx=7: pre=mean(0.001,0.001,0.001)=0.001 → 0.04/0.001=40 → match
    t = detect_t_switch_from_msd(msd, temps, baseline_window=3, min_persistence=3)
    assert t is not None
    assert t == 140.0


def test_oscillating_msd_no_switch() -> None:
    """Oszillierende MSD ohne dauerhaften Sprung → None."""
    # MSD oszilliert zwischen 0.01 und 0.02 (keine dauerhafte Verschiebung)
    msd = [0.01 if i % 2 == 0 else 0.02 for i in range(20)]
    temps = _temps(len(msd))

    t = detect_t_switch_from_msd(msd, temps)
    assert t is None

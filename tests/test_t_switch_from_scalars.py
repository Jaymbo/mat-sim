"""Tests für detect_t_switch_from_scalars()."""

from __future__ import annotations

from mat_sim.metrics import detect_t_switch_from_scalars


# ── Fixtures ────────────────────────────────────────────────────────────────

def _smooth_ramp(
    n_steps: int = 60,
    t_start: float = 0.0,
    delta_t: float = 10.0,
    vol0: float = 100.0,
    thermal_exp: float = 0.001,   # 0.1 % pro Schritt (graduelle Ausdehnung)
    q4_base: float = 0.76,
) -> tuple[list[float], list[float], list[float]]:
    """Glatte Temperaturrampe ohne Phasenwechsel (nur thermische Ausdehnung)."""
    temps = [t_start + i * delta_t for i in range(n_steps)]
    vols = [vol0 * (1 + thermal_exp * i) for i in range(n_steps)]
    q4s = [q4_base + 0.0005 * i for i in range(n_steps)]  # minimale Drift
    return temps, vols, q4s


# ── Tests: Phasenwechsel detektieren ────────────────────────────────────────

def test_volume_jump_detected() -> None:
    """Diskontinuierlicher Volumensprung > 3 % wird als T_switch erkannt."""
    temps, vols, q4s = _smooth_ramp()
    # Volumensprung bei Schritt 30 (T=300 K): +5 %
    for i in range(30, 60):
        vols[i] = vols[i] * 1.05

    t_sw = detect_t_switch_from_scalars(vols, q4s, temps)
    assert t_sw is not None
    assert t_sw == 300.0


def test_q4_jump_detected() -> None:
    """Q4-Sprung > 0.05 wird als T_switch erkannt."""
    temps, vols, q4s = _smooth_ramp()
    # Q4-Sprung bei Schritt 20 (T=200 K): von ~0.77 auf ~0.50
    for i in range(20, 60):
        q4s[i] = 0.50

    t_sw = detect_t_switch_from_scalars(vols, q4s, temps)
    assert t_sw is not None
    assert t_sw == 200.0


def test_both_signals_jump() -> None:
    """Volumen- und Q4-Sprung am selben Schritt → T_switch an dieser Temperatur."""
    temps, vols, q4s = _smooth_ramp()
    for i in range(25, 60):
        vols[i] *= 1.08
        q4s[i] = 0.40

    t_sw = detect_t_switch_from_scalars(vols, q4s, temps)
    assert t_sw is not None
    assert t_sw == 250.0


# ── Tests: Kein Phasenwechsel ───────────────────────────────────────────────

def test_smooth_ramp_no_switch() -> None:
    """Glatte thermische Ausdehnung ohne Sprung → kein T_switch."""
    temps, vols, q4s = _smooth_ramp()
    t_sw = detect_t_switch_from_scalars(vols, q4s, temps)
    assert t_sw is None


def test_small_volume_change_no_switch() -> None:
    """Volumenänderung < 3 % wird nicht als Phasenwechsel gewertet."""
    temps, vols, q4s = _smooth_ramp()
    for i in range(30, 60):
        vols[i] *= 1.02  # 2 % — unter Schwellwert

    t_sw = detect_t_switch_from_scalars(vols, q4s, temps)
    assert t_sw is None


def test_small_q4_change_no_switch() -> None:
    """Q4-Änderung < 0.05 wird nicht als Phasenwechsel gewertet."""
    temps, vols, q4s = _smooth_ramp()
    for i in range(20, 60):
        q4s[i] -= 0.03  # unter Schwellwert (Default 0.05)

    t_sw = detect_t_switch_from_scalars(vols, q4s, temps)
    assert t_sw is None


# ── Tests: Persistenz-Check ─────────────────────────────────────────────────

def test_transient_spike_filtered() -> None:
    """Ein einzelner Volumen-Spike (1 Schritt) wird durch Persistenz-Check verworfen."""
    temps, vols, q4s = _smooth_ramp()
    # Spike bei Schritt 30: nur 1 Schritt lang, dann zurück zur normalen Rampe
    vols[30] *= 1.10
    # Schritt 31+: weiterhin normale Rampe (kein 3 % Shift gegenüber Schritt 29)
    # Da die smooth ramp bei Schritt 31 ~ 100 * (1 + 0.001*31) ≈ 103.1 hat,
    # und Schritt 29 ≈ 102.9 — Differenz < 3 % → nicht persistent

    t_sw = detect_t_switch_from_scalars(vols, q4s, temps, min_persistence=3)
    assert t_sw is None


def test_short_persistence_filtered() -> None:
    """Sprung, der nur 2 Schritte besteht (bei min_persistence=3) → verworfen."""
    temps, vols, q4s = _smooth_ramp()
    # Sprung bei Schritt 30, hält nur 2 Schritte (30, 31), dann zurück
    vols[30] *= 1.10
    vols[31] *= 1.10
    # Ab Schritt 32: zurück zur smooth ramp (rel. Änderung < 3 %)

    t_sw = detect_t_switch_from_scalars(vols, q4s, temps, min_persistence=3)
    assert t_sw is None


# ── Tests: Randfälle ────────────────────────────────────────────────────────

def test_too_few_steps() -> None:
    """Weniger als min_persistence+2 Schritte → None."""
    temps = [0.0, 10.0, 20.0]
    vols = [100.0, 110.0, 120.0]
    q4s = [0.76, 0.50, 0.40]

    t_sw = detect_t_switch_from_scalars(vols, q4s, temps, min_persistence=3)
    assert t_sw is None


def test_custom_thresholds() -> None:
    """Custom-Schwellwerte funktionieren korrekt."""
    temps, vols, q4s = _smooth_ramp()
    # Sprung von 2 % — Default-Schwelle ist 3 %, aber custom ist 1 %
    for i in range(30, 60):
        vols[i] *= 1.02

    # Default → kein Switch
    assert detect_t_switch_from_scalars(vols, q4s, temps) is None

    # Mit volume_threshold=0.01 → Switch detektiert
    t_sw = detect_t_switch_from_scalars(vols, q4s, temps, volume_threshold=0.01)
    assert t_sw is not None
    assert t_sw == 300.0


def test_q4_threshold_boundary() -> None:
    """Q4-Sprung genau an der 0.05-Schwelle wird erkannt."""
    temps, vols, q4s = _smooth_ramp()
    # Q4-Sprung bei Schritt 25 (T=250 K): 0.76 + 25*0.0005 = 0.7725 → 0.72
    # ΔQ4 = 0.0525, knapp über Default-Schwelle 0.05
    for i in range(25, 60):
        q4s[i] = 0.72

    t_sw = detect_t_switch_from_scalars(vols, q4s, temps)
    assert t_sw is not None
    assert t_sw == 250.0


def test_first_step_jump() -> None:
    """Sprung beim ersten Schritt (idx=1) wird korrekt erkannt."""
    n = 10
    temps = [i * 10.0 for i in range(n)]
    vols = [100.0] + [120.0] * (n - 1)  # 20 % Sprung bei Schritt 1
    q4s = [0.76] + [0.50] * (n - 1)     # Q4-Sprung bei Schritt 1

    t_sw = detect_t_switch_from_scalars(vols, q4s, temps, min_persistence=3)
    assert t_sw is not None
    assert t_sw == 10.0


def test_zero_volume_handled() -> None:
    """Volumen = 0 wird sicher behandelt (keine Division durch Null)."""
    temps = [i * 10.0 for i in range(10)]
    vols = [0.0] * 10
    q4s = [0.76] * 10

    # Sollte nicht abstürzen und None zurückgeben (kein Sprung detektierbar)
    t_sw = detect_t_switch_from_scalars(vols, q4s, temps)
    assert t_sw is None

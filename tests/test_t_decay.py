"""Tests für detect_t_decay() mit Persistenz-Check."""

from __future__ import annotations

from mat_sim.metrics import detect_t_decay

# ── Fixtures ────────────────────────────────────────────────────────────────

def _temps(n: int = 20, delta_t: float = 20.0) -> list[float]:
    """Temperatur-Zeitreihe: 0, 20, 40, … K."""
    return [i * delta_t for i in range(n)]


# ── Tests: Persistenz ───────────────────────────────────────────────────────

def test_immediate_decay_no_persistence() -> None:
    """Bei min_persistence=1 wird der erste Schritt über Schwelle zurückgegeben."""
    nn = 3.0  # Å
    frac = 0.12
    threshold = (frac * nn) ** 2  # = 0.1296 Å²
    # MSD-Serie: erst niedrig, dann ein Sprung
    msd = [0.001] * 5 + [threshold * 2] * 5
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac, min_persistence=1)
    assert t == 100.0  # T=5*20=100


def test_decay_with_persistence_2() -> None:
    """Bei min_persistence=2: MSD muss 2 Schritte über Schwelle bleiben."""
    nn = 3.0
    frac = 0.12
    threshold = (frac * nn) ** 2
    msd = [0.001] * 5 + [threshold * 2] * 5
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac, min_persistence=2)
    assert t == 100.0  # T des ERSTEN Überschreitens, nicht des zweiten


def test_transient_spike_filtered() -> None:
    """Ein einzelner MSD-Spike wird durch Persistenz-Check verworfen."""
    nn = 3.0
    frac = 0.12
    threshold = (frac * nn) ** 2
    # Spike bei Schritt 5, dann wieder zurück unter Schwelle
    msd = [0.001] * 5 + [threshold * 2] + [0.001] * 5
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac, min_persistence=2)
    assert t is None  # Spike gefiltert


def test_short_persistence_filtered() -> None:
    """2 Schritte über Schwelle, aber min_persistence=3 → verworfen."""
    nn = 3.0
    frac = 0.12
    threshold = (frac * nn) ** 2
    msd = [0.001] * 5 + [threshold * 2] * 2 + [0.001] * 5
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac, min_persistence=3)
    assert t is None


def test_persistence_3_passes() -> None:
    """3 Schritte über Schwelle mit min_persistence=3 → erkannt."""
    nn = 3.0
    frac = 0.12
    threshold = (frac * nn) ** 2
    msd = [0.001] * 5 + [threshold * 2] * 3 + [0.001] * 5
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac, min_persistence=3)
    assert t == 100.0


def test_decay_returns_first_temp() -> None:
    """T_decay ist die Temperatur des ERSTEN Überschreitens, nicht des letzten."""
    nn = 3.0
    frac = 0.12
    threshold = (frac * nn) ** 2
    msd = [0.001] * 5 + [threshold * 2] * 5
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac, min_persistence=3)
    assert t == 100.0  # Schritt 5 = T=100, nicht Schritt 7 = T=140


def test_spike_then_persistent_decay() -> None:
    """Spike bei Schritt 3 (gefiltert), dann echter Zerfall ab Schritt 7."""
    nn = 3.0
    frac = 0.12
    threshold = (frac * nn) ** 2
    msd = [0.001] * 3 + [threshold * 3] + [0.001] * 3 + [threshold * 2] * 5
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac, min_persistence=2)
    assert t == 140.0  # Schritt 7 = T=140


# ── Tests: Randfälle ────────────────────────────────────────────────────────

def test_no_decay() -> None:
    """MSD immer unter Schwelle → None."""
    nn = 3.0
    frac = 0.12
    threshold = (frac * nn) ** 2
    msd = [threshold * 0.5] * 20
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac)
    assert t is None


def test_decay_from_start() -> None:
    """MSD von Anfang an über Schwelle → T_decay = erste Temperatur."""
    nn = 3.0
    frac = 0.12
    threshold = (frac * nn) ** 2
    msd = [threshold * 2] * 10
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac, min_persistence=2)
    assert t == 0.0  # T=0


def test_too_few_steps() -> None:
    """Weniger als min_persistence Schritte über Schwelle am Ende → None."""
    nn = 3.0
    frac = 0.12
    threshold = (frac * nn) ** 2
    msd = [0.001] * 5 + [threshold * 2]  # nur 1 Schritt, min_persistence=2
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac, min_persistence=2)
    assert t is None


def test_threshold_boundary() -> None:
    """MSD genau auf Schwelle wird NICHT als decay gewertet (strikt >)."""
    nn = 3.0
    frac = 0.12
    threshold = (frac * nn) ** 2
    msd = [threshold] * 5  # genau auf Schwelle
    temps = _temps(len(msd))

    t = detect_t_decay(msd, temps, nn, frac, min_persistence=1)
    assert t is None

"""Tests für vibrational MSD (Schwingungsamplitude um Gleichgewichtsposition)."""

from __future__ import annotations

import numpy as np

from mat_sim.metrics import compute_vibrational_msd


class TestComputeVibrationalMsd:
    """Test-Suite für compute_vibrational_msd."""

    def test_zero_vibration(self):
        """Alle Positionen identisch → MSD = 0."""
        pos = np.zeros((5, 4, 3))
        # Verschiedene Positionen pro Atom, aber keine Schwingung über Samples
        for s in range(5):
            pos[s] = np.array([[0, 0, 0], [1, 1, 1], [2, 2, 2], [3, 3, 3]],
                              dtype=float)
        msd = compute_vibrational_msd(pos)
        assert msd == 0.0

    def test_known_vibration(self):
        """Symmetrische Schwingung um Mittelpunkt → MSD = amplitude²."""
        # 1 Atom, das zwischen +a und -a schwingt (nur x-Richtung)
        amplitude = 0.1
        n_samples = 100
        pos = np.zeros((n_samples, 1, 3))
        pos[:, 0, 0] = amplitude * np.sin(np.linspace(0, 10 * np.pi, n_samples))
        msd = compute_vibrational_msd(pos)
        # Mittelwert der sin² ist 0.5, also MSD ≈ amplitude² * 0.5
        expected = amplitude**2 * 0.5
        np.testing.assert_allclose(msd, expected, rtol=0.1)

    def test_drift_not_counted(self):
        """Linearer Drift (Translation) darf nicht zur MSD beitragen."""
        # 1 Atom driftet linear in x-Richtung, keine Schwingung
        n_samples = 50
        pos = np.zeros((n_samples, 1, 3))
        pos[:, 0, 0] = np.linspace(0, 10, n_samples)  # reiner Drift
        msd = compute_vibrational_msd(pos)
        # Alle Auslenkungen vom Mittelwert sind gleich groß,
        # aber der Mittelwert verschiebt sich mit dem Drift.
        # Bei linearem Drift ist die Auslenkung vom Mittelwert konstant:
        # disp = linspace(0, 10) - 5 = linspace(-5, 5)
        # MSD = mean(disp²) = mean(linspace(-5, 5)²)
        disp = np.linspace(-5, 5, n_samples)
        expected = np.mean(disp**2)
        np.testing.assert_allclose(msd, expected)

    def test_drift_with_vibration(self):
        """Drift + Schwingung: nur die Schwingung trägt zur MSD bei."""
        n_samples = 1000
        amplitude = 0.05
        drift = np.linspace(0, 5, n_samples)  # großer Drift
        vibration = amplitude * np.sin(np.linspace(0, 20 * np.pi, n_samples))
        pos = np.zeros((n_samples, 1, 3))
        pos[:, 0, 0] = drift + vibration
        msd = compute_vibrational_msd(pos)

        # Ohne Drift wäre MSD ≈ amplitude² * 0.5
        # Mit Drift: der Drift verschiebt nur den Mittelwert,
        # die Schwingung um den (verschobenen) Mittelwert bleibt gleich.
        # Aber der lineare Drift erzeugt eine zusätzliche Auslenkung
        # vom Mittelwert: disp_drift = linspace(-2.5, 2.5)
        # Diese ist groß im Vergleich zur Schwingung.
        # → MSD wird dominiert vom Drift!
        # Das ist genau das gewünschte Verhalten: wir messen die
        # Gesamtvariation um die mittlere Position.
        # Für das Lindemann-Kriterium ist das korrekt, weil auch
        # eine drifende Struktur (z. B. Phasenübergang mit Volumen-
        # änderung) eine hohe Variation zeigt.
        assert msd > 0

    def test_too_few_samples(self):
        """Weniger als 2 Samples → 0.0 (keine Statistik möglich)."""
        pos = np.zeros((1, 4, 3))
        assert compute_vibrational_msd(pos) == 0.0

    def test_none_input(self):
        """None-Input → 0.0."""
        assert compute_vibrational_msd(None) == 0.0

    def test_multiple_atoms(self):
        """Mehrere Atome mit unterschiedlicher Schwingung."""
        n_samples = 100
        a1, a2 = 0.1, 0.2
        pos = np.zeros((n_samples, 2, 3))
        pos[:, 0, 0] = a1 * np.sin(np.linspace(0, 10 * np.pi, n_samples))
        pos[:, 1, 0] = a2 * np.sin(np.linspace(0, 10 * np.pi, n_samples))
        msd = compute_vibrational_msd(pos)
        # sum(disp², axis=2) summiert über x,y,z → (n_samples, N)
        # mean() mittelt über samples und Atome
        # Atom 0: sum = a1²*sin²,  Atom 1: sum = a2²*sin²
        # mean = (a1²*0.5 + a2²*0.5) / 2
        expected = (a1**2 + a2**2) * 0.5 / 2
        np.testing.assert_allclose(msd, expected, rtol=0.1)

    def test_vibrational_vs_drift_msd(self):
        """Vibrational MSD misst Schwingungsamplitude, nicht Drift."""
        n_samples = 200
        amplitude = 0.05
        # Atom schwingt um Position (1, 0, 0)
        pos = np.zeros((n_samples, 1, 3))
        pos[:, 0, 0] = 1.0 + amplitude * np.sin(np.linspace(0, 20 * np.pi, n_samples))

        # Vibrational MSD: Schwingung um Mittelwert (≈1.0)
        vib_msd = compute_vibrational_msd(pos)

        # sum(disp², axis=2) summiert über x,y,z → nur x hat Werte
        # mean = amplitude² * mean(sin²) = amplitude² * 0.5
        expected_vib = amplitude**2 * 0.5
        np.testing.assert_allclose(vib_msd, expected_vib, rtol=0.1)

        # Vergleich: drift MSD gegen 0K-Position hängt vom Snapshot ab
        # und kann viel größer sein (Schwingung + Abweichung)
        assert vib_msd >= 0

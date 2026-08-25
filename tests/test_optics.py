"""Tests für die verfeinerte Optik-Modellierung in mat_sim/optics.py.

Getestet werden:
  1. AM1.5 3-Gaussian-Gewichtung (Normierung, Peak-Positionen)
  2. Albedo-basierte Normierung (Q_abs/Q_ext ≤ 1, Q_sca/Q_ext ≤ 1)
  3. Q4→Drude-Mapping (sigmoid-artige Abbildung, Q4 hoch → ω_p=0)
  4. Switching-Kontrast (Identität bei gleichen Zuständen, Max bei Komplementär)
  5. Partikelradius aus RDF-Struktur
  6. compute_optical_scores Rückgabewerte (Struktur-basiert, ohne PyMieScatt)
"""

from __future__ import annotations

import numpy as np
import pytest

from mat_sim.optics import (
    IR_WINDOW_WAVELENGTHS_NM,
    SOLAR_WAVELENGTHS_NM,
    DielectricModel,
    DrudeModel,
    LorentzOscillator,
    _am15_weights,
    _estimate_particle_radius_nm,
    _ir_emissivity,
    _q4_to_drude_strength,
    _rdf_first_peak,
    _solar_absorption,
    _solar_reflectance,
    _switching_contrast,
    compute_optical_scores,
    derive_dielectric_model,
)

# ── 1. AM1.5 3-Gaussian-Gewichtung ──────────────────────────────────────────

class TestAM15Weights:
    """Tests für die 3-Gaussian AM1.5-Approximation."""

    def test_weights_sum_to_one(self) -> None:
        """Gewichte summieren auf 1 (Normierung)."""
        w = _am15_weights(SOLAR_WAVELENGTHS_NM)
        assert np.isclose(w.sum(), 1.0, atol=1e-10)

    def test_weights_non_negative(self) -> None:
        """Alle Gewichte ≥ 0."""
        w = _am15_weights(SOLAR_WAVELENGTHS_NM)
        assert np.all(w >= 0.0)

    def test_peak_near_475nm(self) -> None:
        """Höchstes Gewicht im Bereich des UV-blau Peaks (~475 nm)."""
        w = _am15_weights(SOLAR_WAVELENGTHS_NM)
        peak_idx = int(np.argmax(w))
        peak_wl = SOLAR_WAVELENGTHS_NM[peak_idx]
        # Peak sollte im Bereich 400–600 nm liegen
        assert 400 <= peak_wl <= 600

    def test_nir_contribution_present(self) -> None:
        """NIR-Bereich (1000–2500 nm) hat signifikantes Gewicht (>5%)."""
        w = _am15_weights(SOLAR_WAVELENGTHS_NM)
        mask = SOLAR_WAVELENGTHS_NM >= 1000
        nir_fraction = w[mask].sum()
        assert nir_fraction > 0.05, (
            f"NIR-Anteil {nir_fraction:.3f} sollte > 5% sein"
        )

    def test_empty_wavelengths(self) -> None:
        """Leeres Array → leere Gewichte (kein Crash)."""
        w = _am15_weights(np.array([]))
        assert w.size == 0


# ── 2. Albedo-basierte Normierung ───────────────────────────────────────────

class TestAlbedoNormalization:
    """Tests für Q/Q_ext-basierte Normierung."""

    def test_solar_reflectance_in_unit_interval(self) -> None:
        """Solar-Reflexion ∈ [0, 1] für typische Q-Werte."""
        n = len(SOLAR_WAVELENGTHS_NM)
        q_sca = np.ones(n) * 0.5
        q_ext = np.ones(n) * 1.0
        refl = _solar_reflectance(q_sca, q_ext, SOLAR_WAVELENGTHS_NM)
        assert 0.0 <= refl <= 1.0

    def test_solar_absorption_in_unit_interval(self) -> None:
        """Solar-Absorption ∈ [0, 1] für typische Q-Werte."""
        n = len(SOLAR_WAVELENGTHS_NM)
        q_abs = np.ones(n) * 0.3
        q_ext = np.ones(n) * 1.0
        abs_val = _solar_absorption(q_abs, q_ext, SOLAR_WAVELENGTHS_NM)
        assert 0.0 <= abs_val <= 1.0

    def test_absorption_plus_reflection_leq_one(self) -> None:
        """Q_abs + Q_sca = Q_ext → Absorption + Reflexion ≤ 1."""
        n = len(SOLAR_WAVELENGTHS_NM)
        q_abs = np.ones(n) * 0.3
        q_sca = np.ones(n) * 0.5
        q_ext = q_abs + q_sca  # physikalische Konsistenz
        refl = _solar_reflectance(q_sca, q_ext, SOLAR_WAVELENGTHS_NM)
        abs_val = _solar_absorption(q_abs, q_ext, SOLAR_WAVELENGTHS_NM)
        assert refl + abs_val <= 1.0 + 1e-10

    def test_ir_emissivity_in_unit_interval(self) -> None:
        """IR-Emissivität ∈ [0, 1]."""
        n = len(IR_WINDOW_WAVELENGTHS_NM)
        q_abs = np.ones(n) * 0.8
        q_ext = np.ones(n) * 1.0
        emiss = _ir_emissivity(q_abs, q_ext, IR_WINDOW_WAVELENGTHS_NM)
        assert 0.0 <= emiss <= 1.0

    def test_zero_ext_no_crash(self) -> None:
        """Q_ext = 0 → keine Division durch Null, Rückgabe 0."""
        n = len(SOLAR_WAVELENGTHS_NM)
        q = np.ones(n) * 0.5
        zeros = np.zeros(n)
        refl = _solar_reflectance(q, zeros, SOLAR_WAVELENGTHS_NM)
        abs_val = _solar_absorption(q, zeros, SOLAR_WAVELENGTHS_NM)
        assert refl == 0.0
        assert abs_val == 0.0


# ── 3. Q4→Drude-Mapping ─────────────────────────────────────────────────────

class TestQ4ToDrude:
    """Tests für die Q4→Drude-Parameter-Abbildung."""

    def test_high_q4_zero_omega_p(self) -> None:
        """Q4 ≥ 0.6 (kristallin/isolierend) → ω_p = 0."""
        omega_p, _ = _q4_to_drude_strength(q4=0.7, volume=100.0, n_atoms=12)
        assert omega_p == 0.0

    def test_low_q4_positive_omega_p(self) -> None:
        """Q4 niedrig (amorph/metallisch) → ω_p > 0."""
        omega_p, _ = _q4_to_drude_strength(q4=0.1, volume=100.0, n_atoms=12)
        assert omega_p > 0.0

    def test_monotonic_decrease(self) -> None:
        """ω_p sinkt monoton mit steigendem Q4."""
        q4_vals = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
        omega_ps = [
            _q4_to_drude_strength(q4=q, volume=100.0, n_atoms=12)[0]
            for q in q4_vals
        ]
        for i in range(len(omega_ps) - 1):
            assert omega_ps[i] >= omega_ps[i + 1]

    def test_gamma_in_range(self) -> None:
        """γ_d ∈ [0.2, 0.8] eV."""
        _, gamma = _q4_to_drude_strength(q4=0.1, volume=100.0, n_atoms=12)
        assert 0.2 <= gamma <= 0.8
        _, gamma2 = _q4_to_drude_strength(q4=0.6, volume=100.0, n_atoms=12)
        assert 0.2 <= gamma2 <= 0.8

    def test_density_influence(self) -> None:
        """Höhere Dichte → höhere ω_p (bei gleichem Q4)."""
        # Dichte = n_atoms / volume
        omega_low_density, _ = _q4_to_drude_strength(
            q4=0.2, volume=200.0, n_atoms=12
        )
        omega_high_density, _ = _q4_to_drude_strength(
            q4=0.2, volume=50.0, n_atoms=12
        )
        assert omega_high_density >= omega_low_density


# ── 4. Switching-Kontrast ───────────────────────────────────────────────────

class TestSwitchingContrast:
    """Tests für den Switching-Kontrast-Score."""

    def test_zero_contrast_identical_states(self) -> None:
        """Identische Zustände → Kontrast = 0."""
        c = _switching_contrast(0.5, 0.5, 0.3, 0.3, 0.8, 0.8)
        assert c == 0.0

    def test_max_contrast_complementary(self) -> None:
        """Komplementäre Zustände → Kontrast nahe 100."""
        c = _switching_contrast(0.0, 1.0, 0.0, 1.0, 0.0, 1.0)
        assert c == 100.0

    def test_partial_contrast(self) -> None:
        """Teilweiser Kontrast → Score zwischen 0 und 100."""
        c = _switching_contrast(0.2, 0.8, 0.3, 0.3, 0.5, 0.5)
        # Nur Δ_refl = 0.6 trägt bei, Δ_abs = 0, Δ_ir = 0
        # → (0.6 + 0 + 0) / 3 * 100 = 20
        assert c == pytest.approx(20.0, abs=0.1)

    def test_score_in_unit_interval(self) -> None:
        """Kontrast ∈ [0, 100]."""
        c = _switching_contrast(0.1, 0.9, 0.2, 0.7, 0.3, 0.8)
        assert 0.0 <= c <= 100.0


# ── 5. Partikelradius aus RDF ───────────────────────────────────────────────

class TestParticleRadius:
    """Tests für Partikelradius-Schätzung aus RDF."""

    def test_valid_rdf(self) -> None:
        """RDF mit Peak bei ~2 Å → Radius ~100 nm."""
        r = np.linspace(0.5, 10.0, 200)
        g = np.zeros_like(r)
        g[10] = 5.0  # Scharfer Peak bei r ≈ 1.0 Å
        radius = _estimate_particle_radius_nm((r, g))
        assert 50.0 <= radius <= 500.0

    def test_no_rdf_returns_default(self) -> None:
        """Keine RDF → Default 500 nm."""
        radius = _estimate_particle_radius_nm(None)
        assert radius == 500.0

    def test_clipped_to_range(self) -> None:
        """Radius wird auf [50, 500] nm begrenzt."""
        # Sehr kleiner NN-Abstand → würde <50 nm ergeben, muss geclipped werden
        r = np.linspace(0.1, 10.0, 200)
        g = np.zeros_like(r)
        g[0] = 5.0  # Peak bei r ≈ 0.1 Å → 0.1 * 5 = 0.5 nm
        radius = _estimate_particle_radius_nm((r, g))
        assert radius >= 50.0

    def test_rdf_first_peak_extraction(self) -> None:
        """_rdf_first_peak findet den ersten Peak korrekt."""
        r = np.linspace(0.5, 10.0, 200)
        g = np.zeros_like(r)
        g[20] = 3.0  # Peak bei r ≈ 1.45 Å
        g[50] = 2.0  # Zweiter Peak bei r ≈ 3.0 Å
        peak = _rdf_first_peak((r, g))
        assert peak is not None
        assert peak < 2.0  # Erster Peak


# ── 6. DielectricModel & derive_dielectric_model ────────────────────────────

class TestDielectricModel:
    """Tests für Drude-Lorentz-Modell und Struktur-basierte Ableitung."""

    def test_dielectric_function_insulator(self) -> None:
        """Isolator (ω_p=0) → Im(ε) nur aus Lorentz-Termen."""
        model = DielectricModel(
            eps_inf=3.0,
            drude=DrudeModel(omega_p=0.0),
            lorentz=[LorentzOscillator(omega_0=4.0, gamma=0.5, f=2.0)],
        )
        energies = np.linspace(1.0, 6.0, 50)
        eps = model.dielectric_function(energies)
        # Bei ω_p=0 sollte Im(ε) an der Resonanz (4 eV) einen Peak haben
        peak_idx = int(np.argmax(eps.imag))
        assert 3.5 < energies[peak_idx] < 4.5

    def test_dielectric_function_metal(self) -> None:
        """Metall (ω_p>0) → Im(ε) steigt bei niedrigen Energien (Drude)."""
        model = DielectricModel(
            eps_inf=1.0,
            drude=DrudeModel(omega_p=5.0, gamma_d=0.5),
            lorentz=[],
        )
        energies = np.linspace(0.1, 5.0, 50)
        eps = model.dielectric_function(energies)
        # Drude-Absorption sollte bei niedrigen Energien am stärksten sein
        assert eps.imag[0] > eps.imag[-1]

    def test_refractive_index_imag_non_negative(self) -> None:
        """Im(n) ≥ 0 (physikalische Bedingung: Absorption ≥ 0)."""
        model = DielectricModel(
            eps_inf=3.0,
            drude=DrudeModel(omega_p=3.0, gamma_d=0.5),
            lorentz=[LorentzOscillator(omega_0=4.0, gamma=0.5, f=2.0)],
        )
        energies = np.linspace(0.5, 8.0, 100)
        m = model.refractive_index(energies)
        assert np.all(m.imag >= -1e-10)

    def test_derive_model_insulator(self) -> None:
        """Hoher Q4 → Drude ω_p = 0 (Isolator)."""
        r = np.linspace(0.5, 10.0, 200)
        g = np.zeros_like(r)
        g[30] = 5.0  # Peak bei ~2 Å
        model = derive_dielectric_model(
            rdf=(r, g), q4=0.7, volume=100.0, n_atoms=12,
        )
        assert model.drude.omega_p == 0.0
        # Sollte Lorentz-Oszillatoren haben
        assert len(model.lorentz) >= 3  # Interband + UV + IR-Phononen

    def test_derive_model_metal(self) -> None:
        """Niedriger Q4 → Drude ω_p > 0 (metallisch)."""
        r = np.linspace(0.5, 10.0, 200)
        g = np.zeros_like(r)
        g[30] = 5.0
        model = derive_dielectric_model(
            rdf=(r, g), q4=0.1, volume=100.0, n_atoms=12,
        )
        assert model.drude.omega_p > 0.0

    def test_derive_model_no_rdf(self) -> None:
        """Keine RDF → Fallback-Lorentz-Oszillator wird verwendet."""
        model = derive_dielectric_model(
            rdf=None, q4=0.5, volume=100.0, n_atoms=12,
        )
        # Sollte trotzdem funktionieren (Fallback bei 4.0 eV)
        assert len(model.lorentz) >= 2
        assert model.eps_inf > 1.0


# ── 7. compute_optical_scores (Legacy-Pfad, ohne mat) ──────────────────────

class TestComputeOpticalScores:
    """Tests für compute_optical_scores (Legacy-Pfad ohne StoredMaterial)."""

    def test_legacy_path_returns_all_keys(self) -> None:
        """Legacy-Pfad liefert alle erwarteten Keys."""
        result = compute_optical_scores("VO2", particle_radius_nm=200.0)
        expected_keys = {
            "cooling_score", "heating_score", "total_score",
            "contrast_score", "solar_reflectance", "solar_absorption_cold",
            "ir_emissivity_hot", "ir_emissivity_cold",
            "spectrum_heiss", "spectrum_kalt",
        }
        assert expected_keys.issubset(result.keys())

    def test_scores_in_valid_range(self) -> None:
        """Alle Scores ∈ [0, 100]."""
        result = compute_optical_scores("VO2", particle_radius_nm=200.0)
        for key in ("cooling_score", "heating_score", "total_score", "contrast_score"):
            val = result[key]
            assert 0.0 <= val <= 100.0, f"{key}={val} außerhalb [0, 100]"

    def test_total_is_weighted_average(self) -> None:
        """Total = 0.7 * base + 0.3 * contrast (mit base = (cooling+heating)/2)."""
        result = compute_optical_scores("VO2", particle_radius_nm=200.0)
        base = (result["cooling_score"] + result["heating_score"]) / 2.0
        expected = base * 0.7 + result["contrast_score"] * 0.3
        assert result["total_score"] == pytest.approx(expected, abs=0.2)

    def test_report_string_in_optical_summary(self) -> None:
        """optical_summary enthält einen lesbaren Report-String."""
        from mat_sim.optics import optical_summary
        result = optical_summary("VO2", particle_radius_nm=200.0)
        assert "report" in result
        assert "Total Score" in result["report"]
        assert "Kontrast" in result["report"]

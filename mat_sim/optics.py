"""Optische Modellierung für Switchable Radiative Cooling.

Implementiert:
  1. Drude-Lorentz-Modell für dielektrische Funktion ε(ω) bekannter Oxide
     inkl. Phonon-Resonanzen im fernen IR (10–20 µm)
  2. Mie-Streu-Berechnung mit PyMieScatt (Q_sca, Q_abs, Q_ext)
  3. Cooling-Efficiency-Score (heisser Zustand)  — Solar-Reflexion + IR-Emissivität
  4. Heating-Efficiency-Score (kalter Zustand)   — Solar-Absorption + IR-Rückhaltung
  5. Smart-Textile-Total-Score (arithmetisches Mittel)

Wellenlängen-Grids [nm]:
  - Solar:               300–2500 nm  (0.3–2.5 µm)
  - Atmosphärisches Fenster: 8000–13000 nm  (8–13 µm)
  - Nahtloses Gesamt-Grid: Solar + Gap + IR (dicht, für Plotting)
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Literal

import numpy as np

# ── PyMieScatt: SciPy 1.17 Kompatibilität (trapz → trapezoid) ──────────────
import scipy.integrate as _si
if not hasattr(_si, "trapz"):
    _si.trapz = _si.trapezoid

from PyMieScatt import MieQ  # noqa: E402

logger = logging.getLogger(__name__)

# ── Wellenlängen-Grids [nm] ─────────────────────────────────────────────────
SOLAR_WAVELENGTHS_NM = np.linspace(300, 2500, 200)
IR_WINDOW_WAVELENGTHS_NM = np.linspace(8000, 13000, 200)

# Nahtloses Gesamt-Grid: Solar (200 pts) + Gap 2500–8000 nm (100 pts) + IR (200 pts)
# Keine sichtbaren Sprünge im Dashboard-Plot.
_GAP_WAVELENGTHS_NM = np.linspace(2500, 8000, 100)
FULL_WAVELENGTHS_NM = np.concatenate([
    SOLAR_WAVELENGTHS_NM,
    _GAP_WAVELENGTHS_NM,
    IR_WINDOW_WAVELENGTHS_NM,
])

# Solar-Spektral-Gewichtung (vereinfachte AM1.5-G, normiert)
_AM15_PEAK_NM = 500.0
_AM15_SIGMA_NM = 350.0


# ── Drude-Lorentz-Modell ────────────────────────────────────────────────────

@dataclass
class LorentzOscillator:
    """Ein einzelner Lorentz-Oszillator.

    Parameters
    ----------
    omega_0 : float
        Resonanzfrequenz in eV.
    gamma : float
        Dämpfung (Linienbreite) in eV.
    f : float
        Oszillatorstärke (dimensionslos).
    """
    omega_0: float   # eV
    gamma: float     # eV
    f: float         # strength


@dataclass
class DrudeModel:
    """Drude-Modell für freie Elektronen (metallischer Zustand)."""
    omega_p: float = 0.0   # eV
    gamma_d: float = 0.1   # eV


@dataclass
class DielectricModel:
    """Vollständiges ε(ω)-Modell: ε_inf + Drude + Σ Lorentz."""
    eps_inf: float = 1.0
    drude: DrudeModel = field(default_factory=DrudeModel)
    lorentz: list[LorentzOscillator] = field(default_factory=list)

    def dielectric_function(self, energies_ev: np.ndarray) -> np.ndarray:
        """Komplexes ε(ω) für gegebene Energien [eV] berechnen."""
        eps = np.full_like(energies_ev, self.eps_inf, dtype=complex)

        # Drude-Anteil: -ω_p² / (ω² + iγω)
        if self.drude.omega_p > 0:
            w = energies_ev.astype(complex)
            eps += -self.drude.omega_p**2 / (w**2 + 1j * self.drude.gamma_d * w)

        # Lorentz-Anteile: f·ω₀² / (ω₀² - ω² - iγω)
        for osc in self.lorentz:
            w = energies_ev.astype(complex)
            eps += osc.f * osc.omega_0**2 / (osc.omega_0**2 - w**2 - 1j * osc.gamma * w)

        return eps

    def refractive_index(self, energies_ev: np.ndarray) -> np.ndarray:
        """Komplexer Brechungsindex m = n + ik = √ε(ω) (Im ≥ 0)."""
        eps = self.dielectric_function(energies_ev)
        m = np.sqrt(eps)
        mask = m.imag < 0
        m[mask] = -m[mask]
        return m


# ── IR-Phonon-Oszillatoren (10–20 µm) ──────────────────────────────────────
# Phonon-Resonanzen im fernen IR, typisch für Übergangsmetall-Oxide.
# 10 µm ≈ 0.124 eV, 12 µm ≈ 0.103 eV, 15 µm ≈ 0.083 eV, 18 µm ≈ 0.069 eV
# Diese erzeugen einen starken Anstieg von Im(ε) im atmosphärischen Fenster.

def _ir_phonon_oscillators() -> list[LorentzOscillator]:
    """Standard-IR-Phonon-Resonanzen für Übergangsmetall-Oxide.

    Zwei breite Resonanzen bei ~12 µm und ~16 µm erzeugen realistische
    Peaks in Q_abs innerhalb des 8–13 µm Fensters.
    """
    return [
        LorentzOscillator(omega_0=0.103, gamma=0.015, f=1.5),   # ~12 µm (TO-Phonon)
        LorentzOscillator(omega_0=0.078, gamma=0.012, f=1.0),   # ~16 µm (LO-Phonon)
    ]


# ── Bekannte Oxid-Modelle ──────────────────────────────────────────────────

def _vo2_models() -> dict[str, DielectricModel]:
    """VO₂: Isolierend (kalt, Monoklin) vs. metallisch (heiss, Rutile, T_c ≈ 340 K)."""
    cold = DielectricModel(
        eps_inf=4.0,
        drude=DrudeModel(omega_p=0.0, gamma_d=0.0),
        lorentz=[
            LorentzOscillator(omega_0=1.2, gamma=0.3, f=2.0),   # Interband
            LorentzOscillator(omega_0=3.5, gamma=1.0, f=3.0),   # UV
            LorentzOscillator(omega_0=5.5, gamma=1.5, f=2.0),   # tief-UV
            *_ir_phonon_oscillators(),                           # IR-Phononen
        ],
    )
    hot = DielectricModel(
        eps_inf=4.0,
        drude=DrudeModel(omega_p=4.0, gamma_d=0.8),
        lorentz=[
            LorentzOscillator(omega_0=3.5, gamma=1.0, f=3.0),
            LorentzOscillator(omega_0=5.5, gamma=1.5, f=2.0),
            *_ir_phonon_oscillators(),                           # IR-Phononen
        ],
    )
    return {"kalt": cold, "heiss": hot}


def _tiO2_models() -> dict[str, DielectricModel]:
    """TiO₂: Breitband-Isolator (Bandlücke ~3.2 eV)."""
    cold = DielectricModel(
        eps_inf=5.0,
        drude=DrudeModel(omega_p=0.0, gamma_d=0.0),
        lorentz=[
            LorentzOscillator(omega_0=4.0, gamma=0.5, f=3.0),
            LorentzOscillator(omega_0=6.0, gamma=1.0, f=4.0),
            LorentzOscillator(omega_0=8.0, gamma=2.0, f=3.0),
            *_ir_phonon_oscillators(),
        ],
    )
    hot = DielectricModel(
        eps_inf=5.0,
        drude=DrudeModel(omega_p=0.0, gamma_d=0.0),
        lorentz=[
            LorentzOscillator(omega_0=4.0, gamma=0.8, f=3.0),
            LorentzOscillator(omega_0=6.0, gamma=1.5, f=4.0),
            LorentzOscillator(omega_0=8.0, gamma=2.5, f=3.0),
            *_ir_phonon_oscillators(),
        ],
    )
    return {"kalt": cold, "heiss": hot}


def _v2o3_models() -> dict[str, DielectricModel]:
    """V₂O₃: Metall-Isolator-Übergang bei ~150 K."""
    cold = DielectricModel(
        eps_inf=3.5,
        drude=DrudeModel(omega_p=0.0, gamma_d=0.0),
        lorentz=[
            LorentzOscillator(omega_0=1.0, gamma=0.3, f=1.5),
            LorentzOscillator(omega_0=4.0, gamma=1.0, f=3.0),
            LorentzOscillator(omega_0=6.0, gamma=1.5, f=2.0),
            *_ir_phonon_oscillators(),
        ],
    )
    hot = DielectricModel(
        eps_inf=3.5,
        drude=DrudeModel(omega_p=3.5, gamma_d=0.6),
        lorentz=[
            LorentzOscillator(omega_0=4.0, gamma=1.0, f=3.0),
            LorentzOscillator(omega_0=6.0, gamma=1.5, f=2.0),
            *_ir_phonon_oscillators(),
        ],
    )
    return {"kalt": cold, "heiss": hot}


def _generic_oxide_models() -> dict[str, DielectricModel]:
    """Generic fallback für unbekannte Oxide."""
    cold = DielectricModel(
        eps_inf=3.0,
        drude=DrudeModel(omega_p=0.0, gamma_d=0.0),
        lorentz=[
            LorentzOscillator(omega_0=4.0, gamma=0.8, f=2.5),
            LorentzOscillator(omega_0=6.0, gamma=1.5, f=3.0),
            *_ir_phonon_oscillators(),
        ],
    )
    hot = DielectricModel(
        eps_inf=3.0,
        drude=DrudeModel(omega_p=1.5, gamma_d=0.5),
        lorentz=[
            LorentzOscillator(omega_0=4.0, gamma=1.0, f=2.5),
            LorentzOscillator(omega_0=6.0, gamma=1.5, f=3.0),
            *_ir_phonon_oscillators(),
        ],
    )
    return {"kalt": cold, "heiss": hot}


# Registry: Formel → Modell-Funktion
_MATERIAL_MODELS: dict[str, callable] = {
    "VO2": _vo2_models,
    "TiO2": _tiO2_models,
    "V2O3": _v2o3_models,
}


def get_dielectric_model(formula: str) -> dict[str, DielectricModel]:
    """Drude-Lorentz-Modell für eine gegebene Formel abrufen."""
    if formula in _MATERIAL_MODELS:
        return _MATERIAL_MODELS[formula]()

    key = formula.replace(" ", "")
    for known, func in _MATERIAL_MODELS.items():
        if key.lower() == known.lower():
            return func()

    logger.warning("Kein spezifisches Modell für %s – verwende generisches Oxid-Modell.", formula)
    return _generic_oxide_models()


# ── Mie-Streuung ────────────────────────────────────────────────────────────

def _wavelength_to_energy(wavelength_nm: np.ndarray) -> np.ndarray:
    """Wellenlänge [nm] → Energie [eV]: E = hc/λ."""
    return 1240.0 / wavelength_nm


@dataclass
class MieSpectrum:
    """Ergebnis einer Mie-Spektrums-Berechnung."""
    wavelengths_nm: np.ndarray
    q_ext: np.ndarray
    q_sca: np.ndarray
    q_abs: np.ndarray
    state: str


def simulate_mie_spectrum(
    formula: str,
    state: Literal["kalt", "heiss"] = "heiss",
    particle_radius_nm: float = 500.0,
    wavelengths_nm: np.ndarray | None = None,
) -> MieSpectrum:
    """Mie-Spektrum für eine Partikel-Suspension berechnen."""
    if wavelengths_nm is None:
        wavelengths_nm = FULL_WAVELENGTHS_NM

    models = get_dielectric_model(formula)
    model = models[state]

    energies = _wavelength_to_energy(wavelengths_nm)
    m_complex = model.refractive_index(energies)

    diameter_nm = 2.0 * particle_radius_nm

    q_ext = np.empty(len(wavelengths_nm))
    q_sca = np.empty(len(wavelengths_nm))
    q_abs = np.empty(len(wavelengths_nm))

    for i, (wl, mi) in enumerate(zip(wavelengths_nm, m_complex)):
        if mi.real <= 0 or wl <= 0:
            q_ext[i] = q_sca[i] = q_abs[i] = 0.0
            continue
        try:
            qe, qs, qa, *_ = MieQ(m=mi, wavelength=wl, diameter=diameter_nm)
            q_ext[i] = qe
            q_sca[i] = qs
            q_abs[i] = qa
        except Exception as exc:  # noqa: BLE001
            logger.debug("MieQ Fehler bei λ=%g nm, m=%s: %s", wl, mi, exc)
            q_ext[i] = q_sca[i] = q_abs[i] = 0.0

    return MieSpectrum(
        wavelengths_nm=wavelengths_nm,
        q_ext=q_ext,
        q_sca=q_sca,
        q_abs=q_abs,
        state=state,
    )


# ── Score-Hilfsfunktionen ───────────────────────────────────────────────────

def _am15_weights(wavelengths_nm: np.ndarray) -> np.ndarray:
    """Gauss-Approximation der AM1.5-G Solarstrahlung, normiert."""
    weights = np.exp(-0.5 * ((wavelengths_nm - _AM15_PEAK_NM) / _AM15_SIGMA_NM) ** 2)
    weights /= weights.sum()
    return weights


def _solar_reflectance(q_sca: np.ndarray, q_ext: np.ndarray, wls: np.ndarray) -> float:
    """Gewichtete Solar-Reflexion (0–1) aus Q_sca/Q_ext."""
    mask = np.isin(wls, SOLAR_WAVELENGTHS_NM) | (
        (wls >= SOLAR_WAVELENGTHS_NM[0]) & (wls <= SOLAR_WAVELENGTHS_NM[-1])
    )
    if not np.any(mask):
        return 0.0
    wls_s = wls[mask]
    qs_s = q_sca[mask]
    qe_s = q_ext[mask]
    weights = _am15_weights(wls_s)
    ratio = np.where(qe_s > 1e-10, qs_s / qe_s, 0.0)
    return float(np.clip(np.sum(weights * ratio), 0.0, 1.0))


def _solar_absorption(q_abs: np.ndarray, wls: np.ndarray) -> float:
    """Gewichtete Solar-Absorption (0–1) aus Q_abs."""
    mask = (wls >= SOLAR_WAVELENGTHS_NM[0]) & (wls <= SOLAR_WAVELENGTHS_NM[-1])
    if not np.any(mask):
        return 0.0
    wls_s = wls[mask]
    qa_s = q_abs[mask]
    weights = _am15_weights(wls_s)
    # Q_abs kann > 2; /2 als grobe Normierung
    return float(np.clip(np.sum(weights * qa_s) / 2.0, 0.0, 1.0))


def _ir_emissivity(q_abs: np.ndarray, wls: np.ndarray) -> float:
    """IR-Emissivität (0–1) im atmosphärischen Fenster (8–13 µm)."""
    mask = (wls >= IR_WINDOW_WAVELENGTHS_NM[0]) & (wls <= IR_WINDOW_WAVELENGTHS_NM[-1])
    if not np.any(mask):
        return 0.0
    qa_ir = q_abs[mask]
    return float(np.clip(np.mean(qa_ir) / 2.0, 0.0, 1.0))


# ── Cooling- & Heating-Scores ───────────────────────────────────────────────

def compute_optical_scores(
    formula: str,
    particle_radius_nm: float = 500.0,
) -> dict:
    """Cooling- und Heating-Efficiency-Score sowie Total-Score berechnen.

    Cooling (heisser Zustand):
      - Solar-Reflexion → hoch ist gut   (Gewicht 50 %)
      - IR-Emissivität 8–13 µm → hoch ist gut  (Gewicht 50 %)

    Heating (kalter Zustand):
      - Solar-Absorption → hoch ist gut  (Gewicht 50 %)
      - IR-Emissivität 8–13 µm → niedrig ist gut (Kuscheldecken-Effekt)  (Gewicht 50 %)

    Returns
    -------
    dict
        Alle Scores, Teilmetriken und kombinierte Spektren.
    """
    # Spektren auf dem nahtlosen Gesamt-Grid berechnen
    spec_hot = simulate_mie_spectrum(
        formula, state="heiss", particle_radius_nm=particle_radius_nm,
        wavelengths_nm=FULL_WAVELENGTHS_NM,
    )
    spec_cold = simulate_mie_spectrum(
        formula, state="kalt", particle_radius_nm=particle_radius_nm,
        wavelengths_nm=FULL_WAVELENGTHS_NM,
    )

    wls = spec_hot.wavelengths_nm

    # ── Cooling (heiss) ──
    solar_refl = _solar_reflectance(spec_hot.q_sca, spec_hot.q_ext, wls)
    ir_emiss_hot = _ir_emissivity(spec_hot.q_abs, wls)
    cooling_score = (solar_refl * 0.5 + ir_emiss_hot * 0.5) * 100.0

    # ── Heating (kalt) ──
    solar_abs_cold = _solar_absorption(spec_cold.q_abs, wls)
    ir_emiss_cold = _ir_emissivity(spec_cold.q_abs, wls)
    # Niedrige IR-Emissivität ist gut → (1 - ir_emiss_cold)
    heating_score = (solar_abs_cold * 0.5 + (1.0 - ir_emiss_cold) * 0.5) * 100.0

    # ── Total ──
    total_score = (cooling_score + heating_score) / 2.0

    return {
        "cooling_score": round(cooling_score, 1),
        "heating_score": round(heating_score, 1),
        "total_score": round(total_score, 1),
        "solar_reflectance": round(solar_refl, 3),
        "solar_absorption_cold": round(solar_abs_cold, 3),
        "ir_emissivity_hot": round(ir_emiss_hot, 3),
        "ir_emissivity_cold": round(ir_emiss_cold, 3),
        "spectrum_heiss": spec_hot,
        "spectrum_kalt": spec_cold,
    }


# Abwärtskompatibilität: cooling_efficiency_score als Wrapper
def cooling_efficiency_score(
    formula: str,
    particle_radius_nm: float = 500.0,
) -> dict:
    """Abwärtskompatibler Wrapper um :func:`compute_optical_scores`."""
    return compute_optical_scores(formula, particle_radius_nm)


def _combine_spectra(*specs: MieSpectrum) -> MieSpectrum:
    """Mehrere MieSpectrum-Objekte zusammenfügen."""
    wls = np.concatenate([s.wavelengths_nm for s in specs])
    qe = np.concatenate([s.q_ext for s in specs])
    qs = np.concatenate([s.q_sca for s in specs])
    qa = np.concatenate([s.q_abs for s in specs])
    order = np.argsort(wls)
    return MieSpectrum(
        wavelengths_nm=wls[order],
        q_ext=qe[order],
        q_sca=qs[order],
        q_abs=qa[order],
        state=specs[0].state,
    )


# ── Komfort für analyze.py ──────────────────────────────────────────────────

def _star_rating(total_score: float) -> str:
    """Sterne-Bewertung basierend auf dem Total Score."""
    if total_score >= 70:
        return "★★★ Ausgezeichnet"
    elif total_score >= 50:
        return "★★  Moderat"
    else:
        return "★   Gering"


def optical_summary(formula: str, particle_radius_nm: float = 500.0) -> dict:
    """Vollständige optische Auswertung für ein Material.

    Liefert Cooling-, Heating- und Total-Score sowie einen Text-Report.
    """
    result = compute_optical_scores(formula, particle_radius_nm=particle_radius_nm)

    lines = [
        "=" * 64,
        f"  Optische Mie-Analyse: {formula}",
        "=" * 64,
        f"  Partikelradius: {particle_radius_nm:.0f} nm",
        "",
        "  ── Kühl-Modus (heiss, Sommer) ──",
        f"  Solar-Reflexion:          {result['solar_reflectance']:.1%}",
        f"  IR-Emissivität 8–13 µm:   {result['ir_emissivity_hot']:.1%}",
        f"  Cooling-Score:            {result['cooling_score']:.1f} / 100",
        "",
        "  ── Heiz-Modus (kalt, Winter) ──",
        f"  Solar-Absorption:         {result['solar_absorption_cold']:.1%}",
        f"  IR-Rückhaltung (1-ε):     {1.0 - result['ir_emissivity_cold']:.1%}",
        f"  Heating-Score:            {result['heating_score']:.1f} / 100",
        "",
        "  ── Smart-Textile Gesamt ──",
        f"  Total Score:              {result['total_score']:.1f} / 100",
        f"  Bewertung:                {_star_rating(result['total_score'])}",
        "=" * 64,
    ]

    result["report"] = "\n".join(lines)
    return result

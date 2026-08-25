"""Optische Modellierung für Switchable Radiative Cooling.

Implementiert:
  1. Drude-Lorentz-Modell für dielektrische Funktion ε(ω) bekannter Oxide
     inkl. Phonon-Resonanzen im fernen IR (10–20 µm)
  2. Mie-Streu-Berechnung mit PyMieScatt (Q_sca, Q_abs, Q_ext)
  3. Cooling-Efficiency-Score (heisser Zustand)  — Solar-Reflexion + IR-Emissivität
  4. Heating-Efficiency-Score (kalter Zustand)   — Solar-Absorption + IR-Rückhaltung
  5. Smart-Textile-Total-Score (arithmetisches Mittel + Kontrast-Bonus)

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

from PyMieScatt import MieQ

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

# ── AM1.5-G Solar-Spektral-Gewichtung (3-Gaussian-Approximation) ───────────
# Die AM1.5-G Sonnenspektrale Bestrahlungsstärke (global, 300–2500 nm) wird
# durch drei Gauß-Komponenten approximiert:
#   1. UV-blau   (peak ~475 nm, schmal)  — Hauptmaximum der Sonnenstrahlung
#   2. Visible   (peak ~650 nm, breit)   — roter Schulter
#   3. NIR       (peak ~1000 nm, breit)  — Infrarot-Anteil (wichtig für Reflexion)
# Gewichte approximieren die relativen Integrale der AM1.5G-Spektren.
# Ref: ASTM G173-03, approximiert mit Gaussian Mixture Fit.
_AM15_GAUSSIANS: list[tuple[float, float, float]] = [
    # (weight, peak_nm, sigma_nm)
    (0.45, 475.0, 120.0),   # UV-blau Hauptpeak
    (0.30, 700.0, 200.0),   # Visible/rote Schulter
    (0.25, 1000.0, 350.0),  # NIR-Anteil
]


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


# ── Strukturbasierte Modell-Ableitung ──────────────────────────────────────
#
# Anstatt hardcodierte Drude-Lorentz-Parameter pro Formel zu verwenden,
# leiten wir die dielektrischen Parameter aus den MD-Simulationsdaten ab:
#
#   - RDF-Peak-Position → Nächste-Nachbar-Abstand d_NN
#       → Bandlücken-Schätzung über d-NN-Korrelation
#       → Lorentz-Oszillator-Resonanz (Interband-Übergang)
#
#   - Steinhardt Q4 (Symmetrie-Ordnungsparameter)
#       → Hoher Q4 = geordnete Kristallstruktur = Isolator (kein Drude-Term)
#       → Niedriger Q4 = Symmetriebruch = mögliche Metallisierung (Drude-Term)
#
#   - Volumen / Dichte
#       → ε_inf über Clausius-Mossotti-Näherung
#       → Dichtererhöhung → höhere Polarisierbarkeit → höhere ε_inf
#
#   - IR-Phonon-Resonanzen bleiben generisch (aus _ir_phonon_oscillators)
#     aber ihre Stärke skaliert mit der Anzahl Atome pro Volumen (Dichte)


def _rdf_first_peak(rdf: tuple[np.ndarray, np.ndarray] | None) -> float | None:
    """Ersten RDF-Peak auslesen (Nächste-Nachbar-Abstand in Å)."""
    if rdf is None:
        return None
    from scipy.signal import find_peaks
    r, g = rdf
    peaks, _ = find_peaks(g, height=0.5, distance=5)
    if peaks.size == 0:
        return None
    return float(r[peaks[0]])


def _nn_distance_to_bandgap(d_nn: float) -> float:
    """Nächste-Nachbar-Abstand → Bandlücken-Schätzung (eV).

    Empirische Korrelation für Übergangsmetall-Oxide:
      - d ≈ 1.9 Å (kleine Ionen) → E_gap ≈ 3.5 eV (TiO₂)
      - d ≈ 2.5 Å (mittlere Ionen) → E_gap ≈ 1.0 eV (V₂O₃)
      - d ≈ 2.9 Å (große Ionen) → E_gap ≈ 0.7 eV (VO₂)

    Näherung: E_gap = a / d² + b  (inverse-quadratisch)
    """
    # Fit-Punkte: (d_nn Å, E_gap eV)
    a = 12.0  # eV·Å²
    b = -0.5  # eV (Offset)
    return a / (d_nn ** 2) + b


def _q4_to_drude_strength(q4: float, volume: float, n_atoms: int) -> tuple[float, float]:
    """Steinhardt Q4 → Drude-Parameter (ω_p, γ_d).

    Kontinuierliche, sensitive Abbildung des Q4-Ordnungsparameters auf
    den Metallisierungsgrad:

      - Q4 ≥ 0.6  → eindeutig kristallin/isolierend → ω_p = 0
      - Q4 ≈ 0.5  → schwache Störung → geringe ω_p
      - Q4 ≈ 0.3  → starke Störung → moderate ω_p
      - Q4 ≈ 0.1  → stark amorph/metallisch → hohe ω_p

    Verwendet eine **quadratische** statt linearen Abbildung, um im
    mittleren Bereich (0.3–0.5, wo die meisten Oxide liegen) empfindlicher
    zu sein.

    Returns
    -------
    (omega_p, gamma_d)
        Plasmafrequenz und Dämpfung in eV.
    """
    # Q4-Bereich für Oxide: typisch 0.1–0.8
    # Kontinuierliche quadratische Abbildung:
    #   metallicity = clamp((0.6 - Q4) / 0.5, 0, 1)²
    # Q4=0.6 → 0, Q4=0.35 → 0.25, Q4=0.1 → 1.0
    raw_metallicity = max(0.0, (0.6 - q4) / 0.5)
    metallicity = raw_metallicity ** 2  # quadratisch → empfindlicher im Mittelbereich

    # Dichte-Einfluss: höhere Dichte → mehr freie Elektronen → höhere ω_p
    density = n_atoms / volume if volume > 0 else 0.0  # Atome/Å³
    # Typische Oxid-Dichte: 0.05–0.15 Atome/Å³
    density_factor = min(density / 0.1, 2.0)

    # Maximale ω_p ~12 eV für stark metallische, dichte Oxide
    omega_p = metallicity * density_factor * 6.0
    gamma_d = 0.2 + metallicity * 0.6  # 0.2–0.8 eV
    return omega_p, gamma_d


def _volume_to_eps_inf(volume: float, n_atoms: int) -> float:
    """Volumen/Dichte → ε_inf über vereinfachte Clausius-Mossotti-Näherung.

    Höhere Dichte → höhere Polarisierbarkeit → höhere ε_inf.
    """
    density = n_atoms / volume if volume > 0 else 0.0
    # Typische Oxid-Dichte 0.05–0.15 → ε_inf 2–6
    eps_inf = 1.0 + density * 40.0
    return float(np.clip(eps_inf, 1.5, 8.0))


def _estimate_particle_radius_nm(rdf: tuple[np.ndarray, np.ndarray] | None) -> float:
    """Partikelradius aus RDF-Struktur ableiten.

    Für Nanopartikel-Suspensionen sollte der Radius im Bereich der
    Kohärenzlänge der Kristallstruktur liegen.  Wir approximieren
    den Partikelradius als ~50× den Nächste-Nachbar-Abstand (typisch
    1.5–3 Å → 75–150 nm), was einer Domänengröße von ~10 Einheitszellen
    entspricht — realistisch für sol-gel synthetisierte Nanopartikel.

    Ohne RDF wird der Default-Wert (500 nm) zurückgegeben.
    """
    d_nn = _rdf_first_peak(rdf)
    if d_nn is None or d_nn < 0.5:
        return 500.0
    # 50 × d_nn (Å) → nm:  d_nn [Å] * 50 / 10 = d_nn * 5
    radius = d_nn * 5.0  # z.B. d_nn=2.0 Å → 100 nm
    return float(np.clip(radius, 50.0, 500.0))


def derive_dielectric_model(
    rdf: tuple[np.ndarray, np.ndarray] | None,
    q4: float,
    volume: float,
    n_atoms: int,
) -> DielectricModel:
    """Drude-Lorentz-Modell aus MD-Simulationsdaten ableiten.

    Parameters
    ----------
    rdf
        Radialverteilungsfunktion (r, g) des jeweiligen Zustands.
    q4
        Steinhardt Q4-Ordnungsparameter des jeweiligen Zustands.
    volume
        Zellvolumen in Å³.
    n_atoms
        Anzahl Atome in der Zelle.

    Returns
    -------
    DielectricModel
        Aus Strukturdaten abgeleitetes ε(ω)-Modell.
    """
    # ε_inf aus Dichte
    eps_inf = _volume_to_eps_inf(volume, n_atoms)

    # Drude-Term aus Q4 (Metallisierungsgrad)
    omega_p, gamma_d = _q4_to_drude_strength(q4, volume, n_atoms)

    lorentz: list[LorentzOscillator] = []

    # Interband-Übergang aus NN-Abstand → Bandlücke
    d_nn = _rdf_first_peak(rdf)
    if d_nn is not None and d_nn > 0.5:
        e_gap = max(_nn_distance_to_bandgap(d_nn), 0.3)
        # Lorentz-Oszillator an der Bandlücken-Energie
        # Stärke sinkt mit Drude-Term (Metall schirmt Interband ab)
        interband_strength = 2.0 * (1.0 - omega_p / 12.0) if omega_p > 0 else 2.0
        lorentz.append(LorentzOscillator(
            omega_0=e_gap, gamma=0.3, f=max(interband_strength, 0.5),
        ))
    else:
        # Fallback: generische Interband-Resonanz
        lorentz.append(LorentzOscillator(omega_0=4.0, gamma=0.8, f=2.5))

    # UV-Oszillatoren (immer vorhanden)
    lorentz.append(LorentzOscillator(omega_0=6.0, gamma=1.5, f=3.0))

    # IR-Phononen (Stärke skaliert mit Dichte)
    density = n_atoms / volume if volume > 0 else 0.0
    phonon_scale = min(density / 0.1, 2.0)
    for osc in _ir_phonon_oscillators():
        lorentz.append(LorentzOscillator(
            omega_0=osc.omega_0, gamma=osc.gamma,
            f=osc.f * phonon_scale,
        ))

    return DielectricModel(
        eps_inf=eps_inf,
        drude=DrudeModel(omega_p=omega_p, gamma_d=gamma_d),
        lorentz=lorentz,
    )


# ── Bekannte Oxid-Modelle (Legacy, für Fallback ohne Simulationsdaten) ─────

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


# Registry: Formel → Modell-Funktion (Legacy-Fallback)
_MATERIAL_MODELS: dict[str, callable] = {
    "VO2": _vo2_models,
}


def get_dielectric_model(formula: str) -> dict[str, DielectricModel]:
    """Legacy: Hardcodiertes Drude-Lorentz-Modell für eine Formel.

    Verwendet nur, wenn keine Simulationsdaten verfügbar sind.
    """
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
    model: DielectricModel | None = None,
) -> MieSpectrum:
    """Mie-Spektrum für eine Partikel-Suspension berechnen.

    Parameters
    ----------
    formula
        Chemische Formel (wird nur für Legacy-Fallback verwendet, wenn
        ``model`` None ist).
    state
        ``"kalt"`` oder ``"heiss"`` (nur für Legacy-Fallback).
    particle_radius_nm
        Partikelradius für Mie-Streuung in nm.
    model
        Wenn angegeben, wird dieses DielectricModel direkt verwendet
        (strukturbasierter Pfad).  Wenn *None*, wird das Legacy-Modell
        über ``get_dielectric_model(formula)`` geladen.
    """
    if wavelengths_nm is None:
        wavelengths_nm = FULL_WAVELENGTHS_NM

    if model is not None:
        dielectric = model
    else:
        models = get_dielectric_model(formula)
        dielectric = models[state]

    energies = _wavelength_to_energy(wavelengths_nm)
    m_complex = dielectric.refractive_index(energies)

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
    """3-Gaussian-Approximation der AM1.5-G Solarstrahlung, normiert.

    Drei Gauß-Komponenten approximieren die AM1.5G Sonnenspektrale
    Bestrahlungsstärke im Bereich 300–2500 nm:

      1. UV-blau   (peak ~475 nm, sigma ~120 nm)  — 45% der Energie
      2. Visible   (peak ~700 nm, sigma ~200 nm)  — 30%
      3. NIR       (peak ~1000 nm, sigma ~350 nm) — 25%

    Das NIR-Band (700–2500 nm) ist wichtig für die Solar-Reflexions-Bewertung,
    da hier die Mie-Streuung von Oxid-Nanopartikeln stark variiert.
    """
    weights = np.zeros_like(wavelengths_nm, dtype=float)
    for weight, peak, sigma in _AM15_GAUSSIANS:
        weights += weight * np.exp(
            -0.5 * ((wavelengths_nm - peak) / sigma) ** 2
        )
    norm = weights.sum()
    if norm > 0:
        weights /= norm
    return weights


def _solar_reflectance(q_sca: np.ndarray, q_ext: np.ndarray, wls: np.ndarray) -> float:
    """Gewichtete Solar-Reflexion (0–1) aus Q_sca/Q_ext (Albedo-basiert).

    Die Reflexion eines Partikels ist der Streu-Anteil am Extinktions-
    querschnitt: Q_sca / Q_ext.  Dieser Wert ist von Natur aus 0–1
    (keine willkürliche Normierung nötig).
    """
    mask = (wls >= SOLAR_WAVELENGTHS_NM[0]) & (wls <= SOLAR_WAVELENGTHS_NM[-1])
    if not np.any(mask):
        return 0.0
    wls_s = wls[mask]
    qs_s = q_sca[mask]
    qe_s = q_ext[mask]
    weights = _am15_weights(wls_s)
    ratio = np.divide(qs_s, qe_s, out=np.zeros_like(qs_s), where=qe_s > 1e-10)
    return float(np.clip(np.sum(weights * ratio), 0.0, 1.0))


def _solar_absorption(q_abs: np.ndarray, q_ext: np.ndarray, wls: np.ndarray) -> float:
    """Gewichtete Solar-Absorption (0–1) aus Q_abs/Q_ext (Albedo-basiert).

    Die Absorption ist der Absorptions-Anteil am Extinktionsquerschnitt:
    Q_abs / Q_ext.  Dieser Wert ist von Natur aus 0–1, da
    Q_abs = Q_ext - Q_sca ≤ Q_ext.
    """
    mask = (wls >= SOLAR_WAVELENGTHS_NM[0]) & (wls <= SOLAR_WAVELENGTHS_NM[-1])
    if not np.any(mask):
        return 0.0
    wls_s = wls[mask]
    qa_s = q_abs[mask]
    qe_s = q_ext[mask]
    weights = _am15_weights(wls_s)
    ratio = np.divide(qa_s, qe_s, out=np.zeros_like(qa_s), where=qe_s > 1e-10)
    return float(np.clip(np.sum(weights * ratio), 0.0, 1.0))


def _ir_emissivity(q_abs: np.ndarray, q_ext: np.ndarray, wls: np.ndarray) -> float:
    """IR-Emissivität (0–1) im atmosphärischen Fenster (8–13 µm).

    Verwendet Q_abs/Q_ext statt der rohen /2-Normierung.  Im IR-Fenster
    dominieren Phonon-Resonanzen, die Q_abs/Q_ext nahe 1 bringen können
    (starke Absorption = hohe Emissivität = gute Kühlung).
    """
    mask = (wls >= IR_WINDOW_WAVELENGTHS_NM[0]) & (wls <= IR_WINDOW_WAVELENGTHS_NM[-1])
    if not np.any(mask):
        return 0.0
    qa_ir = q_abs[mask]
    qe_ir = q_ext[mask]
    ratio = np.divide(qa_ir, qe_ir, out=np.zeros_like(qa_ir), where=qe_ir > 1e-10)
    return float(np.clip(np.mean(ratio), 0.0, 1.0))


# ── Cooling- & Heating-Scores ───────────────────────────────────────────────

def _extract_state_params(
    mat,
    state: Literal["kalt", "heiss"],
) -> dict:
    """Simulationsdaten für einen Zustand (kalt/heiss) aus StoredMaterial extrahieren.

    "kalt" = vor T_switch, "heiss" = nach T_switch.
    """
    rdf = mat.rdf_before if state == "kalt" else mat.rdf_after

    # Q4 und Volumen am T_switch-Punkt extrahieren
    temps = np.asarray(mat.temperatures)
    if mat.t_switch is not None and len(temps) > 0:
        idx = int(np.argmin(np.abs(temps - mat.t_switch)))
        if state == "kalt":
            idx = max(idx - 1, 0)
        else:
            idx = min(idx + 1, len(temps) - 1)
    else:
        idx = 0 if state == "kalt" else -1

    q4 = float(mat.ql_values[idx]) if idx < len(mat.ql_values) else 0.5
    volume = float(mat.volumes[idx]) if idx < len(mat.volumes) else 100.0

    # Anzahl Atome aus Struktur oder RDF abschätzen
    if mat.structure_before is not None:
        n_atoms = len(mat.structure_before)
    elif mat.structure_after is not None:
        n_atoms = len(mat.structure_after)
    else:
        n_atoms = 12  # Fallback

    return {"rdf": rdf, "q4": q4, "volume": volume, "n_atoms": n_atoms}


def _switching_contrast(
    solar_refl_cold: float,
    solar_refl_hot: float,
    solar_abs_cold: float,
    solar_abs_hot: float,
    ir_emiss_cold: float,
    ir_emiss_hot: float,
) -> float:
    """Kontrast-Score (0–100) für den optischen Switching-Kontrast.

    Belohnt Materialien, deren optische Eigenschaften sich am T_switch
    **stark** ändern — das ist das Kernkriterium für "switchable"
    radiative cooling.

    Drei Kontrast-Komponenten:
      1. |Δ Solar-Reflexion|  — je mehr Reflexions-Wechsel, desto besser
      2. |Δ Solar-Absorption| — je mehr Absorptions-Wechsel, desto besser
      3. |Δ IR-Emissivität|   — je mehr IR-Emissions-Wechsel, desto besser

    Jede Komponente wird auf 0–1 normiert (Δ max = 1.0), dann gemittelt.
    """
    d_refl = abs(solar_refl_hot - solar_refl_cold)
    d_abs = abs(solar_abs_hot - solar_abs_cold)
    d_ir = abs(ir_emiss_hot - ir_emiss_cold)

    contrast = (d_refl + d_abs + d_ir) / 3.0
    return float(np.clip(contrast * 100.0, 0.0, 100.0))


def compute_optical_scores(
    formula: str,
    particle_radius_nm: float = 500.0,
    mat=None,
) -> dict:
    """Cooling- und Heating-Efficiency-Score sowie Total-Score berechnen.

    Wenn ``mat`` (ein :class:`~mat_sim.storage.StoredMaterial`) übergeben wird,
    werden die Drude-Lorentz-Parameter aus den MD-Simulationsdaten abgeleitet
    (strukturbasierter Pfad).  Andernfalls wird das Legacy-Modell verwendet.

    Scoring-Struktur
    ----------------
    - **Cooling-Score** (heiss): Solar-Reflexion (50%) + IR-Emissivität (50%)
    - **Heating-Score** (kalt):  Solar-Absorption (50%) + IR-Rückhaltung (50%)
    - **Kontrast-Bonus**:        |Δ optische Eigenschaften| zwischen kalt/heiss
    - **Total-Score**:           70% arithmetisches Mittel + 30% Kontrast-Bonus

    Parameters
    ----------
    formula
        Chemische Formel (Legacy-Pfad).
    particle_radius_nm
        Partikelradius für Mie-Streuung.  Wenn 0, wird aus Struktur abgeleitet.
    mat
        Optional: StoredMaterial mit Simulationsdaten.  Wenn angegeben,
        werden die dielektrischen Modelle aus RDF, Q4 und Volumen abgeleitet.

    Returns
    -------
    dict
        Alle Scores, Teilmetriken und kombinierte Spektren.
    """
    # ── Modelle bestimmen ──
    model_cold = None
    model_hot = None

    if mat is not None and mat.t_switch is not None:
        # Strukturbasierter Pfad: Drude-Lorentz aus Simulationsdaten
        params_cold = _extract_state_params(mat, "kalt")
        params_hot = _extract_state_params(mat, "heiss")

        model_cold = derive_dielectric_model(
            rdf=params_cold["rdf"],
            q4=params_cold["q4"],
            volume=params_cold["volume"],
            n_atoms=params_cold["n_atoms"],
        )
        model_hot = derive_dielectric_model(
            rdf=params_hot["rdf"],
            q4=params_hot["q4"],
            volume=params_hot["volume"],
            n_atoms=params_hot["n_atoms"],
        )

        # Partikelradius aus Struktur ableiten, wenn nicht explizit gesetzt
        if particle_radius_nm == 500.0:
            particle_radius_nm = _estimate_particle_radius_nm(params_cold["rdf"])

        logger.info(
            "Strukturbasierte Optik für %s: "
            "kalt(ε_inf=%.1f, ω_p=%.1f, Q4=%.2f) | "
            "heiss(ε_inf=%.1f, ω_p=%.1f, Q4=%.2f) | r=%.0f nm",
            mat.material_id,
            model_cold.eps_inf, model_cold.drude.omega_p, params_cold["q4"],
            model_hot.eps_inf, model_hot.drude.omega_p, params_hot["q4"],
            particle_radius_nm,
        )

    # Spektren auf dem nahtlosen Gesamt-Grid berechnen
    spec_hot = simulate_mie_spectrum(
        formula, state="heiss", particle_radius_nm=particle_radius_nm,
        wavelengths_nm=FULL_WAVELENGTHS_NM, model=model_hot,
    )
    spec_cold = simulate_mie_spectrum(
        formula, state="kalt", particle_radius_nm=particle_radius_nm,
        wavelengths_nm=FULL_WAVELENGTHS_NM, model=model_cold,
    )

    wls = spec_hot.wavelengths_nm

    # ── Cooling (heiss) ── Albedo-basierte Normierung (Q/Q_ext)
    solar_refl_hot = _solar_reflectance(spec_hot.q_sca, spec_hot.q_ext, wls)
    ir_emiss_hot = _ir_emissivity(spec_hot.q_abs, spec_hot.q_ext, wls)
    cooling_score = (solar_refl_hot * 0.5 + ir_emiss_hot * 0.5) * 100.0

    # ── Heating (kalt) ──
    solar_abs_cold = _solar_absorption(spec_cold.q_abs, spec_cold.q_ext, wls)
    ir_emiss_cold = _ir_emissivity(spec_cold.q_abs, spec_cold.q_ext, wls)
    # Niedrige IR-Emissivität ist gut → (1 - ir_emiss_cold)
    heating_score = (solar_abs_cold * 0.5 + (1.0 - ir_emiss_cold) * 0.5) * 100.0

    # ── Kontrast-Bonus ── belohnt Materialien mit starkem optischem Switching
    # Für den Kontrast brauchen wir auch die kalten/heissen Gegenstücke
    solar_refl_cold = _solar_reflectance(spec_cold.q_sca, spec_cold.q_ext, wls)
    solar_abs_hot = _solar_absorption(spec_hot.q_abs, spec_hot.q_ext, wls)
    contrast_score = _switching_contrast(
        solar_refl_cold, solar_refl_hot,
        solar_abs_cold, solar_abs_hot,
        ir_emiss_cold, ir_emiss_hot,
    )

    # ── Total ── 70% Basis + 30% Kontrast-Bonus
    base_score = (cooling_score + heating_score) / 2.0
    total_score = base_score * 0.7 + contrast_score * 0.3

    return {
        "cooling_score": round(cooling_score, 1),
        "heating_score": round(heating_score, 1),
        "total_score": round(total_score, 1),
        "contrast_score": round(contrast_score, 1),
        "solar_reflectance": round(solar_refl_hot, 3),
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


def optical_summary(formula: str, particle_radius_nm: float = 500.0, mat=None) -> dict:
    """Vollständige optische Auswertung für ein Material.

    Liefert Cooling-, Heating- und Total-Score sowie einen Text-Report.
    Wenn ``mat`` übergeben wird, werden strukturbasierte Modelle verwendet.
    """
    result = compute_optical_scores(formula, particle_radius_nm=particle_radius_nm, mat=mat)

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
        "  ── Switching-Kontrast ──",
        f"  Kontrast-Score:           {result['contrast_score']:.1f} / 100",
        "",
        "  ── Smart-Textile Gesamt ──",
        f"  Total Score:              {result['total_score']:.1f} / 100",
        f"  Bewertung:                {_star_rating(result['total_score'])}",
        "=" * 64,
    ]

    result["report"] = "\n".join(lines)
    return result

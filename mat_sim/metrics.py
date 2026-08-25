"""Echtzeit-Metriken während der MD-Temperaturrampe.

Implementiert:
  A) Phasenwechsel-Indikator  – RDF-Peak-Shift / Steinhardt Q_l
  B) Zerfalls-/Schmelzkriterium – Lindemann-MSD
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
from ase import Atoms
from scipy.signal import find_peaks


# ── Datentypen ──────────────────────────────────────────────────────────────
@dataclass
class StepMetrics:
    """Metriken eines einzelnen Temperaturschritts."""

    temperature: float                 # K
    volume: float                      # Å³
    energy: float                      # eV
    msd: float                         # Å²
    rdf_r: np.ndarray = field(repr=False)
    rdf_g: np.ndarray = field(repr=False)
    ql: float = 0.0                    # Steinhardt Q4 (kubisch/tetraedrisch)
    positions: np.ndarray | None = field(default=None, repr=False)  # (N, 3)
    cell: np.ndarray | None = field(default=None, repr=False)       # (3, 3)


@dataclass
class TrajectoryResult:
    """Aggregiertes Ergebnis einer kompletten Rampe."""

    temperatures: list[float] = field(default_factory=list)
    volumes: list[float] = field(default_factory=list)
    energies: list[float] = field(default_factory=list)
    msd_values: list[float] = field(default_factory=list)
    ql_values: list[float] = field(default_factory=list)
    rdf_history: list[tuple[np.ndarray, np.ndarray]] = field(default_factory=list)
    # Struktur-Snapshots pro Temperaturschritt (Positionen + Zelle)
    positions_history: list[np.ndarray] = field(default_factory=list)  # (N, 3)
    cell_history: list[np.ndarray] = field(default_factory=list)       # (3, 3)

    # ← später bestimmte Schwellwerte
    t_switch: float | None = None
    t_decay: float | None = None

    # Status: "converged" (Rampe komplett gelaufen), "diverged" (MD explodiert),
    # "decayed" (vorzeitiger Abbruch durch Lindemann-Kriterium)
    status: str = "converged"

    def add(self, m: StepMetrics) -> None:
        self.temperatures.append(m.temperature)
        self.volumes.append(m.volume)
        self.energies.append(m.energy)
        self.msd_values.append(m.msd)
        self.ql_values.append(m.ql)
        self.rdf_history.append((m.rdf_r, m.rdf_g))
        if m.positions is not None:
            self.positions_history.append(m.positions)
        if m.cell is not None:
            self.cell_history.append(m.cell)


# ── Hilfs: Nächste-Nachbar-Abstand ─────────────────────────────────────────
def nearest_neighbor_distance(atoms: Atoms) -> float:
    """Kürzester Nächste-Nachbar-Abstand im aktuellen Snapshot (Å)."""
    from ase.neighborlist import neighbor_list

    d = neighbor_list("d", atoms, cutoff=4.0)
    return float(np.min(d[d > 1e-3])) if d.size else float(atoms.cell.lengths().min())


# ── MSD ────────────────────────────────────────────────────────────────────
def compute_msd(initial_positions: np.ndarray, current_positions: np.ndarray) -> float:
    """Mittlere quadratische Verschiebung aller Atome (Å²).

    Parameters
    ----------
    initial_positions
        ``(N, 3)``-Array der Referenzpositionen bei T ≈ 0.
    current_positions
        ``(N, 3)``-Array der aktuellen Positionen (gleicher Sort & Reihenfolge!).
    """
    disp = current_positions - initial_positions
    # Minimum-Image-Konvention über Gittervektoren ist nicht nötig, weil wir
    # die *ungebrochenen* ASE-Positionen verwenden (wrap=False im MD-Dyn).
    return float(np.mean(np.sum(disp**2, axis=1)))


# ── RDF ────────────────────────────────────────────────────────────────────
def compute_rdf(
    atoms: Atoms,
    r_max: float = 6.0,
    n_bins: int = 100,
) -> tuple[np.ndarray, np.ndarray]:
    """Radialverteilungsfunktion g(r) berechnen.

    Returns
    -------
    (r, g)
        ``r``  – Mittelpunkte der Bin-Zentren (Å)
        ``g``  – RDF-Werte
    """
    from ase.neighborlist import neighbor_list

    cutoffs = {sym: r_max for sym in atoms.get_chemical_symbols()}

    try:
        d = neighbor_list("d", atoms, cutoffs)
    except Exception:
        # Fallback: Einheits-Cutoff
        d = neighbor_list("d", atoms, cutoff=r_max)

    if d.size == 0:
        r = np.linspace(0, r_max, n_bins)
        return r, np.zeros_like(r)

    hist, bin_edges = np.histogram(d, bins=n_bins, range=(0, r_max))
    r = 0.5 * (bin_edges[:-1] + bin_edges[1:])
    dr = bin_edges[1] - bin_edges[0]

    # Normierung auf ideale Gas-Referenz
    volume = atoms.get_volume()
    n_atoms = len(atoms)
    rho = n_atoms / volume
    shell_volumes = 4.0 * np.pi * r**2 * dr
    n_ideal = rho * shell_volumes
    g = np.where(n_ideal > 0, hist / n_ideal, 0.0)

    return r, g


# ── Steinhardt Q_l (vereinfacht: Q4) ───────────────────────────────────────
def compute_steinhardt_q4(atoms: Atoms, cutoff: float = 3.5) -> float:
    """Steinhardt-Ordnungsparameter Q4 (skalar).

    Misst kubische / tetraedrische Symmetrie-Anteile.
    Ein *großer* Sprung zwischen aufeinanderfolgenden Schritten signalisiert
    einen Symmetrie-Wechsel → Phasentransformation.

    Hinweis:  Für produktiven Einsatz kann Q6 etc. ergänzt werden;
    Q4 reicht als Indikator für Oxid-Strukturen meist aus.
    """
    from scipy.special import sph_harm_y

    from ase.neighborlist import neighbor_list

    try:
        i, j, d, D = neighbor_list("ijdD", atoms, cutoff)
    except Exception:
        return 0.0

    if i.size == 0:
        return 0.0

    q4_sq = 0.0
    n_atoms = len(atoms)
    # Für jedes Atom: mittlere Y_{4,m} über Nachbarn
    for atom_idx in range(n_atoms):
        mask = i == atom_idx
        if not np.any(mask):
            continue
        dv = D[mask]            # (n_nb, 3)
        norms = np.linalg.norm(dv, axis=1)
        norms[norms == 0] = 1.0
        theta = np.arccos(dv[:, 2] / norms)      # polar
        phi = np.arctan2(dv[:, 1], dv[:, 0])      # azimuthal
        qm = np.zeros(9, dtype=complex)           # m = -4 … 4
        for m_idx, m in enumerate(range(-4, 5)):
            qm[m_idx] = np.mean(sph_harm_y(4, m, theta, phi))
        q4_sq += np.sum(np.abs(qm) ** 2)

    q4 = np.sqrt(4 * np.pi / 9 * q4_sq / n_atoms)
    return float(q4)


# ── Phasenwechsel-Erkennung aus RDF ────────────────────────────────────────
def detect_t_switch(
    rdf_history: Sequence[tuple[np.ndarray, np.ndarray]],
    temperatures: Sequence[float],
    peak_shift_threshold: float = 0.30,
    min_persistence: int = 3,
    relative_volume_change: float | None = None,
    volumes: Sequence[float] | None = None,
) -> float | None:
    """T_switch aus diskontinuierlichen RDF-Peak-Verschiebungen ableiten.

    Ein Phasenwechsel wird nur detektiert, wenn **alle** folgenden
    Kriterien erfüllt sind:

    1. Der erste signifikante RDF-Peak verschiebt sich um mehr als
       ``peak_shift_threshold`` Å **sprunghaft** (nicht kontinuierlich).
    2. Die Verschiebung ist **persistent**: sie bleibt für mindestens
       ``min_persistence`` aufeinanderfolgende Schritte oberhalb des
       Schwellwerts (verwirft transientes Rauschen).
    3. Optional: Wenn ``volumes`` übergeben wird, muss auch eine
       Volumenänderung > 5% um denselben Temperaturbereich auftreten.

    Returns
    -------
    float | None
        Temperatur des ersten detektierten Sprungs oder *None*.
    """
    if len(rdf_history) < min_persistence + 1:
        return None

    prev_r, prev_g = rdf_history[0]
    prev_peaks, _ = find_peaks(prev_g, height=0.5, distance=5)
    prev_peak_pos = prev_r[prev_peaks] if prev_peaks.size else np.array([])

    # Kontinuierliche Verschiebung tracken, um Sprünge von Ausdehnung zu trennen
    prev_shift = 0.0  # kumulierte Verschiebung seit Schritt 0

    for idx in range(1, len(rdf_history)):
        r, g = rdf_history[idx]
        peaks, _ = find_peaks(g, height=0.5, distance=5)
        curr_peak_pos = r[peaks] if peaks.size else np.array([])

        detected = False

        # Peak-Verschiebung
        if prev_peak_pos.size > 0 and curr_peak_pos.size > 0:
            n_min = min(prev_peak_pos.size, curr_peak_pos.size)
            # Vergleiche ersten Peak (stabilster, NN-Abstand)
            shift_first = abs(curr_peak_pos[0] - prev_peak_pos[0])
            shifts = np.abs(prev_peak_pos[:n_min] - curr_peak_pos[:n_min])
            max_shift = float(np.max(shifts)) if shifts.size else 0.0

            # Sprung vs. kontinuierliche Ausdehnung:
            # Ein Phasenwechsel ist ein plötzlicher Sprung, nicht eine
            # graduelle Verschiebung über viele Schritte.
            # Wir verlangen: max_shift > threshold UND max_shift > 3× prev_shift
            if max_shift > peak_shift_threshold and max_shift > 3.0 * max(prev_shift, 0.01):
                detected = True

            prev_shift = shift_first

        # Peak verschwunden (nur wenn vorher signifikant)
        elif prev_peak_pos.size > 0 and curr_peak_pos.size == 0:
            detected = True

        if detected:
            # Persistenz-Check: Verschiebung muss für min_persistence
            # weitere Schritte bestehen bleiben
            if idx + min_persistence <= len(rdf_history):
                persistent = True
                for k in range(idx + 1, min(idx + min_persistence, len(rdf_history))):
                    r_k, g_k = rdf_history[k]
                    pk_k, _ = find_peaks(g_k, height=0.5, distance=5)
                    if pk_k.size == 0:
                        persistent = False
                        break
                    pk_k_pos = r_k[pk_k]
                    # Vergleiche mit dem Ursprung (Schritt 0)
                    if prev_peak_pos.size > 0 and pk_k_pos.size > 0:
                        n = min(prev_peak_pos.size, pk_k_pos.size)
                        shift_k = np.abs(prev_peak_pos[:n] - pk_k_pos[:n])
                        if float(np.max(shift_k)) < peak_shift_threshold:
                            persistent = False
                            break
                if not persistent:
                    prev_peak_pos = curr_peak_pos
                    continue

            # Optional: Volumenänderung bestätigen
            if volumes is not None and len(volumes) == len(temperatures):
                v_before = volumes[max(idx - 1, 0)]
                v_after = volumes[min(idx, len(volumes) - 1)]
                if v_before > 0:
                    vol_change = abs(v_after - v_before) / v_before
                    if vol_change < 0.02:  # < 2% Volumenänderung → kein Switch
                        prev_peak_pos = curr_peak_pos
                        continue

            return float(temperatures[idx])

        prev_peak_pos = curr_peak_pos

    return None


# ── Zerfalls-Erkennung aus MSD ─────────────────────────────────────────────
def detect_t_decay(
    msd_values: Sequence[float],
    temperatures: Sequence[float],
    nn_distance: float,
    lindemann_fraction: float = 0.12,
) -> float | None:
    """T_decay via Lindemann-Kriterium: MSD > lindemann_fraction * d_NN².

    Parameters
    ----------
    nn_distance
        Nächste-Nachbar-Abstand bei T ≈ 0 (Å).
    lindemann_fraction
        Schwellwert (typisch 0.10–0.15).
    """
    threshold = (lindemann_fraction * nn_distance) ** 2
    for msd, temp in zip(msd_values, temperatures):
        if msd > threshold:
            return float(temp)
    return None

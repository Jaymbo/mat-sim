"""Visualisierungs- und Analyse-Tool für gespeicherte MD-Ergebnisse.

Erzeugt ein 2×3-Dashboard (Matplotlib) für ein einzelnes Material:
  1. Temperatur-Rampe (T vs. MD-Schritt)
  2. MSD vs. Temperatur mit T_decay-Markierung
  3. Optisches Spektrum: Q_sca kalt vs. heiss mit atmosphärischem Fenster
  4. Volumen & Steinhardt Q4 vs. Temperatur
  5. RDF-Vergleich: vor T_switch, nach T_switch
  6. Optisches Spektrum: Q_abs kalt vs. heiss mit atmosphärischem Fenster

Zusätzlich werden im Terminal ausgegeben:
  - Optische Vor-Evaluierung (Volumen-/Atomabstandsänderung am T_switch)
  - Mie-Streuungs-Analyse mit Cooling-Efficiency-Score

CLI:
    python -m mat_sim.run --analyze --material-id mp-XXXX
    python -m mat_sim.run --analyze --material-id mp-XXXX --db results.db --save plot.png
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Sequence

import numpy as np

from .storage import StoredMaterial, load_result, list_material_ids

logger = logging.getLogger(__name__)

# ── Matplotlib (non-interactive Default) ────────────────────────────────────
import matplotlib
matplotlib.use("Agg")  # sicher für Headless-Server; wird bei --show überschrieben
import matplotlib.pyplot as plt


# ── Plot-Hilfsfunktionen ────────────────────────────────────────────────────

def _plot_temperature_ramp(ax, mat: StoredMaterial) -> None:
    """Plot 1: Temperatur vs. kumulierten MD-Schritten."""
    temps = np.asarray(mat.temperatures)
    # Jeder Temperaturschritt entspricht thermalization_steps MD-Schritten.
    # Da dieser Wert nicht in der DB steht, approximieren wir die x-Achse
    # als fortlaufenden Index und labeln sie als "Schritt-Index".
    steps = np.arange(len(temps))
    ax.plot(steps, temps, "b-o", markersize=3, linewidth=1.2)
    ax.set_xlabel("Temperatur-Schritt-Index")
    ax.set_ylabel("Temperatur (K)")
    ax.set_title("Temperatur-Rampe")
    ax.grid(True, alpha=0.3)


def _plot_msd(ax, mat: StoredMaterial) -> None:
    """Plot 2: MSD vs. Temperatur, mit T_decay als roter gestrichelter Linie."""
    temps = np.asarray(mat.temperatures)
    msd = np.asarray(mat.msd_values)
    ax.plot(temps, msd, "k-", linewidth=1.5, label="MSD")
    ax.set_xlabel("Temperatur (K)")
    ax.set_ylabel("MSD (Å²)")
    ax.set_title("Mittlere quadratische Verschiebung")
    ax.grid(True, alpha=0.3)

    if mat.t_decay is not None:
        ax.axvline(
            mat.t_decay, color="red", linestyle="--", linewidth=1.5,
            label=f"T_decay = {mat.t_decay:.0f} K",
        )
        ax.legend()


def _plot_volume_and_q4(ax, mat: StoredMaterial) -> None:
    """Plot 3: Volumen (links) und Steinhardt Q4 (rechts) vs. Temperatur."""
    temps = np.asarray(mat.temperatures)
    volumes = np.asarray(mat.volumes)
    ql = np.asarray(mat.ql_values)

    color_vol = "tab:blue"
    ax.plot(temps, volumes, color=color_vol, marker="s", markersize=3,
            linewidth=1.2, label="Volumen")
    ax.set_xlabel("Temperatur (K)")
    ax.set_ylabel("Volumen (Å³)", color=color_vol)
    ax.tick_params(axis="y", labelcolor=color_vol)
    ax.grid(True, alpha=0.3)

    # Zweite y-Achse für Q4
    ax2 = ax.twinx()
    color_q4 = "tab:orange"
    ax2.plot(temps, ql, color=color_q4, marker="^", markersize=3,
             linewidth=1.2, label="Q4")
    ax2.set_ylabel("Steinhardt Q4", color=color_q4)
    ax2.tick_params(axis="y", labelcolor=color_q4)

    # T_switch markieren, falls vorhanden
    if mat.t_switch is not None:
        ax.axvline(
            mat.t_switch, color="green", linestyle=":", linewidth=1.5,
            label=f"T_switch = {mat.t_switch:.0f} K",
        )
        ax.legend(loc="upper left")

    ax.set_title("Volumen & Symmetrie (Q4)")


def _plot_rdf_comparison(ax, mat: StoredMaterial) -> None:
    """Plot 4: RDF-Kurven bei T=0 K, vor und nach T_switch."""
    # T=0-Kurve aus dem ersten RDF-Snapshot der Temperatur-Serie
    # (der erste gespeicherte RDF-Punkt entspricht T ≈ 0).
    # rdf_before / rdf_after sind die vom Pipeline gespeicherten Snapshots.
    plotted = False

    if mat.rdf_before is not None:
        r_b, g_b = mat.rdf_before
        ax.plot(r_b, g_b, "-", linewidth=1.5, alpha=0.8,
                label=f"vor T_switch ({_temp_label(mat, 'before')})")
        plotted = True

    if mat.rdf_after is not None:
        r_a, g_a = mat.rdf_after
        ax.plot(r_a, g_a, "-", linewidth=1.5, alpha=0.8,
                label=f"nach T_switch ({_temp_label(mat, 'after')})")
        plotted = True

    # Versuche, die T=0-Kurve aus der earliest-temperature RDF zu rekonstruieren.
    # Da nur rdf_before/after in der DB stehen, dient rdf_before bei
    # fehlendem T_switch als "kalt"-Referenz.
    if not plotted and mat.rdf_before is None:
        ax.text(0.5, 0.5, "Keine RDF-Snapshots verfügbar",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=11, color="gray")
    else:
        ax.set_xlabel("Abstand r (Å)")
        ax.set_ylabel("g(r)")
        ax.set_title("RDF-Vergleich am Phasenwechsel")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_xlim(left=0)


def _temp_label(mat: StoredMaterial, which: str) -> str:
    """Kurzes Temperatur-Label für RDF-Legende generieren."""
    if mat.t_switch is None:
        return "N/A"
    # 'before' = ein Schritt vor T_switch, 'after' = ein Schritt danach
    temps = np.asarray(mat.temperatures)
    idx = int(np.argmin(np.abs(temps - mat.t_switch)))
    if which == "before":
        idx = max(idx - 1, 0)
    else:
        idx = min(idx + 1, len(temps) - 1)
    return f"{temps[idx]:.0f} K"


# ── Optisches Spektrum-Plot ─────────────────────────────────────────────────

def _plot_optical_spectrum(ax, formula: str, quantity: str = "sca") -> None:
    """Optisches Mie-Spektrum: kalt vs. heiss mit atmosphärischem Fenster.

    Parameters
    ----------
    ax
        Matplotlib-Achse.
    formula
        Chemische Formel des Materials.
    quantity
        ``"sca"`` für Q_sca (Streuung/Reflexion) oder
        ``"abs"`` für Q_abs (Absorption/Emissivität).
    """
    from .optics import optical_summary

    try:
        result = optical_summary(formula)
    except Exception as exc:  # noqa: BLE001
        ax.text(0.5, 0.5, f"Optik-Berechnung fehlgeschlagen:\n{exc}",
                transform=ax.transAxes, ha="center", va="center",
                fontsize=9, color="red")
        return

    spec_hot = result["spectrum_heiss"]
    spec_cold = result["spectrum_kalt"]

    wls_um_hot = spec_hot.wavelengths_nm / 1000.0
    wls_um_cold = spec_cold.wavelengths_nm / 1000.0

    if quantity == "sca":
        y_hot = spec_hot.q_sca
        y_cold = spec_cold.q_sca
        ylabel = r"$Q_{\mathrm{sca}}$"
        title = "Mie-Streuung (Solar-Reflexion)"
    else:
        y_hot = spec_hot.q_abs
        y_cold = spec_cold.q_abs
        ylabel = r"$Q_{\mathrm{abs}}$"
        title = "Mie-Absorption (IR-Emissivität)"

    # Nahtloses Grid: beide Linien durchgehend (keine Lücke)
    ax.plot(wls_um_cold, y_cold, "b-", linewidth=1.5, alpha=0.8, label="kalt")
    ax.plot(wls_um_hot, y_hot, "r-", linewidth=1.5, alpha=0.8, label="heiss")

    # Atmosphärisches Fenster (8–13 µm) schraffieren
    ax.axvspan(8.0, 13.0, alpha=0.15, color="green", label="8–13 µm Fenster")

    ax.set_xlabel("Wellenlänge (µm)")
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(left=0)

    # Alle drei Scores als Text-Annotation
    cooling = result["cooling_score"]
    heating = result["heating_score"]
    total = result["total_score"]
    score_text = (
        f"Cooling: {cooling:.0f}%\n"
        f"Heating: {heating:.0f}%\n"
        f"Total:   {total:.0f}%"
    )
    ax.text(0.98, 0.95, score_text,
            transform=ax.transAxes, ha="right", va="top",
            fontsize=8, fontweight="bold", family="monospace",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="wheat", alpha=0.9))


# ── Dashboard ───────────────────────────────────────────────────────────────

def generate_dashboard(
    mat: StoredMaterial,
    save_path: str | Path | None = None,
    show: bool = False,
) -> Path | None:
    """2×3-Dashboard für ein Material erzeugen und optional speichern/anzeigen.

    Parameters
    ----------
    mat
        Geladener Datensatz aus der DB.
    save_path
        Dateipfad für den PNG-Export.  Wenn *None*, wird ein Default-Name
        ``dashboard_<material_id>.png`` im aktuellen Verzeichnis verwendet.
    show
        Wenn *True*, wird ein interaktives Fenster geöffnet (``plt.show()``).

    Returns
    -------
    Path | None
        Pfad der gespeicherten Datei oder *None* bei ``show=True`` ohne Save.
    """
    if show:
        matplotlib.use("TkAgg")  # interaktives Backend

    fig, axes = plt.subplots(2, 3, figsize=(20, 10))
    fig.suptitle(
        f"{mat.material_id} – {mat.formula}\n"
        f"T_switch = {mat.t_switch} K | T_decay = {mat.t_decay} K",
        fontsize=13, fontweight="bold",
    )

    _plot_temperature_ramp(axes[0, 0], mat)
    _plot_msd(axes[0, 1], mat)
    _plot_optical_spectrum(axes[0, 2], mat.formula, quantity="sca")
    _plot_volume_and_q4(axes[1, 0], mat)
    _plot_rdf_comparison(axes[1, 1], mat)
    _plot_optical_spectrum(axes[1, 2], mat.formula, quantity="abs")

    fig.tight_layout(rect=(0, 0, 1, 0.94))

    out_path: Path | None = None
    if save_path is not None or not show:
        out_path = Path(save_path) if save_path else Path(f"dashboard_{mat.material_id}.png")
        fig.savefig(out_path, dpi=150, bbox_inches="tight")
        logger.info("Dashboard gespeichert: %s", out_path)

    if show:
        plt.show()

    plt.close(fig)
    return out_path


# ── Optische Vor-Evaluierung ────────────────────────────────────────────────

def optical_pre_evaluation(mat: StoredMaterial) -> str:
    """Terminal-Zusammenfassung der Volumen- und Atomabstandsänderung am T_switch.

    Gibt einen Hinweis darauf, ob sich die optischen Eigenschaften
    (Brechungsindex, Dielektrizitätsfunktion) beim Phasenwechsel drastisch
    verschoben haben könnten.

    Returns
    -------
    str
        Mehrzeiliger Bewertungs-Text.
    """
    lines: list[str] = []
    lines.append("=" * 64)
    lines.append(f"  Optische Vor-Evaluierung: {mat.material_id} ({mat.formula})")
    lines.append("=" * 64)

    if mat.t_switch is None:
        lines.append("  ⚠ Kein T_switch erkannt – kein Phasenwechsel detektiert.")
        lines.append("    Optische Eigenschaften vermutlich temperaturstabil.")
        lines.append("=" * 64)
        return "\n".join(lines)

    lines.append(f"  Phasenwechsel-Temperatur: T_switch = {mat.t_switch:.1f} K")
    lines.append("")

    # ── Volumenänderung ────────────────────────────────────────────────
    temps = np.asarray(mat.temperatures)
    volumes = np.asarray(mat.volumes)
    idx_sw = int(np.argmin(np.abs(temps - mat.t_switch)))

    v_before = volumes[max(idx_sw - 1, 0)]
    v_after = volumes[min(idx_sw + 1, len(volumes) - 1)]
    delta_v = v_after - v_before
    rel_v = (delta_v / v_before * 100.0) if v_before != 0 else 0.0

    lines.append(f"  Volumen vor T_switch : {v_before:8.2f} Å³")
    lines.append(f"  Volumen nach T_switch: {v_after:8.2f} Å³")
    lines.append(f"  ΔV = {delta_v:+.2f} Å³  ({rel_v:+.2f} %)")
    lines.append("")

    # ── Atomabstandsänderung aus RDF ───────────────────────────────────
    if mat.rdf_before is not None and mat.rdf_after is not None:
        r_b, g_b = mat.rdf_before
        r_a, g_a = mat.rdf_after
        from scipy.signal import find_peaks

        peaks_b, _ = find_peaks(g_b, height=0.5, distance=5)
        peaks_a, _ = find_peaks(g_a, height=0.5, distance=5)

        if peaks_b.size > 0 and peaks_a.size > 0:
            first_b = r_b[peaks_b[0]]
            first_a = r_a[peaks_a[0]]
            delta_r = first_a - first_b
            rel_r = (delta_r / first_b * 100.0) if first_b != 0 else 0.0

            lines.append(f"  1. RDF-Peak vor T_switch : {first_b:.3f} Å")
            lines.append(f"  1. RDF-Peak nach T_switch: {first_a:.3f} Å")
            lines.append(f"  Δr = {delta_r:+.3f} Å  ({rel_r:+.2f} %)")
        else:
            lines.append("  ⚠ RDF-Peak-Erkennung fehlgeschlagen – keine Aussage möglich.")
    else:
        lines.append("  ℹ Keine RDF-Snapshots um T_switch gespeichert.")
    lines.append("")

    # ── Bewertung ──────────────────────────────────────────────────────
    if abs(rel_v) > 5.0 or (mat.rdf_before is not None and mat.rdf_after is not None and
                            "delta_r" in dir() and abs(rel_r) > 5.0):
        rating = "★★★ Stark"
    elif abs(rel_v) > 2.0 or (mat.rdf_before is not None and mat.rdf_after is not None and
                              "delta_r" in dir() and abs(rel_r) > 2.0):
        rating = "★★  Moderat"
    else:
        rating = "★   Gering"

    lines.append(f"  Bewertung der optischen Verschiebung: {rating}")
    lines.append("  → Große ΔV/Δr deuten auf drastische Änderung des Brechungs-")
    lines.append("    index und der dielektrischen Funktion am T_switch hin.")
    lines.append("    Kandidat für detaillierte Mie-Streuungs-Analyse!")
    lines.append("=" * 64)

    return "\n".join(lines)


# ── Ranking ─────────────────────────────────────────────────────────────────

def rank_materials(
    db_path: str | Path,
    top: int = 20,
    only_phase_change: bool = True,
    output_dir: str | Path | None = None,
) -> None:
    """Alle Materialien nach optischem Score sortieren und Top-Kandidaten ausgeben.

    Parameters
    ----------
    db_path
        Pfad zur SQLite-Datenbank.
    top
        Anzahl der Top-Kandidaten, für die Dashboards erzeugt werden.
    only_phase_change
        Wenn *True*, werden nur Materialien mit detektiertem T_switch
        berücksichtigt (Phasenwechsel ist Voraussetzung für Switchable Cooling).
    output_dir
        Verzeichnis für die Dashboard-PNGs der Top-Kandidaten.
    """
    from .optics import compute_optical_scores

    ids = list_material_ids(db_path)
    if not ids:
        logger.warning("Keine Materialien in der Datenbank gefunden.")
        return

    # ── 1. Alle Materialien bewerten ───────────────────────────────────
    # Cache: Formel → Scores (viele Strukturen teilen sich dieselbe Formel)
    score_cache: dict[str, dict] = {}
    rows: list[dict] = []

    for mid in ids:
        mat = load_result(db_path, mid)

        # Filter: Phasenwechsel vorhanden?
        if only_phase_change and mat.t_switch is None:
            continue

        # Status-Filter: nur konvergierte oder dezidierte Materialien
        if mat.status == "diverged":
            continue

        # Optische Scores berechnen (mit Formel-Cache)
        if mat.formula not in score_cache:
            try:
                score_cache[mat.formula] = compute_optical_scores(mat.formula)
            except Exception:
                score_cache[mat.formula] = None

        scores = score_cache[mat.formula]
        if scores is None:
            continue

        rows.append({
            "material_id": mat.material_id,
            "formula": mat.formula,
            "status": mat.status,
            "t_switch": mat.t_switch,
            "t_decay": mat.t_decay,
            "cooling_score": scores["cooling_score"],
            "heating_score": scores["heating_score"],
            "total_score": scores["total_score"],
            "mat": mat,
        })

    if not rows:
        print("Keine Materialien mit Phasenwechsel gefunden.")
        return

    # ── 2. Nach Total-Score sortieren ──────────────────────────────────
    rows.sort(key=lambda r: r["total_score"], reverse=True)

    # ── 3. Tabelle ausgeben ────────────────────────────────────────────
    print("=" * 100)
    print(f"  Ranking — Top {min(top, len(rows))} von {len(rows)} Materialien (sortiert nach Total-Score)")
    if only_phase_change:
        print("  Filter: nur Materialien mit detektiertem Phasenwechsel (T_switch)")
    print("=" * 100)
    print(f"  {'#':>3}  {'MP-ID':<14} {'Formel':<16} {'T_switch':>9} {'T_decay':>9} {'Cool%':>6} {'Heat%':>6} {'Total':>6}")
    print("-" * 100)

    for i, r in enumerate(rows[:top], start=1):
        ts = f"{r['t_switch']:.0f} K" if r['t_switch'] is not None else "—"
        td = f"{r['t_decay']:.0f} K" if r['t_decay'] is not None else "—"
        print(
            f"  {i:>3}  {r['material_id']:<14} {r['formula']:<16} "
            f"{ts:>9} {td:>9} "
            f"{r['cooling_score']:>6.1f} {r['heating_score']:>6.1f} {r['total_score']:>6.1f}"
        )

    print("=" * 100)

    # ── 4. Dashboards für Top-Kandidaten erzeugen ──────────────────────
    if output_dir is not None:
        output_dir = Path(output_dir)
    else:
        output_dir = Path("dashboards_top")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n  Erzeuge Dashboards für Top {top} → {output_dir}/")
    for r in rows[:top]:
        save_path = output_dir / f"dashboard_{r['material_id']}.png"
        generate_dashboard(r["mat"], save_path=save_path, show=False)
        print(f"    {r['material_id']} ({r['formula']}) → {save_path}")

    print(f"\n  Fertig. {len(rows)} Materialien bewertet, Top {top} als Dashboard gespeichert.")


# ── Komfort-API ─────────────────────────────────────────────────────────────

def analyze_material(
    db_path: str | Path,
    material_id: str,
    save_path: str | Path | None = None,
    show: bool = False,
) -> None:
    """Material aus DB laden, Dashboard erzeugen und Auswertungen ausgeben."""
    mat = load_result(db_path, material_id)
    logger.info(
        "Geladen: %s (%s) | T_switch=%s, T_decay=%s",
        mat.material_id, mat.formula, mat.t_switch, mat.t_decay,
    )

    generate_dashboard(mat, save_path=save_path, show=show)

    print(optical_pre_evaluation(mat))

    # Mie-Streuungs-Analyse mit Cooling-Score
    from .optics import optical_summary
    try:
        mie_result = optical_summary(mat.formula)
        print(mie_result["report"])
    except Exception as exc:  # noqa: BLE001
        logger.warning("Mie-Analyse fehlgeschlagen für %s: %s", mat.formula, exc)


def analyze_all(
    db_path: str | Path,
    output_dir: str | Path | None = None,
) -> None:
    """Dashboards für alle in der DB gespeicherten Materialien erzeugen."""
    ids = list_material_ids(db_path)
    if not ids:
        logger.warning("Keine Materialien in der Datenbank gefunden.")
        return

    output_dir = Path(output_dir) if output_dir else Path(".")
    output_dir.mkdir(parents=True, exist_ok=True)

    for mid in ids:
        mat = load_result(db_path, mid)
        save_path = output_dir / f"dashboard_{mid}.png"
        generate_dashboard(mat, save_path=save_path, show=False)
        print(optical_pre_evaluation(mat))

        from .optics import optical_summary
        try:
            mie_result = optical_summary(mat.formula)
            print(mie_result["report"])
        except Exception as exc:  # noqa: BLE001
            logger.warning("Mie-Analyse fehlgeschlagen für %s: %s", mat.formula, exc)

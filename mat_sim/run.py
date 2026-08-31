#!/usr/bin/env python3
"""Kommandozeile-Einstiegspunkt für die mat-sim Pipeline.

Zwei-Phasen-Workflow
---------------------
1. **Ingest** (einmalig):  Strukturen von MP herunterladen → DB.
   Beispiele:
    python -m mat_sim.run --ingest --chemsys V-O Ti-O Cr-O --db results.db

2. **Process** (wiederholt, parallel):  pending-Strukturen aus DB simulieren.
   Beispiele:
    python -m mat_sim.run --db results.db --device auto --duration-min 25

Allgemeine Beispiele
---------------------
Pipeline starten (Ingest + Process in einem Schritt):
    python -m mat_sim.run --chemsys V-O Ti-O --t-max 600 --delta-t 20

Einzelnes Material analysieren (Dashboard + Vor-Evaluierung):
    python -m mat_sim.run --analyze --material-id mp-1234 --db results.db

Alle Materialien der DB analysieren:
    python -m mat_sim.run --analyze --all --db results.db

Ranking — Top 20 nach optischem Score (mit Dashboards):
    python -m mat_sim.run --analyze --rank --db results.db

Ranking — Top 50, auch ohne Phasenwechsel:
    python -m mat_sim.run --analyze --rank --top 50 --no-phase-change-filter --db results.db

Queue-Statistik anzeigen:
    python -m mat_sim.run --queue-stats --db results.db

Stale processing-Einträge zurücksetzen:
    python -m mat_sim.run --reset-stale --db results.db

Optionen (Ingest)
    --ingest          Nur Ingest: Strukturen herunterladen, nicht simulieren
    --chemsys         Ein oder mehrere chemische Systeme
    --include-ternary Ternäre Erweiterung (Default: true)
    --max-results     Max. Strukturen pro System       (Default: 2000)
    --e-hull-max      Max. energy_above_hull eV/atom   (Default: 0.1)

Optionen (Process)
    --device          cpu | cuda | auto  (Default: cpu)
    --t-max           Maximaltemperatur in K          (Default: 1200)
    --delta-t         Temperaturschritt in K          (Default: 20)
    --therm-steps     Thermalisierungs-Schritte pro T (Default: 1000)
    --supercell-min-atoms  Mindestatomzahl für Supercell (Default: 50)
    --duration-min    Max. Laufzeit in Minuten (SLURM) (Default: 25)
    --stale-minutes   processing-Timeout für Reset    (Default: 30)

Optionen (Re-Simulation)
    --resimulate      Material-IDs (mp-xxxx) erneut simulieren
    --version-label   Label für archivierte alte Ergebnisse (Default: v1)

Optionen (Analyse)
    --analyze        Aktiviert den Analyse-Modus
    --material-id    MP-ID des zu analysierenden Materials
    --all            Alle Materialien der DB analysieren
    --save           Dateipfad für den PNG-Export (Default: dashboard_<id>.png)
    --show           Interaktives Fenster öffnen (statt nur PNG zu speichern)
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from dotenv import load_dotenv

# .env-Datei laden, bevor irgendwelche Module den API-Key aus os.environ lesen.
load_dotenv()

# PyTorch-Thread-Limit NUR bei CPU-Nutzung (verhindert Thread-Overhead).
# Bei GPU-Nutzung stören CPU-Thread-Limits nicht, aber wir sparen sie.
import torch

if not torch.cuda.is_available():
    torch.set_num_threads(4)
    torch.set_num_interop_threads(4)

from .md import RampConfig
from .pipeline import PipelineConfig, ingest_phase, run_pipeline

# ── Argument-Parser ─────────────────────────────────────────────────────────

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="High-Throughput Screening-Pipeline für Switchable Radiative Cooling",
    )

    # ── Gemeinsame Optionen ──
    p.add_argument("--db", default="results.db", help="Pfad zur SQLite-DB")
    p.add_argument("-v", "--verbose", action="store_true")

    # ── Analyse-Modus ──
    p.add_argument("--analyze", action="store_true",
                   help="Analyse-Modus: Dashboard & Vor-Evaluierung für gespeicherte Ergebnisse")
    p.add_argument("--material-id", type=str, default=None,
                   help="MP-ID des zu analysierenden Materials (z. B. mp-1234)")
    p.add_argument("--all", action="store_true",
                   help="Alle Materialien der DB analysieren")
    p.add_argument("--rank", action="store_true",
                   help="Ranking-Modus: alle Materialien nach optischem Score sortieren")
    p.add_argument("--top", type=int, default=20,
                   help="Anzahl Top-Kandidaten für Dashboard-Export (Default: 20)")
    p.add_argument("--no-phase-change-filter", action="store_true",
                   help="Auch Materialien ohne T_switch ins Ranking aufnehmen")
    p.add_argument("--recompute", action="store_true",
                   help="Optische Scores neu berechnen (auch bereits gespeicherte)")
    p.add_argument("--recheck-switch", action="store_true",
                   help="T_switch für alle Materialien mit gespeicherten RDFs neu auswerten (strengere Kriterien)")
    p.add_argument("--save", type=str, default=None,
                   help="Dateipfad für PNG-Export (Default: dashboard_<id>.png)")
    p.add_argument("--show", action="store_true",
                   help="Interaktives Matplotlib-Fenster öffnen")

    # ── Ingest-Modus ──
    p.add_argument("--ingest", action="store_true",
                   help="Nur Ingest: Strukturen von MP herunterladen → DB (keine Simulation)")
    p.add_argument("--chemsys", nargs="+", default=None,
                   help="Chemische Systeme, z. B. V-O Ti-O Sr-Ti-O")
    p.add_argument("--include-ternary", action=argparse.BooleanOptionalAction,
                   default=True,
                   help="Ternäre Erweiterung bei binären Systemen (Default: True)")
    p.add_argument("--max-results", type=int, default=2000)
    p.add_argument("--e-hull-max", type=float, default=0.1,
                   help="Max. energy_above_hull in eV/atom (0.0 = nur stabil, 0.1 = inkl. metastabil)")

    # ── Process-Modus ──
    p.add_argument("--mlip", choices=["mace", "chgnet"], default="mace")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--t-max", type=float, default=1200.0,
                   help="Maximaltemperatur in K (Default: 1200)")
    p.add_argument("--delta-t", type=float, default=20.0)
    p.add_argument("--therm-steps", type=int, default=1000,
                   help="Thermalisierungs-Schritte pro T-Stufe (Default: 1000 = 1 ps)")
    p.add_argument("--duration-min", type=int, default=25,
                   help="Max. Laufzeit in Minuten (SLURM Time-Out-Handling, Default: 25)")
    p.add_argument("--stale-minutes", type=int, default=30,
                   help="processing-Einträge älter als N Min. → reset auf pending (Default: 30)")
    p.add_argument("--supercell-min-atoms", type=int, default=50,
                   help="Mindestatomzahl für Supercell (Default: 50, 0 = keine Supercell)")

    # ── Debug ──
    p.add_argument("--debug-convergence", action="store_true",
                   help="Konvergenz-Plots pro T-Schritt speichern (headless PNG)")
    p.add_argument("--debug-plot-dir", type=str, default="debug_convergence",
                   help="Verzeichnis für Debug-Plots (Default: debug_convergence)")

    # ── Queue-Management ──
    p.add_argument("--queue-stats", action="store_true",
                   help="Queue-Statistik anzeigen und beenden")
    p.add_argument("--reset-stale", action="store_true",
                   help="Stale 'processing'-Einträge zurück auf 'pending' setzen")
    p.add_argument("--resimulate", nargs="+", default=None,
                   help="Material-IDs (mp-xxxx) erneut simulieren: 'done' → 'pending' in Queue")
    p.add_argument("--version-label", type=str, default="v1",
                   help="Label für archivierte alte Ergebnisse (Default: v1)")

    return p.parse_args(argv)


# ── Analyse-Modus ───────────────────────────────────────────────────────────

def _run_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze_all, analyze_material, rank_materials

    if args.rank:
        rank_materials(
            db_path=args.db,
            top=args.top,
            only_phase_change=not args.no_phase_change_filter,
            output_dir=args.save,
            recompute=args.recompute,
        )
        return 0

    if args.all:
        analyze_all(args.db, output_dir=Path(args.save) if args.save else None)
        return 0

    if not args.material_id:
        print("[FEHLER] --analyze benötigt entweder --material-id <MP-ID> oder --all.")
        print("Verfügbare Material-IDs in der DB:")
        from .storage import list_material_ids
        ids = list_material_ids(args.db)
        if ids:
            for mid in ids:
                print(f"  {mid}")
        else:
            print("  (Datenbank ist leer)")
        return 1

    analyze_material(
        db_path=args.db,
        material_id=args.material_id,
        save_path=args.save,
        show=args.show,
    )
    return 0


# ── Ingest-Modus ────────────────────────────────────────────────────────────

def _run_ingest(args: argparse.Namespace) -> int:
    if not args.chemsys:
        print("[FEHLER] Ingest-Modus benötigt --chemsys (z. B. --chemsys V-O Ti-O).")
        return 1

    cfg = PipelineConfig(
        chemsys_list=args.chemsys,
        max_results_per_sys=args.max_results,
        e_hull_max=args.e_hull_max,
        include_ternary=args.include_ternary,
        db_path=args.db,
    )
    ingest_phase(cfg)
    return 0


# ── Pipeline-Modus (Process) ────────────────────────────────────────────────

def _run_pipeline(args: argparse.Namespace) -> int:
    # Wenn --chemsys angegeben ist UND nicht --ingest: Ingest + Process in einem
    if args.chemsys and not args.chemsys == []:
        # Erst Ingest, dann Process
        cfg_ingest = PipelineConfig(
            chemsys_list=args.chemsys,
            max_results_per_sys=args.max_results,
            e_hull_max=args.e_hull_max,
            include_ternary=args.include_ternary,
            db_path=args.db,
        )
        ingest_phase(cfg_ingest)

    ramp = RampConfig(
        t_max=args.t_max,
        delta_t=args.delta_t,
        thermalization_steps=args.therm_steps,
        debug_convergence_plots=args.debug_convergence,
        debug_plot_dir=args.debug_plot_dir,
    )
    cfg = PipelineConfig(
        chemsys_list=args.chemsys or [],
        mlip_backend=args.mlip,
        device=args.device,
        ramp=ramp,
        db_path=args.db,
        duration_min=args.duration_min,
        stale_minutes=args.stale_minutes,
        supercell_min_atoms=args.supercell_min_atoms,
    )
    return run_pipeline(cfg)


# ── Queue-Management ────────────────────────────────────────────────────────

def _run_queue_stats(args: argparse.Namespace) -> int:
    from .storage import queue_stats
    stats = queue_stats(args.db)
    print(f"Queue-Statistik ({args.db}):")
    print(f"  pending:    {stats['pending']:>6}")
    print(f"  processing: {stats['processing']:>6}")
    print(f"  done:       {stats['done']:>6}")
    print(f"  error:      {stats['error']:>6}")
    print(f"  total:      {stats['total']:>6}")
    return 0


def _run_reset_stale(args: argparse.Namespace) -> int:
    from .storage import reset_stale
    n = reset_stale(args.db, stale_minutes=args.stale_minutes)
    print(f"{n} stale 'processing'-Einträge zurück auf 'pending' gesetzt.")
    return 0


def _run_resimulate(args: argparse.Namespace) -> int:
    """Materialien für erneute Simulation zurück in die Queue stellen.

    Sichert vorhandene Ergebnisse in ``materials_archive`` (mit version_label),
    bevor sie aus ``materials`` gelöscht werden.
    """
    from .storage import requeue_materials

    material_ids = args.resimulate or []
    n = requeue_materials(args.db, material_ids, version_label=args.version_label)
    print(f"{n} Materialien archiviert ({args.version_label}) und zurück in die Queue gestellt.")
    print(f"Starte Pipeline mit: python -m mat_sim.run --db {args.db} --device auto")
    return 0


# ── Recheck-Switch ─────────────────────────────────────────────────────────

def _run_recheck_switch(args: argparse.Namespace) -> int:
    """T_switch für alle Materialien neu auswerten.

    Zweigeteilte Strategie:
    1. **Mit RDF-History**: Strengere Kriterien (Persistenz-Check, Sprung
       vs. Ausdehnung, Volumen-Bestätigung) auf der gespeicherten RDF-History.
    2. **Ohne RDF-History** (alte DB-Einträge): Fallback über
       ``detect_t_switch_from_scalars()`` — nutzt Volumen- und Q4-Zeitreihen.
    """
    from .metrics import (
        detect_t_switch,
        detect_t_switch_from_msd,
        detect_t_switch_from_scalars,
    )
    from .storage import list_material_ids, load_result, update_t_switch

    ids = list_material_ids(args.db)
    if not ids:
        print("Keine Materialien in der DB.")
        return 0

    n_total = len(ids)
    n_old_switch = 0
    n_new_switch = 0
    n_lost_switch = 0
    n_rdf_based = 0
    n_scalar_based = 0
    n_msd_based = 0
    n_no_data = 0

    print(f"Re-check T_switch für {n_total} Materialien …")

    for i, mid in enumerate(ids, start=1):
        mat = load_result(args.db, mid)
        old_t = mat.t_switch

        new_t: float | None = None

        # ── Strategie 1: RDF-History verfügbar ────────────────────────
        if mat.rdf_history is not None and len(mat.rdf_history) >= 2:
            new_t = detect_t_switch(
                mat.rdf_history,
                mat.temperatures,
                volumes=mat.volumes,
            )
            if new_t is not None:
                n_rdf_based += 1

        # ── Strategie 2: Fallback aus Skalar-Zeitreihen ───────────────
        if new_t is None and len(mat.volumes) >= 2 and len(mat.ql_values) >= 2:
            new_t = detect_t_switch_from_scalars(
                mat.volumes,
                mat.ql_values,
                mat.temperatures,
            )
            if new_t is not None:
                n_scalar_based += 1

        # ── Strategie 3: Fallback aus MSD-Zeitreihe ───────────────────
        if new_t is None and len(mat.msd_values) >= 2:
            new_t = detect_t_switch_from_msd(
                mat.msd_values,
                mat.temperatures,
                min_absolute_msd=0.001,
                min_temperature=100.0,
            )
            if new_t is not None:
                n_msd_based += 1

        if new_t is None and (
            len(mat.volumes) < 2 or len(mat.ql_values) < 2
        ) and (mat.rdf_history is None or len(mat.rdf_history) < 2):
            n_no_data += 1

        if old_t is not None:
            n_old_switch += 1
        if new_t is not None:
            n_new_switch += 1
        if old_t is not None and new_t is None:
            n_lost_switch += 1

        # Nur updaten wenn sich geändert hat
        if new_t != old_t:
            update_t_switch(args.db, mid, new_t)

        if i % 100 == 0:
            print(f"  … {i}/{n_total} geprüft", flush=True)

    print("\nErgebnis Re-check:")
    print(f"  Total geprüft:      {n_total}")
    print(f"  Alte T_switch:      {n_old_switch}")
    print(f"  Neue T_switch:      {n_new_switch}")
    print(f"    davon RDF-basiert:    {n_rdf_based}")
    print(f"    davon Skalar-basiert: {n_scalar_based}")
    print(f"    davon MSD-basiert:    {n_msd_based}")
    print(f"  Verlorene Switches: {n_lost_switch}")
    print(f"  Keine Daten:        {n_no_data}")
    print(f"  → {n_new_switch} gesamt")
    return 0


# ── Einstiegspunkt ──────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.analyze:
        return _run_analyze(args)
    if args.queue_stats:
        return _run_queue_stats(args)
    if args.reset_stale:
        return _run_reset_stale(args)
    if args.resimulate:
        return _run_resimulate(args)
    if args.recheck_switch:
        return _run_recheck_switch(args)
    if args.ingest:
        return _run_ingest(args)
    return _run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())

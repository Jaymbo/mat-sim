#!/usr/bin/env python3
"""Kommandozeile-Einstiegspunkt für die mat-sim Pipeline.

Beispiele
---------
Pipeline starten:
    export MP_API_KEY="dein_key"
    python -m mat_sim.run --chemsys V-O Ti-O Sr-Ti-O --t-max 600 --delta-t 10

Einzelnes Material analysieren (Dashboard + Vor-Evaluierung):
    python -m mat_sim.run --analyze --material-id mp-1234 --db results.db

Alle Materialien der DB analysieren:
    python -m mat_sim.run --analyze --all --db results.db

Optionen (Pipeline)
    --chemsys        Ein oder mehrere chemische Systeme (Leerzeichen-getrennt)
    --mlip           mace | chgnet  (Default: mace)
    --device         cpu | cuda | auto  (Default: cpu)
    --t-max          Maximaltemperatur in K          (Default: 600)
    --delta-t        Temperaturschritt in K          (Default: 10)
    --therm-steps    Thermalisierungs-Schritte pro T (Default: 100)
    --max-results    Max. Strukturen pro System       (Default: 50)
    --duration-min   Max. Laufzeit in Minuten (SLURM) (Default: 25)
    --db             Pfad zur SQLite-DB              (Default: results.db)

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
from .pipeline import PipelineConfig, run_pipeline


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
    p.add_argument("--save", type=str, default=None,
                   help="Dateipfad für PNG-Export (Default: dashboard_<id>.png)")
    p.add_argument("--show", action="store_true",
                   help="Interaktives Matplotlib-Fenster öffnen")

    # ── Pipeline-Optionen ──
    p.add_argument("--chemsys", nargs="+", default=None,
                   help="Chemische Systeme, z. B. V-O Ti-O Sr-Ti-O")
    p.add_argument("--mlip", choices=["mace", "chgnet"], default="mace")
    p.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    p.add_argument("--t-max", type=float, default=600.0)
    p.add_argument("--delta-t", type=float, default=10.0)
    p.add_argument("--therm-steps", type=int, default=100)
    p.add_argument("--max-results", type=int, default=50)
    p.add_argument("--stable-only", action="store_true", default=True)
    p.add_argument("--duration-min", type=int, default=25,
                   help="Max. Laufzeit in Minuten (SLURM Time-Out-Handling, Default: 25)")

    return p.parse_args(argv)


# ── Analyse-Modus ───────────────────────────────────────────────────────────

def _run_analyze(args: argparse.Namespace) -> int:
    from .analyze import analyze_material, analyze_all

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


# ── Pipeline-Modus ──────────────────────────────────────────────────────────

def _run_pipeline(args: argparse.Namespace) -> int:
    if not args.chemsys:
        print("[FEHLER] Pipeline-Modus benötigt --chemsys (z. B. --chemsys V-O Ti-O).")
        return 1

    ramp = RampConfig(
        t_max=args.t_max,
        delta_t=args.delta_t,
        thermalization_steps=args.therm_steps,
    )
    cfg = PipelineConfig(
        chemsys_list=args.chemsys,
        mlip_backend=args.mlip,
        device=args.device,
        max_results_per_sys=args.max_results,
        stable_only=args.stable_only,
        ramp=ramp,
        db_path=args.db,
        duration_min=args.duration_min,
    )
    return run_pipeline(cfg)


# ── Einstiegspunkt ──────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if args.analyze:
        return _run_analyze(args)
    return _run_pipeline(args)


if __name__ == "__main__":
    sys.exit(main())

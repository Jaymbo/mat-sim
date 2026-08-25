#!/bin/bash
#SBATCH --job-name=mat_sim_ingest
#SBATCH --partition=cpu_short
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Vorbereitung ────────────────────────────────────────────────────────────
cd "$HOME/mat-sim"
mkdir -p logs

# ── uv aktivieren ───────────────────────────────────────────────────────────
source "$HOME/.local/bin/env" 2>/dev/null || true
export UV_CACHE_DIR="$HOME/.cache/uv"

# ── Ingest-Phase: Strukturen von MP herunterladen → DB ──────────────────────
echo "[$(date)] Starte mat-sim Ingest auf $(hostname)"
echo ""

# Python aus dem uv venv nutzen
uv run python -m mat_sim.run \
    --ingest \
    --chemsys V-O Ti-O Cr-O Mn-O Sr-Ti-O \
    --include-ternary \
    --e-hull-max 0.1 \
    --max-results 2000 \
    --db results.db

echo ""
echo "[$(date)] Ingest beendet."

# Queue-Statistik
uv run python -m mat_sim.run --queue-stats --db results.db

#!/bin/bash
#SBATCH --job-name=mat_sim_validate
#SBATCH --partition=dev_gpu_h100,dev_gpu_a100_il,gpu_a100_short,gpu_h100_short
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=00:30:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# ── Vorbereitung ────────────────────────────────────────────────────────────
cd "$HOME/mat-sim"
mkdir -p logs

# CPU-Thread-Limits (verhindert Overhead auf Multi-Core-Knoten)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── uv aktivieren ───────────────────────────────────────────────────────────
source "$HOME/.local/bin/env" 2>/dev/null || true
export UV_CACHE_DIR="$HOME/.cache/uv"

# ── Git pull (neuester Stand) ───────────────────────────────────────────────
git pull

echo "[$(date)] Starte mat-sim VALIDATION auf $(hostname)"
echo "  SLURM Job ID:  $SLURM_JOB_ID"
echo "  Partition:     $SLURM_JOB_PARTITION"
echo "  GPU:           $CUDA_VISIBLE_DEVICES"
echo ""

# ── Schritt 1: 3 Test-Materialien als v1 archivieren + re-queue ─────────────
# Nur beim ersten Lauf (kein Checkpoint vorhanden).  Bei Resume wird
# dieser Schritt übersprungen, da die Materialien bereits gequeued sind.
VALIDATE_MIDS="mp-18248 mp-19227 mp-20406"
HAS_CHECKPOINT=$(uv run python -c "
from mat_sim.storage import has_checkpoint
print(any(has_checkpoint('results.db', mid) for mid in ${VALIDATE_MIDS}))
" 2>/dev/null)

if [ "${HAS_CHECKPOINT}" = "False" ]; then
    echo "[$(date)] Schritt 1: Archiviere alte Ergebnisse als v1 …"
    uv run python -m mat_sim.run \
        --resimulate mp-18248 mp-19227 mp-20406 \
        --version-label v1 \
        --db results.db
else
    echo "[$(date)] Schritt 1 übersprungen — Checkpoint vorhanden, Resume wird fortgesetzt."
fi

echo ""

# ── Schritt 2: Mit neuen Parametern simulieren (1200K, 500 steps) ───────────
echo "[$(date)] Schritt 2: Simulation mit neuen Parametern (t_max=1200, therm_steps=500) …"
uv run python -m mat_sim.run \
    --db results.db \
    --device auto \
    --duration-min 25 \
    --t-max 1200 \
    --delta-t 10 \
    --therm-steps 500

SIM_EXIT=$?

echo ""
echo "[$(date)] Simulation beendet. Exit-Code: $SIM_EXIT"

# ── Self-Rescheduling: bei Time-Out (Exit 88) → neuen Job einreichen ────────
# Mit Checkpoint/Resume führt der nächste Job die Simulation fort.
if [ "$SIM_EXIT" -eq 88 ]; then
    echo "[$(date)] Time-Out — reiche neuen Validierungs-Job ein (Resume via Checkpoint) …"
    sbatch validate.sbatch
    exit 88
fi

# ── Schritt 3: Ergebnisse anzeigen ──────────────────────────────────────────
echo ""
echo "══════════════════════════════════════════════════════════════════"
echo "  Validierung: alt (v1) vs. neu (v2)"
echo "══════════════════════════════════════════════════════════════════"
echo ""

echo "── Archivierte v1-Ergebnisse ──"
sqlite3 results.db "
SELECT material_id, formula, t_switch, t_decay
FROM materials_archive
WHERE version_label = 'v1'
ORDER BY material_id;
"

echo ""
echo "── Neue v2-Ergebnisse ──"
sqlite3 results.db "
SELECT material_id, formula, t_switch, t_decay, status
FROM materials
WHERE material_id IN ('mp-18248', 'mp-19227', 'mp-20406')
ORDER BY material_id;
"

echo ""
echo "── Literatur-Vergleich ──"
echo "  mp-18248 (VO2):     v1=? → v2=? | Literatur: ~340 K"
echo "  mp-19227 (YMnO3):   v1=? → v2=? | Literatur: ~900 K"
echo "  mp-20406 (Lu2V2O7): v1=? → v2=? | Literatur: ~130 K (Pyrochlore)"

echo ""
echo "[$(date)] Validierung abgeschlossen."

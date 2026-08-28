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

# ── Schritt 1: 3 Test-Materialien archivieren + re-queue ───────────────────
# NUR beim ersten Lauf (keine Checkpoints vorhanden): requeue + archivieren.
# Bei Rescheduling (Checkpoints vorhanden) überspringen → Pipeline resumed.
HAS_CKPT=$(sqlite3 results.db "
SELECT COUNT(*) FROM checkpoints
WHERE material_id IN ('mp-18248', 'mp-19227', 'mp-20406');
")

if [ "$HAS_CKPT" -eq 0 ]; then
    echo "[$(date)] Schritt 1: Archiviere alte Ergebnisse (erster Lauf) …"
    uv run python -m mat_sim.run \
        --resimulate mp-18248 mp-19227 mp-20406 \
        --version-label v4 \
        --db results.db
else
    echo "[$(date)] Schritt 1 übersprungen — Checkpoints vorhanden, Resume-Modus."
fi

echo ""

# ── Schritt 2: Mit neuen Parametern simulieren (1200K, 500 steps) ───────────
echo "[$(date)] Schritt 2: Simulation mit neuen Parametern (t_max=1200, therm_steps=500) …"
uv run python -m mat_sim.run \
    --db results.db \
    --device auto \
    --duration-min 25 \
    --t-max 1200 \
    --delta-t 20 \
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
echo "  Validierung: alt (v4) vs. neu (mit Persistenz + MSD-basiertem T_switch)"
echo "══════════════════════════════════════════════════════════════════"
echo ""

echo "── Archivierte v4-Ergebnisse (vorheriger Lauf) ──"
sqlite3 results.db "
SELECT material_id, formula, t_switch, t_decay
FROM materials_archive
WHERE version_label = 'v4'
ORDER BY material_id;
"

echo ""
echo "── Neue Ergebnisse (mit Persistenz + MSD-basiertem T_switch) ──"
sqlite3 results.db "
SELECT material_id, formula, t_switch, t_decay, status
FROM materials
WHERE material_id IN ('mp-18248', 'mp-19227', 'mp-20406')
ORDER BY material_id;
"

echo ""
echo "── Literatur-Vergleich ──"
echo "  mp-18248 (VO2):     v4=? → neu=? | Literatur: ~340 K"
echo "  mp-19227 (YMnO3):   v4=? → neu=? | Literatur: ~900 K"
echo "  mp-20406 (Lu2V2O7): v4=? → neu=? | Literatur: ~130 K (Pyrochlore)"

echo ""
echo "[$(date)] Validierung abgeschlossen."

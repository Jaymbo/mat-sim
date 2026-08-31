#!/bin/bash
#SBATCH --job-name=mat_sim_process
#SBATCH --partition=gpu_a100_short,gpu_h100_short,gpu_h100,gpu_a100_il,gpu_h100_il,dev_gpu_h100,dev_gpu_a100_il
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

# ── Worker: claimt pending-Strukturen aus DB, simuliert, wiederholt ─────────
echo "[$(date)] Starte mat-sim Worker auf $(hostname)"
echo "  SLURM Job ID:  $SLURM_JOB_ID"
echo "  Partition:     $SLURM_JOB_PARTITION"
echo "  GPU:           $CUDA_VISIBLE_DEVICES"
echo ""

# Stale-Einträge von abgestürzten Jobs zurücksetzen
uv run python -m mat_sim.run --reset-stale --db results.db

# Queue-Statistik vor dem Start
uv run python -m mat_sim.run --queue-stats --db results.db

# Verarbeitung: claimt pending-Strukturen aus der DB bis Time-Out oder leer
uv run python -m mat_sim.run \
    --db results.db \
    --device auto \
    --duration-min 25 \
    --t-max 1200 \
    --delta-t 50 \
    --therm-steps 1000 \
    --supercell-min-atoms 50

TRAIN_EXIT=$?

echo ""
echo "[$(date)] Worker beendet. Exit-Code: $TRAIN_EXIT"

# Queue-Statistik nach Abschluss
uv run python -m mat_sim.run --queue-stats --db results.db

# ── Self-Rescheduling: wenn noch pending in der Queue → neuen Job einreichen ─
PENDING=$(uv run python -c "
from mat_sim.storage import queue_stats
print(queue_stats('results.db')['pending'])
" 2>/dev/null)

if [ "$TRAIN_EXIT" -eq 88 ] && [ "${PENDING:-0}" -gt 0 ]; then
    echo "[$(date)] Time-Out + $PENDING pending → reiche neuen Job ein …"
    sbatch run_cluster.sbatch
elif [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "[$(date)] Queue leer — kein Neustart nötig."
else
    echo "[$(date)] Exit $TRAIN_EXIT — kein Auto-Rescheduling."
fi

exit $TRAIN_EXIT

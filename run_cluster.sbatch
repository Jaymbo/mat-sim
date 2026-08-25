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
cd "$HOME/Dokumente/mat-sim"
mkdir -p logs

# CPU-Thread-Limits (verhindert Overhead auf Multi-Core-Knoten)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

# ── Conda ───────────────────────────────────────────────────────────────────
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate mat-sim

# ── Worker: claimt pending-Strukturen aus DB, simuliert, wiederholt ─────────
echo "[$(date)] Starte mat-sim Worker auf $(hostname)"
echo "  SLURM Job ID:  $SLURM_JOB_ID"
echo "  Partition:     $SLURM_JOB_PARTITION"
echo "  GPU:           $CUDA_VISIBLE_DEVICES"
echo ""

# Stale-Einträge von abgestürzten Jobs zurücksetzen
python -m mat_sim.run --reset-stale --db results.db

# Queue-Statistik vor dem Start
python -m mat_sim.run --queue-stats --db results.db

# Verarbeitung: claimt pending-Strukturen aus der DB bis Time-Out oder leer
python -m mat_sim.run \
    --db results.db \
    --device auto \
    --duration-min 25 \
    --t-max 500 \
    --delta-t 10

TRAIN_EXIT=$?

echo ""
echo "[$(date)] Worker beendet. Exit-Code: $TRAIN_EXIT"

# Queue-Statistik nach Abschluss
python -m mat_sim.run --queue-stats --db results.db

# Exit-Code 88 = Time-Out (Dispatcher reicht ggf. neuen Job ein)
# Exit-Code 0 = Queue leer
exit $TRAIN_EXIT

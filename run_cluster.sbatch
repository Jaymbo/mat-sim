#!/bin/bash
#SBATCH --job-name=mat_sim_screening
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

# ── Pipeline starten ────────────────────────────────────────────────────────
echo "[$(date)] Starte mat-sim Screening auf $(hostname)"
echo "  SLURM Job ID:  $SLURM_JOB_ID"
echo "  Partition:     $SLURM_JOB_PARTITION"
echo "  GPU:           $CUDA_VISIBLE_DEVICES"
echo ""

python -m mat_sim.run \
    --chemsys V-O Ti-O Cr-O Mn-O Sr-Ti-O \
    --t-max 500 \
    --delta-t 10 \
    --device auto \
    --duration-min 25

TRAIN_EXIT=$?

echo ""
echo "[$(date)] Pipeline beendet. Exit-Code: $TRAIN_EXIT"

# ── Auto-Rescheduling bei Time-Out (Exit 88) ────────────────────────────────
if [ "$TRAIN_EXIT" -eq 88 ]; then
    echo "[$(date)] Time-Out erkannt (Exit 88) – reiche neuen SLURM-Job ein …"
    sbatch run_cluster.sbatch
elif [ "$TRAIN_EXIT" -eq 0 ]; then
    echo "[$(date)] Screening vollständig abgeschlossen. Kein Neustart nötig."
else
    echo "[$(date)] Unerwarteter Exit-Code ($TRAIN_EXIT) – kein Auto-Rescheduling."
fi

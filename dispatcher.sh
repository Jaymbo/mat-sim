#!/bin/bash
# ═══════════════════════════════════════════════════════════════════════════
#  Dispatcher: Überwacht die Queue und reicht SLURM-Worker-Jobs ein.
#
#  Läuft als lokaler Prozess (Terminal, tmux, nohup) oder als langer CPU-Job.
#  Beendet sich automatisch, wenn keine pending-Strukturen mehr vorhanden.
#
#  Verwendung:
#    ./dispatcher.sh                  # Default: max. 5 parallele Jobs
#    ./dispatcher.sh --max-jobs 10    # 10 parallele Jobs
#    ./dispatcher.sh --db results.db --max-jobs 8
#
#  Voraussetzung:
#    1. Ingest ist bereits gelaufen:  sbatch ingest.sbatch
#    2. results.db existiert und hat pending-Einträge
# ═══════════════════════════════════════════════════════════════════════════

set -euo pipefail

# ── Defaults ────────────────────────────────────────────────────────────────
MAX_JOBS=5
DB="results.db"
WORKER_SBATCH="run_cluster.sbatch"
POLL_INTERVAL=60          # Sekunden zwischen Checks
STALE_MINUTES=30          # processing-Timeout für reset

# ── Argumente parsen ────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --max-jobs)     MAX_JOBS="$2"; shift 2 ;;
        --db)           DB="$2"; shift 2 ;;
        --worker)       WORKER_SBATCH="$2"; shift 2 ;;
        --poll)         POLL_INTERVAL="$2"; shift 2 ;;
        --stale-min)    STALE_MINUTES="$2"; shift 2 ;;
        -h|--help)
            echo "Usage: dispatcher.sh [--max-jobs N] [--db PATH] [--worker SBATCH] [--poll SEC] [--stale-min MIN]"
            exit 0 ;;
        *) echo "Unbekanntes Argument: $1"; exit 1 ;;
    esac
done

cd "$HOME/mat-sim"
mkdir -p logs

# ── uv aktivieren (für python -m mat_sim.run) ──────────────────────────────
source "$HOME/.local/bin/env" 2>/dev/null || true
export UV_CACHE_DIR="$HOME/.cache/uv"

# ── Hilfsfunktionen ─────────────────────────────────────────────────────────

# pending + processing aus der Queue auslesen
get_pending() {
    uv run python -c "
from mat_sim.storage import queue_stats
s = queue_stats('$DB')
print(s['pending'] + s['processing'])
" 2>/dev/null
}

# Anzahl laufende mat_sim_process-Jobs via squeue
get_running_jobs() {
    squeue --me --name=mat_sim_process --states=RUNNING,PENDING -h | wc -l
}

# Stale-Einträge zurücksetzen
do_reset_stale() {
    uv run python -m mat_sim.run --reset-stale --stale-minutes "$STALE_MINUTES" --db "$DB" 2>/dev/null || true
}

# Queue-Statistik ausgeben
print_stats() {
    uv run python -m mat_sim.run --queue-stats --db "$DB" 2>/dev/null || true
}

# ── Haupt Schleife ──────────────────────────────────────────────────────────
echo "============================================================"
echo "  Dispatcher gestartet"
echo "  DB:            $DB"
echo "  Worker:        $WORKER_SBATCH"
echo "  Max Jobs:      $MAX_JOBS"
echo "  Poll-Interval: ${POLL_INTERVAL}s"
echo "  Stale-Reset:   ${STALE_MINUTES} min"
echo "============================================================"
echo ""

ITERATION=0

while true; do
    ITERATION=$((ITERATION + 1))

    # 1. Stale-Reset (abgestürzte Worker freigeben)
    do_reset_stale

    # 2. Queue-Status holen
    PENDING=$(get_pending)
    RUNNING=$(get_running_jobs)

    echo "[$(date +%H:%M:%S)] Iteration $ITERATION | pending+processing: $PENDING | laufende Jobs: $RUNNING / $MAX_JOBS"

    # 3. Abbruch: keine Arbeit mehr
    if [ "$PENDING" -eq 0 ]; then
        echo ""
        echo "[$(date)] Keine pending/processing-Strukturen mehr — Dispatcher beendet."
        print_stats
        break
    fi

    # 4. Neue Jobs einreichen, bis MAX_JOBS erreicht
    while [ "$RUNNING" -lt "$MAX_JOBS" ] && [ "$PENDING" -gt 0 ]; do
        JOB_ID=$(sbatch --parsable "$WORKER_SBATCH")
        echo "  -> Job $JOB_ID eingereicht (laufend: $((RUNNING + 1)) / $MAX_JOBS)"
        RUNNING=$((RUNNING + 1))
        # Kurze Pause, damit squeue den neuen Job registriert
        sleep 2
    done

    # 5. Warten bis zur nächsten Iteration
    echo ""
    sleep "$POLL_INTERVAL"
done

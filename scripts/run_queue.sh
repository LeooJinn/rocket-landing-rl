#!/usr/bin/env bash
# Run training jobs with bounded parallelism; retry a job (up to 3 attempts) if it dies.
#   scripts/run_queue.sh JOBS_FILE [MAX_PARALLEL]
# JOBS_FILE lines: "<config-path> <seed> <run-name>"; logs go to logs/<run-name>.log
# A finished run (summary.json present) is skipped, so re-running the queue resumes it.
set -u
JOBS=${1:?jobs file}
PAR=${2:-4}
mkdir -p logs
run_job() {
  cfg=$1; seed=$2; name=$3
  for attempt in 1 2 3; do
    if [ -f "experiments/results/$name/summary.json" ]; then exit 0; fi
    echo "[$(date +%T)] start $name attempt $attempt" >> logs/queue.log
    uv run python -m rocket_landing.train --config "$cfg" --seed "$seed" --run-name "$name" \
      > "logs/$name.log" 2>&1 && { echo "[$(date +%T)] done $name" >> logs/queue.log; exit 0; }
    echo "[$(date +%T)] FAILED $name attempt $attempt" >> logs/queue.log
  done
  exit 1
}
export -f run_job
xargs -P "$PAR" -L 1 bash -c 'run_job "$@"' _ < "$JOBS"

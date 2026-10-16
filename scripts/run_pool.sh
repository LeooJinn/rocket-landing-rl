#!/usr/bin/env bash
# Start jobs from JOBS_FILE (lines "<config> <seed> <run-name>") whenever fewer than MAX training
# processes of THIS worktree are running; each job goes through run_queue.sh (retry on failure,
# skipped if summary.json exists). Returns when every job has finished.
#   scripts/run_pool.sh JOBS_FILE [MAX=8]
set -u
JOBS=${1:?jobs file}
MAX=${2:-8}
ME="$(pwd)/.venv/bin/python -m rocket_landing.train"
TMP=$(mktemp -d)
i=0
while read -r line; do
  [ -z "$line" ] && continue
  while [ "$(pgrep -f "$ME" | wc -l)" -ge "$MAX" ]; do sleep 15; done
  i=$((i+1)); echo "$line" > "$TMP/job_$i.txt"
  scripts/run_queue.sh "$TMP/job_$i.txt" 1 &
  sleep 45   # let the new process appear before re-counting
done < "$JOBS"
wait
rm -rf "$TMP"

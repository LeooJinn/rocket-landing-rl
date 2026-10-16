#!/usr/bin/env bash
# Regenerate every file in raw/ from the 20 headline runs.
#
#   COLLAPSE_RESULTS_DIR=/path/to/experiments/results bash analysis/collapse_diagnosis/run_all.sh [fast|full]
#
# "fast" (default) runs the scripts that only read metrics.jsonl / eval.jsonl (seconds, no checkpoints).
# "full" also rolls out checkpoints (needs the git-ignored checkpoints; roughly 15-30 min on 10 cores).
set -euo pipefail
cd "$(dirname "$0")/../.."          # repo root
A=analysis/collapse_diagnosis
R=$A/raw
mkdir -p "$R"
run() { uv run python "$A/$1.py" "${@:2}"; }

# ---- metrics / eval only ----
run eval_curves          > "$R/eval_curves.txt"
run training_vs_greedy   > "$R/training_vs_greedy.txt"
run fuel_vs_outcome      > "$R/fuel_vs_outcome.txt"
run persistence          > "$R/persistence.txt"
run entropy_fuel_drift   > "$R/entropy_fuel_drift.txt"
run collapse_window hl_v2_s8 1000 1220 > "$R/window_hl_v2_s8_1000_1220.txt"
run collapse_window hl_v2_s6 80 280 5  > "$R/window_hl_v2_s6_80_280.txt"
run collapse_window hl_v1_s5 590 760 6 > "$R/window_hl_v1_s5_590_760.txt"

[ "${1:-fast}" = "full" ] || exit 0

# ---- checkpoint rollouts (held-out reset seeds 50000.., diagnostic only) ----
run final_failure_modes  > "$R/final_failure_modes.txt"
run greedy_vs_sampled hl_v2_s0 hl_v2_s1 hl_v2_s2 hl_v2_s3 hl_v2_s4 hl_v2_s5 hl_v2_s6 hl_v2_s7 hl_v2_s8 hl_v2_s9 \
                         > "$R/greedy_vs_sampled.txt"
uv run python "$A/summarise_greedy_vs_sampled.py" > "$R/greedy_vs_sampled_summary.txt"
run temperature_sweep hl_v2_s8@1000 hl_v2_s8@1100 hl_v2_s8@1150 hl_v2_s8@1200 hl_v2_s8@final \
    hl_v2_s7@1000 hl_v2_s7@1050 hl_v2_s7@1100 hl_v2_s7@1150 hl_v2_s7@1200 hl_v2_s7@final \
    hl_v2_s6@1200 hl_v2_s6@final --tag t1 > "$R/temperature_sweep_t1.txt"
run fuel_race_trace hl_v2_s8 50003 1150 1200 > "$R/fuel_race_trace.txt"
run argmax_margins       > "$R/argmax_margins.txt"
{ run altitude_profile hl_v2_s8@1200 hl_v2_s8@1150 --episodes 30
  run altitude_profile hl_v2_s3@final hl_v2_s2@final hl_v2_s7@1050 hl_v2_s7@1100 --episodes 20
} > "$R/altitude_profile.txt"
run hover_rollout hl_v2_s6@final 50000 50001 > "$R/hover_rollout_hl_v2_s6.txt"
run discounted_returns hl_v2_s6@50 hl_v2_s6@100 hl_v2_s6@150 hl_v2_s6@200 hl_v2_s6@250 hl_v2_s6@final \
    hl_v2_s3@300 hl_v2_s3@600 hl_v2_s3@final hl_v2_s8@final > "$R/discounted_returns.txt"
run advantage_scale hl_v2_s3@300 hl_v2_s3@600 hl_v2_s8@1000 hl_v2_s8@1150 hl_v2_s8@1200 hl_v2_s8@final \
    hl_v2_s6@final > "$R/advantage_scale.txt"

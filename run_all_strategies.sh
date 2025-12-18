#!/usr/bin/env bash
set -euo pipefail

# -----------------------------
# Experiment parameters
# -----------------------------
LAMBDAS=(4.0 5.0 6.0 7.0 8.0)
RUNS=(1 2 3 4 5)

STRATEGIES=(
  least-load-fixed
  least-load-adaptive
  opt_latency_fixed
  opt_power_fixed
  opt_efficiency_fixed
  opt_latency_adaptive
  opt_power_adaptive
  opt_efficiency_adaptive
)

BASE_SEED=41
OUT_DIR="runs"

# -----------------------------
# Strategy-specific flags
# -----------------------------
strategy_flags () {
  local strat="$1"

  case "$strat" in
    *_adaptive)
      echo "--enable-adaptive-freq --fast-decider"
      ;;
    *_fixed|least-load-*)
      echo "--fast-decider"
      ;;
    *)
      echo ""
      ;;
  esac
}

# -----------------------------
# Main loop
# -----------------------------
for STRAT in "${STRATEGIES[@]}"; do
  for L in "${LAMBDAS[@]}"; do
    for r in "${RUNS[@]}"; do

      PREGEN="runs/exp1_L${L}__run${r}_pregen.csv"
      SEED=$((BASE_SEED + r - 1))
      PREFIX="exp1_${STRAT}_L${L}__run${r}"

      echo ">>> STRATEGY=$STRAT | L=$L | run=$r | seed=$SEED"

      python queuing_system_simulation_incremental.py \
        --config config.json \
        --out-dir "$OUT_DIR" \
        --prefix "$PREFIX" \
        --strategy "$STRAT" \
        --pregen-in "$PREGEN" \
        --admission soft \
        --opt-module-latency    live_min_latency_optimizer_incremental \
        --opt-module-power      live_min_power_optimizer_incremental \
        --opt-module-efficiency live_max_efficiency_optimizer_incremental \
        --seed "$SEED" \
        $(strategy_flags "$STRAT")

    done
  done
done

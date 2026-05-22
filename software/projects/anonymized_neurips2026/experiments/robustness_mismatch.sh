#!/bin/bash
# Robustness to parameter mismatch.
#
# Trains cmos_bmru, mingru, lru on smnist_cmos, pmnist_cmos, real_audio_digits
# (all three tasks use mean pooling by definition), then evaluates the trained
# models with multiplicative Gaussian parameter noise at 5%, 10% and 15%
# (±p = 3σ convention, N noisy trials per level).
#
# Architecture choices fixed for this study:
#   num_recs=2, model_dim=64, norm=none, state_dim=64, seed=1
#   cmos_bmru: epsilon=0.0   |   mingru, lru: epsilon not applicable
#
# Control N from command line:   NOISE_N_TRIALS=20 ./robustness_mismatch.sh
#   or just rely on the default below (N=10).
#
# Flags:
#   --plot-only         skip training/robustness, only regenerate figures from
#                       existing *_robustness.json files.
#   --robustness-only   skip training and testing; run only the noise sweep on
#                       already-trained models (requires *_best.pkl to exist).
#   --noise-n=N         override number of noisy trials per noise level.
#
# WandB project: cmos_mismatch_analysis

set -e

BASE="$(cd "$(dirname "$0")/.." && pwd)"
CFGDIR="$BASE/configs/generated"
MODELDIR="$BASE/models/trained"
PLOTDIR="$BASE/plots/robustness_mismatch"
SEED=1
NUM_RECS=2
MODEL_DIM=64
STATE_DIM=64
WANDB_PROJECT="DOUBLE_BLIND_REVIEW"

# Number of noisy trials per noise level (override with env var or --noise-n arg)
NOISE_N="${NOISE_N_TRIALS:-10}"

TASKS=(smnist_cmos pmnist_cmos real_audio_digits real_audio_binary)

declare -A TASK_EPOCHS
TASK_EPOCHS["smnist_cmos"]=30000
TASK_EPOCHS["pmnist_cmos"]=30000
TASK_EPOCHS["real_audio_digits"]=35000
TASK_EPOCHS["real_audio_binary"]=35000

source "$BASE/experiments/lib/dispatch.sh"
parse_dispatch_args "$@"

# Flags
_PLOT_ONLY=0
_ROBUSTNESS_ONLY=0
for _arg in "$@"; do
    case "$_arg" in
        --plot-only)       _PLOT_ONLY=1 ;;
        --robustness-only) _ROBUSTNESS_ONLY=1 ;;
        --noise-n=*)       NOISE_N="${_arg#--noise-n=}" ;;
    esac
done

mkdir -p "$CFGDIR" "$MODELDIR" "$PLOTDIR" "$BASE/logs"

if [[ "$_PLOT_ONLY" == "1" ]]; then
    echo "[plot-only] Regenerating figures from existing JSON files..."
    python3 "$BASE/plot_robustness_mismatch.py" \
        --results_dir "$MODELDIR" \
        --output_dir  "$PLOTDIR"
    echo "Done."
    exit 0
fi

init_dispatch

if [[ "$_ROBUSTNESS_ONLY" == "1" ]]; then
    echo "[robustness-only] N=${NOISE_N} noise trials per level (skipping train+test)"
else
    echo "[robustness_mismatch] N=${NOISE_N} noise trials per level"
fi

for TASK in "${TASKS[@]}"; do
    NUM_EPOCHS="${TASK_EPOCHS[$TASK]}"

    for ARCH in cmos_bmru mingru lru; do
        if [[ "$ARCH" == "cmos_bmru" ]]; then
            EPS_ARG="0.0"
        else
            EPS_ARG="none"
        fi

        CFG=$(python3 "$BASE/seeder.py" \
            "$BASE/configs/base/config_${ARCH}.json" \
            "$CFGDIR" \
            "$TASK" "$SEED" \
            "$NUM_RECS" "$MODEL_DIM" none "$STATE_DIM" "$EPS_ARG" \
            relu "$NUM_EPOCHS" none none none none none 2>&1 | grep "^Created:" | cut -d' ' -f2)

        python3 -c "
import json, sys
with open(sys.argv[1]) as f: c = json.load(f)
c['model']['positional_encodings_dims'] = 16
c['early_stop_patience'] = 90000
with open(sys.argv[1], 'w') as f: json.dump(c, f, indent=2)
" "$CFG"

        MODEL_PATH="$MODELDIR/$(basename "$CFG" .json | sed 's/^config_//').pkl"

        if [[ "$_ROBUSTNESS_ONLY" == "1" ]]; then
            # Skip train+test; go straight to noise sweep on existing model
            run_robustness_only "$CFG" "$TASK" "$MODEL_PATH" "$WANDB_PROJECT" "$NOISE_N"
        else
            # Full pipeline: train + test + robustness (args 7=RUN_ROBUSTNESS, 8=NOISE_N_TRIALS)
            run_experiment "$CFG" "$TASK" "$MODEL_PATH" "$WANDB_PROJECT" "" 16 1 "$NOISE_N"
        fi
    done
done

finalize_dispatch

dispatch_robustness_plot "$MODELDIR" "$PLOTDIR"

echo ""
echo "Robustness mismatch study done."

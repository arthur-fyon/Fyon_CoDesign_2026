#!/bin/bash
# CMOS KWS: CMOS BMRU on KWS Digits.
# state_dim=4, model_dim=256, num_recs=2, single seed, epsilon annealing.
# Outputs: trained model, test metrics, 16 publication-ready PDF logit plots.

set -e

BASE="$(cd "$(dirname "$0")/.." && pwd)"
CFGDIR="$BASE/configs/generated"
MODELDIR="$BASE/models/trained"
PLOTDIR="$BASE/plots/cmos_kws"
TASK="real_audio_digits"
SEED=1
NUM_RECS=2
MODEL_DIM=256
STATE_DIM=32
NUM_EPOCHS=15000

source "$BASE/experiments/lib/dispatch.sh"
parse_dispatch_args "$@"

# --plot-only: skip training and testing, rerun plotting only.
_PLOT_ONLY=0
for _arg in "$@"; do [[ "$_arg" == "--plot-only" ]] && _PLOT_ONLY=1; done

mkdir -p "$CFGDIR" "$MODELDIR" "$PLOTDIR" "$BASE/logs"
init_dispatch

# Generate config: cmos_bmru, state_dim=4, model_dim=256, num_recs=2,
# no normalization, relu activation, epsilon=0 (no decaying).
CFG=$(python3 "$BASE/seeder.py" \
    "$BASE/configs/base/config_cmos_bmru.json" \
    "$CFGDIR" \
    "$TASK" "$SEED" \
    "$NUM_RECS" "$MODEL_DIM" none "$STATE_DIM" none \
    relu "$NUM_EPOCHS" none none none none none 2>&1 | grep "^Created:" | cut -d' ' -f2)

echo "Config: $CFG"
MODEL_PATH="$MODELDIR/$(basename "$CFG" .json | sed 's/^config_//').pkl"
echo "Model:  $MODEL_PATH"

# Train + test + plot (plot enabled via PLOTDIR; skipped with --plot-only).
if [[ "$_PLOT_ONLY" == "0" ]]; then
    run_experiment "$CFG" "$TASK" "$MODEL_PATH" "cmos_mru" "$PLOTDIR" 16
    finalize_dispatch
else
    # --plot-only: skip training and testing, dispatch plot job only.
    BEST_MODEL="${MODEL_PATH%.pkl}_best.pkl"
    if [[ ! -f "$BEST_MODEL" ]]; then
        echo "ERROR: --plot-only specified but trained model not found: $BEST_MODEL" >&2
        exit 1
    fi
    echo "[plot-only] Skipping training and testing. Using: $BEST_MODEL"
    dispatch_plot "plot_cmos_kws" "$CFG" "$BEST_MODEL" "$TASK" "$PLOTDIR" 16
fi

echo ""
echo "CMOS KWS done."

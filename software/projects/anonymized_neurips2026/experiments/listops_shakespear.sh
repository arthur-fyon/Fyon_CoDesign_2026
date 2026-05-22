#!/bin/bash
# Table 1 missing results: ListOps, Pathfinder32, Shakespeare.
# BMRU = cmru, FQ BMRU = cmos_bmru (both with epsilon annealing), LRU, minGRU.
# Classification (ListOps, Pathfinder32): num_recs=2, model_dim=256, layer norm, state_dim=64, glu, PE=32, 5 seeds.
# Shakespeare: num_recs=6, model_dim=256, state_dim=256, layer norm, glu, no PE, 5 seeds.
# WandB project: missing_results

set -e

BASE="$(cd "$(dirname "$0")/.." && pwd)"
CFGDIR="$BASE/configs/generated/missing_results"
MODELDIR="$BASE/models/trained"
SEEDS=(1 2 3 4 5)
WANDB_PROJECT="DOUBLE_BLIND_REVIEW"

source "$BASE/experiments/lib/dispatch.sh"
parse_dispatch_args "$@"

mkdir -p "$CFGDIR" "$MODELDIR" "$BASE/logs"
init_dispatch

# ---------------------------------------------------------------------------
# Classification: ListOps
# num_recs=2, model_dim=256, layer norm, state_dim=64, glu, PE=32, 100000 epochs
# ---------------------------------------------------------------------------
for SEED in "${SEEDS[@]}"; do
    for TASK in listops; do
        for ARCH in cmru cmos_bmru lru mingru; do
            case "$ARCH" in
                cmru|cmos_bmru) EPS_ARG="1.0" ; EPS_DECAY="true" ;;
                *)               EPS_ARG="none" ; EPS_DECAY="false" ;;
            esac
            CFG=$(python3 "$BASE/seeder.py" \
                "$BASE/configs/base/config_${ARCH}.json" \
                "$CFGDIR" \
                "$TASK" "$SEED" \
                2 256 layer 64 "$EPS_ARG" \
                none 100000 none none none none none 2>&1 | grep "^Created:" | cut -d' ' -f2)
            python3 -c "
import json, sys
with open(sys.argv[1]) as f: c = json.load(f)
c['model']['positional_encodings_dims'] = 32
c['early_stop_patience'] = 20000
if sys.argv[2] == 'true':
    c['epsilon_decay'] = True
    c['epsilon_const_epochs'] = 1000
    c['epsilon_decay_epochs'] = 5000
with open(sys.argv[1], 'w') as f: json.dump(c, f, indent=2)
" "$CFG" "$EPS_DECAY"
            MODEL_PATH="$MODELDIR/$(basename "$CFG" .json | sed 's/^config_//').pkl"
            run_experiment "$CFG" "$TASK" "$MODEL_PATH" "$WANDB_PROJECT"
        done
    done
done

# ---------------------------------------------------------------------------
# Shakespeare: depth-6, model_dim=256, state_dim=256, layer norm, glu, no PE, 50000 epochs, 5 seeds
# lm_cmru / lm_cmos_bmru use epsilon annealing; lm_lru / lm_mingru: none
# ---------------------------------------------------------------------------
LM_CFGDIR="$BASE/configs/generated/missing_results_lm"
mkdir -p "$LM_CFGDIR"

declare -A SEED1_CFGS
declare -A SEED1_BEST_PKLS

for SEED in "${SEEDS[@]}"; do
    for ARCH_EPS in "lm_cmru:1.0:true" "lm_cmos_bmru:1.0:true" "lm_lru:none:false" "lm_mingru:none:false"; do
        ARCH="${ARCH_EPS%%:*}"
        REST="${ARCH_EPS#*:}"
        EPS_ARG="${REST%%:*}"
        EPS_DECAY="${REST##*:}"
        CFG=$(python3 "$BASE/seeder.py" \
            "$BASE/configs/base/config_${ARCH}.json" \
            "$LM_CFGDIR" \
            fullshakespeare "$SEED" \
            none 256 layer 256 "$EPS_ARG" \
            none 50000 1e-4 1e-4 none none none 2>&1 | grep "^Created:" | cut -d' ' -f2)
        python3 -c "
import json, sys
with open(sys.argv[1]) as f: c = json.load(f)
c['model']['positional_encodings_dims'] = 0
c['early_stop_patience'] = 20000
if sys.argv[2] == 'true':
    c['epsilon_decay'] = True
    c['epsilon_const_epochs'] = 1000
    c['epsilon_decay_epochs'] = 5000
with open(sys.argv[1], 'w') as f: json.dump(c, f, indent=2)
" "$CFG" "$EPS_DECAY"
        MODEL_PATH="$MODELDIR/$(basename "$CFG" .json | sed 's/^config_//').pkl"
        run_experiment "$CFG" fullshakespeare "$MODEL_PATH" "$WANDB_PROJECT"
        if [[ "$SEED" == "1" ]]; then
            SEED1_CFGS[$ARCH]="$CFG"
            SEED1_BEST_PKLS[$ARCH]="${MODEL_PATH%.pkl}_best.pkl"
        fi
    done
done

finalize_dispatch

for ARCH_EPS in "lm_cmru:0.0" "lm_cmos_bmru:0.0" "lm_lru:none" "lm_mingru:none"; do
    ARCH="${ARCH_EPS%%:*}"
    dispatch_eval_cmd "gen_${ARCH}" \
        generate \
        --config        "${SEED1_CFGS[$ARCH]}" \
        --task          fullshakespeare \
        --model_path    "${SEED1_BEST_PKLS[$ARCH]}" \
        --wandb_project "$WANDB_PROJECT" \
        --gen_prompt    $'\n' \
        --gen_num_chars   1000 \
        --gen_temperature 0.8 \
        --gen_num_samples 3
done

echo "Table 1 missing results submitted."

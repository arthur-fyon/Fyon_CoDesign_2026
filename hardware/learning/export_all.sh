#!/bin/bash

# Loop from seed 45 to 95
for seed in $(seq 45 94); do
    python3 main.py export \
        --task real_audio_binary \
        --config config_cmos_4.json \
        --model_path models/real_audio_binary_cmos_4.pkl \
        --export_dir exports \
        --export_seed $seed
    
    sleep 1
done
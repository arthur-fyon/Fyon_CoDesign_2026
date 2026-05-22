#!/bin/bash

DIMS=(4 8 16 32 64 128 256 512)

for dim in "${DIMS[@]}"; do
    echo "=========================================="
    echo "Training and plotting: state_dim=$dim"
    echo "=========================================="
    
    python3 main.py train \
        --task real_audio_binary \
        --config config_cmos_${dim}.json \
        --model_path models/real_audio_binary_cmos_${dim}.pkl
    
    sleep 2
    
    python3 main.py plot_test \
        --task real_audio_binary \
        --config config_cmos_${dim}.json \
        --model_path models/real_audio_binary_cmos_${dim}.pkl
    
    sleep 2
done

echo "=========================================="
echo "All training complete!"
echo "=========================================="
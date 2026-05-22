#!/bin/bash
# Run quantization tests for all state dimensions and bit widths


STATE_DIMS=(4 8 16 32 64)
QUANT_BITS=(2 4 6 8)

echo "=========================================="
echo "Starting quantization sweep"
echo "State dimensions: ${STATE_DIMS[*]}"
echo "Quantization bits: ${QUANT_BITS[*]}"
echo "=========================================="

for dim in "${STATE_DIMS[@]}"; do
    for bits in "${QUANT_BITS[@]}"; do
        echo ""
        echo "=========================================="
        echo "Running: state_dim=$dim, quantize_bits=$bits"
        echo "=========================================="
        
        python3 main.py validate \
            --task real_audio_binary \
            --config config_cmos_${dim}.json \
            --model_path models/real_audio_binary_cmos_${dim}.pkl \
            --quantize_bits $bits \
            --no_confusion_matrix
        
        sleep 2
    done
done

echo ""
echo "=========================================="
echo "Quantization sweep complete!"
echo "=========================================="
#!/bin/bash
# Run validation multiple times for each state dimension to compute mean and variance
# Resumable: Will skip already completed trials if script is restarted

STATE_DIMS=(4 8 16 32 64 128 256 512)
NUM_TRIALS=100

OUTPUT_FILE="validation_results.csv"

# Create CSV header if file doesn't exist
if [ ! -f $OUTPUT_FILE ]; then
    echo "state_dim,trial,accuracy,loss" > $OUTPUT_FILE
fi

echo "=========================================="
echo "Starting validation sweep (RESUMABLE)"
echo "State dimensions: ${STATE_DIMS[*]}"
echo "Trials per model: $NUM_TRIALS"
echo "Results will be saved to: $OUTPUT_FILE"
echo "=========================================="

for dim in "${STATE_DIMS[@]}"; do
    echo ""
    echo "Testing state_dim=$dim..."
    
    for trial in $(seq 1 $NUM_TRIALS); do
        # Check if this trial already exists in the CSV
        if grep -q "^${dim},${trial}," $OUTPUT_FILE; then
            existing=$(grep "^${dim},${trial}," $OUTPUT_FILE | tail -1)
            acc=$(echo $existing | cut -d',' -f3)
            printf "  Trial %2d/%d: SKIPPED (already done, acc=%s)\n" $trial $NUM_TRIALS $acc
            continue
        fi
        
        # Run validation and capture only the accuracy/loss lines
        output=$(python3 main.py validate \
            --task real_audio_binary \
            --config config_cmos_${dim}.json \
            --model_path models/real_audio_binary_cmos_${dim}.pkl \
            --no_confusion_matrix 2>&1)
        
        # Extract accuracy and loss using grep and awk
        accuracy=$(echo "$output" | grep "Accuracy:" | tail -1 | awk '{print $2}')
        loss=$(echo "$output" | grep "Loss:" | tail -1 | awk '{print $2}')
        
        # Append to CSV
        echo "${dim},${trial},${accuracy},${loss}" >> $OUTPUT_FILE
        
        # Print progress (compact)
        printf "  Trial %2d/%d: acc=%.4f loss=%.4f\n" $trial $NUM_TRIALS $accuracy $loss
        
        sleep 1
    done
done

echo ""
echo "=========================================="
echo "Sweep complete! Results saved to $OUTPUT_FILE"
echo "=========================================="

# Compute and display summary statistics
echo ""
echo "=========================================="
echo "SUMMARY STATISTICS"
echo "=========================================="
printf "%-10s %-12s %-12s %-12s %-12s\n" "StateDim" "MeanAcc" "StdAcc" "MeanLoss" "StdLoss"
echo "----------------------------------------------------------"

for dim in "${STATE_DIMS[@]}"; do
    # Extract data for this dimension and compute stats using awk
    stats=$(grep "^${dim}," $OUTPUT_FILE | awk -F',' '
    {
        acc[NR] = $3
        loss[NR] = $4
        sum_acc += $3
        sum_loss += $4
        n++
    }
    END {
        if (n > 0) {
            mean_acc = sum_acc / n
            mean_loss = sum_loss / n
            
            for (i = 1; i <= n; i++) {
                var_acc += (acc[i] - mean_acc)^2
                var_loss += (loss[i] - mean_loss)^2
            }
            std_acc = sqrt(var_acc / n)
            std_loss = sqrt(var_loss / n)
            
            printf "%.4f %.4f %.4f %.4f %d", mean_acc, std_acc, mean_loss, std_loss, n
        } else {
            printf "N/A N/A N/A N/A 0"
        }
    }')
    
    mean_acc=$(echo $stats | awk '{print $1}')
    std_acc=$(echo $stats | awk '{print $2}')
    mean_loss=$(echo $stats | awk '{print $3}')
    std_loss=$(echo $stats | awk '{print $4}')
    count=$(echo $stats | awk '{print $5}')
    
    printf "%-10s %-12s %-12s %-12s %-12s (n=%s)\n" "$dim" "$mean_acc" "$std_acc" "$mean_loss" "$std_loss" "$count"
done

echo "=========================================="
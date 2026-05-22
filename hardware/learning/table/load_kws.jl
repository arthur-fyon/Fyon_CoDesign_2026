using CSV
using DataFrames
using Statistics

"""
    load_kws_results(filepath::String) -> (results, seeds, dims)

Load KWS binary results from CSV and return a 3D array.

# Returns
- `results`: 3D array of shape (n_seeds, n_dims, 100) containing test accuracies
- `seeds`: Vector of seed values (sorted)
- `dims`: Vector of state dimensions (sorted)

# Usage
```julia
results, seeds, dims = load_kws_results("KWS_binary.csv")
# Access: results[seed_idx, dim_idx, test_idx]
```
"""
function load_kws_results(filepath::String)
    # Read CSV
    df = CSV.read(filepath, DataFrame)
    
    # Extract unique seeds and dimensions
    seeds = sort(unique(df.seed))
    dims = sort(unique(df.state_dim))
    
    n_seeds = length(seeds)
    n_dims = length(dims)
    n_tests = 100
    
    # Create mapping for indexing
    seed_to_idx = Dict(s => i for (i, s) in enumerate(seeds))
    dim_to_idx = Dict(d => i for (i, d) in enumerate(dims))
    
    # Initialize results array
    results = fill(NaN, n_seeds, n_dims, n_tests)
    
    # Extract accuracy column names (test_seed1/accuracy to test_seed100/accuracy)
    acc_cols = ["test_seed$(i)/accuracy" for i in 1:100]
    
    # Fill the results array
    for row in eachrow(df)
        seed_idx = seed_to_idx[row.seed]
        dim_idx = dim_to_idx[row.state_dim]
        
        for (test_idx, col) in enumerate(acc_cols)
            val = row[col]
            # Handle Missing values by keeping NaN
            if !ismissing(val)
                results[seed_idx, dim_idx, test_idx] = val
            end
        end
    end
    
    return results, seeds, dims
end

# Example usage
if abspath(PROGRAM_FILE) == @__FILE__
    filepath = ARGS[1]  # or hardcode your path
    
    results, seeds, dims = load_kws_results(filepath)
    
    println("Results array shape: ", size(results))
    println("Seeds: ", seeds)
    println("Dimensions: ", dims)
    println()
    
    # Example: mean accuracy for each (seed, dim) combination
    println("Mean accuracy per (seed, dim):")
    for (i, s) in enumerate(seeds)
        for (j, d) in enumerate(dims)
            mean_acc = mean(filter(!isnan, results[i, j, :]))
            println("  seed=$s, dim=$d: $(round(mean_acc * 100, digits=2))%")
        end
    end
end

function print_result(results)
    # Mean and std over the 100 tests, ignoring NaN values
    mean_matrix = zeros(length(seeds), length(dims))
    std_matrix = zeros(length(seeds), length(dims))

    for i in 1:length(seeds)
        for j in 1:length(dims)
            vals = filter(!isnan, results[i, j, :])
            mean_matrix[i, j] = isempty(vals) ? NaN : mean(vals)
            std_matrix[i, j] = isempty(vals) ? NaN : std(vals)
        end
    end

    # For each dimension: mean across seeds (with min/max interval), ignoring NaN
    mean_of_means = [mean(filter(!isnan, mean_matrix[:, j])) for j in 1:length(dims)]
    min_of_means = [minimum(filter(!isnan, mean_matrix[:, j])) for j in 1:length(dims)]
    max_of_means = [maximum(filter(!isnan, mean_matrix[:, j])) for j in 1:length(dims)]

    mean_of_stds = [mean(filter(!isnan, std_matrix[:, j])) for j in 1:length(dims)]
    min_of_stds = [minimum(filter(!isnan, std_matrix[:, j])) for j in 1:length(dims)]
    max_of_stds = [maximum(filter(!isnan, std_matrix[:, j])) for j in 1:length(dims)]

    # Display results
    println("Per dimension (across 5 seeds):")
    println("-"^60)
    for (i, d) in enumerate(dims)
        println("dim=$d:")
        println("  Accuracy: $(round(mean_of_means[i]*100, digits=2))% [$(round(min_of_means[i]*100, digits=2)) - $(round(max_of_means[i]*100, digits=2))]")
        println("  Std:      $(round(mean_of_stds[i]*100, digits=3))% [$(round(min_of_stds[i]*100, digits=3)) - $(round(max_of_stds[i]*100, digits=3))]")
    end
end
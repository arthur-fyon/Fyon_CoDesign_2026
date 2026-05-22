#!/usr/bin/env julia
"""
Paper-quality visualization of neural network classification results.
Creates a 3-panel figure showing output currents, predicted class, and confidence.
Currents are in nA (1 ML unit = 1 nA).
"""

using Plots
using Printf
using LaTeXStrings

# Use GR backend with better defaults for publication
gr(
    fontfamily="Computer Modern",
    titlefontsize=12,
    guidefontsize=10,
    tickfontsize=9,
    legendfontsize=9,
    linewidth=1.5,
    framestyle=:box,
    grid=true,
    minorgrid=false,
    gridlinewidth=0.5,
    gridalpha=0.3,
    foreground_color_grid=:gray70,
    dpi=300
)

"""
Compute softmax probabilities from output currents.
Uses currents as logits for classification.
"""
function softmax(logits::Matrix{Float64})
    # logits (currents) is 2×T, compute softmax along first dimension
    probs = similar(logits)
    for t in 1:size(logits, 2)
        exp_logits = exp.(logits[:, t] .- maximum(logits[:, t]))
        probs[:, t] = exp_logits ./ sum(exp_logits)
    end
    return probs
end

"""
Create paper-quality figure for classification results.

Parameters:
- logits: 2×T matrix of output currents in nA (row 1 = "no", row 2 = "yes")
- time: time vector (length T) - if nothing, computed from sampling_rate_hz
- sampling_rate_hz: sampling rate in Hz (default 100, only used if time=nothing)
- save_path: path to save figure (default "classification_results.pdf")
- title: optional overall title
- show_plot: whether to display the plot (default true)
"""
function plot_classification_results(
    logits::Matrix{Float64};
    time::Union{Vector{Float64}, Nothing}=nothing,
    sampling_rate_hz::Float64=100.0,
    save_path::String="classification_results.pdf",
    title::String="",
    show_plot::Bool=true,
    figsize::Tuple{Int,Int}=(800, 600)
)
    num_classes, num_timesteps = size(logits)

    # Time axis
    if time === nothing
        time_s = (0:num_timesteps-1) ./ sampling_rate_hz
    else
        time_s = time
    end

    # Compute probabilities and predictions
    probs = softmax(logits)
    predictions = [argmax(logits[:, t]) - 1 for t in 1:num_timesteps]  # 0 or 1
    confidence = [maximum(probs[:, t]) for t in 1:num_timesteps]

    # Define colors (colorblind-friendly)
    color_class0 = RGB(0.0, 0.447, 0.698)  # Blue
    color_class1 = RGB(0.835, 0.369, 0.0)  # Orange
    color_conf = RGB(0.337, 0.706, 0.314)  # Green
    color_pred = RGB(0.8, 0.2, 0.2)        # Red

    # Create 3-panel figure
    p1 = plot(
        time_s, logits[1, :],
        label="Class 0 (No)",
        color=color_class0,
        linewidth=1.5,
        ylabel="Current (nA)",
        title="(a) Output Currents",
        titleloc=:left,
        legend=:topright,
        xlims=(minimum(time_s), maximum(time_s)),
        xformatter=_->"",  # Hide x labels for top panels
    )
    plot!(p1, time_s, logits[2, :],
        label="Class 1 (Yes)",
        color=color_class1,
        linewidth=1.5
    )
    hline!(p1, [0], color=:gray50, linestyle=:dash, linewidth=0.8, label="")

    # Compute majority vote (weighted by time intervals for non-uniform dt)
    # For each class, sum the time spent in that class
    time_class0 = 0.0
    time_class1 = 0.0
    for t in 1:num_timesteps-1
        dt = time_s[t+1] - time_s[t]
        if predictions[t] == 0
            time_class0 += dt
        else
            time_class1 += dt
        end
    end
    display("Majority vote output:")
    display(time_class1)
    majority_class = time_class1 > time_class0 ? 1 : 0
    majority_label = majority_class == 1 ? "Yes" : "No"

    # Panel 2: Predicted class
    p2 = plot(
        time_s, predictions,
        seriestype=:steppost,
        color=color_pred,
        linewidth=1.8,
        fill=(0, 0.2, color_pred),
        ylabel="Predicted class",
        title="(b) Classification Output",
        titleloc=:left,
        legend=false,
        yticks=([0, 1], ["No (0)", "Yes (1)"]),
        ylims=(-0.1, 1.1),
        xlims=(minimum(time_s), maximum(time_s)),
        xformatter=_->"",
    )
    # Add majority vote annotation
    annotate!(p2, maximum(time_s) * 0.98, 0.5,
        text("Predicted: $majority_label", 9, :right, :gray30))

    # Panel 3: Confidence (probability of predicted class)
    p3 = plot(
        time_s, confidence,
        color=color_conf,
        linewidth=1.5,
        fill=(0.5, 0.25, color_conf),
        ylabel="Confidence",
        xlabel="Time (s)",
        title="(c) Prediction Confidence",
        titleloc=:left,
        legend=false,
        ylims=(0.45, 1.02),
        xlims=(minimum(time_s), maximum(time_s)),
    )
    hline!(p3, [0.5], color=:gray50, linestyle=:dash, linewidth=0.8, label="")

    # Combine panels
    fig = plot(p1, p2, p3,
        layout=(3, 1),
        size=figsize,
        left_margin=5Plots.mm,
        right_margin=3Plots.mm,
        top_margin=2Plots.mm,
        bottom_margin=3Plots.mm,
    )

    # Add overall title if provided
    if !isempty(title)
        plot!(fig, plot_title=title, plot_titlefontsize=14)
    end

    # Save figure
    savefig(fig, save_path)
    println("✓ Figure saved to: $save_path")

    # Also save as PNG for quick viewing
    png_path = replace(save_path, r"\.[^.]+$" => ".png")
    savefig(fig, png_path)
    println("✓ Figure saved to: $png_path")

    # Print summary statistics
    println("\n" * "="^50)
    println("CLASSIFICATION SUMMARY")
    println("="^50)
    println("Total timesteps: $num_timesteps")
    println("Duration: $(round(maximum(time_s), digits=2)) s")
    println("")

    num_class0 = count(==(0), predictions)
    num_class1 = count(==(1), predictions)
    @printf("Class 0 (No):  %d timesteps (%.1f%%)\n", num_class0, 100*num_class0/num_timesteps)
    @printf("Class 1 (Yes): %d timesteps (%.1f%%)\n", num_class1, 100*num_class1/num_timesteps)
    println("")
    @printf("Mean confidence: %.3f\n", mean(confidence))
    @printf("Min confidence:  %.3f\n", minimum(confidence))
    @printf("Max confidence:  %.3f\n", maximum(confidence))

    # Find transitions
    transitions = findall(diff(predictions) .!= 0)
    if !isempty(transitions)
        println("\nClass transitions at:")
        for t in transitions
            @printf("  t = %.3f s (step %d): %d → %d\n",
                time_s[t], t, predictions[t], predictions[t+1])
        end
    end

    if show_plot
        display(fig)
    end

    return fig
end

"""
Alternative: Create a more compact 2-panel figure.
"""
function plot_classification_compact(
    logits::Matrix{Float64};
    time::Union{Vector{Float64}, Nothing}=nothing,
    sampling_rate_hz::Float64=100.0,
    save_path::String="classification_compact.pdf",
    figsize::Tuple{Int,Int}=(700, 400)
)
    num_classes, num_timesteps = size(logits)

    # Time axis
    if time === nothing
        time_s = (0:num_timesteps-1) ./ sampling_rate_hz
    else
        time_s = time
    end

    probs = softmax(logits)
    predictions = [argmax(logits[:, t]) - 1 for t in 1:num_timesteps]

    color_class0 = RGB(0.0, 0.447, 0.698)
    color_class1 = RGB(0.835, 0.369, 0.0)

    # Panel 1: Currents with shaded prediction regions
    p1 = plot(
        time_s, logits[1, :],
        label="Class 0 (No)",
        color=color_class0,
        linewidth=1.5,
        ylabel="Current (nA)",
        title="Output Currents",
        legend=:topright,
        xlims=(minimum(time_s), maximum(time_s)),
        xformatter=_->"",
    )
    plot!(p1, time_s, logits[2, :],
        label="Class 1 (Yes)",
        color=color_class1,
        linewidth=1.5
    )

    # Add shaded regions for predictions
    for t in 1:num_timesteps-1
        if predictions[t] == 1
            vspan!(p1, [time_s[t], time_s[t+1]],
                color=color_class1, alpha=0.1, label="")
        end
    end

    # Panel 2: Softmax probabilities
    p2 = plot(
        time_s, probs[2, :],
        color=color_class1,
        linewidth=1.5,
        fill=(0, 0.2, color_class1),
        ylabel="P(Yes)",
        xlabel="Time (s)",
        title="Classification Probability",
        legend=false,
        ylims=(0, 1.02),
        xlims=(minimum(time_s), maximum(time_s)),
    )
    hline!(p2, [0.5], color=:gray50, linestyle=:dash, linewidth=0.8)

    fig = plot(p1, p2,
        layout=(2, 1),
        size=figsize,
        left_margin=5Plots.mm,
        right_margin=3Plots.mm,
    )

    savefig(fig, save_path)
    println("✓ Figure saved to: $save_path")

    png_path = replace(save_path, r"\.[^.]+$" => ".png")
    savefig(fig, png_path)
    println("✓ Figure saved to: $png_path")

    return fig
end

"""
Create a single-panel figure showing only probabilities (most compact).
"""
function plot_probability_only(
    logits::Matrix{Float64};
    time::Union{Vector{Float64}, Nothing}=nothing,
    sampling_rate_hz::Float64=100.0,
    save_path::String="classification_probability.pdf",
    figsize::Tuple{Int,Int}=(600, 250)
)
    num_timesteps = size(logits, 2)

    # Time axis
    if time === nothing
        time_s = (0:num_timesteps-1) ./ sampling_rate_hz
    else
        time_s = time
    end

    probs = softmax(logits)

    color_yes = RGB(0.835, 0.369, 0.0)

    fig = plot(
        time_s, probs[2, :],
        color=color_yes,
        linewidth=1.8,
        fill=(0, 0.25, color_yes),
        ylabel="P(Keyword)",
        xlabel="Time (s)",
        legend=false,
        ylims=(0, 1.02),
        xlims=(minimum(time_s), maximum(time_s)),
        size=figsize,
        left_margin=5Plots.mm,
        bottom_margin=4Plots.mm,
    )
    hline!(fig, [0.5], color=:gray50, linestyle=:dash, linewidth=1.0)

    # Add annotations for high-confidence regions
    threshold = 0.8
    in_detection = false
    detection_start = 0.0

    for t in 1:num_timesteps
        if probs[2, t] > threshold && !in_detection
            in_detection = true
            detection_start = time_s[t]
        elseif probs[2, t] <= threshold && in_detection
            in_detection = false
            detection_end = time_s[t]
            mid = (detection_start + detection_end) / 2
            annotate!(fig, mid, 0.95, text("detected", 8, :gray40))
        end
    end

    savefig(fig, save_path)
    println("✓ Figure saved to: $save_path")

    return fig
end

"""
    plot_classification_MC(
        full_data_out1, full_data_out2, full_data_out3, full_data_out4,
        full_data_pos1, full_data_pos2, full_data_pos3, full_data_pos4,
        full_data_neg1, full_data_neg2, full_data_neg3, full_data_neg4;
        w_class, n_mc, save_path, figsize
    )

Plot all MC samples overlaid + a panel showing the classification vote
breakdown (% Yes vs % No) at each time step.
"""
function plot_classification_MC(
    all_logits, all_times, votes;
    w_class::Matrix{Float64} = [0.7664272785 0.6018738747 0.0906422287 0.0430877917;
                                 0.0619407333 -0.7988973260 -0.6309837103 0.5674794912],
    n_mc::Int = 200,
    alpha::Float64 = 0.06,
    save_path::String = "classification_MC.pdf",
    figsize::Tuple{Int,Int} = (800, 400)
)
    n_valid = length(votes)
    n_yes = count(==(1), votes)
    n_no  = count(==(0), votes)
    println("Valid MC samples: $n_valid / $n_mc")
    println("Classification: $n_yes Yes, $n_no No")

    # Colors
    color_no  = myRedNeurIPS
    color_yes = myGreenNeurIPS

    # --- Panel 1: all MC logit traces overlaid ---
    p1 = plot(
        titleloc=:left,
        ylabel=L"I_{yes}/I_{other}\,\,\mathrm{(nA)}",
        xlabel=L"\mathrm{Time}\,\,\mathrm{(s)}",
        xticks=[0, 0.2, 0.4, 0.6, 0.8, 1],
        legend=:topright,
    )
    # Dummy series for legend (full opacity)
    plot!(p1, [NaN], [NaN], color=color_no, linewidth=1.5, label=L"\mathrm{Class}\,\,{0}\,\,\mathrm{(No)}")
    plot!(p1, [NaN], [NaN], color=color_yes, linewidth=1.5, label=L"\mathrm{Class}\,\,{1}\,\,\mathrm{(Yes)}")
    for k = 1 : n_valid
        t_k = all_times[k, :]
        t_k = Float64[x for x in t_k if x != ""]

        logits_k = all_logits[k, :]
        logits_k = Float64[x for x in logits_k if x != ""]
        logits1 = logits_k[1:2:end]
        logits2 = logits_k[2:2:end]

        plot!(p1, t_k, logits1,
              color=color_no, alpha=alpha, linewidth=0.4, label="")
        plot!(p1, t_k, logits2,
              color=color_yes, alpha=alpha, linewidth=0.4, label="")
    end
    hline!(p1, [0], color=:gray50, linestyle=:dash, linewidth=0.8, label="")

    # --- Panel 2: bar chart of final votes ---
    p2 = bar(
        [L"\mathrm{No}", L"\mathrm{Yes}"], [n_no, n_yes],
        color=[color_no, color_yes],
        ylabel=L"\mathrm{MC}\,\,\mathrm{samples}",
        titleloc=:left,
        legend=false,
        ylims=(0, n_valid * 1.15),
        bar_width=0.5,
    )
    # annotate!(p2, 1, n_no + n_valid * 0.04,
        # text(@sprintf("%d (%.1f%%)", n_no, 100 * n_no / n_valid), 10, :center))
    # annotate!(p2, 2, n_yes + n_valid * 0.04,
        # text(@sprintf("%d (%.1f%%)", n_yes, 100 * n_yes / n_valid), 10, :center))

    # Combine
    fig = plot(p1, p2,
        layout=grid(2, 1),
        size=figsize,
        left_margin=5Plots.mm,
        right_margin=5Plots.mm,
        top_margin=2Plots.mm,
        bottom_margin=3Plots.mm,
        dpi=450,
    )

    # savefig(fig, save_path)
    # println("✓ Figure saved to: $save_path")
    png_path = replace(save_path, r"\.[^.]+$" => ".png")
    savefig(fig, png_path)
    println("✓ Figure saved to: $png_path")

    # display(fig)
    return fig
end

# Example usage and main execution
if abspath(PROGRAM_FILE) == @__FILE__
    # Demo with sample data
    println("Generating demo figure...")

    # Create sample logits (simulate a keyword detection scenario)
    T = 500
    logits = zeros(2, T)

    # Class 0 dominant at start, Class 1 spike in middle
    for t in 1:T
        if 200 < t < 350
            logits[1, t] = -1.0 + 0.3*randn()
            logits[2, t] = 2.0 + 0.3*randn()
        else
            logits[1, t] = 1.5 + 0.2*randn()
            logits[2, t] = -2.0 + 0.2*randn()
        end
    end

    # Generate figures
    plot_classification_results(logits; save_path="demo_classification.pdf")
    plot_classification_compact(logits; save_path="demo_compact.pdf")
    plot_probability_only(logits; save_path="demo_probability.pdf")
end

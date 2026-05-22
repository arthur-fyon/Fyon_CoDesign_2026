#!/usr/bin/env julia
"""
Paper-quality visualization of power consumption and energy during inference.
Power in nW, Energy in nJ.
"""

using Plots
using Printf

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
Compute cumulative energy from power using trapezoidal integration.
P in nW, t in seconds → E in nJ
"""
function cumulative_energy(P::Vector{Float64}, t::Vector{Float64})
    E = zeros(length(t))
    for i in 2:length(t)
        dt = t[i] - t[i-1]
        # Trapezoidal rule: area = (P[i] + P[i-1]) / 2 * dt
        E[i] = E[i-1] + (P[i] + P[i-1]) / 2 * dt
    end
    return E  # in nJ (nW × s = nJ)
end

"""
Create paper-quality figure for power consumption and energy.

Parameters:
- P: power vector in nW (length T)
- t: time vector in seconds (length T)
- save_path: path to save figure (default "power_energy.pdf")
- title: optional overall title
- show_plot: whether to display the plot (default false)
"""
function plot_power_energy(
    P::Vector{Float64},
    t::Vector{Float64};
    save_path::String="power_energy.pdf",
    title::String="",
    show_plot::Bool=false,
    figsize::Tuple{Int,Int}=(800, 500)
)
    # Compute cumulative energy
    E = cumulative_energy(P, t)
    total_energy = E[end]

    # Statistics
    mean_power = sum((P[1:end-1] .+ P[2:end]) ./ 2 .* diff(t)) / (t[end] - t[1])
    peak_power = maximum(P)
    min_power = minimum(P)

    # Colors
    color_power = RGB(0.0, 0.447, 0.698)   # Blue
    color_energy = RGB(0.835, 0.369, 0.0)  # Orange

    # Panel 1: Power vs time
    p1 = plot(
        t, P,
        color=color_power,
        linewidth=1.2,
        fill=(0, 0.2, color_power),
        ylabel="Power (nW)",
        title="(a) Instantaneous Power",
        titleloc=:left,
        legend=false,
        xlims=(minimum(t), maximum(t)),
        xformatter=_->"",
    )

    # Add mean power line
    hline!(p1, [mean_power], color=:gray40, linestyle=:dash, linewidth=1.2, label="")

    # Annotate mean power
    annotate!(p1, maximum(t) * 0.98, mean_power * 1.1,
        text(@sprintf("mean = %.2f nW", mean_power), 8, :right, :gray40))

    # Panel 2: Cumulative energy
    p2 = plot(
        t, E,
        color=color_energy,
        linewidth=1.8,
        fill=(0, 0.2, color_energy),
        ylabel="Cumulative Energy (nJ)",
        xlabel="Time (s)",
        title="(b) Energy Consumption",
        titleloc=:left,
        legend=false,
        xlims=(minimum(t), maximum(t)),
        ylims=(0, maximum(E) * 1.05),
    )

    # Annotate total energy
    annotate!(p2, maximum(t) * 0.98, total_energy * 0.9,
        text(@sprintf("Total = %.3f nJ", total_energy), 9, :right, color_energy))

    # Combine panels
    fig = plot(p1, p2,
        layout=(2, 1),
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

    # Also save as PNG
    png_path = replace(save_path, r"\.[^.]+$" => ".png")
    savefig(fig, png_path)
    println("✓ Figure saved to: $png_path")

    # Print summary statistics
    println("\n" * "="^50)
    println("POWER & ENERGY SUMMARY")
    println("="^50)
    @printf("Inference duration:  %.4f s\n", t[end] - t[1])
    @printf("Number of samples:   %d\n", length(t))
    println("")
    println("POWER:")
    @printf("  Mean power:        %.4f nW\n", mean_power)
    @printf("  Peak power:        %.4f nW\n", peak_power)
    @printf("  Min power:         %.4f nW\n", min_power)
    println("")
    println("ENERGY:")
    @printf("  Total energy:      %.6f nJ\n", total_energy)
    @printf("  Total energy:      %.6f pJ\n", total_energy * 1000)

    # Energy per inference (assuming single inference)
    println("")
    println("EFFICIENCY METRICS:")
    @printf("  Energy/inference:  %.6f nJ\n", total_energy)
    @printf("  Energy/inference:  %.3f pJ\n", total_energy * 1000)

    if show_plot
        display(fig)
    end

    return fig, total_energy
end

"""
Create a compact single-panel power plot with energy annotation.
"""
function plot_power_compact(
    P::Vector{Float64},
    t::Vector{Float64};
    save_path::String="power_compact.pdf",
    figsize::Tuple{Int,Int}=(600, 300)
)
    E = cumulative_energy(P, t)
    total_energy = E[end]
    mean_power = sum((P[1:end-1] .+ P[2:end]) ./ 2 .* diff(t)) / (t[end] - t[1])

    color_power = RGB(0.0, 0.447, 0.698)

    fig = plot(
        t, P,
        color=color_power,
        linewidth=1.2,
        fill=(0, 0.25, color_power),
        ylabel="Power (nW)",
        xlabel="Time (s)",
        legend=false,
        xlims=(minimum(t), maximum(t)),
        size=figsize,
        left_margin=5Plots.mm,
        bottom_margin=4Plots.mm,
        right_margin=8Plots.mm,
    )

    # Add mean line
    hline!(fig, [mean_power], color=:gray40, linestyle=:dash, linewidth=1.0)

    # Add text box with stats
    x_pos = maximum(t) * 0.98
    y_pos = maximum(P) * 0.95
    annotate!(fig, x_pos, y_pos,
        text(@sprintf("E = %.3f nJ\nP̄ = %.2f nW", total_energy, mean_power),
             9, :right, :gray30))

    savefig(fig, save_path)
    println("✓ Figure saved to: $save_path")

    png_path = replace(save_path, r"\.[^.]+$" => ".png")
    savefig(fig, png_path)
    println("✓ Figure saved to: $png_path")

    return fig, total_energy
end

"""
Create a 3-panel figure with power, cumulative energy, and energy breakdown.
"""
function plot_power_detailed(
    P::Vector{Float64},
    t::Vector{Float64};
    save_path::String="power_detailed.pdf",
    figsize::Tuple{Int,Int}=(800, 650)
)
    E = cumulative_energy(P, t)
    total_energy = E[end]
    mean_power = sum((P[1:end-1] .+ P[2:end]) ./ 2 .* diff(t)) / (t[end] - t[1])

    # Colors
    color_power = RGB(0.0, 0.447, 0.698)
    color_energy = RGB(0.835, 0.369, 0.0)
    color_hist = RGB(0.337, 0.706, 0.314)

    # Panel 1: Power vs time
    p1 = plot(
        t, P,
        color=color_power,
        linewidth=1.2,
        fill=(0, 0.2, color_power),
        ylabel="Power (nW)",
        title="(a) Instantaneous Power",
        titleloc=:left,
        legend=false,
        xlims=(minimum(t), maximum(t)),
        xformatter=_->"",
    )
    hline!(p1, [mean_power], color=:gray40, linestyle=:dash, linewidth=1.0)

    # Panel 2: Cumulative energy
    p2 = plot(
        t, E,
        color=color_energy,
        linewidth=1.8,
        fill=(0, 0.2, color_energy),
        ylabel="Cumulative Energy (nJ)",
        title="(b) Energy Consumption",
        titleloc=:left,
        legend=false,
        xlims=(minimum(t), maximum(t)),
        xformatter=_->"",
    )

    # Panel 3: Power histogram
    p3 = histogram(
        P,
        bins=50,
        color=color_hist,
        alpha=0.7,
        xlabel="Power (nW)",
        ylabel="Count",
        title="(c) Power Distribution",
        titleloc=:left,
        legend=false,
    )
    vline!(p3, [mean_power], color=:gray40, linestyle=:dash, linewidth=1.5)

    # Combine panels
    fig = plot(p1, p2, p3,
        layout=(3, 1),
        size=figsize,
        left_margin=5Plots.mm,
        right_margin=3Plots.mm,
    )

    savefig(fig, save_path)
    println("✓ Figure saved to: $save_path")

    png_path = replace(save_path, r"\.[^.]+$" => ".png")
    savefig(fig, png_path)
    println("✓ Figure saved to: $png_path")

    # Print stats
    println("\n" * "="^50)
    println("POWER & ENERGY SUMMARY")
    println("="^50)
    @printf("Total energy:    %.6f nJ (%.3f pJ)\n", total_energy, total_energy * 1000)
    @printf("Mean power:      %.4f nW\n", mean_power)
    @printf("Peak power:      %.4f nW\n", maximum(P))
    @printf("Std power:       %.4f nW\n", std(P))

    return fig, total_energy
end

# Add std function if not available
function std(x)
    m = sum(x) / length(x)
    return sqrt(sum((x .- m).^2) / (length(x) - 1))
end

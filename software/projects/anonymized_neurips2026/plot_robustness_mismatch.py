#!/usr/bin/env python3
"""
Publication-ready robustness-to-parameter-mismatch figures.

Reads *_robustness.json files written by `main.py robustness` and produces:

  robustness_mismatch.pdf   — 1 × 3 panel (one subplot per task):
        x-axis : parameter noise level  (0 %, 5 %, 10 %, 15 %)
        y-axis : classification accuracy (%)
        lines  : one per architecture, ±1σ shaded band over N trials

Usage (from the project root):
    python3 plot_robustness_mismatch.py
    python3 plot_robustness_mismatch.py --results_dir models/trained \\
                                        --output_dir  plots/robustness_mismatch
"""

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D


# ─── Visual identity ─────────────────────────────────────────────────────────

ARCH_STYLE: Dict[str, dict] = {
    "cmos_bmru": dict(color="#d62728", marker="o", label="FQ BMRU", zorder=4),
    "mingru":    dict(color="#1f77b4", marker="s", label="minGRU",    zorder=3),
    "lru":       dict(color="#2ca02c", marker="^", label="LRU",       zorder=2),
}

TASK_LABEL: Dict[str, str] = {
    "smnist_cmos":       "sMNIST",
    "pmnist_cmos":       "pMNIST",
    "real_audio_digits": "KWS Digits",
    "real_audio_binary": "KWS Binary",
}

# Publication-quality matplotlib settings
RC = {
    "font.family":        "serif",
    "font.size":          10,
    "axes.labelsize":     11,
    "axes.titlesize":     12,
    "axes.titleweight":   "bold",
    "xtick.labelsize":    9,
    "ytick.labelsize":    9,
    "legend.fontsize":    9,
    "figure.dpi":         150,
    "lines.linewidth":    1.8,
    "lines.markersize":   5.5,
    "axes.spines.top":    False,
    "axes.spines.right":  False,
}


# ─── Data loading ─────────────────────────────────────────────────────────────

def load_results(
    results_dir: Path,
    tasks: Optional[Set[str]] = None,
    archs: Optional[Set[str]] = None,
) -> Dict[str, Dict[str, dict]]:
    """
    Glob all *_robustness.json files under results_dir and return:
        data[task][arch] = {
            'noise_pcts' : [0.0, 5.0, 10.0, 15.0],   # includes nominal (0)
            'mean_accs'  : [...],                       # as fractions
            'std_accs'   : [...],
            'n_trials'   : int,
        }
    If multiple seeds are present for the same (task, arch), their per-trial
    accuracy lists are pooled before computing mean/std.
    """
    # Pool per-trial accuracies across seeds: data_raw[task][arch][noise_key] = [acc, ...]
    data_raw: Dict[str, Dict[str, Dict[str, List[float]]]] = defaultdict(
        lambda: defaultdict(lambda: defaultdict(list))
    )
    nominal_raw: Dict[str, Dict[str, List[float]]] = defaultdict(lambda: defaultdict(list))
    n_trials_map: Dict[str, Dict[str, int]] = defaultdict(dict)

    for p in sorted(results_dir.glob("**/*_robustness.json")):
        try:
            with open(p) as f:
                rec = json.load(f)
        except Exception as e:
            print(f"  WARNING: could not read {p}: {e}")
            continue

        task = rec.get("task")
        arch = rec.get("arch")
        if not task or not arch:
            continue
        if tasks and task not in tasks:
            continue
        if archs and arch not in archs:
            continue

        nominal_raw[task][arch].append(rec["nominal_accuracy"])
        n_trials_map[task][arch] = rec.get("n_trials", 10)

        for noise_key, res in rec["results"].items():
            # Each JSON result may contain the per-trial list; fall back to
            # reconstructing a single synthetic trial from mean if absent.
            per_trial: List[float] = res.get("accuracies") or [res["mean_accuracy"]]
            data_raw[task][arch][noise_key].extend(per_trial)

    # Build the final structure
    data: Dict[str, Dict[str, dict]] = defaultdict(dict)

    for task, arch_map in data_raw.items():
        for arch, noise_map in arch_map.items():
            noise_keys_sorted: List[str] = sorted(noise_map.keys(), key=float)

            mean_accs = [float(np.mean(nominal_raw[task][arch]))]
            std_accs  = [0.0]  # nominal has no noise variance
            noise_pcts = [0.0]

            for nk in noise_keys_sorted:
                trials = noise_map[nk]
                mean_accs.append(float(np.mean(trials)))
                std_accs.append(float(np.std(trials)))
                noise_pcts.append(float(nk) * 100.0)

            data[task][arch] = {
                "noise_pcts": noise_pcts,
                "mean_accs":  mean_accs,
                "std_accs":   std_accs,
                "n_trials":   n_trials_map[task][arch],
            }

    return data


# ─── Plotting ─────────────────────────────────────────────────────────────────

def _plot_panel(
    ax: plt.Axes,
    task: str,
    task_data: Dict[str, dict],
    arch_order: List[str],
    show_ylabel: bool,
    x_max: float = 100.0,
) -> None:
    """Draw one subplot for a single task."""
    for arch in arch_order:
        if arch not in task_data:
            continue
        style = ARCH_STYLE.get(arch, {})
        d = task_data[arch]

        xs    = [v / 10.0 for v in d["noise_pcts"]]
        means = np.array(d["mean_accs"]) * 100.0   # → percentage
        stds  = np.array(d["std_accs"])  * 100.0

        ax.plot(
            xs, means,
            marker=style.get("marker", "o"),
            color=style.get("color", "k"),
            zorder=style.get("zorder", 2),
            label=style.get("label", arch),
        )
        ax.fill_between(
            xs,
            means - stds,
            means + stds,
            alpha=0.15,
            color=style.get("color", "gray"),
            linewidth=0,
            zorder=style.get("zorder", 2) - 1,
        )

    ax.set_title(TASK_LABEL.get(task, task))
    ax.set_xlabel(r"Noise level ($\times$ measured analog noise)")
    if show_ylabel:
        ax.set_ylabel("Accuracy (%)")
    else:
        ax.tick_params(labelleft=False)
    ax.set_ylim(0, 100)
    if x_max <= 3:
        tick_candidates = [0, 0.5, 1, 2, 3, 4, 5]
    else:
        tick_candidates = list(range(0, int(x_max) + 1))
    ax.set_xticks([t for t in tick_candidates if t <= x_max])
    ax.xaxis.set_major_formatter(mticker.FormatStrFormatter("%g"))
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.0f"))
    ax.grid(axis="y", linestyle="--", alpha=0.35, zorder=0)
    ax.set_xlim(-x_max * 0.01, x_max * 1.01)


def plot_robustness(
    data: Dict[str, Dict[str, dict]],
    tasks: List[str],
    archs: List[str],
    output_path: Path,
    x_max: float = 100.0,
) -> None:
    """Generate and save the publication-ready figure."""
    matplotlib.rcParams.update(RC)

    present_tasks = [t for t in tasks if t in data]
    if not present_tasks:
        print("No data found for the requested tasks.")
        return

    n = len(present_tasks)
    fig, axes = plt.subplots(
        1, n,
        figsize=(3.5 * n + 0.4, 3.4),
        sharey=True,
        squeeze=False,
    )
    axes = axes[0]

    for col, task in enumerate(present_tasks):
        _plot_panel(
            axes[col],
            task,
            data[task],
            archs,
            show_ylabel=(col == 0),
            x_max=x_max,
        )

    # Build a custom legend using present architectures
    legend_handles = []
    for arch in archs:
        if not any(arch in data[t] for t in present_tasks):
            continue
        style = ARCH_STYLE.get(arch, {})
        legend_handles.append(
            Line2D(
                [0], [0],
                color=style.get("color", "k"),
                marker=style.get("marker", "o"),
                linewidth=RC["lines.linewidth"],
                markersize=RC["lines.markersize"],
                label=style.get("label", arch),
            )
        )

    fig.legend(
        handles=legend_handles,
        loc="lower center",
        ncol=len(legend_handles),
        frameon=True,
        framealpha=0.92,
        edgecolor="lightgray",
        bbox_to_anchor=(0.5, -0.14),
    )

    fig.tight_layout(rect=[0, 0.05, 1, 1])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {output_path}")


# ─── Entry point ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot robustness-to-parameter-mismatch results."
    )
    parser.add_argument(
        "--results_dir", default="models/trained",
        help="Directory (recursive) to search for *_robustness.json files.",
    )
    parser.add_argument(
        "--output_dir", default="plots/robustness_mismatch",
        help="Output directory for the PDF figure.",
    )
    parser.add_argument(
        "--tasks", nargs="+",
        default=["smnist_cmos", "pmnist_cmos", "real_audio_digits", "real_audio_binary"],
        help="Task names to include (in subplot order).",
    )
    parser.add_argument(
        "--archs", nargs="+",
        default=["cmos_bmru", "mingru", "lru"],
        help="Architecture names to include (in legend order).",
    )
    args = parser.parse_args()

    results_dir = Path(args.results_dir)
    output_dir  = Path(args.output_dir)

    print(f"Loading results from: {results_dir}")
    data = load_results(results_dir, set(args.tasks), set(args.archs))

    if not data:
        print("No *_robustness.json files found. Run the robustness experiments first.")
        return

    # Summary
    for task, arch_map in data.items():
        for arch, d in arch_map.items():
            nominal = d["mean_accs"][0] * 100
            worst   = d["mean_accs"][-1] * 100
            print(f"  {task:22s}  {arch:12s}  "
                  f"nominal={nominal:.2f}%  "
                  f"@{d['noise_pcts'][-1]:.0f}%noise={worst:.2f}%  "
                  f"(N={d['n_trials']})")

    # Full range (0–10× measured analog noise)
    plot_robustness(data, args.tasks, args.archs,
                    output_dir / "robustness_mismatch.pdf",
                    x_max=10.0)

    # Zoomed view (0–2× measured analog noise)
    plot_robustness(data, args.tasks, args.archs,
                    output_dir / "robustness_mismatch_zoom.pdf",
                    x_max=2.0)

    print("Done.")


if __name__ == "__main__":
    main()

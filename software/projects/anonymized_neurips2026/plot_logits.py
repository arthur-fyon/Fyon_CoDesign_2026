#!/usr/bin/env python3
"""
Generate publication-ready PDF plots of CMOS BMRU logits over time.

For each of N test samples:
  - x axis : time step (0 … T-1)
  - y axis : raw logit value (pre-softmax)
  - one coloured line per class
  - dashed horizontal line at the per-sample mean logit, which is the
    "equal-logit" level: when all logits equal this value the softmax
    output is uniform (random-chance prediction).

Usage:
    python plot_logits.py \\
        --config   configs/generated/config_cmos_bmru_real_audio_digits_seed1_…json \\
        --model_path models/trained/cmos_bmru_real_audio_digits_seed1_…_best.pkl \\
        --task     real_audio_digits \\
        --plot_dir plots/cmos_kws \\
        --num_plots 16
"""

import argparse
import json
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from benchmarks.registry import BENCHMARK_TASKS
from mru.training.checkpointing import load_model


# ── Publication style ──────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":       "serif",
    "font.size":         10,
    "axes.labelsize":    11,
    "axes.titlesize":    10,
    "legend.fontsize":   7.5,
    "xtick.labelsize":   9,
    "ytick.labelsize":   9,
    "axes.grid":         True,
    "grid.linewidth":    0.4,
    "grid.alpha":        0.35,
    "lines.linewidth":   1.1,
    "figure.dpi":        150,
})

# ── KWS Digits class metadata ─────────────────────────────────────────────────
CLASS_NAMES = [
    "silence", "zero", "one", "two", "three", "four",
    "five",    "six",  "seven", "eight", "nine",
]
NUM_CLASSES = len(CLASS_NAMES)  # 11

# 11 distinct colours: pair-skip through tab20 (maximises perceptual distance),
# then add one colour from Set2 for the 11th class.
_t20 = plt.cm.tab20
CLASS_COLORS = [_t20(i / 20.0) for i in [0, 2, 4, 6, 8, 10, 12, 14, 16, 18]] \
             + [plt.cm.Set2(4 / 8.0)]


# ── Single-figure renderer ─────────────────────────────────────────────────────
def make_logit_plot(
    logits: np.ndarray,
    true_class: int,
    random_level: float,
    sample_idx: int,
    save_path: Path,
    pred_class: int = None,
) -> None:
    """
    Render and save one publication-ready PDF with three subplots:
      left   — raw logits over time
      centre — softmax probabilities over time
      right  — running average of probabilities over time

    Args:
        logits:       [T, C] raw logit array (numpy, CPU).
        true_class:   Ground-truth class index for this sample.
        random_level: Mean logit over all time steps and classes; when all
                      logits equal this value softmax output is uniform.
        sample_idx:   0-based figure index (shown in title).
        save_path:    Destination .pdf path.
    """
    T = logits.shape[0]
    time_steps = np.arange(T)

    # Stable softmax
    shifted = logits - logits.max(axis=1, keepdims=True)
    exp_l   = np.exp(shifted)
    probs   = exp_l / exp_l.sum(axis=1, keepdims=True)   # [T, C]

    # (Old: mean probabilities.)
    # We'll compute the running integral (cumulative sum) of the raw logits
    # over time for each class and plot that in the rightmost subplot.
    running_integral = np.cumsum(logits, axis=0)  # [T, C]

    if pred_class is None:
        pred_class = int(np.argmax(logits.mean(axis=0)))
    correct    = pred_class == true_class
    verdict    = "✓ correct" if correct else "✗ wrong"
    title_color = "#2a7a2a" if correct else "#b22222"

    fig, (ax_l, ax_r, ax_avg) = plt.subplots(1, 3, figsize=(17, 3.6))
    fig.suptitle(
        f"Sample {sample_idx + 1}  —  true: {CLASS_NAMES[true_class]}"
        f"   pred: {CLASS_NAMES[pred_class]}   {verdict}",
        fontsize=10,
        y=1.01,
        color=title_color,
        fontweight="bold",
    )

    # ── Left: logits ──────────────────────────────────────────────────────────
    for c in range(NUM_CLASSES):
        is_true = (c == true_class)
        ax_l.plot(
            time_steps, logits[:, c],
            color=CLASS_COLORS[c],
            linewidth=1.8 if is_true else 0.9,
            alpha=1.0  if is_true else 0.65,
            zorder=3   if is_true else 2,
            label=CLASS_NAMES[c],
        )
    ax_l.axhline(
        y=random_level, color="black", linestyle="--",
        linewidth=1.2, alpha=0.72, zorder=4, label="random (equal logits)",
    )
    ax_l.set_xlabel("Time step")
    ax_l.set_ylabel("Logit")
    ax_l.set_title("Logits", pad=4)
    ax_l.set_xlim(0, T - 1)

    # ── Centre: probabilities ─────────────────────────────────────────────────
    for c in range(NUM_CLASSES):
        is_true = (c == true_class)
        ax_r.plot(
            time_steps, probs[:, c],
            color=CLASS_COLORS[c],
            linewidth=1.8 if is_true else 0.9,
            alpha=1.0  if is_true else 0.65,
            zorder=3   if is_true else 2,
        )
    ax_r.axhline(
        y=1.0 / NUM_CLASSES, color="black", linestyle="--",
        linewidth=1.2, alpha=0.72, zorder=4,
    )
    ax_r.set_xlabel("Time step")
    ax_r.set_ylabel("Probability")
    ax_r.set_title("Probabilities (softmax)", pad=4)
    ax_r.set_xlim(0, T - 1)
    ax_r.set_ylim(0, 1)

    # ── Right: running integral (cumulative sum) of logits per class
    for c in range(NUM_CLASSES):
        is_true = (c == true_class)
        ax_avg.plot(
            time_steps, running_integral[:, c],
            color=CLASS_COLORS[c],
            linewidth=1.8 if is_true else 0.9,
            alpha=1.0  if is_true else 0.65,
            zorder=3   if is_true else 2,
        )
    # Show cumulative baseline corresponding to the 'random_level' (i.e.
    # the integral of a constant random-level logit across time).
    baseline = random_level * np.arange(1, T + 1)
    ax_avg.plot(
        time_steps, baseline,
        color="black", linestyle="--",
        linewidth=1.2, alpha=0.72, zorder=4,
    )
    ax_avg.set_xlabel("Time step")
    ax_avg.set_ylabel("Running integral of logits")
    ax_avg.set_title("Integrated logits (running integral)", pad=4)
    ax_avg.set_xlim(0, T - 1)

    # ── Shared legend (external, right of rightmost panel) ────────────────────
    handles, labels = ax_l.get_legend_handles_labels()
    ax_avg.legend(
        handles, labels,
        loc="upper left",
        bbox_to_anchor=(1.01, 1.0),
        borderaxespad=0.0,
        frameon=True,
        framealpha=0.9,
        edgecolor="0.8",
        fontsize=7.5,
    )

    fig.tight_layout()
    fig.savefig(save_path, format="pdf", bbox_inches="tight", dpi=300)
    plt.close(fig)


# ── Main ──────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Plot CMOS KWS logits over time (publication-ready PDFs)."
    )
    parser.add_argument("--config",     required=True,  help="Config JSON path.")
    parser.add_argument("--model_path", required=True,  help="Trained model .pkl path.")
    parser.add_argument("--task",       default="real_audio_digits",
                        help="Benchmark task name (default: real_audio_digits).")
    parser.add_argument("--plot_dir",   required=True,  help="Output directory for PDFs.")
    parser.add_argument("--num_plots",  type=int, default=16,
                        help="Number of test samples to plot (default: 16).")
    args = parser.parse_args()

    plot_dir = Path(args.plot_dir)
    plot_dir.mkdir(parents=True, exist_ok=True)

    # ── Task ─────────────────────────────────────────────────────────────────
    if args.task not in BENCHMARK_TASKS:
        print(f"Unknown task '{args.task}'. Available: {list(BENCHMARK_TASKS.keys())}")
        sys.exit(1)

    task_config = BENCHMARK_TASKS[args.task]
    task_params = task_config["default_params"]
    data_fn     = task_config["data_fn"]

    # ── Model ─────────────────────────────────────────────────────────────────
    print(f"Loading model: {args.model_path}")
    model, _ = load_model(args.model_path)
    model.eval()

    # ── Generate plots ────────────────────────────────────────────────────────
    print(f"Generating {args.num_plots} PDFs → {plot_dir}/")
    print()

    for i in range(args.num_plots):
        # Spread seeds deterministically across the test set.
        key = jax.random.PRNGKey(i * 137 + 7)

        # Draw one test example. targets shape: [1, 1, C] (time dim broadcast).
        inputs, targets = data_fn(
            1,
            task_params["seq_length"],
            task_params["input_size"],
            key,
            split="test",
        )

        # Forward pass (batch size 1, no gradients).
        initial_state = model.init_state(1)
        model_output  = model(inputs, initial_state, training=False)

        # logits: [T, C] — raw pre-softmax values for every time step.
        logits      = np.array(model_output["outputs"][0])   # remove batch dim
        true_class  = int(jnp.argmax(targets[0, 0]))        # targets: [B,1,C]

        # Prediction: mean-pooled logits over time, matching use_mean_pooling_accuracy=True.
        mean_logits = logits.mean(axis=0)                    # [C]
        pred_class  = int(np.argmax(mean_logits))

        # Random baseline: the mean logit over time and classes.
        # Rationale: if all logits equal this value the softmax output is
        # uniform (1/C per class), i.e. the model is at random chance.
        random_level = float(np.mean(logits))

        save_path = plot_dir / f"logits_{i:02d}.pdf"
        make_logit_plot(logits, true_class, random_level, i, save_path, pred_class)
        correct    = "✓" if pred_class == true_class else "✗"
        print(
            f"  [{i + 1:2d}/{args.num_plots}]  "
            f"true={CLASS_NAMES[true_class]:<8s}  "
            f"pred={CLASS_NAMES[pred_class]:<8s}  {correct}  "
            f"→ {save_path.name}"
        )

    print()
    print(f"Done. {args.num_plots} PDFs saved in {plot_dir}/")


if __name__ == "__main__":
    print(f"JAX devices: {jax.devices()}", flush=True)
    main()

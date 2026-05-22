#!/usr/bin/env python3
"""
Main script for CMOS-SBC RNN model training and evaluation.
Supports training, testing, validation, and state visualization for keyword spotting.

Usage:
    python main.py train --task real_audio_binary --config config.json --model_path model.pkl
    python main.py test --task real_audio_binary --model_path model.pkl
    python main.py validate --task real_audio_binary --model_path model.pkl
    python main.py plot_test --task real_audio_binary --model_path model.pkl
    python main.py export --task real_audio_binary --model_path model.pkl --export_seed 42
"""

import argparse
import json
from pathlib import Path

import jax
import jax.numpy as jnp

from tasks import BENCHMARK_TASKS
from train_test import (
    train_model,
    test_model,
    plot_network_states,
    test_quantized_model,
    export_confusion_matrix,
    export_inference_data,
)
from utils import load_model


def main():
    parser = argparse.ArgumentParser(
        description="CMOS-SBC RNN Model Training and Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  Train a model:
    python main.py train --task real_audio_binary --config config.json --model_path model.pkl

  Test a trained model:
    python main.py test --task real_audio_binary --model_path model.pkl

  Validate on validation set:
    python main.py validate --task real_audio_binary --model_path model.pkl

  Test with weight quantization:
    python main.py validate --task real_audio_binary --model_path model.pkl --quantize_bits 4

  Plot network states:
    python main.py plot_test --task real_audio_binary --model_path model.pkl

  Export inference data for hardware validation:
    python main.py export --task real_audio_binary --model_path model.pkl --export_seed 42
        """
    )
    
    # Mode selection
    parser.add_argument(
        "mode",
        choices=["train", "test", "validate", "plot_test", "list_tasks", "export"],
        help="Mode of operation"
    )
    
    # Task and model arguments
    parser.add_argument(
        "--task", type=str, default="real_audio_binary",
        help="Task to train/test on (default: real_audio_binary)"
    )
    parser.add_argument(
        "--config", type=str,
        help="Path to JSON configuration file"
    )
    parser.add_argument(
        "--model_path", type=str,
        help="Path to save/load model checkpoint"
    )
    parser.add_argument(
        "--plot_path", type=str,
        help="Path to save plot (for plot_test mode)"
    )
    
    # Training hyperparameters
    parser.add_argument(
        "--epochs", type=int, default=300,
        help="Number of training epochs (default: 300)"
    )
    parser.add_argument(
        "--batch_size", type=int, default=16,
        help="Batch size (default: 16)"
    )
    parser.add_argument(
        "--learning_rate", type=float, default=1e-3,
        help="Learning rate (default: 0.001)"
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed (default: 42)"
    )
    
    # Quantization testing
    parser.add_argument(
        "--quantize_bits", type=int, default=None,
        help="Number of bits for weight quantization (e.g., 8, 4, 2). None = no quantization"
    )
    
    # Confusion matrix arguments
    parser.add_argument(
        "--no_confusion_matrix", action="store_true",
        help="Disable confusion matrix export for test/validate modes"
    )
    parser.add_argument(
        "--cm_dir", type=str, default="confusion_matrices",
        help="Directory to save confusion matrices (default: confusion_matrices)"
    )
    
    # Export arguments
    parser.add_argument(
        "--export_dir", type=str, default="inference_exports",
        help="Directory to save inference exports (default: inference_exports)"
    )
    parser.add_argument(
        "--export_seed", type=int, default=None,
        help="Random seed for inference export (None = random)"
    )

    args = parser.parse_args()

    # Load configuration from file or use defaults
    if args.config:
        with open(args.config, 'r') as f:
            config = json.load(f)
    else:
        config = {
            "num_epochs": args.epochs,
            "batch_size": args.batch_size,
            "learning_rate": args.learning_rate,
            "seed": args.seed,
        }

    # Execute based on mode
    if args.mode == "list_tasks":
        print("\nAvailable benchmark tasks:")
        print("=" * 60)
        for task_name, task_info in BENCHMARK_TASKS.items():
            print(f"  {task_name}:")
            print(f"    {task_info['description']}")
        print("=" * 60)

    elif args.mode == "train":
        print("=" * 60)
        print("Training mode")
        print("=" * 60)
        
        model, losses, metrics = train_model(args.task, config, args.model_path)

        # Show example prediction
        if args.task in BENCHMARK_TASKS:
            task_config = BENCHMARK_TASKS[args.task]
            task_params = task_config["default_params"]

            test_key = jax.random.PRNGKey(123)
            test_inputs, test_targets = task_config["data_fn"](
                1, task_params["seq_length"], task_params["input_size"],
                test_key, split="validation"
            )

            initial_state = model.init_state(1)
            model_output = model(test_inputs, initial_state)
            predictions = model_output["outputs"]

            print(f"\nExample prediction for {task_config['name']}:")
            print(f"  Input shape: {test_inputs.shape}")
            print(f"  Target shape: {test_targets.shape}")
            print(f"  Prediction shape: {predictions.shape}")

            # Show final prediction
            final_pred = jax.nn.softmax(predictions[0, -1, :])
            true_class = jnp.argmax(test_targets[0, -1, :])
            pred_class = jnp.argmax(final_pred)

            print(f"  True class: {true_class}")
            print(f"  Predicted class: {pred_class}")
            print(f"  Confidence: {final_pred[pred_class]:.4f}")

    elif args.mode == "test":
        if not args.model_path:
            raise ValueError("Model path required for testing (--model_path)")

        # Load model
        model, checkpoint = load_model(args.model_path)

        print("=" * 60)
        print("Test mode")
        print("=" * 60)
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Original task: {checkpoint.get('task_name', 'unknown')}")

        # Determine task to test
        test_task = args.task
        model_name = Path(args.model_path).stem

        # Check if quantization testing is requested
        if args.quantize_bits is not None:
            print(f"\nTesting with {args.quantize_bits}-bit quantization...")
            test_quantized_model(
                model,
                test_task,
                num_bits=args.quantize_bits,
                batch_size=args.batch_size
            )
        else:
            # Regular testing
            test_model(model, test_task, batch_size=args.batch_size)

            # Export confusion matrix if enabled
            if not args.no_confusion_matrix:
                export_confusion_matrix(
                    model, test_task,
                    batch_size=args.batch_size,
                    split="test",
                    cm_save_dir=args.cm_dir,
                    model_name=model_name
                )

    elif args.mode == "validate":
        if not args.model_path:
            raise ValueError("Model path required for validation (--model_path)")

        # Load model
        model, checkpoint = load_model(args.model_path)

        print("=" * 60)
        print("Validation mode")
        print("=" * 60)
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Original task: {checkpoint.get('task_name', 'unknown')}")

        validate_task = args.task
        model_name = Path(args.model_path).stem

        if validate_task in BENCHMARK_TASKS:
            task_config = BENCHMARK_TASKS[validate_task]

            # Check if quantization testing is requested
            if args.quantize_bits is not None:
                print(f"\nTesting with {args.quantize_bits}-bit quantization...")
                test_quantized_model(
                    model,
                    validate_task,
                    num_bits=args.quantize_bits,
                    batch_size=args.batch_size
                )

            else:
                print("\nUsing validation split...")

                from train_test import evaluate_model
                test_key = jax.random.PRNGKey(999)
                metrics = evaluate_model(
                    model, task_config, test_key, args.batch_size,
                    num_eval_batches=50, split="validation"
                )

                print(f"\nValidation Results for {task_config['name']}:")
                print(f"  Loss: {metrics['loss']:.6f}")
                print(f"  Accuracy: {metrics['accuracy']:.4f}")

                # Export confusion matrix if enabled
                if not args.no_confusion_matrix:
                    export_confusion_matrix(
                        model, validate_task,
                        batch_size=args.batch_size,
                        split="validation",
                        cm_save_dir=args.cm_dir,
                        model_name=model_name
                    )
        else:
            print(f"Unknown task: {validate_task}")

    elif args.mode == "plot_test":
        if not args.model_path:
            raise ValueError("Model path required for plot testing (--model_path)")

        # Load model
        model, checkpoint = load_model(args.model_path)

        print("=" * 60)
        print("Plot test mode")
        print("=" * 60)
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Plotting network states for task: {args.task}")

        # Generate plot path if not provided
        plot_path = args.plot_path
        if not plot_path:
            model_name = Path(args.model_path).stem
            plot_path = f"plots/{model_name}_{args.task}_states.png"
            Path("plots").mkdir(exist_ok=True)

        # Create state visualization
        plot_network_states(model, args.task, plot_path)

    elif args.mode == "export":
        if not args.model_path:
            raise ValueError("Model path required for export (--model_path)")
        if not args.task:
            raise ValueError("Task required for export (--task)")

        # Load model
        model, checkpoint = load_model(args.model_path)

        print("=" * 60)
        print("Export mode")
        print("=" * 60)
        print(f"Loaded model from epoch {checkpoint['epoch']}")
        print(f"Original task: {checkpoint.get('task_name', 'unknown')}")

        # Run export
        export_bundle = export_inference_data(
            model=model,
            task_name=args.task,
            seed=args.export_seed,
            export_dir=args.export_dir,
            batch_size=1,
            verbose=True
        )

        print(f"\n[OK] Inference data ready for hardware validation!")
        print(f"  Files saved in: {args.export_dir}")


if __name__ == "__main__":
    main()

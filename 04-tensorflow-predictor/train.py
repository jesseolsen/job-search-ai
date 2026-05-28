#!/usr/bin/env python3
"""Train the salary band predictor model."""

from pathlib import Path
from predictor import SalaryBandPredictor
from generate_data import generate_synthetic_salary_data


def main():
    """Train and save the model."""
    print("Generating synthetic salary data...")
    descriptions, salary_bands, seniorities = generate_synthetic_salary_data(n=500)

    # Split into train/val (80/20)
    split_idx = int(0.8 * len(descriptions))
    train_texts = descriptions[:split_idx]
    val_texts = descriptions[split_idx:]
    train_labels = salary_bands[:split_idx]
    val_labels = salary_bands[split_idx:]
    train_seniorities = seniorities[:split_idx]
    val_seniorities = seniorities[split_idx:]

    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Create and train Sequential model
    print("\nTraining Sequential model...")
    predictor = SalaryBandPredictor(use_functional=False)
    history = predictor.train(
        train_texts,
        train_labels,
        val_texts=val_texts,
        val_labels=val_labels,
        epochs=10,
        batch_size=32,
        early_stopping=True,
    )

    print(f"\nTraining complete!")
    final_val_acc = history['val_accuracy'][-1]
    print(f"Final validation accuracy: {final_val_acc:.4f}")

    # Save model
    model_path = Path(__file__).parent / "model"
    predictor.save(str(model_path))

    print(f"\n✓ Model saved to {model_path}")

    # Also train Functional model
    print("\n" + "="*70)
    print("Training Functional model (with seniority input)...")
    predictor_func = SalaryBandPredictor(use_functional=True)
    history_func = predictor_func.train(
        train_texts,
        train_labels,
        seniorities=train_seniorities,
        val_texts=val_texts,
        val_labels=val_labels,
        val_seniorities=val_seniorities,
        epochs=10,
        batch_size=32,
        early_stopping=True,
    )

    final_val_acc_func = history_func['val_accuracy'][-1]
    print(f"Final validation accuracy: {final_val_acc_func:.4f}")

    # Save functional model
    model_func_path = Path(__file__).parent / "model_functional"
    predictor_func.save(str(model_func_path))

    print(f"✓ Functional model saved to {model_func_path}")


if __name__ == "__main__":
    main()

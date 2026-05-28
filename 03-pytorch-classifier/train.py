#!/usr/bin/env python3
"""Train the resume section classifier."""

import random
from pathlib import Path

from classifier import ResumeSectionClassifier
from generate_data import generate_synthetic_data


def main():
    """Train and save the model."""
    print("Generating synthetic training data...")
    texts, labels = generate_synthetic_data(num_per_section=50)

    # Split into train/val (80/20)
    split_idx = int(0.8 * len(texts))
    train_texts, val_texts = texts[:split_idx], texts[split_idx:]
    train_labels, val_labels = labels[:split_idx], labels[split_idx:]

    print(f"Train samples: {len(train_texts)}")
    print(f"Validation samples: {len(val_texts)}")

    # Create and train classifier
    print("\nTraining classifier...")
    classifier = ResumeSectionClassifier()
    metrics = classifier.train(
        train_texts,
        train_labels,
        val_texts,
        val_labels,
        epochs=10,
        batch_size=32,
        learning_rate=0.001,
    )

    # Print summary
    print(f"\nTraining complete!")
    print(f"Final validation accuracy: {metrics['val_accuracies'][-1]:.4f}")

    # Save model
    model_path = Path(__file__).parent / "model"
    classifier.save(str(model_path))

    print(f"\n✓ Model saved to {model_path}")


if __name__ == "__main__":
    main()

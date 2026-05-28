#!/usr/bin/env python3
"""Demo: Resume section classifier predictions."""

from pathlib import Path
from classifier import ResumeSectionClassifier
from generate_data import generate_synthetic_data


def main():
    """Run demo with trained model."""
    print("\n" + "="*70)
    print("Resume Section Classifier — Demo")
    print("="*70)

    model_path = Path(__file__).parent / "model"

    if not model_path.exists():
        print(f"\n⚠ Model not found at {model_path}")
        print("Please run: python3 train.py")
        return

    # Load model
    print("\nLoading trained model...")
    classifier = ResumeSectionClassifier()
    classifier.load(str(model_path))

    # Test samples
    test_samples = [
        "Senior Software Engineer at TechCorp with 5 years experience",
        "BS Computer Science from State University 2020",
        "Proficient in Python PostgreSQL and AWS",
        "Built e-commerce platform processing millions of transactions",
        "Strong background in distributed systems and microservices",
        "References available upon request",
    ]

    print("\nMaking predictions:")
    print("-" * 70)

    for text in test_samples:
        label, confidence = classifier.predict(text)
        print(f"\nText: {text}")
        print(f"Predicted: {label:12} (confidence: {confidence:.2%})")

    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()

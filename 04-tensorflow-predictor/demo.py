#!/usr/bin/env python3
"""Demo: Salary band predictor predictions."""

from pathlib import Path
from predictor import SalaryBandPredictor


def main():
    """Run demo with trained models."""
    print("\n" + "="*70)
    print("Salary Band Predictor — Demo")
    print("="*70)

    model_path = Path(__file__).parent / "model"
    model_func_path = Path(__file__).parent / "model_functional"

    if not model_path.exists():
        print(f"\n⚠ Model not found at {model_path}")
        print("Please run: python3 train.py")
        return

    # Test samples
    test_samples = [
        ("Senior Machine Learning Engineer with 5+ years in deep learning", "senior"),
        ("Data Scientist building predictive models", "mid"),
        ("Junior Python Developer eager to learn web development", "junior"),
        ("Staff Software Engineer leading technical strategy", "staff"),
        ("DevOps Engineer managing cloud infrastructure", "mid"),
        ("Backend Engineer specializing in microservices", "senior"),
    ]

    # Demo Sequential model
    print("\n" + "-"*70)
    print("Sequential Model Predictions:")
    print("-"*70)

    predictor = SalaryBandPredictor(use_functional=False)
    predictor.load(str(model_path))

    for text, seniority in test_samples:
        band, confidence = predictor.predict(text)
        print(f"\nText: {text}")
        print(f"Expected seniority: {seniority}")
        print(f"Predicted band: {band:12} (confidence: {confidence:.2%})")

    # Demo Functional model if available
    if model_func_path.exists():
        print("\n" + "="*70)
        print("Functional Model Predictions (with seniority input):")
        print("="*70)

        predictor_func = SalaryBandPredictor(use_functional=True)
        predictor_func.load(str(model_func_path))

        for text, seniority in test_samples:
            band, confidence = predictor_func.predict(text, seniority=seniority)
            print(f"\nText: {text}")
            print(f"Seniority: {seniority}")
            print(f"Predicted band: {band:12} (confidence: {confidence:.2%})")

    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()

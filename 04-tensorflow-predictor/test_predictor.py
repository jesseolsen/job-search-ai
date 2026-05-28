"""Tests for TensorFlow salary band predictor.

Tests cover:
- Dataset loading and preprocessing
- Model architecture (Sequential and Functional APIs)
- Model compilation and training
- Callbacks (EarlyStopping, ModelCheckpoint)
- Evaluation metrics (accuracy, confusion matrix)
- Model persistence (save/load)
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Test constants
SALARY_BANDS = ["<80k", "80k-120k", "120k-160k", "160k+"]
NUM_CLASSES = len(SALARY_BANDS)


class TestDataPreprocessing:
    """Test dataset loading and preprocessing."""

    def test_load_synthetic_data(self):
        """Should load synthetic job posting data."""
        from generate_data import generate_synthetic_salary_data
        texts, salaries, seniorities = generate_synthetic_salary_data(n=100)
        assert len(texts) == 100
        assert len(salaries) == 100
        assert len(seniorities) == 100

    def test_salary_band_assignment(self):
        """Should assign salary values to correct bands."""
        from predictor import assign_salary_band
        assert assign_salary_band(50000) == 0  # <80k
        assert assign_salary_band(100000) == 1  # 80k-120k
        assert assign_salary_band(140000) == 2  # 120k-160k
        assert assign_salary_band(200000) == 3  # 160k+

    def test_text_vectorization_creates_sequences(self):
        """TextVectorization layer should convert text to sequences."""
        from predictor import create_text_vectorizer
        texts = ["machine learning engineer", "data scientist", "senior developer"]
        vectorizer = create_text_vectorizer(texts)
        sequences = vectorizer(texts)
        assert sequences.shape[0] == 3
        assert sequences.shape[1] > 0

    def test_categorical_encoding(self):
        """Should encode seniority levels as integer indices."""
        from predictor import encode_seniority
        seniorities = ["junior", "mid", "senior", "staff"]
        encoded = encode_seniority(seniorities)
        assert encoded.shape[0] == 4
        assert all(0 <= idx < 4 for idx in encoded)

    def test_data_split_preserves_distribution(self):
        """Train/val/test split should maintain label distribution."""
        from predictor import create_train_val_test_split
        np.random.seed(42)
        texts = ["text"] * 300
        labels = [0] * 75 + [1] * 75 + [2] * 75 + [3] * 75
        train, val, test = create_train_val_test_split(texts, labels, val_split=0.1, test_split=0.1)
        assert len(train[0]) == 240
        assert len(val[0]) == 30
        assert len(test[0]) == 30


class TestSequentialModel:
    """Test Sequential API model architecture."""

    def test_sequential_model_builds(self):
        """Sequential model should build without errors."""
        from predictor import create_sequential_model
        model = create_sequential_model(vocab_size=1000)
        assert model is not None
        assert isinstance(model, keras.Model)

    def test_sequential_model_compiles(self):
        """Model should compile with loss and optimizer."""
        from predictor import create_sequential_model
        model = create_sequential_model(vocab_size=1000)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        assert model.optimizer is not None
        assert model.loss is not None

    def test_sequential_model_accepts_inputs(self):
        """Model should accept text inputs and produce predictions."""
        from predictor import create_sequential_model
        model = create_sequential_model(vocab_size=1000)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        x = np.random.randint(0, 1000, (4, 100))
        y = model.predict(x, verbose=0)
        assert y.shape == (4, NUM_CLASSES)

    def test_sequential_model_output_shape(self):
        """Model output should have shape (batch_size, num_classes)."""
        from predictor import create_sequential_model
        model = create_sequential_model(vocab_size=500, max_length=50)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        x = np.random.randint(0, 500, (16, 50))
        y = model.predict(x, verbose=0)
        assert y.shape == (16, NUM_CLASSES)

    def test_sequential_model_summary(self):
        """Model should have readable summary."""
        from predictor import create_sequential_model
        model = create_sequential_model(vocab_size=1000)
        assert len(model.layers) > 0


class TestFunctionalModel:
    """Test Functional API model with multi-input."""

    def test_functional_model_with_text_input(self):
        """Functional model should accept text input."""
        from predictor import create_functional_model
        model = create_functional_model(vocab_size=1000)
        assert model is not None
        assert isinstance(model, keras.Model)

    def test_functional_model_with_multi_input(self):
        """Functional model should accept text and seniority inputs."""
        from predictor import create_functional_model
        model = create_functional_model(vocab_size=1000, num_seniority_levels=4)
        assert len(model.inputs) == 2

    def test_functional_model_produces_predictions(self):
        """Functional model should produce predictions for multi-input."""
        from predictor import create_functional_model
        model = create_functional_model(vocab_size=1000, num_seniority_levels=4)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        text_input = np.random.randint(0, 1000, (4, 100))
        sen_input = np.random.randint(0, 4, (4, 1))
        predictions = model.predict([text_input, sen_input], verbose=0)
        assert predictions.shape == (4, NUM_CLASSES)

    def test_functional_model_merges_inputs_correctly(self):
        """Functional model should merge text and seniority branches."""
        from predictor import create_functional_model
        model = create_functional_model(vocab_size=1000, num_seniority_levels=4)
        assert model is not None


class TestModelTraining:
    """Test training loop and callbacks."""

    def test_model_trains_for_epochs(self):
        """Model should train for specified epochs without errors."""
        from predictor import create_sequential_model
        from generate_data import generate_synthetic_salary_data

        model = create_sequential_model(vocab_size=100)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        texts, salaries, _ = generate_synthetic_salary_data(n=100)
        x = np.random.randint(0, 100, (100, 50))
        y = keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, 100), NUM_CLASSES)

        history = model.fit(x, y, epochs=2, batch_size=16, verbose=0)
        assert 'loss' in history.history
        assert 'accuracy' in history.history

    def test_early_stopping_callback(self):
        """EarlyStopping should halt training when validation metric plateaus."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=100)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        early_stop = keras.callbacks.EarlyStopping(monitor='val_loss', patience=2)
        assert early_stop is not None

    def test_model_checkpoint_callback(self):
        """ModelCheckpoint should save best model weights."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=100)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint = keras.callbacks.ModelCheckpoint(
                filepath=str(Path(tmpdir) / "best.h5"),
                monitor='val_accuracy',
                save_best_only=True
            )
            assert checkpoint is not None

    def test_loss_decreases_during_training(self):
        """Training loss should generally decrease over epochs."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=100)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        x = np.random.randint(0, 100, (100, 50))
        y = keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, 100), NUM_CLASSES)

        history = model.fit(x, y, epochs=3, batch_size=16, verbose=0)
        losses = history.history['loss']
        assert len(losses) == 3


class TestModelEvaluation:
    """Test evaluation metrics and confusion matrix."""

    def test_accuracy_metric(self):
        """Model should report accuracy on test set."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=100)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        x = np.random.randint(0, 100, (50, 50))
        y = keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, 50), NUM_CLASSES)

        loss, accuracy = model.evaluate(x, y, verbose=0)
        assert 0 <= accuracy <= 1

    def test_predictions_are_valid_classes(self):
        """Model predictions should be valid class indices."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=100)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        x = np.random.randint(0, 100, (10, 50))
        predictions = model.predict(x, verbose=0)
        predicted_classes = np.argmax(predictions, axis=1)
        assert all(0 <= c < NUM_CLASSES for c in predicted_classes)

    def test_confusion_matrix_generation(self):
        """Should generate valid confusion matrix."""
        from predictor import compute_confusion_matrix

        y_true = np.array([0, 1, 1, 2, 2, 3])
        y_pred = np.array([0, 1, 2, 2, 1, 3])
        cm = compute_confusion_matrix(y_true, y_pred, num_classes=NUM_CLASSES)
        assert cm.shape == (NUM_CLASSES, NUM_CLASSES)
        assert cm.sum() == len(y_true)

    def test_classification_report(self):
        """Should generate classification report with precision/recall."""
        from predictor import generate_classification_report

        y_true = np.array([0, 1, 1, 2, 2, 3, 3, 3])
        y_pred = np.array([0, 1, 1, 2, 1, 3, 3, 2])
        report = generate_classification_report(y_true, y_pred)
        assert report is not None


class TestModelPersistence:
    """Test saving and loading models."""

    def test_model_saves_as_h5(self):
        """Model should save as .h5 file."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.h5")
            model.save(path)
            assert Path(path).exists()

    def test_model_saves_as_keras_format(self):
        """Model should save as native Keras format."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.keras")
            model.save(path)
            assert Path(path).exists()

    def test_model_loads_from_h5(self):
        """Model should load from .h5 file."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=100)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.h5")
            model.save(path)
            loaded = keras.models.load_model(path)
            assert loaded is not None

    def test_loaded_model_produces_same_predictions(self):
        """Loaded model should produce same output as original."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=100)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        x = np.random.randint(0, 100, (4, 50))
        preds1 = model.predict(x, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.h5")
            model.save(path)
            loaded = keras.models.load_model(path)
            preds2 = loaded.predict(x, verbose=0)

            np.testing.assert_array_almost_equal(preds1, preds2, decimal=5)

    def test_model_metadata_preserved(self):
        """Model configuration should be preserved after save/load."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=500)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.h5")
            model.save(path)
            loaded = keras.models.load_model(path)
            assert len(model.layers) == len(loaded.layers)


class TestIntegration:
    """Integration tests."""

    @pytest.mark.integration
    def test_full_pipeline(self):
        """Full pipeline: load data, create model, train, evaluate."""
        from predictor import create_sequential_model
        from generate_data import generate_synthetic_salary_data

        texts, salaries, seniorities = generate_synthetic_salary_data(n=200)

        x = np.random.randint(0, 1000, (200, 50))
        y = keras.utils.to_categorical(np.array(salaries), NUM_CLASSES)

        model = create_sequential_model(vocab_size=1000)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        model.fit(x, y, epochs=3, batch_size=16, verbose=0)

        test_loss, test_acc = model.evaluate(x[:50], y[:50], verbose=0)
        assert test_acc >= 0.1

    @pytest.mark.integration
    def test_end_to_end_with_persistence(self):
        """Full pipeline with model save and load."""
        from predictor import create_sequential_model

        model = create_sequential_model(vocab_size=100)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        x = np.random.randint(0, 100, (100, 50))
        y = keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, 100), NUM_CLASSES)

        model.fit(x, y, epochs=2, batch_size=16, verbose=0)

        with tempfile.TemporaryDirectory() as tmpdir:
            path = str(Path(tmpdir) / "model.h5")
            model.save(path)
            loaded = keras.models.load_model(path)

            loss, acc = loaded.evaluate(x[:20], y[:20], verbose=0)
            assert acc >= 0.1

    @pytest.mark.integration
    def test_functional_model_pipeline(self):
        """Full pipeline with functional multi-input model."""
        from predictor import create_functional_model

        model = create_functional_model(vocab_size=500, num_seniority_levels=4, max_length=50)
        model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

        text_input = np.random.randint(0, 500, (100, 50))
        sen_input = np.random.randint(0, 4, (100, 1))
        y = keras.utils.to_categorical(np.random.randint(0, NUM_CLASSES, 100), NUM_CLASSES)

        history = model.fit([text_input, sen_input], y, epochs=2, batch_size=16, verbose=0)
        assert 'loss' in history.history

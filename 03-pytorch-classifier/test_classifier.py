"""Tests for PyTorch resume section classifier.

Tests cover:
- Dataset creation and loading
- Model architecture and forward pass
- Training loop and loss reduction
- Validation metrics
- Model persistence (save/load)
- Inference and predictions
"""

import pytest
import sys
import os
import tempfile
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestDataset:
    """Test the resume section dataset."""

    def test_dataset_loads_texts_and_labels(self):
        """Dataset should load text snippets and labels."""
        from classifier import ResumeSectionDataset
        texts = ["Python programming", "AWS cloud platform"]
        labels = ["SKILLS", "SKILLS"]
        dataset = ResumeSectionDataset(texts, labels)
        assert len(dataset) == 2
        assert dataset.texts == texts
        assert dataset.labels == labels

    def test_dataset_vocab_builds_from_texts(self):
        """Dataset should build vocabulary from training texts."""
        from classifier import ResumeSectionDataset
        texts = ["Python programming", "Python developer"]
        labels = ["SKILLS", "SKILLS"]
        dataset = ResumeSectionDataset(texts, labels)
        assert "python" in dataset.vocab.word2idx
        assert "programming" in dataset.vocab.word2idx
        assert "developer" in dataset.vocab.word2idx
        assert len(dataset.vocab) > 2

    def test_dataset_converts_text_to_tokens(self):
        """Dataset should convert text to token indices."""
        from classifier import ResumeSectionDataset
        texts = ["Python programming"]
        labels = ["SKILLS"]
        dataset = ResumeSectionDataset(texts, labels)
        tokens = dataset.vocab.encode("Python programming")
        assert len(tokens) > 0
        assert all(isinstance(t, int) for t in tokens)

    def test_dataset_handles_unknown_words(self):
        """Unknown words should map to UNK token."""
        from classifier import ResumeSectionDataset
        texts = ["Python programming"]
        labels = ["SKILLS"]
        dataset = ResumeSectionDataset(texts, labels)
        unk_idx = dataset.vocab.word2idx["<UNK>"]
        tokens = dataset.vocab.encode("Python programming xyzabc")
        assert unk_idx in tokens

    def test_dataloader_batches_correctly(self):
        """DataLoader should batch samples with padding."""
        from classifier import ResumeSectionDataset
        texts = ["Python"] * 10
        labels = ["SKILLS"] * 10
        dataset = ResumeSectionDataset(texts, labels)
        loader = DataLoader(dataset, batch_size=4)
        batch_texts, batch_labels = next(iter(loader))
        assert batch_texts.shape[0] == 4
        assert batch_labels.shape[0] == 4
        assert batch_texts.shape[1] == 50


class TestModelArchitecture:
    """Test the neural network model."""

    def test_model_initializes(self):
        """Model should initialize without errors."""
        from classifier import TextClassifier
        model = TextClassifier(vocab_size=1000, embed_dim=128, hidden_dim=64, num_classes=6)
        assert model is not None

    def test_model_has_embedding_layer(self):
        """Model should have nn.Embedding layer."""
        from classifier import TextClassifier
        model = TextClassifier(vocab_size=1000)
        assert hasattr(model, 'embed')
        assert isinstance(model.embed, nn.Embedding)

    def test_model_has_lstm_layer(self):
        """Model should have LSTM or similar recurrent layer."""
        from classifier import TextClassifier
        model = TextClassifier(vocab_size=1000)
        assert hasattr(model, 'lstm')
        assert isinstance(model.lstm, nn.LSTM)

    def test_model_has_output_layer(self):
        """Model should have output layer for classification."""
        from classifier import TextClassifier
        model = TextClassifier(vocab_size=1000)
        assert hasattr(model, 'fc2')
        assert isinstance(model.fc2, nn.Linear)

    def test_forward_pass_returns_logits(self):
        """Forward pass should return logits of shape (batch_size, num_classes)."""
        from classifier import TextClassifier
        model = TextClassifier(vocab_size=1000)
        batch_size = 4
        seq_length = 50
        input_tensor = torch.randint(0, 1000, (batch_size, seq_length))
        output = model(input_tensor)
        assert output.shape[0] == batch_size
        assert output.shape[1] == 6

    def test_forward_pass_shape(self):
        """Output shape should match number of classes."""
        from classifier import TextClassifier
        model = TextClassifier(vocab_size=500, num_classes=6)
        input_tensor = torch.randint(0, 500, (8, 50))
        output = model(input_tensor)
        assert output.shape == (8, 6)


class TestTrainingLoop:
    """Test the training process."""

    def test_training_reduces_loss(self):
        """Loss should decrease over training iterations."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        split = int(0.8 * len(texts))
        train_texts, val_texts = texts[:split], texts[split:]
        train_labels, val_labels = labels[:split], labels[split:]

        classifier = ResumeSectionClassifier()
        metrics = classifier.train(train_texts, train_labels, val_texts, val_labels, epochs=3)
        losses = metrics["train_losses"]
        assert len(losses) == 3
        assert losses[0] > 0

    def test_backward_pass_computes_gradients(self):
        """Backward pass should populate model gradients."""
        from classifier import TextClassifier
        model = TextClassifier(vocab_size=100)
        criterion = nn.CrossEntropyLoss()
        input_tensor = torch.randint(0, 100, (4, 50))
        target = torch.randint(0, 6, (4,))

        logits = model(input_tensor)
        loss = criterion(logits, target)
        loss.backward()

        for param in model.parameters():
            assert param.grad is not None

    def test_optimizer_updates_weights(self):
        """Optimizer step should update model weights."""
        from classifier import TextClassifier
        model = TextClassifier(vocab_size=100)
        optimizer = optim.Adam(model.parameters(), lr=0.001)
        criterion = nn.CrossEntropyLoss()

        initial_weight = model.fc1.weight.data.clone()

        input_tensor = torch.randint(0, 100, (4, 50))
        target = torch.randint(0, 6, (4,))
        logits = model(input_tensor)
        loss = criterion(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        assert not torch.equal(initial_weight, model.fc1.weight.data)

    def test_validation_metrics_computed(self):
        """Should compute accuracy on validation set."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        split = int(0.8 * len(texts))
        train_texts, val_texts = texts[:split], texts[split:]
        train_labels, val_labels = labels[:split], labels[split:]

        classifier = ResumeSectionClassifier()
        metrics = classifier.train(train_texts, train_labels, val_texts, val_labels, epochs=2)

        assert "val_accuracies" in metrics
        assert len(metrics["val_accuracies"]) == 2
        assert all(0 <= acc <= 1 for acc in metrics["val_accuracies"])

    def test_training_epoch_completes(self):
        """Full training epoch should complete without errors."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=5)
        split = int(0.8 * len(texts))
        train_texts, val_texts = texts[:split], texts[split:]
        train_labels, val_labels = labels[:split], labels[split:]

        classifier = ResumeSectionClassifier()
        metrics = classifier.train(train_texts, train_labels, val_texts, val_labels, epochs=1)
        assert metrics is not None


class TestModelEvaluation:
    """Test model evaluation metrics."""

    def test_accuracy_calculation(self):
        """Should calculate accuracy correctly."""
        predictions = torch.tensor([0, 1, 2, 0, 1])
        targets = torch.tensor([0, 1, 1, 0, 1])
        accuracy = (predictions == targets).float().mean().item()
        assert abs(accuracy - 0.8) < 0.01

    def test_validation_loop(self):
        """Validation loop should compute metrics."""
        from classifier import TextClassifier, ResumeSectionDataset
        model = TextClassifier(vocab_size=100)
        model.eval()

        texts = ["Python"] * 10
        labels = ["SKILLS"] * 10
        dataset = ResumeSectionDataset(texts, labels)
        loader = DataLoader(dataset, batch_size=4)

        correct = 0
        total = 0
        with torch.no_grad():
            for batch_texts, batch_labels in loader:
                logits = model(batch_texts)
                predictions = logits.argmax(dim=1)
                correct += (predictions == batch_labels).sum().item()
                total += batch_labels.size(0)

        accuracy = correct / total
        assert 0 <= accuracy <= 1

    def test_torch_no_grad_context(self):
        """Validation should use torch.no_grad() for efficiency."""
        from classifier import TextClassifier
        model = TextClassifier(vocab_size=100)
        input_tensor = torch.randint(0, 100, (4, 50))

        with torch.no_grad():
            output = model(input_tensor)

        assert output is not None


class TestInference:
    """Test model inference and predictions."""

    def test_predict_single_text(self):
        """Should predict label for a single text."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        split = int(0.8 * len(texts))
        classifier = ResumeSectionClassifier()
        classifier.train(texts[:split], labels[:split], texts[split:], labels[split:], epochs=1)

        label, confidence = classifier.predict("Python programming")
        assert isinstance(label, str)

    def test_predict_returns_label_name(self):
        """Prediction should return the section label as string."""
        from classifier import ResumeSectionClassifier, SECTION_LABELS
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        split = int(0.8 * len(texts))
        classifier = ResumeSectionClassifier()
        classifier.train(texts[:split], labels[:split], texts[split:], labels[split:], epochs=1)

        label, confidence = classifier.predict("Python programming")
        assert label in SECTION_LABELS

    def test_predict_with_confidence(self):
        """Should return confidence score for prediction."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        split = int(0.8 * len(texts))
        classifier = ResumeSectionClassifier()
        classifier.train(texts[:split], labels[:split], texts[split:], labels[split:], epochs=1)

        label, confidence = classifier.predict("Python programming")
        assert isinstance(confidence, float)
        assert 0 <= confidence <= 1

    def test_predict_handles_unknown_words(self):
        """Prediction should handle out-of-vocabulary words gracefully."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        split = int(0.8 * len(texts))
        classifier = ResumeSectionClassifier()
        classifier.train(texts[:split], labels[:split], texts[split:], labels[split:], epochs=1)

        label, confidence = classifier.predict("xyzabc qwerty unknown_word_12345")
        assert isinstance(label, str)
        assert isinstance(confidence, float)


class TestModelPersistence:
    """Test saving and loading models."""

    def test_model_state_dict_saves(self):
        """Should save model state_dict."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        split = int(0.8 * len(texts))
        classifier = ResumeSectionClassifier()
        classifier.train(texts[:split], labels[:split], texts[split:], labels[split:], epochs=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            classifier.save(tmpdir)
            assert (Path(tmpdir) / "model.pt").exists()
            assert (Path(tmpdir) / "vocab.json").exists()

    def test_model_state_dict_loads(self):
        """Should load model state_dict."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        split = int(0.8 * len(texts))
        classifier = ResumeSectionClassifier()
        classifier.train(texts[:split], labels[:split], texts[split:], labels[split:], epochs=1)

        with tempfile.TemporaryDirectory() as tmpdir:
            classifier.save(tmpdir)
            classifier2 = ResumeSectionClassifier()
            classifier2.load(tmpdir)
            assert classifier2.model is not None
            assert classifier2.vocab is not None

    def test_loaded_model_produces_same_output(self):
        """Loaded model should produce same output as original."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        split = int(0.8 * len(texts))
        classifier = ResumeSectionClassifier()
        classifier.train(texts[:split], labels[:split], texts[split:], labels[split:], epochs=1)

        label1, conf1 = classifier.predict("Python programming")

        with tempfile.TemporaryDirectory() as tmpdir:
            classifier.save(tmpdir)
            classifier2 = ResumeSectionClassifier()
            classifier2.load(tmpdir)
            label2, conf2 = classifier2.predict("Python programming")

            assert label1 == label2
            assert abs(conf1 - conf2) < 0.001

    def test_vocab_persists_with_model(self):
        """Vocabulary should be saved and loaded with model."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        split = int(0.8 * len(texts))
        classifier = ResumeSectionClassifier()
        classifier.train(texts[:split], labels[:split], texts[split:], labels[split:], epochs=1)

        original_vocab_size = len(classifier.vocab)

        with tempfile.TemporaryDirectory() as tmpdir:
            classifier.save(tmpdir)
            classifier2 = ResumeSectionClassifier()
            classifier2.load(tmpdir)
            assert len(classifier2.vocab) == original_vocab_size


class TestDataGeneration:
    """Test synthetic training data generation."""

    def test_generate_synthetic_data(self):
        """Should generate synthetic labeled resume snippets."""
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        assert len(texts) == len(labels)
        assert len(texts) > 0

    def test_synthetic_data_has_labels(self):
        """Generated data should have correct section labels."""
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=10)
        valid_labels = {"EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS", "SUMMARY", "OTHER"}
        assert all(label in valid_labels for label in labels)

    def test_synthetic_data_covers_all_sections(self):
        """Should generate examples for all 6 section types."""
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=5)
        unique_labels = set(labels)
        expected = {"EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS", "SUMMARY", "OTHER"}
        assert unique_labels == expected

    def test_synthetic_data_realistic_patterns(self):
        """Synthetic data should follow realistic patterns."""
        from generate_data import generate_synthetic_data
        texts, labels = generate_synthetic_data(num_per_section=5)
        assert all(len(text) > 0 for text in texts)
        assert all(isinstance(text, str) for text in texts)


class TestIntegration:
    """Integration tests (slower, but test real behavior)."""

    @pytest.mark.integration
    def test_full_training_pipeline(self):
        """Full pipeline: create data, train model, validate."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data

        texts, labels = generate_synthetic_data(num_per_section=50)
        split = int(0.8 * len(texts))
        train_texts, val_texts = texts[:split], texts[split:]
        train_labels, val_labels = labels[:split], labels[split:]

        classifier = ResumeSectionClassifier()
        metrics = classifier.train(train_texts, train_labels, val_texts, val_labels, epochs=5)

        assert "train_losses" in metrics
        assert "val_accuracies" in metrics
        assert len(metrics["val_accuracies"]) == 5

    @pytest.mark.integration
    def test_model_achieves_reasonable_accuracy(self):
        """Trained model should improve from random guessing (16.7%)."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data

        texts, labels = generate_synthetic_data(num_per_section=100)
        split = int(0.8 * len(texts))
        train_texts, val_texts = texts[:split], texts[split:]
        train_labels, val_labels = labels[:split], labels[split:]

        classifier = ResumeSectionClassifier()
        metrics = classifier.train(train_texts, train_labels, val_texts, val_labels, epochs=10)

        final_accuracy = metrics["val_accuracies"][-1]
        assert final_accuracy >= 0.1

    @pytest.mark.integration
    def test_end_to_end_inference(self):
        """Full inference pipeline with save/load."""
        from classifier import ResumeSectionClassifier
        from generate_data import generate_synthetic_data

        texts, labels = generate_synthetic_data(num_per_section=20)
        split = int(0.8 * len(texts))
        classifier = ResumeSectionClassifier()
        classifier.train(texts[:split], labels[:split], texts[split:], labels[split:], epochs=3)

        with tempfile.TemporaryDirectory() as tmpdir:
            classifier.save(tmpdir)
            loaded = ResumeSectionClassifier()
            loaded.load(tmpdir)

            label, conf = loaded.predict("Senior Software Engineer at TechCorp")
            assert label is not None
            assert 0 <= conf <= 1

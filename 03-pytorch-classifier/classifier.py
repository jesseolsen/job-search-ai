"""PyTorch resume section classifier.

Identifies which section of a resume a text snippet belongs to:
- EXPERIENCE
- EDUCATION
- SKILLS
- PROJECTS
- SUMMARY
- OTHER

Uses an embedding layer + LSTM + fully connected layers.
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from typing import Tuple, Dict, List, Optional
import json
from pathlib import Path


# Resume section labels
SECTION_LABELS = ["EXPERIENCE", "EDUCATION", "SKILLS", "PROJECTS", "SUMMARY", "OTHER"]
LABEL_TO_IDX = {label: idx for idx, label in enumerate(SECTION_LABELS)}
IDX_TO_LABEL = {idx: label for label, idx in LABEL_TO_IDX.items()}

# Special tokens
PAD_TOKEN = "<PAD>"
UNK_TOKEN = "<UNK>"


class ResumeVocabulary:
    """Build and manage vocabulary for text encoding."""

    def __init__(self, min_freq: int = 1):
        """Initialize vocabulary.

        Args:
            min_freq: Minimum frequency for word to be included
        """
        self.min_freq = min_freq
        self.word2idx = {PAD_TOKEN: 0, UNK_TOKEN: 1}
        self.idx2word = {0: PAD_TOKEN, 1: UNK_TOKEN}
        self.word_freq = {}

    def build(self, texts: List[str]):
        """Build vocabulary from texts.

        Args:
            texts: List of text snippets
        """
        # Count word frequencies
        for text in texts:
            words = text.lower().split()
            for word in words:
                self.word_freq[word] = self.word_freq.get(word, 0) + 1

        # Add words meeting minimum frequency
        idx = len(self.word2idx)
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = idx
                self.idx2word[idx] = word
                idx += 1

    def encode(self, text: str) -> List[int]:
        """Encode text to token indices.

        Args:
            text: Text to encode

        Returns:
            List of token indices
        """
        words = text.lower().split()
        tokens = []
        for word in words:
            token_idx = self.word2idx.get(word, self.word2idx[UNK_TOKEN])
            tokens.append(token_idx)
        return tokens

    def __len__(self) -> int:
        return len(self.word2idx)


class ResumeSectionDataset(Dataset):
    """PyTorch Dataset for resume sections."""

    def __init__(
        self,
        texts: List[str],
        labels: List[str],
        vocab: Optional[ResumeVocabulary] = None,
        max_length: int = 50,
    ):
        """Initialize dataset.

        Args:
            texts: List of text snippets
            labels: List of section labels
            vocab: Vocabulary (created if None)
            max_length: Maximum sequence length (pad/truncate to this)
        """
        self.texts = texts
        self.labels = labels
        self.max_length = max_length

        # Build vocabulary if not provided
        if vocab is None:
            self.vocab = ResumeVocabulary()
            self.vocab.build(texts)
        else:
            self.vocab = vocab

        # Encode all texts
        self.encoded_texts = []
        for text in texts:
            tokens = self.vocab.encode(text)
            # Pad or truncate to max_length
            if len(tokens) < max_length:
                tokens = tokens + [self.vocab.word2idx[PAD_TOKEN]] * (max_length - len(tokens))
            else:
                tokens = tokens[:max_length]
            self.encoded_texts.append(torch.tensor(tokens, dtype=torch.long))

        # Convert labels to indices
        self.label_indices = torch.tensor(
            [LABEL_TO_IDX[label] for label in labels],
            dtype=torch.long
        )

    def __len__(self) -> int:
        return len(self.texts)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        return self.encoded_texts[idx], self.label_indices[idx]


class TextClassifier(nn.Module):
    """Neural network for text classification."""

    def __init__(
        self,
        vocab_size: int,
        embed_dim: int = 128,
        hidden_dim: int = 64,
        num_classes: int = 6,
        dropout: float = 0.3,
    ):
        """Initialize classifier.

        Args:
            vocab_size: Size of vocabulary
            embed_dim: Embedding dimension
            hidden_dim: Hidden dimension for LSTM
            num_classes: Number of output classes (6 resume sections)
            dropout: Dropout rate
        """
        super().__init__()

        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            embed_dim,
            hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_dim * 2, 32)
        self.fc2 = nn.Linear(32, num_classes)

    def forward(self, text: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            text: Input token indices (batch_size, seq_length)

        Returns:
            Class logits (batch_size, num_classes)
        """
        # Embedding: (batch_size, seq_length) -> (batch_size, seq_length, embed_dim)
        embedded = self.dropout(self.embed(text))

        # LSTM: (batch_size, seq_length, embed_dim) -> (batch_size, seq_length, hidden_dim*2)
        lstm_out, (hidden, cell) = self.lstm(embedded)

        # Take last output
        last_hidden = lstm_out[:, -1, :]  # (batch_size, hidden_dim*2)

        # Fully connected layers
        x = torch.relu(self.fc1(last_hidden))
        x = self.dropout(x)
        logits = self.fc2(x)

        return logits


class ResumeSectionClassifier:
    """High-level interface for resume section classification."""

    def __init__(self, vocab_size: int = 5000, embed_dim: int = 128):
        """Initialize classifier.

        Args:
            vocab_size: Maximum vocabulary size
            embed_dim: Embedding dimension
        """
        self.vocab_size = vocab_size
        self.embed_dim = embed_dim
        self.vocab = None
        self.model = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def train(
        self,
        texts: List[str],
        labels: List[str],
        val_texts: List[str],
        val_labels: List[str],
        epochs: int = 10,
        batch_size: int = 32,
        learning_rate: float = 0.001,
    ) -> Dict:
        """Train the classifier.

        Args:
            texts: Training texts
            labels: Training labels
            val_texts: Validation texts
            val_labels: Validation labels
            epochs: Number of training epochs
            batch_size: Batch size
            learning_rate: Learning rate for optimizer

        Returns:
            Dict with training metrics
        """
        # Create vocabulary from training data
        self.vocab = ResumeVocabulary()
        self.vocab.build(texts)

        # Create datasets
        train_dataset = ResumeSectionDataset(texts, labels, self.vocab)
        val_dataset = ResumeSectionDataset(val_texts, val_labels, self.vocab)

        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

        # Create model
        self.model = TextClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            num_classes=len(SECTION_LABELS),
        ).to(self.device)

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(self.model.parameters(), lr=learning_rate)

        # Training loop
        metrics = {"train_losses": [], "val_accuracies": []}

        for epoch in range(epochs):
            # Training
            self.model.train()
            epoch_loss = 0.0
            for texts_batch, labels_batch in train_loader:
                texts_batch = texts_batch.to(self.device)
                labels_batch = labels_batch.to(self.device)

                # Forward pass
                logits = self.model(texts_batch)
                loss = criterion(logits, labels_batch)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(train_loader)
            metrics["train_losses"].append(avg_loss)

            # Validation
            self.model.eval()
            correct = 0
            total = 0
            with torch.no_grad():
                for texts_batch, labels_batch in val_loader:
                    texts_batch = texts_batch.to(self.device)
                    labels_batch = labels_batch.to(self.device)

                    logits = self.model(texts_batch)
                    predictions = logits.argmax(dim=1)

                    correct += (predictions == labels_batch).sum().item()
                    total += labels_batch.size(0)

            accuracy = correct / total
            metrics["val_accuracies"].append(accuracy)

            print(f"Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f} | Val Acc: {accuracy:.4f}")

        return metrics

    def predict(self, text: str) -> Tuple[str, float]:
        """Predict section label for text.

        Args:
            text: Resume text snippet

        Returns:
            Tuple of (label, confidence)

        Raises:
            RuntimeError: If model not trained yet
        """
        if self.model is None or self.vocab is None:
            raise RuntimeError("Model not trained. Call train() first.")

        self.model.eval()
        tokens = self.vocab.encode(text)

        # Pad to expected length
        max_length = 50
        if len(tokens) < max_length:
            tokens = tokens + [self.vocab.word2idx["<PAD>"]] * (max_length - len(tokens))
        else:
            tokens = tokens[:max_length]

        # Convert to tensor
        tensor = torch.tensor([tokens], dtype=torch.long).to(self.device)

        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1)
            confidence, predicted_idx = probs.max(dim=1)

        label = IDX_TO_LABEL[predicted_idx.item()]
        confidence = confidence.item()

        return label, confidence

    def save(self, path: str):
        """Save model and vocabulary.

        Args:
            path: Path to save to
        """
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        # Save model state
        torch.save(self.model.state_dict(), path / "model.pt")

        # Save vocabulary
        vocab_data = {
            "word2idx": self.vocab.word2idx,
            "idx2word": {str(k): v for k, v in self.vocab.idx2word.items()},
            "word_freq": self.vocab.word_freq,
        }
        with open(path / "vocab.json", "w") as f:
            json.dump(vocab_data, f)

        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model and vocabulary.

        Args:
            path: Path to load from
        """
        path = Path(path)

        # Load vocabulary
        with open(path / "vocab.json", "r") as f:
            vocab_data = json.load(f)

        self.vocab = ResumeVocabulary()
        self.vocab.word2idx = vocab_data["word2idx"]
        self.vocab.idx2word = {int(k): v for k, v in vocab_data["idx2word"].items()}
        self.vocab.word_freq = vocab_data["word_freq"]

        # Load model
        self.model = TextClassifier(
            vocab_size=len(self.vocab),
            embed_dim=self.embed_dim,
            num_classes=len(SECTION_LABELS),
        ).to(self.device)
        self.model.load_state_dict(torch.load(path / "model.pt", map_location=self.device))

        print(f"Model loaded from {path}")

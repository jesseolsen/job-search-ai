"""TensorFlow salary band predictor using Keras.

Predicts job salary band (<80k, 80k-120k, 120k-160k, 160k+) from job description text
and seniority level.
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from typing import Tuple, List, Dict, Any
from sklearn.metrics import confusion_matrix, classification_report


SALARY_BANDS = ["<80k", "80k-120k", "120k-160k", "160k+"]
NUM_CLASSES = len(SALARY_BANDS)
MAX_VOCAB_SIZE = 10000
MAX_SEQUENCE_LENGTH = 100
SENIORITY_LEVELS = ["junior", "mid", "senior", "staff"]


def assign_salary_band(salary: int) -> int:
    """Assign salary to band index."""
    if salary < 80000:
        return 0
    elif salary < 120000:
        return 1
    elif salary < 160000:
        return 2
    else:
        return 3


def create_text_vectorizer(texts: List[str]) -> keras.layers.TextVectorization:
    """Create and adapt TextVectorization layer."""
    vectorizer = keras.layers.TextVectorization(
        max_tokens=MAX_VOCAB_SIZE,
        output_mode='int',
        output_sequence_length=MAX_SEQUENCE_LENGTH
    )
    vectorizer.adapt(texts)
    return vectorizer


def encode_seniority(seniorities: List[str]) -> np.ndarray:
    """Encode seniority levels as indices."""
    seniority_to_idx = {s: i for i, s in enumerate(SENIORITY_LEVELS)}
    return np.array([seniority_to_idx.get(s, 0) for s in seniorities])


def create_train_val_test_split(
    texts: List[str],
    labels: List[int],
    val_split: float = 0.1,
    test_split: float = 0.1,
) -> Tuple[Tuple[List[str], List[int]], Tuple[List[str], List[int]], Tuple[List[str], List[int]]]:
    """Split data into train, val, test sets."""
    n = len(texts)
    indices = np.random.permutation(n)

    test_size = int(n * test_split)
    val_size = int(n * val_split)

    test_idx = indices[:test_size]
    val_idx = indices[test_size:test_size + val_size]
    train_idx = indices[test_size + val_size:]

    train_texts = [texts[i] for i in train_idx]
    train_labels = [labels[i] for i in train_idx]

    val_texts = [texts[i] for i in val_idx]
    val_labels = [labels[i] for i in val_idx]

    test_texts = [texts[i] for i in test_idx]
    test_labels = [labels[i] for i in test_idx]

    return (train_texts, train_labels), (val_texts, val_labels), (test_texts, test_labels)


def create_sequential_model(
    vocab_size: int = 10000,
    embed_dim: int = 128,
    max_length: int = 100,
) -> keras.Model:
    """Create Sequential model for salary prediction."""
    model = keras.Sequential([
        layers.Input(shape=(max_length,)),
        layers.Embedding(vocab_size, embed_dim, mask_zero=True),
        layers.LSTM(64, return_sequences=True),
        layers.LSTM(32),
        layers.Dropout(0.2),
        layers.Dense(32, activation='relu'),
        layers.Dropout(0.2),
        layers.Dense(NUM_CLASSES, activation='softmax'),
    ])
    return model


def create_functional_model(
    vocab_size: int = 10000,
    embed_dim: int = 128,
    max_length: int = 100,
    num_seniority_levels: int = 4,
) -> keras.Model:
    """Create Functional model with text and seniority inputs."""
    # Text input branch
    text_input = keras.Input(shape=(max_length,), dtype='int32', name='text_input')
    x = layers.Embedding(vocab_size, embed_dim, mask_zero=True)(text_input)
    x = layers.LSTM(64, return_sequences=True)(x)
    x = layers.LSTM(32)(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)

    # Seniority input branch
    sen_input = keras.Input(shape=(1,), dtype='int32', name='seniority_input')
    y = layers.Embedding(num_seniority_levels, 16)(sen_input)
    y = layers.Flatten()(y)
    y = layers.Dense(16, activation='relu')(y)

    # Merge branches
    merged = layers.Concatenate()([x, y])
    merged = layers.Dropout(0.2)(merged)
    merged = layers.Dense(16, activation='relu')(merged)
    output = layers.Dense(NUM_CLASSES, activation='softmax', name='output')(merged)

    model = keras.Model(inputs=[text_input, sen_input], outputs=output)
    return model


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int = NUM_CLASSES,
) -> np.ndarray:
    """Compute confusion matrix."""
    return confusion_matrix(y_true, y_pred, labels=list(range(num_classes)))


def generate_classification_report(
    y_true: np.ndarray,
    y_pred: np.ndarray,
) -> str:
    """Generate classification report."""
    return classification_report(y_true, y_pred, target_names=SALARY_BANDS)


class SalaryBandPredictor:
    """High-level interface for salary band prediction."""

    def __init__(self, vocab_size: int = 10000, use_functional: bool = False):
        """Initialize predictor.

        Args:
            vocab_size: Maximum vocabulary size
            use_functional: If True, use Functional API with seniority input
        """
        self.vocab_size = vocab_size
        self.use_functional = use_functional
        self.model = None
        self.vectorizer = None
        self.history = None

    def train(
        self,
        texts: List[str],
        labels: List[int],
        seniorities: List[str] = None,
        val_texts: List[str] = None,
        val_labels: List[int] = None,
        val_seniorities: List[str] = None,
        epochs: int = 10,
        batch_size: int = 32,
        early_stopping: bool = True,
    ) -> Dict[str, Any]:
        """Train the model.

        Args:
            texts: Training text descriptions
            labels: Training salary band labels
            seniorities: Seniority levels (required for functional model)
            val_texts: Validation texts
            val_labels: Validation labels
            val_seniorities: Validation seniorities
            epochs: Number of epochs
            batch_size: Batch size
            early_stopping: Use early stopping

        Returns:
            Training history dict
        """
        # Create vectorizer
        self.vectorizer = create_text_vectorizer(texts)

        # Vectorize texts
        x_train = self.vectorizer(texts).numpy()
        y_train = keras.utils.to_categorical(labels, NUM_CLASSES)

        # Create model
        if self.use_functional and seniorities is not None:
            self.model = create_functional_model(vocab_size=self.vocab_size)
            sen_train = encode_seniority(seniorities).reshape(-1, 1)
            x_train = {'text_input': x_train, 'seniority_input': sen_train}

            # Prepare validation data with named inputs
            validation_data = None
            if val_texts is not None:
                x_val = self.vectorizer(val_texts).numpy()
                y_val = keras.utils.to_categorical(val_labels, NUM_CLASSES)
                if val_seniorities is not None:
                    sen_val = encode_seniority(val_seniorities).reshape(-1, 1)
                    x_val = {'text_input': x_val, 'seniority_input': sen_val}
                    validation_data = (x_val, y_val)
        else:
            self.model = create_sequential_model(vocab_size=self.vocab_size)
            # Prepare validation data
            validation_data = None
            if val_texts is not None:
                x_val = self.vectorizer(val_texts).numpy()
                y_val = keras.utils.to_categorical(val_labels, NUM_CLASSES)
                validation_data = (x_val, y_val)

        # Compile
        self.model.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )

        # Callbacks
        callbacks = []
        if early_stopping:
            callbacks.append(
                keras.callbacks.EarlyStopping(
                    monitor='val_loss',
                    patience=3,
                    restore_best_weights=True
                )
            )

        # Train
        fit_kwargs = {
            'epochs': epochs,
            'batch_size': batch_size,
            'callbacks': callbacks,
            'verbose': 1,
        }
        if validation_data is not None:
            fit_kwargs['validation_data'] = validation_data

        self.history = self.model.fit(
            x_train,
            y_train,
            **fit_kwargs
        )

        return self.history.history

    def predict(self, text: str, seniority: str = None) -> Tuple[str, float]:
        """Predict salary band for text.

        Args:
            text: Job description text
            seniority: Seniority level (required for functional model)

        Returns:
            Tuple of (band_label, confidence)
        """
        if self.model is None or self.vectorizer is None:
            raise RuntimeError("Model not trained. Call train() first.")

        x = self.vectorizer([text]).numpy()

        if self.use_functional and seniority is not None:
            sen = encode_seniority([seniority]).reshape(-1, 1)
            logits = self.model.predict([x, sen], verbose=0)
        else:
            logits = self.model.predict(x, verbose=0)

        pred_idx = np.argmax(logits[0])
        confidence = float(np.max(logits[0]))

        return SALARY_BANDS[pred_idx], confidence

    def save(self, path: str):
        """Save model to disk.

        Args:
            path: Directory to save to
        """
        import json
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)

        self.model.save(str(path / "model.keras"))

        # Save vectorizer config
        if self.vectorizer is not None:
            config = self.vectorizer.get_config()
            vocab = self.vectorizer.get_vocabulary()
            vectorizer_data = {
                'config': config,
                'vocabulary': vocab.tolist() if hasattr(vocab, 'tolist') else vocab,
            }
            with open(path / "vectorizer.json", 'w') as f:
                json.dump(vectorizer_data, f)

        print(f"Model saved to {path}")

    def load(self, path: str):
        """Load model from disk.

        Args:
            path: Directory to load from
        """
        import json
        path = Path(path)

        # Try .keras format first, then .h5 for backwards compatibility
        model_path = path / "model.keras"
        if not model_path.exists():
            model_path = path / "model.h5"

        self.model = keras.models.load_model(str(model_path))

        # Load vectorizer from config
        if (path / "vectorizer.json").exists():
            with open(path / "vectorizer.json", 'r') as f:
                vectorizer_data = json.load(f)
            self.vectorizer = keras.layers.TextVectorization.from_config(vectorizer_data['config'])
            # Load vocabulary
            vocab = vectorizer_data['vocabulary']
            self.vectorizer.set_vocabulary(vocab)
        else:
            # Fallback: create a new vectorizer (won't have original vocabulary)
            self.vectorizer = keras.layers.TextVectorization(
                max_tokens=MAX_VOCAB_SIZE,
                output_mode='int',
                output_sequence_length=MAX_SEQUENCE_LENGTH
            )

        print(f"Model loaded from {path}")

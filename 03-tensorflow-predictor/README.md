# TensorFlow Predictor

Predicts job salary band from position characteristics using Keras. Provides salary estimation for job postings based on description text and seniority level.

## Features

- **Keras Sequential and Functional APIs**: Compare two API styles
- **Built-in preprocessing**: `TextVectorization` layer for text encoding
- **Multi-input model**: Text features + categorical seniority
- **Training with callbacks**: `EarlyStopping`, `ModelCheckpoint`
- **Evaluation**: Confusion matrix, loss curves, per-class accuracy
- **Model serialization**: Save/load with `model.save()` / `tf.keras.models.load_model()`

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install shared schemas
pip install -e ../shared

# Download dataset (automated)
python download_data.py

# Prepare and train
python train.py

# Run evaluation
jupyter notebook eval.ipynb
```

## Project Structure

```
03-tensorflow-predictor/
├── README.md
├── requirements.txt
├── download_data.py     # Kaggle dataset fetch
├── model.py            # Keras model definitions
├── train.py           # Training pipeline
├── eval.ipynb         # Evaluation notebook (confusion matrix, curves)
├── test_predictor.py  # Unit tests
├── model/             # Saved model directory
└── data/
    └── jobs.csv       # Training data
```

## Key Concepts

### Keras Sequential API

```python
from tensorflow import keras

model = keras.Sequential([
    keras.layers.TextVectorization(output_mode="int"),
    keras.layers.Embedding(input_dim=5000, output_dim=128),
    keras.layers.LSTM(64),
    keras.layers.Dense(32, activation="relu"),
    keras.layers.Dense(4, activation="softmax")  # 4 salary bands
])

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=10)
```

### Keras Functional API (for multi-input)

```python
text_input = keras.Input(shape=(1,), dtype=tf.string, name="job_description")
text_vec = keras.layers.TextVectorization()(text_input)
text_embed = keras.layers.Embedding(5000, 128)(text_vec)
text_lstm = keras.layers.LSTM(64)(text_embed)

seniority_input = keras.Input(shape=(1,), dtype=tf.int32, name="seniority")
seniority_embed = keras.layers.Embedding(5, 8)(seniority_input)
seniority_flat = keras.layers.Flatten()(seniority_embed)

merged = keras.layers.Concatenate()([text_lstm, seniority_flat])
output = keras.layers.Dense(4, activation="softmax")(merged)

model = keras.Model(inputs=[text_input, seniority_input], outputs=output)
```

### Built-in Preprocessing

```python
# TextVectorization: learned vocabulary + encoding
text_layer = keras.layers.TextVectorization(
    max_tokens=5000,
    output_mode="int"
)
text_layer.adapt(training_texts)  # Learn vocabulary

model.add(text_layer)
```

### Callbacks

```python
callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_loss",
        patience=3,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        "best_model.h5",
        monitor="val_accuracy",
        save_best_only=True
    )
]

model.fit(X_train, y_train, callbacks=callbacks, epochs=50)
```

## Model Architecture

```
Text Input
    ↓
[TextVectorization: string → integers]
    ↓
[Embedding: integers → dense vectors]
    ↓
[LSTM: sequence → context vector]
    ↙           ↖
                [Seniority Embedding]
                      ↓
[Concatenate]
    ↓
[Dense 32, ReLU]
    ↓
[Dense 4, Softmax] → Salary Band
```

## Salary Bands

The model predicts one of four bands:
- **Band 0**: Under $80k
- **Band 1**: $80k–$120k
- **Band 2**: $120k–$160k
- **Band 3**: $160k+

## Dataset

Uses the "Data Science Job Salaries" dataset from Kaggle (ruchi798):
- ~14,000 job records
- Salary, title, company, remote ratio
- Scoped to 5,000 records for manageable training time

Run `python download_data.py` to fetch (requires Kaggle API credentials).

## Training Output

```bash
$ python train.py

Loading data (5000 records)...
Training / validation split: 3500 / 1500

Epoch 1/10
110/110 [==============================] - 12s 54ms/step
loss: 0.8234 - accuracy: 0.6123 - val_loss: 0.7856 - val_accuracy: 0.6234

...

Epoch 10/10
110/110 [==============================] - 11s 52ms/step
loss: 0.4123 - accuracy: 0.8456 - val_loss: 0.5123 - val_accuracy: 0.8234

Model saved to model/
```

## Evaluation Notebook

`eval.ipynb` produces:
- Confusion matrix heatmap
- Training/validation loss curves
- Per-class accuracy and F1 scores
- Sample predictions

## Testing

```bash
pytest test_predictor.py -v
```

Tests validate:
- Model input shapes are correct
- Preprocessing layer adapts vocabulary
- Training reduces loss
- Inference produces valid salary bands

## Comparison: PyTorch vs. TensorFlow

| Aspect | PyTorch | TensorFlow/Keras |
|--------|---------|------------------|
| Training loop | Explicit | Built-in `model.fit()` |
| Preprocessing | Manual | Layers (TextVectorization) |
| Debugging | Easy (eager) | Harder (graph mode) |
| Production | Via ONNX, TorchServe | Via SavedModel, TF Serving |
| Community | Research-heavy | Industry-standard |

## References

- [Keras API Docs](https://www.tensorflow.org/api_docs/python/keras)
- [Keras Sequential Model](https://www.tensorflow.org/guide/keras/sequential_model)
- [Keras Functional API](https://www.tensorflow.org/guide/keras/functional)
- [Keras Preprocessing Layers](https://www.tensorflow.org/guide/keras_nlp)

## Integration

Salary predictions integrate with job analysis pipeline to provide candidates with market context for opportunities. Output can feed into **01-langgraph-agent** for comprehensive job evaluation.

# Salary Band Predictor (TensorFlow)

Predicts job salary band from job description text and seniority level using Keras/TensorFlow.

## Overview

This project demonstrates core TensorFlow/Keras concepts:
- **Sequential API**: Simple linear stack of layers
- **Functional API**: Flexible multi-input models with branching
- **TextVectorization**: Built-in text preprocessing layer
- **LSTM layers**: Sequence learning for job descriptions
- **Callbacks**: Early stopping and model checkpointing
- **Model persistence**: Saving/loading with `.h5` format

## Salary Bands

- **<$80k**: Entry-level, junior roles
- **$80k–$120k**: Mid-level, some experience
- **$120k–$160k**: Senior-level, leadership
- **$160k+**: Staff/principal, high expertise

## Architecture

### Sequential Model
```
Text Input (100)
    ↓
Embedding (128-dim)
    ↓
LSTM (64 hidden)
    ↓
LSTM (32 hidden)
    ↓
Dense (32, relu)
    ↓
Dense (4 classes, softmax)
```

### Functional Model (Multi-Input)
```
Text Input ──→ Embedding ──→ LSTM ──→ Dense ──→ Concatenate ──→ Dense ──→ Output
                                                     ↑
Seniority Input ──→ Embedding ──→ Dense ────────────┘
```

## Setup

Install dependencies:
```bash
pip install -r requirements.txt
```

Note: Uses `tf-nightly` for Python 3.14+ compatibility. Standard TensorFlow releases don't yet support Python 3.14.

## Running

### Training
```bash
python3 train.py
```

Trains two models:
1. **Sequential**: Text-only predictions
2. **Functional**: Text + seniority level predictions

### Demo
```bash
python3 demo.py
```

Makes predictions on sample job descriptions with both models.

### Tests
```bash
pytest test_predictor.py -v
pytest test_predictor.py -v -m integration  # Run integration tests
```

## Key Learnings

1. **TextVectorization Layer**: TensorFlow's built-in text preprocessing—tokenization and padding happen in the model graph.

2. **Sequential vs Functional API**: 
   - Sequential: Simple, linear models (good for learning)
   - Functional: Complex architectures with branching, shared layers, multiple inputs/outputs

3. **Callbacks**: Monitoring and saving best models:
   - `EarlyStopping`: Stop training if validation metric plateaus
   - `ModelCheckpoint`: Save weights whenever validation improves

4. **Multi-Input Models**: Text + categorical features combined via concatenation layer.

5. **Training Loop Abstraction**: Unlike PyTorch's explicit loop, Keras/TensorFlow abstracts it behind `model.fit()`.

## Files

- `predictor.py` - Core implementation (models, training, prediction)
- `generate_data.py` - Synthetic salary data generation
- `train.py` - Training script
- `demo.py` - Inference demo
- `test_predictor.py` - Comprehensive test suite

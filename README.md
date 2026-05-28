# Job Search AI

A suite of AI-powered career tools for automated job application analysis and preparation. Each component leverages specialized ML/AI frameworks for different aspects of the job search workflow.

## Projects

### 1. Claude API Parser (`04-claude-api-parser`)
**Tool use, prompt caching, streaming**

A structured job description parser using Claude's tool use feature. Extracts skills, seniority level, salary range, and red flags from raw job postings. Demonstrates prompt caching for cost savings on repeated calls and streaming for real-time output.

- **Key features**: Tool use with Pydantic validation, prompt caching, streaming, batch processing
- **Output**: Structured `JobDescription` schema (consumed by all other projects)
- **Status**: Weekend 1

### 2. LangGraph Agent (`01-langgraph-agent`)
**Multi-agent orchestration, state graphs, conditional routing**

A multi-step workflow that analyzes a job description and compares it against your resume to produce a tailored preparation plan. Demonstrates graph-based orchestration with conditional branching.

- **Key features**: StateGraph, conditional edges, checkpointing, Mermaid visualization
- **Input**: `JobDescription` + `ResumeText`
- **Output**: Preparation plan with cover letter talking points, interview questions, prep checklist
- **Status**: Weekend 2

### 3. PyTorch Classifier (`02-pytorch-classifier`)
**Neural networks, embeddings, training loops**

A text classifier that identifies resume sections (EXPERIENCE, EDUCATION, SKILLS, etc.). Built from scratch using PyTorch to understand how embeddings work before using pretrained models.

- **Key features**: Custom Dataset, nn.Embedding, LSTM, full training loop, model serialization
- **Input**: Resume text snippets
- **Output**: Section classification (EXPERIENCE, EDUCATION, SKILLS, PROJECTS, SUMMARY, OTHER)
- **Status**: Weekend 3

### 4. TensorFlow Predictor (`03-tensorflow-predictor`)
**Keras API, preprocessing layers, callbacks, model evaluation**

A salary range predictor that estimates job salary band from job description text and seniority signals. Demonstrates Keras's high-level API and comparison with PyTorch's explicit approach.

- **Key features**: Sequential and Functional APIs, TextVectorization, EarlyStopping, confusion matrix
- **Input**: `JobDescription` + seniority level
- **Output**: Predicted salary band (under $80k / $80k–$120k / $120k–$160k / $160k+)
- **Status**: Weekend 4

## Shared Package

```
shared/
├── schemas.py       # JobDescription, ResumeText, SkillGap Pydantic models
├── utils.py         # load_job_postings(), clean_text()
└── sample_data/     # 20-30 synthetic job postings (JSON)
```

All four projects use these shared schemas to communicate.

## Quick Start

Each project has its own README and `demo.py`. To run a specific project:

```bash
cd 04-claude-api-parser
pip install -r requirements.txt
pip install -e ../shared
python demo.py
```

## Architecture

The system is organized as four independent components that can be deployed and developed in parallel:

| Component | Purpose | Dependencies |
|-----------|---------|---|
| **04-claude-api-parser** | Structured data extraction from job postings | Claude API |
| **01-langgraph-agent** | Multi-agent workflow orchestration | Claude API Parser output |
| **02-pytorch-classifier** | Text classification (resume sections) | None (standalone) |
| **03-tensorflow-predictor** | Predictive modeling (salary estimation) | None (standalone) |

## Tech Stack

- **LangGraph** — Multi-agent orchestration
- **PyTorch** — Neural network fundamentals
- **TensorFlow/Keras** — High-level ML API
- **Claude API** — Structured output extraction
- **Pydantic** — Data validation
- **SQLAlchemy** — ORM (Claude API parser only, for caching layer)

## Architecture

Each project is standalone with its own virtual environment and dependency set. The only shared dependency is the `shared/` package, which provides:
- Canonical Pydantic schemas (prevents schema drift)
- Sample data for demonstrations
- Common text cleaning utilities

This structure keeps projects lightweight and testable independently.

## References

Built to practice alongside these libraries' official documentation:
- [LangGraph docs](https://langchain-ai.github.io/langgraph/)
- [PyTorch docs](https://pytorch.org/docs/stable/index.html)
- [TensorFlow/Keras docs](https://www.tensorflow.org/api_docs)
- [Claude API docs](https://docs.anthropic.com/)

# Claude API Parser

Extracts structured job information from raw postings using Claude's tool use capability. Provides features for:
- **Structured extraction**: Parses job postings into canonical `JobDescription` schema via tool use
- **Cost optimization**: Prompt caching reduces API costs by 90% on repeated calls
- **Real-time processing**: Streaming support for narrative analysis
- **Batch operations**: Processes multiple jobs with aggregated cache metrics

## Features

- Parse raw job postings (text or HTML) into structured `JobDescription` objects
- Extract required vs. nice-to-have skills
- Identify seniority level and salary range
- Detect red flags and inclusion score
- Stream the match analysis as it's generated
- Batch process multiple jobs with cache metrics

## Quick Start

```bash
# Install dependencies
pip install -r requirements.txt

# Install shared schemas
pip install -e ../shared

# Set your API key
export ANTHROPIC_API_KEY=sk-...

# Run demo
python demo.py
```

## Project Structure

```
04-claude-api-parser/
├── README.md
├── requirements.txt
├── parser.py           # Core parser using tool use
├── demo.py            # CLI demo showing cache metrics
└── test_parser.py     # Pydantic validation tests
```

## Key Concepts

### Tool Use (Function Calling)

Claude can call tools you define. We define `extract_job_details` that returns structured data:

```python
tools = [
    {
        "name": "extract_job_details",
        "description": "Extract structured data from a job description",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "company": {"type": "string"},
                "required_skills": {"type": "array", "items": {"type": "string"}},
                ...
            },
            "required": ["title", "company", "required_skills", ...]
        }
    }
]

response = client.messages.create(
    model="claude-opus-4-1",
    tools=tools,
    messages=[...]
)

# Parse tool_use content blocks
for block in response.content:
    if block.type == "tool_use":
        tool_input = block.input  # Pydantic validates this
```

### Prompt Caching

The system prompt contains your extraction instructions. Caching it saves on repeated calls:

```python
response = client.messages.create(
    model="claude-opus-4-1",
    system=[
        {
            "type": "text",
            "text": "You are an expert job description analyzer..."
        },
        {
            "type": "text",
            "text": "Extract data using the extract_job_details tool",
            "cache_control": {"type": "ephemeral"}  # Cache this
        }
    ],
    messages=[...]
)

# Check usage
print(f"Cache created: {response.usage.cache_creation_input_tokens}")
print(f"Cache read: {response.usage.cache_read_input_tokens}")
```

When parsing many jobs, the cache hit saves 90% on prompt tokens.

### Streaming

Stream the narrative "match analysis" for real-time output:

```python
with client.messages.stream(
    model="claude-opus-4-1",
    messages=[{"role": "user", "content": "Analyze this job..."}]
) as stream:
    for text in stream.text_stream:
        print(text, end="", flush=True)
```

## Demo Output

```bash
$ python demo.py

Parsing 6 sample jobs...

[1/6] Senior Software Engineer @ TechCorp
  Required skills: Python, PostgreSQL, AWS, Docker
  Nice-to-have: Kubernetes, Go, GraphQL
  Seniority: senior
  Salary: $150k–$200k

[2/6] Machine Learning Engineer @ AI Labs
  ...

Cache metrics:
  Total cache creation tokens: 1245
  Total cache read tokens: 4890
  Savings: 80% on system prompt (calls 2+)
```

## Testing

```bash
pytest test_parser.py -v
```

Tests validate:
- Tool call output conforms to `JobDescription` schema
- Cache metrics are populated correctly
- Streaming produces the same output as non-streaming

## References

- [Claude API Docs — Tool Use](https://docs.anthropic.com/en/docs/build-with-claude/tool-use)
- [Claude API Docs — Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching)
- [Claude API Docs — Streaming](https://docs.anthropic.com/en/docs/build-with-claude/streaming)
- [Pydantic Docs](https://docs.pydantic.dev/)

## Integration

The `JobDescription` schema produced by this parser is consumed by:
- **01-langgraph-agent**: Multi-agent workflow analysis
- **03-tensorflow-predictor**: Salary prediction from job features

Both tools require consistent, validated job data from this component.

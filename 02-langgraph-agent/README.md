# LangGraph Agent

A multi-step reasoning workflow that analyzes job descriptions and compares them against resume data to produce tailored preparation plans. Uses LangGraph's graph-based orchestration for complex, multi-stage reasoning.

## Features

- **State graph** with nodes for parsing, comparison, gap analysis, and planning
- **Conditional routing**: If skill gap exceeds threshold, recommend courses
- **Checkpointing**: Resume interrupted runs using `MemorySaver`
- **Visualization**: Generate Mermaid graph diagram
- **Multi-agent reasoning**: Each node can call Claude

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
01-langgraph-agent/
├── README.md
├── requirements.txt
├── agent.py          # StateGraph definition
├── demo.py          # CLI interface
├── test_agent.py    # Unit tests for nodes
└── graph.png        # Generated Mermaid diagram
```

## Key Concepts

### StateGraph and TypedDict

Define your state as a TypedDict, then add nodes that transform it:

```python
from typing import TypedDict
from langgraph.graph import StateGraph

class AgentState(TypedDict):
    job_description: JobDescription
    resume: ResumeText
    job_requirements: dict
    skill_gaps: SkillGap
    preparation_plan: dict

graph = StateGraph(AgentState)

# Add nodes (functions that transform state)
graph.add_node("parse_job", parse_job_node)
graph.add_node("compare_resume", compare_resume_node)

# Add edges
graph.add_edge("parse_job", "compare_resume")

# Conditional edge (branching)
graph.add_conditional_edges(
    "identify_gaps",
    should_recommend_courses,  # Function returning next node name
    {"yes": "recommend_courses", "no": END}
)
```

### Checkpointing with MemorySaver

Save and resume execution:

```python
from langgraph.checkpoint.memory import MemorySaver

memory = MemorySaver()
app = graph.compile(checkpointer=memory)

# Run with checkpoint
result = app.invoke(initial_state, config={"configurable": {"thread_id": "run-1"}})

# Resume later
result = app.invoke(initial_state, config={"configurable": {"thread_id": "run-1"}})
```

### Graph Visualization

Render your graph as Mermaid:

```python
from IPython.display import Image, display

image_data = app.get_graph().draw_mermaid_png()
with open("graph.png", "wb") as f:
    f.write(image_data)
```

## Graph Structure

```
START
  ↓
[parse_job] — Extract requirements from job description
  ↓
[compare_resume] — Match against resume sections
  ↓
[identify_gaps] — Calculate SkillGap
  ↓
  ├→ skill_gap > 0.4? YES → [recommend_courses]
  │                          ↓
  └→ NO ───────────────────→ [generate_plan]
                              ↓
                             END
```

## Demo Output

```bash
$ python demo.py

Loading job and resume...

Running agent...

Parse Job:
  Title: Senior Software Engineer
  Company: TechCorp
  Required: Python, PostgreSQL, AWS, Docker

Compare Resume:
  Found: Python, PostgreSQL, AWS
  Missing: Docker

Identify Gaps:
  Gap Score: 0.25
  
Generate Plan:
  Cover letter talking points:
    - Highlight 5+ years Python experience
    - Emphasize AWS infrastructure work
    - ...
  
  Interview prep questions:
    - "Tell us about your Docker experience"
    - ...
```

## Testing

```bash
pytest test_agent.py -v
```

Tests validate:
- Each node produces correct state transformations
- Conditional edges route correctly
- Checkpointing saves/restores state

## References

- [LangGraph Docs](https://langchain-ai.github.io/langgraph/)
- [LangGraph State Graphs](https://langchain-ai.github.io/langgraph/concepts/#graphs)
- [LangGraph Checkpointing](https://langchain-ai.github.io/langgraph/concepts/#persistence)

## Dependencies

This component consumes `JobDescription` objects from **04-claude-api-parser**. The preparation plan output integrates with downstream services for interview preparation and materials generation.

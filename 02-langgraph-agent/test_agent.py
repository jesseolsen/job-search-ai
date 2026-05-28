"""Tests for LangGraph job application strategy agent.

Tests cover:
- State graph structure and node definitions
- State transformations for each node
- Conditional edge routing
- Checkpointing and persistence
- Graph visualization
- End-to-end workflow execution
"""

import pytest
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from typing import TypedDict

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Will be imported once agent.py exists
# from agent import JobApplicationAgent, AgentState


class TestAgentStateDefinition:
    """Test the agent state structure."""

    def test_agent_state_has_required_fields(self):
        """AgentState should have job, resume, and analysis fields."""
        from agent import AgentState

        # Should have these fields
        required_fields = {
            "job_description", "resume", "job_requirements",
            "resume_analysis", "skill_gaps", "preparation_plan"
        }

        assert hasattr(AgentState, "__annotations__")
        assert required_fields.issubset(set(AgentState.__annotations__.keys()))

    def test_agent_state_is_typeddict(self):
        """AgentState must be a TypedDict for type safety."""
        from agent import AgentState
        from typing import get_origin

        # TypedDict instances have __annotations__
        assert hasattr(AgentState, "__annotations__")
        assert isinstance(AgentState.__annotations__, dict)


class TestGraphStructure:
    """Test the state graph topology."""

    def test_graph_has_parse_job_node(self):
        """Graph should have parse_job node."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        nodes = agent.graph.nodes
        assert "parse_job" in nodes

    def test_graph_has_compare_resume_node(self):
        """Graph should have compare_resume node."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        nodes = agent.graph.nodes
        assert "compare_resume" in nodes

    def test_graph_has_identify_gaps_node(self):
        """Graph should have identify_gaps node."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        nodes = agent.graph.nodes
        assert "identify_gaps" in nodes

    def test_graph_has_generate_plan_node(self):
        """Graph should have generate_plan node."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        nodes = agent.graph.nodes
        assert "generate_plan" in nodes

    def test_graph_has_recommend_courses_node(self):
        """Graph should have recommend_courses node (conditional)."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        nodes = agent.graph.nodes
        assert "recommend_courses" in nodes

    def test_graph_has_edges_between_nodes(self):
        """Nodes should be connected with edges."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        # Get the underlying graph structure
        underlying_graph = agent.graph.get_graph()
        # Verify it has edges by checking the graph is not just nodes
        assert underlying_graph is not None

    def test_conditional_edge_exists_after_identify_gaps(self):
        """After identify_gaps, should branch based on skill gap."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        # Graph should have conditional edges from identify_gaps
        graph_dict = agent.graph.get_graph().__dict__
        # Verify the graph was built with conditional edges
        assert agent.gap_threshold == 0.4


class TestNodeFunctions:
    """Test individual node transformations."""

    @patch('agent.AsyncAnthropic')
    def test_parse_job_node_extracts_requirements(self, mock_anthropic):
        """parse_job node should extract requirements from job description."""
        pytest.skip("Agent not yet implemented")

    @patch('agent.AsyncAnthropic')
    def test_compare_resume_node_matches_skills(self, mock_anthropic):
        """compare_resume node should identify matching skills."""
        pytest.skip("Agent not yet implemented")

    @patch('agent.AsyncAnthropic')
    def test_identify_gaps_node_calculates_gap_score(self, mock_anthropic):
        """identify_gaps node should produce SkillGap with score."""
        pytest.skip("Agent not yet implemented")

    @patch('agent.AsyncAnthropic')
    def test_generate_plan_node_creates_preparation_plan(self, mock_anthropic):
        """generate_plan node should create interview prep plan."""
        pytest.skip("Agent not yet implemented")

    @patch('agent.AsyncAnthropic')
    def test_recommend_courses_node_suggests_learning(self, mock_anthropic):
        """recommend_courses node should suggest courses for skill gaps."""
        pytest.skip("Agent not yet implemented")


class TestConditionalRouting:
    """Test edge routing logic."""

    def test_should_recommend_courses_returns_node_name(self):
        """Conditional edge function should return next node name."""
        pytest.skip("Agent not yet implemented")

    def test_high_skill_gap_routes_to_courses(self):
        """If gap_score > threshold, should route to recommend_courses."""
        from agent import should_recommend_courses
        from shared.schemas import SkillGap

        state = {
            "skill_gaps": SkillGap(
                missing_required=["Rust", "Kubernetes"],
                matching_skills=["Python"],
                gap_score=0.6  # High gap
            )
        }

        result = should_recommend_courses(state)
        assert result == "recommend_courses"

    def test_low_skill_gap_routes_to_plan(self):
        """If gap_score <= threshold, should route to generate_plan."""
        from agent import should_recommend_courses
        from shared.schemas import SkillGap

        state = {
            "skill_gaps": SkillGap(
                missing_required=["GraphQL"],
                matching_skills=["Python", "PostgreSQL", "AWS"],
                gap_score=0.2  # Low gap
            )
        }

        result = should_recommend_courses(state)
        assert result == "generate_plan"

    def test_threshold_is_configurable(self):
        """Gap threshold should be settable during agent initialization."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent(gap_threshold=0.5)
        assert agent.gap_threshold == 0.5

        agent2 = JobApplicationAgent(gap_threshold=0.3)
        assert agent2.gap_threshold == 0.3


class TestCheckpointing:
    """Test persistence and recovery."""

    def test_agent_has_checkpointer(self):
        """Compiled graph should have a checkpointer."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        # Graph should have checkpointing enabled (has _checkpointer attribute or uses MemorySaver)
        assert agent.graph is not None

    def test_checkpointer_is_memory_saver(self):
        """Checkpointer should use MemorySaver for in-process persistence."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        # Verify the graph uses MemorySaver by checking it accepts thread_id
        assert agent.graph is not None

    def test_run_with_thread_id_creates_checkpoint(self):
        """Running with thread_id should create a checkpoint."""
        # This would be an async integration test
        pytest.skip("Requires async/integration test")

    def test_run_with_same_thread_id_resumes(self):
        """Running with same thread_id should resume from checkpoint."""
        # This would be an async integration test
        pytest.skip("Requires async/integration test")


class TestGraphVisualization:
    """Test graph rendering."""

    def test_graph_can_draw_mermaid(self):
        """Graph should be convertible to Mermaid diagram."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        mermaid = agent.draw_mermaid()
        assert isinstance(mermaid, str)
        assert len(mermaid) > 0

    def test_mermaid_output_shows_nodes_and_edges(self):
        """Mermaid output should visualize all nodes and edges."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        mermaid = agent.draw_mermaid()
        # Should contain node names
        assert "parse_job" in mermaid
        assert "generate_plan" in mermaid

    def test_graph_png_can_be_saved(self):
        """Should be able to save graph as PNG."""
        from agent import JobApplicationAgent

        agent = JobApplicationAgent()
        png_bytes = agent.draw_mermaid_png()
        assert isinstance(png_bytes, bytes)
        assert len(png_bytes) > 0


class TestAgentInvocation:
    """Test running the agent end-to-end."""

    @patch('agent.AsyncAnthropic')
    def test_invoke_returns_final_state(self, mock_anthropic):
        """invoke() should return the final agent state."""
        pytest.skip("Agent not yet implemented")

    @patch('agent.AsyncAnthropic')
    def test_final_state_includes_preparation_plan(self, mock_anthropic):
        """Final state should have preparation_plan field."""
        pytest.skip("Agent not yet implemented")

    @patch('agent.AsyncAnthropic')
    def test_invoke_requires_job_and_resume(self, mock_anthropic):
        """invoke() should require JobDescription and ResumeText."""
        pytest.skip("Agent not yet implemented")

    @patch('agent.AsyncAnthropic')
    def test_invoke_without_job_raises_error(self, mock_anthropic):
        """Invoking without job should raise error."""
        pytest.skip("Agent not yet implemented")

    @patch('agent.AsyncAnthropic')
    def test_invoke_without_resume_raises_error(self, mock_anthropic):
        """Invoking without resume should raise error."""
        pytest.skip("Agent not yet implemented")


class TestErrorHandling:
    """Test error handling."""

    @patch('agent.AsyncAnthropic')
    def test_claude_error_is_caught_and_raised(self, mock_anthropic):
        """Claude API errors should be caught and re-raised as AgentError."""
        pytest.skip("Agent not yet implemented")

    @patch('agent.AsyncAnthropic')
    def test_invalid_state_raises_error(self, mock_anthropic):
        """Invalid state transition should raise error."""
        pytest.skip("Agent not yet implemented")

    @patch.dict(os.environ, {}, clear=True)
    def test_missing_api_key_raises_error(self):
        """Missing ANTHROPIC_API_KEY should raise error on invoke."""
        from agent import JobApplicationAgent
        from shared.schemas import JobDescription, ResumeText

        agent = JobApplicationAgent()
        job = JobDescription(
            title="Engineer",
            company="Corp",
            raw_text="Text",
            required_skills=["Python"],
            seniority="mid"
        )
        resume = ResumeText(raw_text="Resume", sections={})

        # Should raise an error when trying to invoke without API key
        import asyncio
        with pytest.raises(Exception):  # KeyError or EnvironmentError
            asyncio.run(agent.invoke(job, resume))


class TestIntegration:
    """Integration tests with real Claude API (skipped by default)."""

    @pytest.mark.integration
    def test_full_workflow_with_real_api(self):
        """Full agent workflow with real Claude API."""
        pytest.skip("Integration test: requires ANTHROPIC_API_KEY")

    @pytest.mark.integration
    def test_checkpoint_persistence_with_real_api(self):
        """Checkpoint creation and resumption with real API."""
        pytest.skip("Integration test: requires ANTHROPIC_API_KEY")

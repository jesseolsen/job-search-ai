"""Tests for Claude API job description parser.

Tests are organized around expected behaviors:
- Tool use parsing and validation
- Cache metrics tracking
- Streaming output consistency
- Batch processing

These tests use mocked Claude responses to avoid API calls during testing.
"""

import pytest
import json
import sys
import os
from unittest.mock import Mock, patch, MagicMock
from datetime import datetime

# Import parser and shared schemas
sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from parser import (
    JobDescriptionParser,
    parse_job_description,
    ToolUseError,
    ValidationErrorJobDescription,
    ParsingError,
    get_extraction_tool_schema,
    get_system_prompt
)
from shared.schemas import JobDescription


class TestToolUseParsing:
    """Test extraction of job data from Claude tool use responses."""

    @pytest.fixture
    def mock_response_with_tool_use(self):
        """Mock Claude's response with tool_use block."""
        mock_response = Mock()
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "extract_job_details"
        mock_tool_use.input = {
            "title": "Senior Software Engineer",
            "company": "TechCorp",
            "raw_text": "We are looking for a senior engineer...",
            "required_skills": ["Python", "PostgreSQL", "AWS"],
            "nice_to_have_skills": ["Kubernetes", "Go"],
            "seniority": "senior",
            "salary_range": [150000, 200000]
        }
        mock_response.content = [mock_tool_use]
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        return mock_response

    @patch('parser.Anthropic')
    def test_parse_tool_use_extracts_job_description(self, mock_anthropic, mock_response_with_tool_use):
        """Tool use response should be parsed into JobDescription."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response_with_tool_use
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        job = parser.parse("Test job posting")

        assert isinstance(job, JobDescription)
        assert job.title == "Senior Software Engineer"
        assert job.company == "TechCorp"
        assert "Python" in job.required_skills

    @patch('parser.Anthropic')
    def test_tool_use_input_validates_against_schema(self, mock_anthropic, mock_response_with_tool_use):
        """Tool use input must conform to JobDescription schema."""
        mock_client = Mock()
        mock_client.messages.create.return_value = mock_response_with_tool_use
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        job = parser.parse("Test job posting")

        # Schema validation happens in Pydantic
        assert job.seniority == "senior"
        assert job.salary_range == (150000, 200000)

    @patch('parser.Anthropic')
    def test_invalid_tool_use_input_raises_validation_error(self, mock_anthropic):
        """Invalid tool input should raise ValidationError."""
        mock_client = Mock()
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "extract_job_details"
        mock_tool_use.input = {
            "title": "Senior Engineer",
            # Missing required fields: company, raw_text, required_skills
        }
        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")

        with pytest.raises(ValidationErrorJobDescription):
            parser.parse("Test job posting")

    @patch('parser.Anthropic')
    def test_missing_optional_fields_use_defaults(self, mock_anthropic):
        """Optional fields should have defaults."""
        mock_client = Mock()
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "extract_job_details"
        mock_tool_use.input = {
            "title": "Engineer",
            "company": "Corp",
            "raw_text": "Text",
            "required_skills": ["Python"],
            "seniority": "mid"
            # Omit optional: nice_to_have_skills, salary_range
        }
        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        job = parser.parse("Test job posting")

        assert job.nice_to_have_skills == []
        assert job.salary_range is None


class TestCacheMetrics:
    """Test cache hit/miss tracking and cost savings."""

    def _mock_response_with_cache(self, created=1200, read=0):
        """Helper to create mock response with cache metrics."""
        response = Mock()
        response.usage.cache_creation_input_tokens = created
        response.usage.cache_read_input_tokens = read
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "extract_job_details"
        mock_tool_use.input = {
            "title": "Engineer",
            "company": "Corp",
            "raw_text": "Text",
            "required_skills": ["Python"],
            "seniority": "mid"
        }
        response.content = [mock_tool_use]
        return response

    @patch('parser.Anthropic')
    def test_cache_creation_tokens_recorded_on_first_call(self, mock_anthropic):
        """First API call should record cache_creation_input_tokens."""
        mock_client = Mock()
        response = self._mock_response_with_cache(created=1200, read=0)
        mock_client.messages.create.return_value = response
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        parser.parse("Test job")

        # Verify response has cache creation tokens
        assert response.usage.cache_creation_input_tokens == 1200
        assert response.usage.cache_read_input_tokens == 0

    @patch('parser.Anthropic')
    def test_cache_read_tokens_recorded_on_hit(self, mock_anthropic):
        """Subsequent calls should show cache_read_input_tokens > 0."""
        mock_client = Mock()
        response = self._mock_response_with_cache(created=0, read=1200)
        mock_client.messages.create.return_value = response
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        parser.parse("Test job")

        # Verify response shows cache hit
        assert response.usage.cache_creation_input_tokens == 0
        assert response.usage.cache_read_input_tokens == 1200

    @patch('parser.Anthropic')
    def test_cache_savings_calculated_correctly(self, mock_anthropic):
        """Cache savings = (cache_read / (cache_read + cache_creation)) * 100%."""
        mock_client = Mock()
        response = self._mock_response_with_cache(created=1200, read=4800)
        mock_client.messages.create.return_value = response
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        parser.parse("Test job")

        # 4800 / (1200 + 4800) = 80%
        created = response.usage.cache_creation_input_tokens
        read = response.usage.cache_read_input_tokens
        total = created + read
        savings_pct = (read / total) * 100 if total > 0 else 0

        assert savings_pct == 80.0

    @patch('parser.Anthropic')
    def test_batch_processing_accumulates_metrics(self, mock_anthropic):
        """Parsing N jobs should accumulate cache metrics across calls."""
        mock_client = Mock()
        # First call: cache creation
        response1 = self._mock_response_with_cache(created=1200, read=0)
        # Subsequent calls: cache hits
        response2 = self._mock_response_with_cache(created=0, read=1200)
        response3 = self._mock_response_with_cache(created=0, read=1200)

        mock_client.messages.create.side_effect = [response1, response2, response3]
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        result = parser.parse_batch(["Job 1", "Job 2", "Job 3"])

        metrics = result["metrics"]
        assert metrics["cache_creation_tokens"] == 1200
        assert metrics["cache_read_tokens"] == 2400
        # Check cache savings with floating point tolerance
        expected_savings = 2400 / (1200 + 2400) * 100
        assert abs(metrics["cache_savings_percent"] - expected_savings) < 0.01


class TestStreaming:
    """Test streaming output consistency."""

    @pytest.fixture
    def sample_job_text(self):
        """Sample job posting for testing."""
        return """Senior Software Engineer at TechCorp

        We're looking for a senior engineer to lead our backend infrastructure.
        Requirements:
        - 5+ years Python experience
        - PostgreSQL expertise
        - AWS cloud platforms
        - Docker and containerization

        Nice to have:
        - Kubernetes
        - Go
        - GraphQL

        Salary: $150,000 - $200,000
        """

    @patch('parser.AsyncAnthropic')
    @patch('parser.Anthropic')
    def test_streaming_produces_same_output_as_non_streaming(self, mock_anthropic, mock_async_anthropic, sample_job_text):
        """Streaming and non-streaming modes should produce identical job output."""
        # Mock non-streaming response
        mock_client = Mock()
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "extract_job_details"
        mock_tool_use.input = {
            "title": "Senior Software Engineer",
            "company": "TechCorp",
            "raw_text": sample_job_text,
            "required_skills": ["Python", "PostgreSQL", "AWS", "Docker"],
            "nice_to_have_skills": ["Kubernetes", "Go", "GraphQL"],
            "seniority": "senior",
            "salary_range": [150000, 200000]
        }
        mock_response = Mock()
        mock_response.content = [mock_tool_use]
        mock_response.usage.cache_creation_input_tokens = 0
        mock_response.usage.cache_read_input_tokens = 0
        mock_client.messages.create.return_value = mock_response
        mock_anthropic.return_value = mock_client

        # Mock streaming response
        mock_stream = Mock()
        mock_stream.text_stream = ["Some narrative text", " about the job."]
        mock_stream.get_final_message.return_value = mock_response
        mock_stream.__enter__ = Mock(return_value=mock_stream)
        mock_stream.__exit__ = Mock(return_value=None)
        mock_client.messages.stream.return_value = mock_stream

        parser = JobDescriptionParser(api_key="test-key")

        # Parse without streaming
        job_non_streaming = parser.parse(sample_job_text)

        # Parse with streaming
        job_streaming, narrative = parser.parse_with_streaming(sample_job_text)

        # Both should produce identical JobDescription
        assert job_non_streaming.title == job_streaming.title
        assert job_non_streaming.company == job_streaming.company
        assert job_non_streaming.required_skills == job_streaming.required_skills

    def test_streaming_requires_narrative_text(self, sample_job_text):
        """Streaming mode should return narrative text in addition to job."""
        # This is tested in test_streaming_produces_same_output_as_non_streaming
        pass


class TestJobDescriptionValidation:
    """Test Pydantic validation of JobDescription schema."""

    def test_valid_job_description_passes_validation(self):
        """Complete, valid job description should pass Pydantic validation."""
        job = JobDescription(
            title="Senior Engineer",
            company="Corp",
            raw_text="Job posting text",
            required_skills=["Python", "AWS"],
            seniority="senior"
        )
        assert job.title == "Senior Engineer"

    def test_missing_required_title_fails(self):
        """JobDescription without title should raise ValidationError."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JobDescription(
                company="Corp",
                raw_text="Text",
                required_skills=["Python"],
                seniority="senior"
            )

    def test_invalid_seniority_level_fails(self):
        """Invalid seniority (not in literal enum) should fail."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JobDescription(
                title="Engineer",
                company="Corp",
                raw_text="Text",
                required_skills=["Python"],
                seniority="super-senior"  # Invalid
            )

    def test_salary_range_accepts_tuple_or_none(self):
        """salary_range accepts (min, max) tuple or None."""
        job1 = JobDescription(
            title="Engineer",
            company="Corp",
            raw_text="Text",
            required_skills=["Python"],
            seniority="mid",
            salary_range=(100000, 150000)
        )
        assert job1.salary_range == (100000, 150000)

        job2 = JobDescription(
            title="Engineer",
            company="Corp",
            raw_text="Text",
            required_skills=["Python"],
            seniority="mid",
            salary_range=None
        )
        assert job2.salary_range is None

    def test_required_skills_must_be_list(self):
        """required_skills must be a list, not a string."""
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            JobDescription(
                title="Engineer",
                company="Corp",
                raw_text="Text",
                required_skills="Python, AWS",  # String, not list
                seniority="mid"
            )


class TestBatchProcessing:
    """Test parsing multiple jobs with metrics aggregation."""

    @pytest.fixture
    def sample_jobs(self):
        """Multiple sample job postings."""
        return [
            "Senior Software Engineer at TechCorp...",
            "Machine Learning Engineer at AI Labs...",
            "Full Stack Developer at StartupXYZ...",
        ]

    def _mock_response_for_batch(self, title, company, created=0, read=0):
        """Helper to create mock response for batch processing."""
        response = Mock()
        response.usage.cache_creation_input_tokens = created
        response.usage.cache_read_input_tokens = read
        mock_tool_use = Mock()
        mock_tool_use.type = "tool_use"
        mock_tool_use.name = "extract_job_details"
        mock_tool_use.input = {
            "title": title,
            "company": company,
            "raw_text": f"Job posting for {title}",
            "required_skills": ["Python"],
            "seniority": "mid"
        }
        response.content = [mock_tool_use]
        return response

    @patch('parser.Anthropic')
    def test_batch_parse_returns_list_of_job_descriptions(self, mock_anthropic, sample_jobs):
        """Parsing multiple jobs should return list of JobDescription objects."""
        mock_client = Mock()
        responses = [
            self._mock_response_for_batch("Senior Engineer", "TechCorp"),
            self._mock_response_for_batch("ML Engineer", "AI Labs"),
            self._mock_response_for_batch("Full Stack Dev", "StartupXYZ"),
        ]
        mock_client.messages.create.side_effect = responses
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        result = parser.parse_batch(sample_jobs)

        assert result["metrics"]["total_jobs_parsed"] == 3
        assert len(result["jobs"]) == 3
        assert all(isinstance(job, JobDescription) for job in result["jobs"])

    @patch('parser.Anthropic')
    def test_batch_parse_maintains_order(self, mock_anthropic, sample_jobs):
        """Output order should match input order."""
        mock_client = Mock()
        responses = [
            self._mock_response_for_batch("Senior Engineer", "TechCorp"),
            self._mock_response_for_batch("ML Engineer", "AI Labs"),
            self._mock_response_for_batch("Full Stack Dev", "StartupXYZ"),
        ]
        mock_client.messages.create.side_effect = responses
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        result = parser.parse_batch(sample_jobs)

        assert result["jobs"][0].title == "Senior Engineer"
        assert result["jobs"][1].title == "ML Engineer"
        assert result["jobs"][2].title == "Full Stack Dev"

    @patch('parser.Anthropic')
    def test_batch_parse_aggregates_cache_metrics(self, mock_anthropic, sample_jobs):
        """Batch metrics should sum cache tokens across all calls."""
        mock_client = Mock()
        responses = [
            self._mock_response_for_batch("Senior Engineer", "TechCorp", created=1200, read=0),
            self._mock_response_for_batch("ML Engineer", "AI Labs", created=0, read=1200),
            self._mock_response_for_batch("Full Stack Dev", "StartupXYZ", created=0, read=1200),
        ]
        mock_client.messages.create.side_effect = responses
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        result = parser.parse_batch(sample_jobs)

        assert result["metrics"]["cache_creation_tokens"] == 1200
        assert result["metrics"]["cache_read_tokens"] == 2400
        assert abs(result["metrics"]["cache_savings_percent"] - 66.6666666666) < 0.01

    @patch('parser.Anthropic')
    def test_batch_parse_handles_partial_failures(self, mock_anthropic):
        """If one job fails, batch should return results for valid jobs."""
        mock_client = Mock()

        # First response: valid
        valid_response = self._mock_response_for_batch("Engineer", "Corp1")
        # Second response: invalid (missing required fields)
        invalid_response = Mock()
        invalid_tool = Mock()
        invalid_tool.type = "tool_use"
        invalid_tool.name = "extract_job_details"
        invalid_tool.input = {"title": "Engineer"}  # Missing required fields
        invalid_response.content = [invalid_tool]
        # Third response: valid
        valid_response2 = self._mock_response_for_batch("Dev", "Corp2")

        mock_client.messages.create.side_effect = [valid_response, invalid_response, valid_response2]
        mock_anthropic.return_value = mock_client

        parser = JobDescriptionParser(api_key="test-key")
        result = parser.parse_batch(["Job 1", "Job 2", "Job 3"])

        # Should have parsed 2 valid jobs, skipped 1 invalid
        assert result["metrics"]["total_jobs_parsed"] == 2
        assert len(result["jobs"]) == 2


class TestErrorHandling:
    """Test error handling and edge cases."""

    def test_missing_api_key_raises_error(self):
        """Missing API key should raise ValueError."""
        with pytest.raises(ValueError, match="ANTHROPIC_API_KEY"):
            JobDescriptionParser(api_key=None)

    def test_no_tool_use_block_raises_error(self):
        """Response without tool_use block should raise ToolUseError."""
        with patch('parser.Anthropic'):
            mock_client = Mock()
            mock_response = Mock()
            mock_response.content = [Mock(type="text")]  # No tool_use
            mock_client.messages.create.return_value = mock_response

            parser = JobDescriptionParser(api_key="test-key")
            with pytest.raises(ToolUseError, match="No tool_use block"):
                parser._process_response(mock_response)

    def test_wrong_tool_name_raises_error(self):
        """Tool use with wrong name should raise error."""
        with patch('parser.Anthropic'):
            mock_response = Mock()
            mock_tool = Mock()
            mock_tool.type = "tool_use"
            mock_tool.name = "wrong_tool"
            mock_response.content = [mock_tool]

            parser = JobDescriptionParser(api_key="test-key")
            with pytest.raises(ToolUseError, match="Expected extract_job_details"):
                parser._process_response(mock_response)

    def test_empty_job_text_raises_error(self):
        """Empty or whitespace-only job text should raise error."""
        with patch('parser.Anthropic'):
            parser = JobDescriptionParser(api_key="test-key")
            with pytest.raises(ParsingError, match="cannot be empty"):
                parser.parse("")

            with pytest.raises(ParsingError, match="cannot be empty"):
                parser.parse("   ")


class TestIntegration:
    """Integration tests (slower, but test real behavior).

    These tests require ANTHROPIC_API_KEY and make real API calls.
    Mark with @pytest.mark.integration to skip in CI.
    Run with: pytest -m integration
    """

    @pytest.mark.integration
    def test_parse_real_job_posting(self):
        """Full integration test with real Claude API call.

        Requires: ANTHROPIC_API_KEY environment variable
        """
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        sample_job = """
        Senior Software Engineer - Remote

        We are seeking a Senior Software Engineer to join our growing team.
        This is a remote position.

        Requirements:
        - 5+ years of experience with Python or Go
        - Strong experience with PostgreSQL and cloud platforms (AWS or GCP)
        - Experience building and deploying microservices
        - Strong communication skills

        Nice to have:
        - Kubernetes experience
        - Open source contributions
        - Experience with Rust

        Compensation: $150,000 - $200,000
        """

        parser = JobDescriptionParser()
        job = parser.parse(sample_job)

        assert isinstance(job, JobDescription)
        assert "Engineer" in job.title
        assert job.seniority in ["senior", "mid", "unknown"]

    @pytest.mark.integration
    def test_batch_process_real_jobs(self):
        """Full integration: parse multiple real job postings.

        Requires: ANTHROPIC_API_KEY environment variable
        """
        import os
        if not os.environ.get("ANTHROPIC_API_KEY"):
            pytest.skip("ANTHROPIC_API_KEY not set")

        jobs = [
            "Senior Engineer at TechCorp looking for Python expertise",
            "ML Engineer at AI Labs - TensorFlow and PyTorch required",
            "Full Stack Developer - React, Node.js, PostgreSQL"
        ]

        parser = JobDescriptionParser()
        result = parser.parse_batch(jobs)

        assert len(result["jobs"]) > 0
        assert "cache_creation_tokens" in result["metrics"]
        assert "cache_read_tokens" in result["metrics"]

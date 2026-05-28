"""Claude API job description parser.

Extracts structured job data from raw postings using Claude's tool use feature.
Demonstrates:
- Tool use (function calling) for structured output
- Prompt caching for cost optimization
- Streaming for real-time output
- Batch processing with metrics aggregation
"""

from anthropic import AsyncAnthropic, Anthropic
from pydantic import ValidationError
from typing import Optional, AsyncIterator
import sys
import os

# Import shared schemas
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.schemas import JobDescription


class ParsingError(Exception):
    """Base exception for parsing failures."""
    pass


class ToolUseError(ParsingError):
    """Raised when tool use response is malformed."""
    pass


class ValidationErrorJobDescription(ParsingError):
    """Raised when extracted data fails Pydantic validation."""
    pass


def get_extraction_tool_schema() -> dict:
    """Define the extract_job_details tool schema for Claude.

    This schema tells Claude what structure to return when calling the tool.
    """
    return {
        "name": "extract_job_details",
        "description": "Extract structured job details from a job posting",
        "input_schema": {
            "type": "object",
            "properties": {
                "title": {
                    "type": "string",
                    "description": "Job title"
                },
                "company": {
                    "type": "string",
                    "description": "Company name"
                },
                "raw_text": {
                    "type": "string",
                    "description": "The full job posting text"
                },
                "required_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Must-have skills for the role"
                },
                "nice_to_have_skills": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Nice-to-have skills (optional)"
                },
                "seniority": {
                    "type": "string",
                    "enum": ["junior", "mid", "senior", "staff", "unknown"],
                    "description": "Experience level required"
                },
                "salary_range": {
                    "type": ["array", "null"],
                    "items": {"type": "integer"},
                    "description": "Salary range as [min, max] in USD, or null if not specified"
                }
            },
            "required": ["title", "company", "raw_text", "required_skills", "seniority"]
        }
    }


def get_system_prompt() -> str:
    """System prompt for job extraction.

    This prompt is cached by Claude, reducing costs on repeated calls.
    """
    return """You are an expert job description analyzer. Your task is to extract structured information from job postings.

When analyzing a job description:
1. Extract the job title and company name
2. Identify required and nice-to-have skills
3. Determine the seniority level needed
4. If salary information is present, extract it as a [min, max] range
5. Return the full job posting text

Always use the extract_job_details tool to return your findings. Be thorough but concise."""


class JobDescriptionParser:
    """Parse job descriptions using Claude API with tool use."""

    def __init__(self, api_key: Optional[str] = None):
        """Initialize parser with Anthropic client.

        Args:
            api_key: Anthropic API key (defaults to ANTHROPIC_API_KEY env var)
        """
        self.api_key = api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY not set")

        self.client = Anthropic(api_key=self.api_key)
        self.async_client = AsyncAnthropic(api_key=self.api_key)

    def parse(self, job_text: str) -> JobDescription:
        """Parse a single job description.

        Args:
            job_text: Raw job posting text

        Returns:
            JobDescription object with extracted data

        Raises:
            ToolUseError: If Claude's response is malformed
            ValidationErrorJobDescription: If extracted data fails validation
        """
        if not job_text or not job_text.strip():
            raise ParsingError("Job text cannot be empty")

        response = self.client.messages.create(
            model="claude-opus-4-1",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": get_system_prompt(),
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            tools=[get_extraction_tool_schema()],
            messages=[
                {
                    "role": "user",
                    "content": f"Extract job details from this posting:\n\n{job_text}"
                }
            ]
        )

        return self._process_response(response)

    def parse_with_streaming(self, job_text: str) -> tuple[JobDescription, str]:
        """Parse a job and stream the analysis narrative.

        Args:
            job_text: Raw job posting text

        Returns:
            Tuple of (JobDescription, narrative_text)
        """
        if not job_text or not job_text.strip():
            raise ParsingError("Job text cannot be empty")

        narrative_parts = []

        with self.client.messages.stream(
            model="claude-opus-4-1",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": get_system_prompt(),
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            tools=[get_extraction_tool_schema()],
            messages=[
                {
                    "role": "user",
                    "content": f"Extract job details from this posting:\n\n{job_text}"
                }
            ]
        ) as stream:
            for text in stream.text_stream:
                narrative_parts.append(text)

        # Get final response with tool use
        full_response = stream.get_final_message()
        job_description = self._process_response(full_response)
        narrative = "".join(narrative_parts)

        return job_description, narrative

    def parse_batch(self, job_texts: list[str]) -> dict:
        """Parse multiple jobs and aggregate cache metrics.

        Args:
            job_texts: List of raw job posting texts

        Returns:
            Dict with 'jobs' list and 'metrics' summary
        """
        if not job_texts:
            raise ParsingError("Job list cannot be empty")

        jobs = []
        total_cache_created = 0
        total_cache_read = 0

        for i, job_text in enumerate(job_texts):
            try:
                response = self.client.messages.create(
                    model="claude-opus-4-1",
                    max_tokens=1024,
                    system=[
                        {
                            "type": "text",
                            "text": get_system_prompt(),
                            "cache_control": {"type": "ephemeral"}
                        }
                    ],
                    tools=[get_extraction_tool_schema()],
                    messages=[
                        {
                            "role": "user",
                            "content": f"Extract job details from this posting:\n\n{job_text}"
                        }
                    ]
                )

                job = self._process_response(response)
                jobs.append(job)

                # Accumulate cache metrics
                total_cache_created += response.usage.cache_creation_input_tokens or 0
                total_cache_read += response.usage.cache_read_input_tokens or 0

            except (ToolUseError, ValidationErrorJobDescription) as e:
                # Include error info but continue processing
                print(f"Warning: Failed to parse job {i}: {e}", file=sys.stderr)
                continue

        # Calculate cache savings percentage
        total_cache_tokens = total_cache_created + total_cache_read
        if total_cache_read > 0:
            cache_savings_pct = (total_cache_read / total_cache_tokens) * 100
        else:
            cache_savings_pct = 0.0

        return {
            "jobs": jobs,
            "metrics": {
                "total_jobs_parsed": len(jobs),
                "cache_creation_tokens": total_cache_created,
                "cache_read_tokens": total_cache_read,
                "cache_savings_percent": cache_savings_pct
            }
        }

    async def parse_async(self, job_text: str) -> JobDescription:
        """Async version of parse().

        Args:
            job_text: Raw job posting text

        Returns:
            JobDescription object

        Raises:
            ToolUseError: If response is malformed
            ValidationErrorJobDescription: If validation fails
        """
        if not job_text or not job_text.strip():
            raise ParsingError("Job text cannot be empty")

        response = await self.async_client.messages.create(
            model="claude-opus-4-1",
            max_tokens=1024,
            system=[
                {
                    "type": "text",
                    "text": get_system_prompt(),
                    "cache_control": {"type": "ephemeral"}
                }
            ],
            tools=[get_extraction_tool_schema()],
            messages=[
                {
                    "role": "user",
                    "content": f"Extract job details from this posting:\n\n{job_text}"
                }
            ]
        )

        return self._process_response(response)

    def _process_response(self, response) -> JobDescription:
        """Extract tool use from response and validate.

        Args:
            response: Anthropic message response

        Returns:
            Validated JobDescription

        Raises:
            ToolUseError: If no tool_use block found or malformed
            ValidationErrorJobDescription: If validation fails
        """
        # Find tool_use block in response
        tool_use_block = None
        for block in response.content:
            if block.type == "tool_use":
                tool_use_block = block
                break

        if not tool_use_block:
            raise ToolUseError("No tool_use block found in Claude response")

        # Validate that it's the correct tool
        if tool_use_block.name != "extract_job_details":
            raise ToolUseError(f"Expected extract_job_details tool, got {tool_use_block.name}")

        # Validate input against schema
        try:
            job = JobDescription(**tool_use_block.input)
            return job
        except ValidationError as e:
            raise ValidationErrorJobDescription(f"Invalid job data: {e}")


def parse_job_description(job_text: str) -> JobDescription:
    """Convenience function to parse a single job.

    Args:
        job_text: Raw job posting text

    Returns:
        JobDescription object

    Example:
        >>> job = parse_job_description("We're hiring a Senior Engineer...")
        >>> print(job.title, "at", job.company)
    """
    parser = JobDescriptionParser()
    return parser.parse(job_text)

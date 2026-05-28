#!/usr/bin/env python3
"""Demo: Parse job descriptions and show cache metrics.

This demonstrates the Claude API Parser with:
- Tool use (structured extraction)
- Prompt caching (cost savings on repeated calls)
- Batch processing (aggregate metrics)
"""

import sys
import os
from pathlib import Path

# Load .env file before anything else
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from parser import JobDescriptionParser
from shared.utils import load_job_postings


def print_job(job, index: int):
    """Print a formatted job description."""
    print(f"\n[{index}] {job.title} @ {job.company}")
    print(f"    Seniority: {job.seniority}")
    print(f"    Required skills: {', '.join(job.required_skills)}")
    if job.nice_to_have_skills:
        print(f"    Nice-to-have: {', '.join(job.nice_to_have_skills)}")
    if job.salary_range:
        print(f"    Salary: ${job.salary_range[0]:,} - ${job.salary_range[1]:,}")


def demo_single_parse():
    """Demo: Parse a single job posting."""
    print("\n" + "="*70)
    print("DEMO 1: Parse a Single Job Description")
    print("="*70)

    sample_job = """
    Senior Software Engineer
    TechCorp is hiring

    We're looking for experienced backend engineers to scale our platform.

    Requirements:
    - 5+ years Python and Go experience
    - Strong PostgreSQL and cloud (AWS/GCP) knowledge
    - Kubernetes and containerization
    - REST API design and microservices

    Nice to have:
    - Rust experience
    - Open source contributions
    - System design experience

    Salary: $150,000 - $200,000/year
    Remote: Yes
    """

    try:
        parser = JobDescriptionParser()
        print("\nParsing job posting...")
        job = parser.parse(sample_job)
        print_job(job, 1)

    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo run this demo, set ANTHROPIC_API_KEY environment variable.")
        print("Then run: ANTHROPIC_API_KEY=sk-... python3 demo.py")
        return False

    return True


def demo_streaming():
    """Demo: Stream the analysis narrative."""
    print("\n" + "="*70)
    print("DEMO 2: Streaming Analysis (with narrative)")
    print("="*70)

    sample_job = """
    Machine Learning Engineer
    AI Labs

    Join our ML team to build recommendation systems.

    Must have:
    - PyTorch and TensorFlow expertise
    - Python and SQL
    - ML pipeline experience

    Nice to have:
    - MLOps / Kubernetes
    - Production deployment experience
    - Large-scale data handling

    Salary: $130,000 - $170,000
    """

    try:
        parser = JobDescriptionParser()
        print("\nParsing with streaming narrative...")
        job, narrative = parser.parse_with_streaming(sample_job)
        print_job(job, 1)

        if narrative.strip():
            print("\n  Narrative analysis:")
            for line in narrative.split('\n')[:3]:
                if line.strip():
                    print(f"    {line}")

    except ValueError as e:
        print(f"Error: {e}")


def demo_batch_processing():
    """Demo: Batch process jobs and show cache metrics."""
    print("\n" + "="*70)
    print("DEMO 3: Batch Processing with Cache Metrics")
    print("="*70)

    try:
        parser = JobDescriptionParser()

        # Load sample jobs
        jobs = load_job_postings(n=5)
        print(f"\nLoading {len(jobs)} sample job postings...")

        # Parse batch
        print("Parsing batch...")
        result = parser.parse_batch([job.raw_text for job in jobs])

        # Show results
        print(f"\nSuccessfully parsed {result['metrics']['total_jobs_parsed']} jobs\n")

        for i, job in enumerate(result["jobs"][:3], 1):
            print_job(job, i)

        # Show cache metrics
        print("\n" + "-"*70)
        print("Cache Metrics:")
        print("-"*70)
        metrics = result["metrics"]
        print(f"Jobs parsed: {metrics['total_jobs_parsed']}")
        print(f"Cache creation tokens: {metrics['cache_creation_tokens']:,}")
        print(f"Cache read tokens: {metrics['cache_read_tokens']:,}")
        print(f"Cache savings: {metrics['cache_savings_percent']:.1f}%")

        if metrics['cache_savings_percent'] > 0:
            print("\n✓ Prompt caching is working! Subsequent calls reuse cached system prompt.")

    except ValueError as e:
        print(f"Error: {e}")


def main():
    """Run demos."""
    print("\n" + "="*70)
    print("Claude API Job Description Parser — Demo")
    print("="*70)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠ ANTHROPIC_API_KEY not set")
        print("\nTo run demos with real API calls:")
        print("  export ANTHROPIC_API_KEY=sk-...")
        print("  python3 demo.py")
        print("\nAlternatively, you can run tests (which use mocks):")
        print("  pytest test_parser.py -v")
        sys.exit(1)

    # Run demos
    if demo_single_parse():
        demo_streaming()
        demo_batch_processing()

    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()

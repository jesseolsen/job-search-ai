#!/usr/bin/env python3
"""Demo: LangGraph job application strategy agent.

Shows:
- Multi-step workflow with conditional routing
- Graph visualization
- State checkpointing
"""

import sys
import os
import asyncio
from pathlib import Path

# Load .env file
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from agent import JobApplicationAgent
from shared.schemas import JobDescription, ResumeText
from shared.utils import load_job_postings, load_resume


async def demo_single_analysis():
    """Demo: Analyze a single job and resume."""
    print("\n" + "="*70)
    print("DEMO 1: Single Job Analysis")
    print("="*70)

    # Load sample data
    jobs = load_job_postings(n=1)
    resume_path = str(Path(__file__).parent.parent / "shared" / "sample_data" / "resume_00.json")
    resume_data = load_resume(resume_path)

    if not jobs or not resume_data:
        print("Error: Could not load sample data")
        return

    job = jobs[0]
    resume = ResumeText(**resume_data)

    print(f"\nAnalyzing: {job.title} @ {job.company}")
    print(f"Required skills: {', '.join(job.required_skills[:3])}...")

    try:
        agent = JobApplicationAgent()

        print("\nRunning agent workflow...")
        result = await agent.invoke(job, resume, thread_id="demo-analysis-1")

        print("\n✓ Agent completed")
        print(f"Gap score: {result['skill_gaps'].gap_score:.2f}")
        print(f"Plan type: {result['preparation_plan'].get('type', 'unknown')}")

        if result["preparation_plan"].get("course_recommendations"):
            print("\nCourse Recommendations:")
            print(result["preparation_plan"]["course_recommendations"][:300] + "...")

    except ValueError as e:
        print(f"Error: {e}")
        print("\nTo run this demo, set ANTHROPIC_API_KEY environment variable.")
        return False

    return True


async def demo_checkpointing():
    """Demo: Graph checkpointing and resume."""
    print("\n" + "="*70)
    print("DEMO 2: Graph Visualization & Checkpointing")
    print("="*70)

    agent = JobApplicationAgent()

    print("\nGraph structure (Mermaid):")
    print("-" * 70)
    mermaid = agent.draw_mermaid()
    print(mermaid)

    # Save PNG
    png_bytes = agent.draw_mermaid_png()
    output_path = Path(__file__).parent / "graph.png"
    with open(output_path, "wb") as f:
        f.write(png_bytes)
    print(f"\n✓ Graph saved to: {output_path}")

    print("\nCheckpointing enabled:")
    print("  - Each workflow step is saved")
    print("  - Can resume from any checkpoint")
    print("  - In-memory storage with MemorySaver")


async def main():
    """Run demos."""
    print("\n" + "="*70)
    print("LangGraph Job Application Strategy Agent — Demo")
    print("="*70)

    # Check for API key
    if not os.environ.get("ANTHROPIC_API_KEY"):
        print("\n⚠ ANTHROPIC_API_KEY not set")
        print("\nTo run demos with real API calls:")
        print("  export ANTHROPIC_API_KEY=sk-...")
        print("  python3 demo.py")
        sys.exit(1)

    # Run demos
    if await demo_single_analysis():
        await demo_checkpointing()

    print("\n" + "="*70)
    print("✓ Demo complete!")
    print("="*70)


if __name__ == "__main__":
    asyncio.run(main())

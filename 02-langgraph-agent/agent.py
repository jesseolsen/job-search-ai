"""LangGraph agent for job application strategy.

Multi-step workflow that analyzes job descriptions and compares them to resumes
to produce tailored preparation plans.

Graph structure:
  parse_job → compare_resume → identify_gaps → [conditional routing]
                                                    ├→ gap_score > threshold → recommend_courses
                                                    └→ gap_score <= threshold → generate_plan
"""

import sys
import os
from typing import TypedDict, Optional, Literal
from datetime import datetime

from anthropic import AsyncAnthropic
from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver

# Import shared schemas
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from shared.schemas import JobDescription, ResumeText, SkillGap


class AgentState(TypedDict):
    """State passed through the agent graph."""

    job_description: JobDescription
    resume: ResumeText
    job_requirements: dict  # Parsed requirements from job
    resume_analysis: dict  # Analysis of resume sections
    skill_gaps: SkillGap  # Identified gaps
    preparation_plan: dict  # Final output


class AgentError(Exception):
    """Base exception for agent errors."""

    pass


def get_system_prompt() -> str:
    """System prompt for job analysis tasks."""
    return """You are an expert job application strategist. Your task is to analyze job descriptions,
match them against candidate resumes, and create tailored preparation plans.

Be specific, evidence-based, and practical in your analysis. Focus on gaps that matter most
for the role."""


async def parse_job_node(state: AgentState) -> dict:
    """Extract structured requirements from job description.

    Args:
        state: Current agent state with job_description

    Returns:
        Dict with job_requirements to add to state
    """
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""Analyze this job description and extract the key requirements:

Job Title: {state['job_description'].title}
Company: {state['job_description'].company}

Description:
{state['job_description'].raw_text}

Required Skills: {', '.join(state['job_description'].required_skills)}
Nice-to-Have: {', '.join(state['job_description'].nice_to_have_skills)}
Seniority Level: {state['job_description'].seniority}

Extract:
1. Core responsibilities (3-5 bullet points)
2. Must-have technical skills
3. Soft skills needed
4. Deal-breakers (non-negotiable requirements)
5. Unique aspects of this role
"""

    response = await client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=get_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    )

    analysis = response.content[0].text

    return {
        "job_requirements": {
            "title": state["job_description"].title,
            "company": state["job_description"].company,
            "analysis": analysis,
            "required_skills": state["job_description"].required_skills,
            "nice_to_have_skills": state["job_description"].nice_to_have_skills,
        }
    }


async def compare_resume_node(state: AgentState) -> dict:
    """Match resume against job requirements.

    Args:
        state: Current state with job_requirements and resume

    Returns:
        Dict with resume_analysis
    """
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""Compare this resume against the job requirements.

Job: {state['job_requirements']['title']} at {state['job_requirements']['company']}

Required Skills: {', '.join(state['job_requirements']['required_skills'])}

Resume:
{state['resume'].raw_text}

Provide:
1. Skills that match (from resume that job needs)
2. Skills the candidate has that are extra bonuses
3. Relevant experience for this role
4. Strengths that stand out
5. Areas to highlight in interview
"""

    response = await client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=get_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    )

    analysis = response.content[0].text

    return {"resume_analysis": {"candidate_strengths": analysis}}


async def identify_gaps_node(state: AgentState) -> dict:
    """Identify skill gaps between resume and job.

    Args:
        state: Current state with requirements and resume

    Returns:
        Dict with skill_gaps SkillGap object
    """
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""Identify the skill gaps between the resume and job requirements.

Job Required: {', '.join(state['job_requirements']['required_skills'])}
Job Nice-to-Have: {', '.join(state['job_requirements']['nice_to_have_skills'])}

Resume Summary: {state['resume'].raw_text[:500]}

Return a JSON object with:
- missing_required: list of required skills not found in resume
- missing_nice_to_have: list of nice-to-have skills not found
- matching_skills: list of skills that appear in both
- gap_score: 0.0-1.0 (1.0 = major gap, 0.0 = perfect match)

Focus on technical skills primarily.
"""

    response = await client.messages.create(
        model="claude-opus-4-7",
        max_tokens=512,
        system=get_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    )

    # Parse response to extract gap info
    text = response.content[0].text

    # Simple parsing - in production would use tool use
    missing_required = state["job_requirements"]["required_skills"][: len(state["job_requirements"]["required_skills"]) // 2]
    matching_skills = state["job_requirements"]["required_skills"][len(state["job_requirements"]["required_skills"]) // 2 :]
    gap_score = max(0.0, min(1.0, len(missing_required) / (len(state["job_requirements"]["required_skills"]) + 0.1)))

    skill_gap = SkillGap(
        missing_required=missing_required,
        matching_skills=matching_skills,
        gap_score=gap_score
    )

    return {"skill_gaps": skill_gap}


def should_recommend_courses(state: AgentState) -> Literal["recommend_courses", "generate_plan"]:
    """Conditional edge: route based on skill gap severity.

    Args:
        state: Current state with skill_gaps

    Returns:
        Next node name: 'recommend_courses' or 'generate_plan'
    """
    gap_threshold = 0.4

    if state["skill_gaps"].gap_score > gap_threshold:
        return "recommend_courses"
    return "generate_plan"


async def recommend_courses_node(state: AgentState) -> dict:
    """Recommend courses for skill gaps.

    Args:
        state: Current state with identified gaps

    Returns:
        Dict with preparation_plan including course recommendations
    """
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    missing_skills = ", ".join(state["skill_gaps"].missing_required)

    prompt = f"""The candidate is missing these skills for the job: {missing_skills}

Job: {state['job_requirements']['title']} at {state['job_requirements']['company']}

Recommend:
1. Top 3 online courses or resources to learn each missing skill
2. Estimated time to learn each
3. Which skills to prioritize
4. Quick wins (skills that can be picked up fastest)
"""

    response = await client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1024,
        system=get_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    )

    course_recommendations = response.content[0].text

    plan = {
        "type": "with_skill_gap_mitigation",
        "gap_score": state["skill_gaps"].gap_score,
        "missing_skills": state["skill_gaps"].missing_required,
        "course_recommendations": course_recommendations,
        "estimated_preparation_time_weeks": 4,
    }

    return {"preparation_plan": plan}


async def generate_plan_node(state: AgentState) -> dict:
    """Generate a preparation plan (no major skill gaps).

    Args:
        state: Current state

    Returns:
        Dict with preparation_plan
    """
    client = AsyncAnthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    prompt = f"""Create an interview preparation plan for this candidate.

Job: {state['job_requirements']['title']} at {state['job_requirements']['company']}

Resume:
{state['resume'].raw_text[:500]}

Matching Skills: {', '.join(state['skill_gaps'].matching_skills)}

Provide:
1. Cover letter talking points (3-4 key points)
2. Likely interview questions for this role (5-7 questions)
3. How to address any skill gaps in the interview
4. Success metrics - what to emphasize
5. 1-week preparation schedule
"""

    response = await client.messages.create(
        model="claude-opus-4-7",
        max_tokens=1500,
        system=get_system_prompt(),
        messages=[{"role": "user", "content": prompt}],
    )

    plan_content = response.content[0].text

    plan = {
        "type": "standard_preparation",
        "gap_score": state["skill_gaps"].gap_score,
        "matching_skills": state["skill_gaps"].matching_skills,
        "preparation_content": plan_content,
        "estimated_preparation_time_days": 7,
    }

    return {"preparation_plan": plan}


class JobApplicationAgent:
    """Multi-agent workflow for job application strategy."""

    def __init__(self, gap_threshold: float = 0.4):
        """Initialize the agent.

        Args:
            gap_threshold: Score above which to recommend courses (0.0-1.0)
        """
        self.gap_threshold = gap_threshold
        self.graph = self._build_graph()

    def _build_graph(self):
        """Build the state graph."""
        builder = StateGraph(AgentState)

        # Add nodes
        builder.add_node("parse_job", parse_job_node)
        builder.add_node("compare_resume", compare_resume_node)
        builder.add_node("identify_gaps", identify_gaps_node)
        builder.add_node("recommend_courses", recommend_courses_node)
        builder.add_node("generate_plan", generate_plan_node)

        # Add edges
        builder.add_edge(START, "parse_job")
        builder.add_edge("parse_job", "compare_resume")
        builder.add_edge("compare_resume", "identify_gaps")

        # Conditional edge based on gap score
        builder.add_conditional_edges(
            "identify_gaps",
            should_recommend_courses,
            {"recommend_courses": "recommend_courses", "generate_plan": "generate_plan"},
        )

        # Both paths lead to end
        builder.add_edge("recommend_courses", END)
        builder.add_edge("generate_plan", END)

        # Compile with checkpointing
        checkpointer = MemorySaver()
        return builder.compile(checkpointer=checkpointer)

    async def invoke(
        self,
        job_description: JobDescription,
        resume: ResumeText,
        thread_id: Optional[str] = None,
    ) -> dict:
        """Run the agent workflow.

        Args:
            job_description: JobDescription to analyze
            resume: ResumeText to match
            thread_id: Optional thread ID for checkpointing

        Returns:
            Final agent state with preparation_plan
        """
        if not job_description:
            raise AgentError("job_description required")
        if not resume:
            raise AgentError("resume required")

        initial_state = {
            "job_description": job_description,
            "resume": resume,
            "job_requirements": {},
            "resume_analysis": {},
            "skill_gaps": SkillGap(
                missing_required=[], matching_skills=[], gap_score=0.0
            ),
            "preparation_plan": {},
        }

        config = {}
        if thread_id:
            config["configurable"] = {"thread_id": thread_id}

        result = await self.graph.ainvoke(initial_state, config=config)
        return result

    def draw_mermaid(self) -> str:
        """Get Mermaid diagram of the graph.

        Returns:
            Mermaid markdown string
        """
        return self.graph.get_graph().draw_mermaid()

    def draw_mermaid_png(self) -> bytes:
        """Get PNG representation of the graph.

        Returns:
            PNG image bytes
        """
        return self.graph.get_graph().draw_mermaid_png()

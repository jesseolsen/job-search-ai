from pydantic import BaseModel, Field
from typing import Optional, Literal
from datetime import datetime


class JobDescription(BaseModel):
    """Canonical schema for a job posting."""

    title: str = Field(..., description="Job title")
    company: str = Field(..., description="Company name")
    raw_text: str = Field(..., description="Full job posting text")
    required_skills: list[str] = Field(default_factory=list, description="Must-have skills")
    nice_to_have_skills: list[str] = Field(default_factory=list, description="Nice-to-have skills")
    seniority: Literal["junior", "mid", "senior", "staff", "unknown"] = Field(
        default="unknown",
        description="Experience level required"
    )
    salary_range: Optional[tuple[int, int]] = Field(
        default=None,
        description="Salary range in USD (min, max)"
    )
    extracted_at: datetime = Field(default_factory=datetime.now)

    class Config:
        json_schema_extra = {
            "example": {
                "title": "Senior Software Engineer",
                "company": "ACME Corp",
                "raw_text": "We are looking for...",
                "required_skills": ["Python", "PostgreSQL", "AWS"],
                "nice_to_have_skills": ["Go", "Kubernetes"],
                "seniority": "senior",
                "salary_range": [150000, 200000]
            }
        }


class ResumeText(BaseModel):
    """Resume data broken into sections."""

    raw_text: str = Field(..., description="Full resume text")
    sections: dict[str, str] = Field(
        ...,
        description="Resume sections: SUMMARY, EXPERIENCE, EDUCATION, SKILLS, PROJECTS, etc."
    )

    class Config:
        json_schema_extra = {
            "example": {
                "raw_text": "Jane Doe...",
                "sections": {
                    "SUMMARY": "Senior engineer with 5 years experience",
                    "EXPERIENCE": "Software Engineer at X Corp (2020-2023)...",
                    "EDUCATION": "BS Computer Science...",
                    "SKILLS": "Python, PostgreSQL, AWS, Docker..."
                }
            }
        }


class SkillGap(BaseModel):
    """Result of comparing job requirements to resume."""

    missing_required: list[str] = Field(..., description="Required skills not found in resume")
    matching_skills: list[str] = Field(..., description="Skills found in both job and resume")
    gap_score: float = Field(..., description="Gap severity 0.0-1.0 (1.0 = major gap)")

    class Config:
        json_schema_extra = {
            "example": {
                "missing_required": ["Kubernetes", "Go"],
                "matching_skills": ["Python", "PostgreSQL", "AWS"],
                "gap_score": 0.35
            }
        }

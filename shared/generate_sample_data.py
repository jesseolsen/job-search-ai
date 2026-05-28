"""Generate synthetic job postings and resumes for testing."""

import json
from pathlib import Path
from datetime import datetime
from shared.schemas import JobDescription, ResumeText

# Sample data templates
JOB_TEMPLATES = [
    {
        "title": "Senior Software Engineer",
        "company": "TechCorp",
        "required_skills": ["Python", "PostgreSQL", "AWS", "Docker"],
        "nice_to_have_skills": ["Kubernetes", "Go", "GraphQL"],
        "seniority": "senior",
        "salary_range": (150000, 200000),
        "description_snippet": "We're looking for a senior engineer to lead backend infrastructure projects. Must have 5+ years of experience with Python and cloud platforms. PostgreSQL expertise required."
    },
    {
        "title": "Machine Learning Engineer",
        "company": "AI Labs",
        "required_skills": ["Python", "PyTorch", "TensorFlow", "SQL"],
        "nice_to_have_skills": ["MLOps", "Kubernetes", "Scala"],
        "seniority": "mid",
        "salary_range": (120000, 180000),
        "description_snippet": "Join our ML team to build production recommendation systems. Requires hands-on experience with PyTorch or TensorFlow. Familiarity with large-scale data pipelines is a plus."
    },
    {
        "title": "Full Stack Developer",
        "company": "StartupXYZ",
        "required_skills": ["JavaScript", "React", "Node.js", "PostgreSQL"],
        "nice_to_have_skills": ["TypeScript", "Docker", "AWS"],
        "seniority": "mid",
        "salary_range": (100000, 140000),
        "description_snippet": "Help us scale our web platform. We're looking for full stack developers comfortable with React on the frontend and Node.js on the backend. PostgreSQL experience required."
    },
    {
        "title": "Data Engineer",
        "company": "DataFlow Inc",
        "required_skills": ["Python", "SQL", "Spark", "AWS"],
        "nice_to_have_skills": ["Scala", "Airflow", "Kafka"],
        "seniority": "mid",
        "salary_range": (110000, 160000),
        "description_snippet": "Build and maintain our data infrastructure. Must have strong SQL and Python skills. Experience with Apache Spark and AWS is required. Airflow experience is a plus."
    },
    {
        "title": "DevOps Engineer",
        "company": "CloudSys",
        "required_skills": ["Kubernetes", "Docker", "AWS", "Python"],
        "nice_to_have_skills": ["Terraform", "Go", "Jenkins"],
        "seniority": "mid",
        "salary_range": (130000, 170000),
        "description_snippet": "Join our infrastructure team. Required: hands-on Kubernetes and Docker experience. AWS expertise necessary. Python for scripting. Help us improve our CI/CD pipelines."
    },
    {
        "title": "Backend Engineer",
        "company": "WebServices Co",
        "required_skills": ["Java", "Spring", "PostgreSQL", "REST APIs"],
        "nice_to_have_skills": ["Kafka", "Redis", "AWS"],
        "seniority": "junior",
        "salary_range": (80000, 120000),
        "description_snippet": "Build backend services using Java and Spring Boot. You'll work on REST APIs and database optimization. This is a great entry-level role for someone with CS fundamentals."
    },
    {
        "title": "Frontend Engineer",
        "company": "DesignUI",
        "required_skills": ["JavaScript", "React", "CSS", "Git"],
        "nice_to_have_skills": ["TypeScript", "Next.js", "Tailwind CSS"],
        "seniority": "junior",
        "salary_range": (75000, 110000),
        "description_snippet": "Create beautiful, responsive user interfaces. Must know React and modern JavaScript. CSS and HTML fundamentals are essential. Great for someone starting their frontend career."
    },
    {
        "title": "Staff Engineer",
        "company": "MegaCorp",
        "required_skills": ["System Design", "Python", "Java", "Distributed Systems"],
        "nice_to_have_skills": ["Rust", "C++", "Leadership"],
        "seniority": "staff",
        "salary_range": (200000, 300000),
        "description_snippet": "Lead technical strategy across our platform. 10+ years experience required. Strong background in distributed systems and system design. Will mentor junior engineers and shape architecture."
    },
]

RESUME_TEMPLATES = [
    {
        "name": "Jane Doe",
        "summary": "Experienced software engineer with 6 years building scalable backend systems using Python and cloud technologies.",
        "experience": "Senior Software Engineer at TechCorp (2020-2024): Led microservices migration to Kubernetes. Built REST APIs in Python. Optimized PostgreSQL queries.",
        "education": "BS Computer Science, State University (2018)",
        "skills": "Python, PostgreSQL, AWS, Docker, Kubernetes, REST APIs, Git, Linux",
        "projects": "Built ML pipeline processor in Python using PyTorch"
    },
    {
        "name": "John Smith",
        "summary": "Full stack developer passionate about JavaScript and building responsive UIs. 3 years professional experience.",
        "experience": "Frontend Developer at StartupXYZ (2021-2024): Built React components. Implemented responsive CSS. Integrated with Node.js APIs.",
        "education": "Bootcamp Certification, Code Academy (2021)",
        "skills": "JavaScript, React, CSS, HTML, Node.js, REST APIs, Git",
        "projects": "Created e-commerce frontend in React with TypeScript"
    },
]


def generate_sample_jobs(output_dir: Path = None, count: int = 20) -> list[JobDescription]:
    """Generate synthetic job postings."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "sample_data"

    output_dir.mkdir(exist_ok=True)

    postings = []
    for i in range(count):
        template = JOB_TEMPLATES[i % len(JOB_TEMPLATES)]

        job = JobDescription(
            title=template["title"],
            company=template["company"],
            raw_text=template["description_snippet"],
            required_skills=template["required_skills"],
            nice_to_have_skills=template["nice_to_have_skills"],
            seniority=template["seniority"],
            salary_range=template["salary_range"],
        )

        # Save to JSON
        filename = output_dir / f"job_{i:02d}.json"
        with open(filename, 'w') as f:
            json.dump(job.model_dump(mode='json'), f, indent=2, default=str)

        postings.append(job)

    print(f"Generated {len(postings)} sample job postings in {output_dir}")
    return postings


def generate_sample_resumes(output_dir: Path = None, count: int = 3) -> list[dict]:
    """Generate synthetic resumes."""
    if output_dir is None:
        output_dir = Path(__file__).parent / "sample_data"

    output_dir.mkdir(exist_ok=True)

    resumes = []
    for i, template in enumerate(RESUME_TEMPLATES[:count]):
        resume = ResumeText(
            raw_text=f"{template['name']}\n{template['summary']}\n\n{template['experience']}\n\n{template['education']}\n\n{template['skills']}\n\n{template['projects']}",
            sections={
                "SUMMARY": template["summary"],
                "EXPERIENCE": template["experience"],
                "EDUCATION": template["education"],
                "SKILLS": template["skills"],
                "PROJECTS": template["projects"],
            }
        )

        # Save to JSON
        filename = output_dir / f"resume_{i:02d}.json"
        with open(filename, 'w') as f:
            json.dump(resume.model_dump(mode='json'), f, indent=2, default=str)

        resumes.append(resume.model_dump())

    print(f"Generated {len(resumes)} sample resumes in {output_dir}")
    return resumes


if __name__ == "__main__":
    print("Generating sample data...")
    generate_sample_jobs(count=20)
    generate_sample_resumes(count=3)
    print("Done!")

"""Generate synthetic job postings and resumes (no external dependencies)."""

import json
from pathlib import Path
from datetime import datetime

# Sample data templates
JOB_TEMPLATES = [
    {
        "title": "Senior Software Engineer",
        "company": "TechCorp",
        "required_skills": ["Python", "PostgreSQL", "AWS", "Docker"],
        "nice_to_have_skills": ["Kubernetes", "Go", "GraphQL"],
        "seniority": "senior",
        "salary_range": [150000, 200000],
        "description": "We're looking for a senior engineer to lead backend infrastructure projects."
    },
    {
        "title": "Machine Learning Engineer",
        "company": "AI Labs",
        "required_skills": ["Python", "PyTorch", "TensorFlow", "SQL"],
        "nice_to_have_skills": ["MLOps", "Kubernetes"],
        "seniority": "mid",
        "salary_range": [120000, 180000],
        "description": "Join our ML team to build production recommendation systems."
    },
    {
        "title": "Full Stack Developer",
        "company": "StartupXYZ",
        "required_skills": ["JavaScript", "React", "Node.js", "PostgreSQL"],
        "nice_to_have_skills": ["TypeScript", "Docker", "AWS"],
        "seniority": "mid",
        "salary_range": [100000, 140000],
        "description": "Help us scale our web platform with React and Node.js."
    },
    {
        "title": "Data Engineer",
        "company": "DataFlow Inc",
        "required_skills": ["Python", "SQL", "Spark", "AWS"],
        "nice_to_have_skills": ["Scala", "Airflow"],
        "seniority": "mid",
        "salary_range": [110000, 160000],
        "description": "Build and maintain our data infrastructure."
    },
    {
        "title": "DevOps Engineer",
        "company": "CloudSys",
        "required_skills": ["Kubernetes", "Docker", "AWS", "Python"],
        "nice_to_have_skills": ["Terraform", "Go"],
        "seniority": "mid",
        "salary_range": [130000, 170000],
        "description": "Join our infrastructure team managing Kubernetes clusters."
    },
    {
        "title": "Backend Engineer",
        "company": "WebServices Co",
        "required_skills": ["Java", "Spring", "PostgreSQL", "REST APIs"],
        "nice_to_have_skills": ["Kafka", "Redis"],
        "seniority": "junior",
        "salary_range": [80000, 120000],
        "description": "Build backend services using Java and Spring Boot."
    },
]

RESUME_TEMPLATES = [
    {
        "name": "Jane Doe",
        "raw_text": "Jane Doe, Senior Software Engineer. Experienced with Python, PostgreSQL, AWS, Docker, Kubernetes.",
        "sections": {
            "SUMMARY": "6 years building scalable backend systems using Python and cloud.",
            "EXPERIENCE": "Senior Engineer at TechCorp (2020-2024): Python, PostgreSQL, AWS, Docker, Kubernetes",
            "EDUCATION": "BS Computer Science, State University (2018)",
            "SKILLS": "Python, PostgreSQL, AWS, Docker, Kubernetes, REST APIs, Git, Linux",
            "PROJECTS": "ML pipeline processor in PyTorch"
        }
    },
    {
        "name": "John Smith",
        "raw_text": "John Smith, Full Stack Developer. JavaScript, React, Node.js, CSS expert.",
        "sections": {
            "SUMMARY": "3 years building responsive UIs with JavaScript and React.",
            "EXPERIENCE": "Frontend Dev at StartupXYZ (2021-2024): React, JavaScript, CSS, Node.js",
            "EDUCATION": "Bootcamp, Code Academy (2021)",
            "SKILLS": "JavaScript, React, CSS, HTML, Node.js, REST APIs, Git",
            "PROJECTS": "E-commerce frontend in React with TypeScript"
        }
    },
    {
        "name": "Alice Chen",
        "raw_text": "Alice Chen, Data Engineer. Proficient in Python, SQL, Spark, and AWS.",
        "sections": {
            "SUMMARY": "4 years in data infrastructure and analytics.",
            "EXPERIENCE": "Data Engineer at BigData Corp (2020-2024): Python, SQL, Spark, AWS, Airflow",
            "EDUCATION": "MS Data Science, Tech University (2020)",
            "SKILLS": "Python, SQL, Apache Spark, AWS, Airflow, Kafka, Git",
            "PROJECTS": "Built data pipeline processing 100M+ events daily"
        }
    },
]


def generate():
    """Generate sample data."""
    data_dir = Path(__file__).parent / "sample_data"
    data_dir.mkdir(exist_ok=True)

    # Generate jobs
    for i, template in enumerate(JOB_TEMPLATES):
        job_data = {
            "title": template["title"],
            "company": template["company"],
            "raw_text": template["description"],
            "required_skills": template["required_skills"],
            "nice_to_have_skills": template["nice_to_have_skills"],
            "seniority": template["seniority"],
            "salary_range": template["salary_range"],
            "extracted_at": datetime.now().isoformat()
        }
        filename = data_dir / f"job_{i:02d}.json"
        with open(filename, 'w') as f:
            json.dump(job_data, f, indent=2)

    # Generate resumes
    for i, template in enumerate(RESUME_TEMPLATES):
        resume_data = {
            "raw_text": template["raw_text"],
            "sections": template["sections"]
        }
        filename = data_dir / f"resume_{i:02d}.json"
        with open(filename, 'w') as f:
            json.dump(resume_data, f, indent=2)

    print(f"Generated {len(JOB_TEMPLATES)} jobs and {len(RESUME_TEMPLATES)} resumes in {data_dir}")


if __name__ == "__main__":
    generate()

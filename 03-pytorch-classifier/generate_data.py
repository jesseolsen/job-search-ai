"""Generate synthetic training data for resume section classifier."""

import random
from typing import Tuple, List

# Realistic patterns for each section - more diverse
EXPERIENCE_PATTERNS = [
    "Senior Software Engineer at TechCorp",
    "Led backend infrastructure projects",
    "Managed team of five engineers",
    "Machine Learning Engineer building recommendation systems",
    "Full Stack Developer working with React",
    "Responsible for system architecture",
    "Data Engineer maintaining pipelines",
    "DevOps Engineer managing deployment",
    "Developed APIs using Spring Boot",
    "Architect for microservices platform",
    "Technical lead for 10 person team",
    "Worked on distributed systems",
    "Implemented CI/CD pipeline",
    "Mentored junior developers",
    "Owned payment processing service",
]

EDUCATION_PATTERNS = [
    "BS Computer Science",
    "MS Data Science",
    "MBA from Business School",
    "BA in Mathematics",
    "Bootcamp Certification",
    "PhD in Computer Science",
    "Associate Degree in IT",
    "Graduated with honors",
    "State University 2020",
    "Stanford University 2019",
    "Online degree from Coursera",
    "Technical certification exam",
    "Advanced degree from MIT",
    "Double major in CS and Math",
    "Summa Cum Laude graduate",
]

SKILLS_PATTERNS = [
    "Python programming",
    "PostgreSQL database",
    "AWS cloud platform",
    "Docker containerization",
    "React frontend",
    "TensorFlow machine learning",
    "Git version control",
    "REST API design",
    "Kubernetes orchestration",
    "JavaScript expertise",
    "SQL querying",
    "Node.js backend",
    "MongoDB NoSQL",
    "Linux system administration",
    "Java Spring Boot",
]

PROJECTS_PATTERNS = [
    "Built e-commerce platform",
    "Developed ML system",
    "Created data pipeline",
    "Built chat application",
    "Implemented cluster management",
    "Developed Python package",
    "Built analytics dashboard",
    "Created mobile app",
    "Designed database schema",
    "Implemented search engine",
    "Built monitoring system",
    "Developed testing framework",
    "Created automation tool",
    "Built recommendation system",
    "Implemented caching layer",
]

SUMMARY_PATTERNS = [
    "Experienced software engineer",
    "Data scientist specializing in ML",
    "Full stack developer",
    "DevOps specialist",
    "Technical architect",
    "Junior developer",
    "Engineering manager",
    "Product engineer",
    "Platform engineer",
    "Security specialist",
    "Performance expert",
    "Database administrator",
    "Systems engineer",
    "AI researcher",
    "Infrastructure engineer",
]

OTHER_PATTERNS = [
    "References available upon request",
    "Certifications and Awards",
    "Publications in conferences",
    "Open source contributions",
    "Languages spoken",
    "Contact information provided",
    "Portfolio website",
    "GitHub profile",
    "LinkedIn profile",
    "Awards and recognitions",
    "Speaking engagements",
    "Teaching experience",
    "Committee memberships",
    "Professional affiliations",
    "Volunteer work",
]


def generate_synthetic_data(
    num_per_section: int = 100,
) -> Tuple[List[str], List[str]]:
    """Generate synthetic labeled resume snippets.

    Args:
        num_per_section: Number of examples per section

    Returns:
        Tuple of (texts, labels)
    """
    texts = []
    labels = []

    # Generate examples for each section
    for _ in range(num_per_section):
        texts.append(random.choice(EXPERIENCE_PATTERNS))
        labels.append("EXPERIENCE")

        texts.append(random.choice(EDUCATION_PATTERNS))
        labels.append("EDUCATION")

        texts.append(random.choice(SKILLS_PATTERNS))
        labels.append("SKILLS")

        texts.append(random.choice(PROJECTS_PATTERNS))
        labels.append("PROJECTS")

        texts.append(random.choice(SUMMARY_PATTERNS))
        labels.append("SUMMARY")

        texts.append(random.choice(OTHER_PATTERNS))
        labels.append("OTHER")

    # Shuffle together
    combined = list(zip(texts, labels))
    random.shuffle(combined)
    texts, labels = zip(*combined)

    return list(texts), list(labels)


if __name__ == "__main__":
    texts, labels = generate_synthetic_data(num_per_section=50)
    print(f"Generated {len(texts)} samples")
    print(f"Distribution:")
    for label in set(labels):
        count = labels.count(label)
        print(f"  {label}: {count}")
    print(f"\nExample samples:")
    for i in range(3):
        print(f"  {labels[i]}: {texts[i][:60]}...")

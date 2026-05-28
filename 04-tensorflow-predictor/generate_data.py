"""Generate synthetic salary data for TensorFlow predictor."""

import random
from typing import Tuple, List


JOB_DESCRIPTIONS = [
    "Senior Machine Learning Engineer with 5+ years experience in neural networks and deep learning",
    "Data Scientist building predictive models and dashboards for business intelligence",
    "Junior Python Developer eager to learn full-stack web development",
    "DevOps Engineer managing cloud infrastructure and CI/CD pipelines",
    "Backend Engineer specializing in microservices and distributed systems",
    "Analytics Engineer designing data warehouses and ETL pipelines",
    "Machine Learning Operations specialist optimizing model deployment",
    "Frontend Developer building responsive web applications with React",
    "Staff Software Engineer leading technical strategy and architecture",
    "Data Engineer maintaining large-scale data processing systems",
]

SENIORITY_LEVELS = ["junior", "mid", "senior", "staff"]

# Salary distributions per seniority level (in USD)
SALARY_DISTRIBUTIONS = {
    "junior": [45000, 65000, 55000, 60000, 50000, 58000, 62000, 52000, 56000, 48000],
    "mid": [85000, 110000, 95000, 120000, 105000, 115000, 100000, 108000, 98000, 112000],
    "senior": [140000, 170000, 155000, 180000, 160000, 175000, 165000, 168000, 158000, 172000],
    "staff": [200000, 240000, 220000, 250000, 230000, 245000, 235000, 225000, 215000, 260000],
}


def assign_salary_band(salary: int) -> int:
    """Assign salary to band index.

    Args:
        salary: Salary in USD

    Returns:
        Band index: 0=<80k, 1=80k-120k, 2=120k-160k, 3=160k+
    """
    if salary < 80000:
        return 0
    elif salary < 120000:
        return 1
    elif salary < 160000:
        return 2
    else:
        return 3


def generate_synthetic_salary_data(
    n: int = 100,
) -> Tuple[List[str], List[int], List[str]]:
    """Generate synthetic job posting data with salaries.

    Args:
        n: Number of samples to generate

    Returns:
        Tuple of (descriptions, salary_bands, seniorities)
    """
    descriptions = []
    salary_bands = []
    seniorities = []

    for _ in range(n):
        seniority = random.choice(SENIORITY_LEVELS)
        description = random.choice(JOB_DESCRIPTIONS)
        salary = random.choice(SALARY_DISTRIBUTIONS[seniority])
        band = assign_salary_band(salary)

        descriptions.append(description)
        salary_bands.append(band)
        seniorities.append(seniority)

    return descriptions, salary_bands, seniorities


if __name__ == "__main__":
    descriptions, bands, seniorities = generate_synthetic_salary_data(n=50)
    print(f"Generated {len(descriptions)} job postings")
    print(f"\nSeniority distribution:")
    for seniority in SENIORITY_LEVELS:
        count = seniorities.count(seniority)
        print(f"  {seniority}: {count}")

    print(f"\nSalary band distribution:")
    for band in range(4):
        count = bands.count(band)
        labels = ["<80k", "80k-120k", "120k-160k", "160k+"]
        print(f"  {labels[band]}: {count}")

    print(f"\nExample samples:")
    for i in range(3):
        labels = ["<80k", "80k-120k", "120k-160k", "160k+"]
        print(f"  {seniorities[i]:6} → {labels[bands[i]]}: {descriptions[i][:50]}...")

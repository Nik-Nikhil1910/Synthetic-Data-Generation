"""
Project Bootstrap Script
========================

Purpose:
--------
Initializes the directory and file structure for the
Synthetic Data Generator project.

- Creates folders and placeholder files
- Safe to run multiple times
- Creates structure relative to project root
- Does NOT overwrite existing files
"""

from pathlib import Path
from typing import List


# -------------------------------------------------------------------
# Resolve Project Root (robust to execution location)
# -------------------------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent
if PROJECT_ROOT.name == "src":
    PROJECT_ROOT = PROJECT_ROOT.parent


# -------------------------------------------------------------------
# Final Project Structure (ROOT RELATIVE)
# -------------------------------------------------------------------

PROJECT_STRUCTURE: List[str] = [
    # =========================
    # Application Core
    # =========================
    "src/app/__init__.py",
    "src/app/main.py",
    "src/app/config.py",
    "src/app/exceptions.py",

    # =========================
    # Schema Inference
    # =========================
    "src/schema_inference/__init__.py",
    "src/schema_inference/infer_schema.py",
    "src/schema_inference/validators.py",

    # =========================
    # Generation Plan
    # =========================
    "src/generation_plan/__init__.py",
    "src/generation_plan/plan_generator.py",
    "src/generation_plan/validators.py",

    # =========================
    # Code Generation
    # =========================
    "src/code_generation/__init__.py",
    "src/code_generation/faker_generator.py",
    "src/code_generation/validators.py",

    # =========================
    # Shared Utilities
    # =========================
    "src/common/__init__.py",
    "src/common/contract_utils.py",
    "src/common/typing_defs.py",

    # =========================
    # API Layer
    # =========================
    "src/api/__init__.py",
    "src/api/routes.py",
    "src/api/schemas.py",

    # =========================
    # Tests
    # =========================
    "tests/test_schema_inference.py",
    "tests/test_generation_plan.py",
    "tests/test_code_generation.py",

    # =========================
    # Deployment
    # =========================
    "deploy/Dockerfile",
    "deploy/docker-compose.yml",
    "deploy/entrypoint.sh",

    # =========================
    # Root-Level Files
    # =========================
    ".env.example",
    "requirements.txt",
    "README.md",
]


# -------------------------------------------------------------------
# Core Logic
# -------------------------------------------------------------------

def create_path(relative_path: str) -> None:
    """
    Create a file path relative to project root.
    """
    path = PROJECT_ROOT / relative_path
    path.parent.mkdir(parents=True, exist_ok=True)

    if not path.exists():
        path.touch()

    print(f"✓ {path.relative_to(PROJECT_ROOT)}")


def bootstrap_project() -> None:
    """
    Bootstrap the project structure.
    """
    print(f"\n🚀 Initializing project at:\n{PROJECT_ROOT}\n")

    for relative_path in PROJECT_STRUCTURE:
        create_path(relative_path)

    print("\n✅ Project structure initialized successfully.")


# -------------------------------------------------------------------
# Entry Point
# -------------------------------------------------------------------

if __name__ == "__main__":
    bootstrap_project()

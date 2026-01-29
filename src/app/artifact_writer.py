"""
Artifact Writer
================

Manages writing internal artifacts to the artifacts/ directory.
Artifacts are for debugging/audit, never exposed to users.
"""

import json
import os
from pathlib import Path
from datetime import datetime
from typing import Any
import uuid


# =============================================================================
# CONFIGURATION
# =============================================================================

# Default artifacts directory (can be overridden via environment variable)
ARTIFACTS_DIR = os.environ.get(
    "SDG_ARTIFACTS_DIR",
    str(Path(__file__).parent.parent.parent / "artifacts")
)


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def get_run_id() -> str:
    """Generate a unique run ID."""
    return str(uuid.uuid4())


def get_artifacts_dir(run_id: str) -> Path:
    """
    Get the artifacts directory for a specific run.
    Creates the directory if it doesn't exist.
    """
    run_dir = Path(ARTIFACTS_DIR) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def write_artifact(run_id: str, filename: str, content: Any) -> str:
    """
    Write an artifact file.
    
    Args:
        run_id: Unique run identifier.
        filename: Name of the artifact file.
        content: Content to write (str or dict for JSON).
        
    Returns:
        Path to the written file.
    """
    artifacts_path = get_artifacts_dir(run_id)
    file_path = artifacts_path / filename
    
    if isinstance(content, dict):
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(content, f, indent=2, default=str)
    else:
        with open(file_path, "w", encoding="utf-8") as f:
            f.write(str(content))
    
    return str(file_path)


# =============================================================================
# STAGE-SPECIFIC ARTIFACT WRITERS
# =============================================================================

def write_raw_input(run_id: str, description: str) -> str:
    """Write raw_input.txt artifact."""
    return write_artifact(run_id, "raw_input.txt", description)


def write_refined_description(run_id: str, description: str, attempt: int) -> str:
    """Write refined_description_attempt_N.txt artifact."""
    filename = f"refined_description_attempt_{attempt}.txt"
    return write_artifact(run_id, filename, description)


def write_final_refined_description(run_id: str, description: str) -> str:
    """Write final_refined_description.txt artifact."""
    return write_artifact(run_id, "final_refined_description.txt", description)


def write_schema_validation_feedback(run_id: str, feedback: dict) -> str:
    """Write schema_validation_feedback.json artifact."""
    return write_artifact(run_id, "schema_validation_feedback.json", feedback)


def write_schema(run_id: str, schema: dict) -> str:
    """Write schema.json artifact."""
    return write_artifact(run_id, "schema.json", schema)


def write_generation_plan(run_id: str, plan: dict) -> str:
    """Write generation_plan.json artifact."""
    return write_artifact(run_id, "generation_plan.json", plan)


def write_generated_code(run_id: str, code: str) -> str:
    """Write generated_code.py artifact."""
    return write_artifact(run_id, "generated_code.py", code)


def write_execution_log(run_id: str, log: dict) -> str:
    """Write execution_log.json artifact."""
    return write_artifact(run_id, "execution_log.json", log)


def write_export_manifest(run_id: str, manifest: dict) -> str:
    """Write export_manifest.json artifact."""
    return write_artifact(run_id, "export_manifest.json", manifest)

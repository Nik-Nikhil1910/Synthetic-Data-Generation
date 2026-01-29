"""
App Module
==========

FastAPI application initialization.
"""

from .config import VERSION, APP_NAME, OUTPUT_DIR, get_output_dir
from .exceptions import get_http_exception, global_exception_handler
from .artifact_writer import (
    get_run_id,
    get_artifacts_dir,
    write_artifact,
    write_raw_input,
    write_refined_description,
    write_final_refined_description,
    write_schema_validation_feedback,
    write_schema,
    write_generation_plan,
    write_generated_code,
    write_execution_log,
    write_export_manifest,
)

__all__ = [
    "VERSION",
    "APP_NAME",
    "OUTPUT_DIR",
    "get_output_dir",
    "get_http_exception",
    "global_exception_handler",
    "get_run_id",
    "get_artifacts_dir",
    "write_artifact",
    "write_raw_input",
    "write_refined_description",
    "write_final_refined_description",
    "write_schema_validation_feedback",
    "write_schema",
    "write_generation_plan",
    "write_generated_code",
    "write_execution_log",
    "write_export_manifest",
]

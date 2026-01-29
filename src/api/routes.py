"""
API Routes
==========

Endpoint definitions for the Synthetic Data Generator API.
Implements the interactive two-step workflow:
  1. POST /schema/refine - Clarify business description
  2. POST /schema/generate - Generate schema and execute pipeline

This module orchestrates Stages 0-C without adding business logic.
"""

from fastapi import APIRouter

from .schemas import (
    RefineRequest,
    RefineResponse,
    GenerateRequest,
    GenerateResponse,
    GeneratedFiles,
    SqlFiles,
    DebugInfo,
    HealthResponse,
    VersionResponse,
)

# Import core pipeline functions
from src.schema_inference import (
    refine_understanding,
    refine_understanding_with_feedback,
    generate_schema_from_text,
)
from src.schema_inference.infer_schema_validator import (
    validate_schema_with_feedback,
    TerminationError,
)
from src.generation_plan import generate_plan
from src.code_generation import generate_python_code, execute_code
from src.export import export_to_json, export_to_csv, export_to_sql

# Import config and utilities
from src.app import config as app_config
from src.app import exceptions as app_exceptions
from src.app import artifact_writer


# =============================================================================
# ROUTER
# =============================================================================

router = APIRouter()


# =============================================================================
# ATTEMPT TRACKING (run_id-based)
# =============================================================================

# Track attempts per run_id to enforce 2-attempt limit
# Key: run_id, Value: {"attempt": int, "description": str}
_attempt_tracker: dict[str, dict] = {}


def _register_run(run_id: str, description: str) -> None:
    """Register a new run with attempt 1."""
    _attempt_tracker[run_id] = {
        "attempt": 1,
        "description": description
    }


def _get_attempt(run_id: str) -> int:
    """Get current attempt number for a run. Returns 0 if not found."""
    if run_id in _attempt_tracker:
        return _attempt_tracker[run_id].get("attempt", 0)
    return 0


def _increment_attempt(run_id: str) -> int:
    """Increment attempt counter for a run."""
    if run_id in _attempt_tracker:
        _attempt_tracker[run_id]["attempt"] = 2
        return 2
    return 0


def _clear_run(run_id: str) -> None:
    """Clear run tracking after completion or termination."""
    _attempt_tracker.pop(run_id, None)


# =============================================================================
# ENDPOINTS
# =============================================================================

@router.post("/schema/refine", response_model=RefineResponse)
def refine_schema(request: RefineRequest) -> RefineResponse:
    """
    Step 1: Refine raw business description into clear, structured text.
    
    No schema generation occurs here. Returns run_id for use in /schema/generate.
    User must review and approve the refined description before proceeding.
    """
    try:
        # Generate run_id for artifact and attempt tracking
        run_id = artifact_writer.get_run_id()
        
        # Persist raw input
        artifact_writer.write_raw_input(run_id, request.description)
        
        # Call Layer 1 (Refinement AI)
        understanding = refine_understanding(request.description)
        refined_text = understanding.authoritative_description
        
        # Persist refined description
        artifact_writer.write_refined_description(run_id, refined_text, attempt=1)
        
        # Register run for attempt tracking
        _register_run(run_id, refined_text)
        
        return RefineResponse(
            status="needs_review",
            run_id=run_id,
            refined_description=refined_text
        )
    
    except Exception as e:
        raise app_exceptions.get_http_exception(e)


@router.post("/schema/generate", response_model=GenerateResponse)
def generate_data(request: GenerateRequest) -> GenerateResponse:
    """
    Step 2: Generate schema from approved description and execute pipeline.
    
    Requires run_id from /schema/refine. Implements two-attempt logic:
    - Attempt 1: If validation fails, returns improved description for review.
    - Attempt 2: If validation fails again, terminates.
    """
    try:
        run_id = request.run_id
        attempt = _get_attempt(run_id)
        
        # If run_id not found, treat as attempt 1 (new run)
        if attempt == 0:
            _register_run(run_id, request.approved_description)
            attempt = 1
        
        # Persist the approved description
        artifact_writer.write_final_refined_description(run_id, request.approved_description)
        
        # --- PHASE 2: Generate Schema ---
        schema = generate_schema_from_text(request.approved_description)
        
        # --- PHASE 3: Validate Schema ---
        feedback = validate_schema_with_feedback(schema, attempt)
        
        if not feedback.is_valid:
            # Persist validation feedback
            artifact_writer.write_schema_validation_feedback(run_id, {
                "attempt": attempt,
                "errors": feedback.errors,
                "suggestions": feedback.suggestions
            })
            
            if attempt == 1:
                # First failure: refine and return for user review
                _increment_attempt(run_id)
                
                # Use Layer 1 to improve description based on feedback
                improved = refine_understanding_with_feedback(
                    request.approved_description,
                    feedback
                )
                improved_text = improved.authoritative_description
                
                # Persist improved description
                artifact_writer.write_refined_description(run_id, improved_text, attempt=2)
                
                return GenerateResponse(
                    status="needs_review",
                    run_id=run_id,
                    refined_description=improved_text
                )
            else:
                # Second failure: terminate
                _clear_run(run_id)
                return GenerateResponse(
                    status="terminated",
                    run_id=run_id,
                    message="Please re-explain the business problem from scratch with more clarity."
                )
        
        # --- SCHEMA VALID: Execute Full Pipeline ---
        
        # Persist schema
        artifact_writer.write_schema(run_id, schema)
        
        # Stage A: Generation Plan
        plan = generate_plan(schema)
        artifact_writer.write_generation_plan(run_id, plan)
        
        # Stage B: Code Generation + Execution
        code = generate_python_code(plan)
        artifact_writer.write_generated_code(run_id, code)
        
        exec_result = execute_code(code)
        data_store = exec_result.data_store
        
        artifact_writer.write_execution_log(run_id, {
            "tables": list(data_store.keys()),
            "row_counts": {k: len(v) for k, v in data_store.items()}
        })
        
        # Stage C: Export
        output_dir = app_config.get_output_dir()
        files = GeneratedFiles()
        sql_files = SqlFiles()
        
        if "json" in request.export_formats:
            files.json = export_to_json(data_store, output_dir)
        
        if "csv" in request.export_formats:
            files.csv = export_to_csv(data_store, output_dir, schema)
        
        if "sql" in request.export_formats:
            for dialect in request.sql_dialects:
                sql_path = export_to_sql(data_store, schema, output_dir, dialect)
                if dialect == "sqlite":
                    sql_files.sqlite = sql_path
                elif dialect == "postgresql":
                    sql_files.postgresql = sql_path
                elif dialect == "mysql":
                    sql_files.mysql = sql_path
            files.sql = sql_files
        
        # Write export manifest
        artifact_writer.write_export_manifest(run_id, {
            "json": files.json,
            "csv": files.csv,
            "sql": {
                "sqlite": sql_files.sqlite,
                "postgresql": sql_files.postgresql,
                "mysql": sql_files.mysql
            }
        })
        
        # Clear run tracking
        _clear_run(run_id)
        
        # Build response
        response = GenerateResponse(
            status="success",
            run_id=run_id,
            files=files
        )
        
        # Include debug info if requested
        if request.include_debug:
            response.debug_info = DebugInfo(
                schema_json=schema,
                plan_json=plan
            )
        
        return response
    
    except TerminationError:
        # Hard termination from validator
        _clear_run(request.run_id)
        return GenerateResponse(
            status="terminated",
            run_id=request.run_id,
            message="Please re-explain the business problem from scratch with more clarity."
        )
    
    except Exception as e:
        raise app_exceptions.get_http_exception(e)


@router.get("/health", response_model=HealthResponse)
def health_check() -> HealthResponse:
    """Health check endpoint."""
    return HealthResponse(status="ok")


@router.get("/version", response_model=VersionResponse)
def get_version() -> VersionResponse:
    """Get API version."""
    return VersionResponse(version=app_config.VERSION, name=app_config.APP_NAME)

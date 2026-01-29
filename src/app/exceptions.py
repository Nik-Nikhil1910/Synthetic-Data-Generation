"""
Application Exceptions
======================

Maps internal exceptions to HTTP status codes.
"""

from fastapi import HTTPException, Request
from fastapi.responses import JSONResponse


# =============================================================================
# EXCEPTION MAPPING
# =============================================================================

# Maps exception class names to (status_code, user_message)
EXCEPTION_MAP = {
    # Stage 0: Schema Inference
    "ContractViolationError": (400, "Could not infer valid schema from description."),
    "TerminationError": (400, "Schema inference was terminated."),
    
    # Stage A: Generation Plan
    "InvalidSchemaError": (422, "The inferred schema is not valid for plan generation."),
    "CircularDependencyError": (422, "The schema contains circular dependencies."),
    "GenerationPlanError": (422, "Failed to generate a valid plan from schema."),
    "PlanValidationError": (422, "Generated plan failed validation."),
    
    # Stage B: Code Generation
    "CodeGenerationError": (500, "Failed to compile generation code."),
    "SyntaxValidationError": (500, "Generated code has syntax errors."),
    "ExecutionValidationError": (500, "Generated code failed to execute."),
    "CodeValidationError": (500, "Generated code failed validation."),
    
    # Stage C: Export
    "JsonExportError": (500, "Failed to export JSON."),
    "CsvExportError": (500, "Failed to export CSV."),
    "SqlExportError": (500, "Failed to export SQL."),
    "ExportValidationError": (500, "Exported files failed validation."),
}


def get_http_exception(exc: Exception) -> HTTPException:
    """
    Convert an internal exception to an HTTPException.
    
    Args:
        exc: The caught exception.
        
    Returns:
        HTTPException with appropriate status code and detail.
    """
    exc_name = type(exc).__name__
    
    if exc_name in EXCEPTION_MAP:
        status_code, user_message = EXCEPTION_MAP[exc_name]
        return HTTPException(
            status_code=status_code,
            detail={
                "status": "error",
                "message": user_message,
                "detail": str(exc)
            }
        )
    
    # Fallback for unknown exceptions
    return HTTPException(
        status_code=500,
        detail={
            "status": "error",
            "message": "Internal system error.",
            "detail": str(exc)
        }
    )


async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
    """
    Global exception handler for FastAPI.
    
    Catches all unhandled exceptions and returns structured error response.
    """
    exc_name = type(exc).__name__
    
    if exc_name in EXCEPTION_MAP:
        status_code, user_message = EXCEPTION_MAP[exc_name]
    else:
        status_code = 500
        user_message = "Internal system error."
    
    return JSONResponse(
        status_code=status_code,
        content={
            "status": "error",
            "message": user_message,
            "detail": str(exc)
        }
    )

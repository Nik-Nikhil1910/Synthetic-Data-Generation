"""
API Request/Response Schemas
============================

Pydantic models for API request and response validation.
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field


# =============================================================================
# REQUEST SCHEMAS
# =============================================================================

class RefineRequest(BaseModel):
    """Request body for POST /schema/refine endpoint."""
    
    description: str = Field(
        ...,
        min_length=5,
        description="Raw business requirement to refine"
    )


class GenerateRequest(BaseModel):
    """Request body for POST /schema/generate endpoint."""
    
    run_id: str = Field(
        ...,
        description="Run ID returned by /schema/refine (required for attempt tracking)"
    )
    approved_description: str = Field(
        ...,
        min_length=10,
        description="User-approved refined description from /schema/refine"
    )
    export_formats: list[Literal["json", "csv", "sql"]] = Field(
        default=["json", "csv", "sql"],
        description="Output formats to generate"
    )
    sql_dialects: list[Literal["sqlite", "postgresql", "mysql"]] = Field(
        default=["sqlite"],
        description="SQL dialects to generate (only used if 'sql' in export_formats)"
    )
    include_debug: bool = Field(
        default=False,
        description="Include schema and plan in response (debug mode only)"
    )


# =============================================================================
# RESPONSE SCHEMAS
# =============================================================================

class SqlFiles(BaseModel):
    """SQL file paths grouped by dialect."""
    
    sqlite: Optional[str] = None
    postgresql: Optional[str] = None
    mysql: Optional[str] = None


class GeneratedFiles(BaseModel):
    """File paths grouped by format."""
    
    json: list[str] = Field(default_factory=list)
    csv: list[str] = Field(default_factory=list)
    sql: Optional[SqlFiles] = None


class DebugInfo(BaseModel):
    """Debug information (schema and plan). Only included if requested."""
    
    schema_json: dict = Field(default_factory=dict)
    plan_json: dict = Field(default_factory=dict)


class RefineResponse(BaseModel):
    """Response for POST /schema/refine (always needs_review)."""
    
    status: Literal["needs_review"] = "needs_review"
    run_id: str
    refined_description: str


class GenerateResponse(BaseModel):
    """
    Response for POST /schema/generate.
    
    Three possible outcomes:
    1. status="success" → Pipeline completed, files available
    2. status="needs_review" → Validation failed (attempt 1), user must review improved description
    3. status="terminated" → Validation failed twice, user must restart from /schema/refine
    """
    
    status: Literal["success", "needs_review", "terminated"]
    
    # Always present
    run_id: Optional[str] = None
    
    # Present on "success"
    files: Optional[GeneratedFiles] = None
    
    # Present on "needs_review" (retry)
    refined_description: Optional[str] = None
    
    # Present on "terminated"
    message: Optional[str] = None
    
    # Present if include_debug=True and status="success"
    debug_info: Optional[DebugInfo] = None


class HealthResponse(BaseModel):
    """Response for GET /health endpoint."""
    
    status: str = "ok"


class VersionResponse(BaseModel):
    """Response for GET /version endpoint."""
    
    version: str
    name: str = "Synthetic Data Generator"


class ErrorResponse(BaseModel):
    """Standard error response."""
    
    status: str = "error"
    message: str
    detail: Optional[str] = None

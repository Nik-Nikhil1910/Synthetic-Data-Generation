"""
Schema Inference Module
=======================

This module provides GenAI-powered schema inference using a multi-phase
human-in-the-loop workflow.

Usage:
------
1. Call `infer_schema_interactive(description)` to get Phase 1 output
2. User reviews/edits the `authoritative_description`
3. Call `finalize_schema(approved_description)` to get the final schema

The legacy heuristic-based `infer_schema()` function is deprecated.
"""

from .infer_schema import (
    # Interactive workflow entry points (RECOMMENDED)
    infer_schema_interactive,
    finalize_schema,
    refine_understanding,
    generate_schema_from_text,
    
    # Data structures (workflow-specific)
    RefinedUnderstanding,
    
    # Exceptions (workflow-specific)
    SchemaInferenceError,
    NoEntitiesFoundError,
    AmbiguousRelationshipError,
    LLMUnavailableError,
    
    # Deprecated (raises DeprecationWarning)
    infer_schema,
)

# Validation-related imports from dedicated validator module
from .infer_schema_validator import (
    # Validation functions
    validate_schema_with_feedback,
    validate_and_handle_termination,
    
    # Data structures
    ValidationFeedback,
    
    # Exceptions
    ContractViolationError,
    TerminationError,
)

__all__ = [
    # Primary API
    "infer_schema_interactive",
    "finalize_schema",
    "refine_understanding",
    "generate_schema_from_text",
    "validate_schema_with_feedback",
    "validate_and_handle_termination",
    
    # Data structures
    "RefinedUnderstanding",
    "ValidationFeedback",
    
    # Exceptions
    "SchemaInferenceError",
    "NoEntitiesFoundError",
    "ContractViolationError",
    "AmbiguousRelationshipError",
    "LLMUnavailableError",
    "TerminationError",
    
    # Deprecated
    "infer_schema",
]

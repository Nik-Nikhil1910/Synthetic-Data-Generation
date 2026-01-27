"""
Generation Plan Module
======================

This module generates deterministic data generation plans from
validated Stage 0 schema output.

Public Interface:
-----------------
- generate_plan(schema, row_count=50) -> dict
- validate_generation_plan(plan) -> ValidationResult
"""

from .plan_generator import (
    # Primary API
    generate_plan,
    
    # Exceptions
    GenerationPlanError,
    CircularDependencyError,
    InvalidSchemaError,
)

from .generated_plan_validator import (
    # Validation API
    validate_generation_plan,
    validate_and_raise,
    
    # Data structures
    ValidationResult,
    
    # Exceptions
    PlanValidationError,
)

from .validators import (
    # Pydantic models (for type checking and advanced usage)
    GenerationPlan,
    TablePlan,
    PrimaryKeyColumn,
    ForeignKeyColumn,
    FakerColumn,
    PlanMeta,
)

__all__ = [
    # Primary API
    "generate_plan",
    "validate_generation_plan",
    "validate_and_raise",
    
    # Data structures
    "ValidationResult",
    "GenerationPlan",
    "TablePlan",
    "PrimaryKeyColumn",
    "ForeignKeyColumn",
    "FakerColumn",
    "PlanMeta",
    
    # Exceptions
    "GenerationPlanError",
    "CircularDependencyError",
    "InvalidSchemaError",
    "PlanValidationError",
]

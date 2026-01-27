"""
Code Generation Module
======================

Stage B of the Synthetic Data Generator pipeline.
Converts validated Generation Plan JSON into executable Python code.

This is a deterministic compiler with NO AI, NO heuristics.
"""

from .faker_generator import (
    generate_python_code,
    CodeGenerationError,
)

from .generated_code_validators import (
    validate_syntax,
    execute_code,
    validate_execution_against_plan,
    ExecutionResult,
    CodeValidationError,
    SyntaxValidationError,
    ExecutionValidationError,
)

__all__ = [
    # Primary API
    "generate_python_code",
    
    # Validation API
    "validate_syntax",
    "execute_code",
    "validate_execution_against_plan",
    
    # Data structures
    "ExecutionResult",
    
    # Exceptions
    "CodeGenerationError",
    "CodeValidationError",
    "SyntaxValidationError",
    "ExecutionValidationError",
]

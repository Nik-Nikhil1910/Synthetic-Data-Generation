"""
Schema Validation Module
========================

This module contains all schema validation logic for the Schema Inference workflow.
It is purely deterministic and contains no LLM usage.

Validation is performed against the Schema Inference Contract.
"""

from typing import Any
from dataclasses import dataclass


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ContractViolationError(Exception):
    """Raised when the inferred schema violates the contract."""
    pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ValidationFeedback:
    """
    Phase 3 output on validation failure.
    Returned to user to guide their revision.
    """
    is_valid: bool
    errors: list[str]
    suggestions: list[str]
    attempt_number: int  # 1 or 2


class TerminationError(Exception):
    """
    Raised after 2 failed schema validation attempts.
    Caller must prompt user to restart from scratch.
    """
    def __init__(self, final_errors: list[str]):
        self.final_errors = final_errors
        super().__init__(
            "Schema inference failed after 2 attempts. "
            "Please re-explain the business problem from scratch."
        )


# =============================================================================
# CONSTANTS
# =============================================================================

ALLOWED_TYPES = frozenset({
    "INTEGER", "VARCHAR", "TEXT", "BOOLEAN", "DATE", "DATETIME", "FLOAT"
})


# =============================================================================
# VALIDATION LOGIC
# =============================================================================

def _validate_schema_internal(schema: dict[str, Any]) -> None:
    """
    Validate the inferred schema against the System Contract rules.
    Raises ContractViolationError on any violation.
    
    This is an internal function. Use validate_schema_with_feedback() for
    the public interface that returns structured feedback.
    """
    table_names: set[str] = set()
    table_columns: dict[str, set[str]] = {}
    table_col_types: dict[str, dict[str, str]] = {}

    for table in schema["tables"]:
        name = table["name"]

        # Unique Entity Names.
        if name in table_names:
            raise ContractViolationError(f"Duplicate table name: '{name}'.")
        table_names.add(name)

        col_names: set[str] = set()
        col_types: dict[str, str] = {}
        pk_count = 0

        for col in table["columns"]:
            col_name = col["name"]

            # Duplicate column check.
            if col_name in col_names:
                raise ContractViolationError(
                    f"Duplicate column '{col_name}' in table '{name}'."
                )
            col_names.add(col_name)
            col_types[col_name] = col["type"]

            # Allowed Types.
            if col["type"] not in ALLOWED_TYPES:
                raise ContractViolationError(
                    f"Invalid type '{col['type']}' for column "
                    f"'{name}.{col_name}'. Allowed: {sorted(ALLOWED_TYPES)}."
                )

            # Primary Key Validation.
            if col["is_pk"]:
                pk_count += 1
                if col["is_nullable"]:
                    raise ContractViolationError(
                        f"Primary key '{name}.{col_name}' cannot be nullable."
                    )
                if col["type"] not in ("INTEGER", "VARCHAR"):
                    raise ContractViolationError(
                        f"Primary key '{name}.{col_name}' must be INTEGER or VARCHAR."
                    )

        # Exactly one PK per table.
        if pk_count != 1:
            raise ContractViolationError(
                f"Table '{name}' must have exactly one primary key, found {pk_count}."
            )

        table_columns[name] = col_names
        table_col_types[name] = col_types

    # FK Validation (second pass).
    for table in schema["tables"]:
        name = table["name"]
        local_cols = table_columns[name]
        local_types = table_col_types[name]

        seen_fks: set[tuple[str, str, str]] = set()

        for fk in table["foreign_keys"]:
            fk_col = fk["column"]
            ref_table = fk["references_table"]
            ref_col = fk["references_column"]

            # Duplicate FK constraint check.
            fk_key = (fk_col, ref_table, ref_col)
            if fk_key in seen_fks:
                raise ContractViolationError(
                    f"Duplicate FK constraint in '{name}': {fk_key}."
                )
            seen_fks.add(fk_key)

            # FK column must exist locally.
            if fk_col not in local_cols:
                raise ContractViolationError(
                    f"FK column '{fk_col}' does not exist in table '{name}'."
                )

            # Referenced table must exist.
            if ref_table not in table_names:
                raise ContractViolationError(
                    f"FK in '{name}' references non-existent table '{ref_table}'."
                )

            # Referenced column must exist in target table.
            if ref_col not in table_columns[ref_table]:
                raise ContractViolationError(
                    f"FK in '{name}' references non-existent column '{ref_table}.{ref_col}'."
                )

            # Type Matching.
            local_type = local_types[fk_col]
            target_type = table_col_types[ref_table][ref_col]
            if local_type != target_type:
                raise ContractViolationError(
                    f"FK type mismatch: '{name}.{fk_col}' ({local_type}) != "
                    f"'{ref_table}.{ref_col}' ({target_type})."
                )


def _generate_suggestions(error_msg: str) -> list[str]:
    """Generate helpful suggestions based on validation error."""
    suggestions = []
    
    if "Duplicate table" in error_msg:
        suggestions.append("Ensure each entity is mentioned only once in your description.")
    if "foreign_keys" in error_msg.lower() or "FK" in error_msg:
        suggestions.append("Check that relationships between entities are clearly stated.")
    if "primary key" in error_msg.lower():
        suggestions.append("Each table should have exactly one identifier.")
    if "type" in error_msg.lower():
        suggestions.append("Ensure data types are appropriate for each field.")
    
    if not suggestions:
        suggestions.append("Please review your description for clarity and completeness.")
    
    return suggestions


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def validate_schema_with_feedback(schema: dict, attempt_number: int) -> ValidationFeedback:
    """
    Validate schema using deterministic logic and return structured feedback.
    
    This is the primary public interface for Phase 3 validation.
    
    Args:
        schema: Schema dict to validate.
        attempt_number: Current attempt (1 or 2).
        
    Returns:
        ValidationFeedback with validation results.
    """
    try:
        _validate_schema_internal(schema)
        return ValidationFeedback(
            is_valid=True,
            errors=[],
            suggestions=[],
            attempt_number=attempt_number,
        )
    except ContractViolationError as e:
        error_msg = str(e)
        suggestions = _generate_suggestions(error_msg)
        return ValidationFeedback(
            is_valid=False,
            errors=[error_msg],
            suggestions=suggestions,
            attempt_number=attempt_number,
        )


def validate_and_handle_termination(
    schema: dict,
    attempt_number: int
) -> ValidationFeedback:
    """
    Validate schema and handle termination logic.
    
    Args:
        schema: Schema dict to validate.
        attempt_number: Current attempt (1 or 2).
        
    Returns:
        ValidationFeedback if validation passes or on first failure.
        
    Raises:
        TerminationError: On second validation failure.
    """
    feedback = validate_schema_with_feedback(schema, attempt_number)
    
    if feedback.is_valid:
        return feedback
    
    # Handle failure based on attempt number
    if attempt_number == 1:
        # First failure: return feedback for user to revise
        return feedback
    else:
        # Second failure: terminate
        raise TerminationError(feedback.errors)

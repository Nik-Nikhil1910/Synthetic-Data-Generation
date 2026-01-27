"""
Generation Plan Validator
=========================

This module validates Generation Plan JSON output from plan_generator.py.
It is purely deterministic and contains NO LLM/AI logic.

Validation is performed against the Generation Plan Contract.
"""

from typing import Any
from dataclasses import dataclass


# =============================================================================
# EXCEPTIONS
# =============================================================================

class PlanValidationError(Exception):
    """Raised when the generation plan fails validation."""
    pass


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class ValidationResult:
    """Result of validating a Generation Plan."""
    is_valid: bool
    errors: list[str]
    
    @property
    def error_summary(self) -> str:
        """Get a formatted summary of all errors."""
        if self.is_valid:
            return "Plan is valid."
        return "; ".join(self.errors)


# =============================================================================
# CONSTANTS
# =============================================================================

VALID_GENERATOR_TYPES = frozenset({"primary_key", "foreign_key", "faker"})
VALID_PK_STRATEGIES = frozenset({"sequential_integer"})


# =============================================================================
# INTERNAL VALIDATION FUNCTIONS
# =============================================================================

def _validate_structure(plan: dict) -> list[str]:
    """Validate basic plan structure."""
    errors = []
    
    if not isinstance(plan, dict):
        errors.append("Plan must be a dictionary.")
        return errors
    
    if "execution_order" not in plan:
        errors.append("Plan missing required 'execution_order' field.")
    elif not isinstance(plan["execution_order"], list):
        errors.append("'execution_order' must be a list.")
    elif len(plan["execution_order"]) == 0:
        errors.append("'execution_order' cannot be empty.")
    
    if "tables" not in plan:
        errors.append("Plan missing required 'tables' field.")
    elif not isinstance(plan["tables"], list):
        errors.append("'tables' must be a list.")
    elif len(plan["tables"]) == 0:
        errors.append("'tables' cannot be empty.")
    
    return errors


def _validate_execution_order_matches_tables(plan: dict) -> list[str]:
    """R1 & R2: Validate execution_order matches table names."""
    errors = []
    
    table_names = {t["name"] for t in plan.get("tables", [])}
    order_names = set(plan.get("execution_order", []))
    
    missing_in_order = table_names - order_names
    extra_in_order = order_names - table_names
    
    if missing_in_order:
        errors.append(f"Tables missing in execution_order: {sorted(missing_in_order)}")
    if extra_in_order:
        errors.append(f"Unknown tables in execution_order: {sorted(extra_in_order)}")
    
    return errors


def _validate_execution_order_dependencies(plan: dict) -> list[str]:
    """R3: Validate FK parents appear before children in execution order."""
    errors = []
    
    execution_order = plan.get("execution_order", [])
    table_position = {name: i for i, name in enumerate(execution_order)}
    
    for table in plan.get("tables", []):
        table_name = table.get("name", "")
        table_pos = table_position.get(table_name, -1)
        
        for col in table.get("columns", []):
            if col.get("generator_type") == "foreign_key":
                references = col.get("references", {})
                parent_table = references.get("table", "")
                parent_pos = table_position.get(parent_table, -1)
                
                if parent_pos >= table_pos:
                    errors.append(
                        f"Invalid execution order: '{parent_table}' must precede "
                        f"'{table_name}' (FK column: '{col.get('name', 'unknown')}')"
                    )
    
    return errors


def _validate_fk_references(plan: dict) -> list[str]:
    """R4: Validate FK references point to existing tables/columns."""
    errors = []
    
    # Build lookup of table -> columns
    table_columns: dict[str, set[str]] = {}
    for table in plan.get("tables", []):
        table_name = table.get("name", "")
        columns = {col.get("name", "") for col in table.get("columns", [])}
        table_columns[table_name] = columns
    
    all_tables = set(table_columns.keys())
    
    for table in plan.get("tables", []):
        table_name = table.get("name", "")
        
        for col in table.get("columns", []):
            if col.get("generator_type") == "foreign_key":
                references = col.get("references", {})
                ref_table = references.get("table", "")
                ref_column = references.get("column", "")
                
                if not ref_table:
                    errors.append(
                        f"FK column '{table_name}.{col.get('name')}' missing "
                        f"'references.table'"
                    )
                elif ref_table not in all_tables:
                    errors.append(
                        f"FK column '{table_name}.{col.get('name')}' references "
                        f"non-existent table '{ref_table}'"
                    )
                elif not ref_column:
                    errors.append(
                        f"FK column '{table_name}.{col.get('name')}' missing "
                        f"'references.column'"
                    )
                elif ref_column not in table_columns.get(ref_table, set()):
                    errors.append(
                        f"FK column '{table_name}.{col.get('name')}' references "
                        f"non-existent column '{ref_table}.{ref_column}'"
                    )
    
    return errors


def _validate_generator_types(plan: dict) -> list[str]:
    """R5: Validate generator_type values."""
    errors = []
    
    for table in plan.get("tables", []):
        table_name = table.get("name", "")
        
        for col in table.get("columns", []):
            col_name = col.get("name", "")
            gen_type = col.get("generator_type", "")
            
            if not gen_type:
                errors.append(
                    f"Column '{table_name}.{col_name}' missing 'generator_type'"
                )
            elif gen_type not in VALID_GENERATOR_TYPES:
                errors.append(
                    f"Column '{table_name}.{col_name}' has invalid generator_type "
                    f"'{gen_type}'. Valid: {sorted(VALID_GENERATOR_TYPES)}"
                )
    
    return errors


def _validate_pk_columns(plan: dict) -> list[str]:
    """R6: Validate PK columns have correct strategy."""
    errors = []
    
    for table in plan.get("tables", []):
        table_name = table.get("name", "")
        pk_count = 0
        
        for col in table.get("columns", []):
            if col.get("generator_type") == "primary_key":
                pk_count += 1
                col_name = col.get("name", "")
                strategy = col.get("strategy", "")
                
                if not strategy:
                    errors.append(
                        f"PK column '{table_name}.{col_name}' missing 'strategy'"
                    )
                elif strategy not in VALID_PK_STRATEGIES:
                    errors.append(
                        f"PK column '{table_name}.{col_name}' has invalid strategy "
                        f"'{strategy}'. Valid: {sorted(VALID_PK_STRATEGIES)}"
                    )
        
        if pk_count == 0:
            errors.append(f"Table '{table_name}' has no primary key column")
        elif pk_count > 1:
            errors.append(f"Table '{table_name}' has {pk_count} PK columns (expected 1)")
    
    return errors


def _validate_fk_columns(plan: dict) -> list[str]:
    """R7: Validate FK columns have valid references object."""
    errors = []
    
    for table in plan.get("tables", []):
        table_name = table.get("name", "")
        
        for col in table.get("columns", []):
            if col.get("generator_type") == "foreign_key":
                col_name = col.get("name", "")
                references = col.get("references")
                
                if references is None:
                    errors.append(
                        f"FK column '{table_name}.{col_name}' missing 'references'"
                    )
                elif not isinstance(references, dict):
                    errors.append(
                        f"FK column '{table_name}.{col_name}' 'references' must be object"
                    )
    
    return errors


def _validate_faker_columns(plan: dict) -> list[str]:
    """R8: Validate Faker columns have faker_provider."""
    errors = []
    
    for table in plan.get("tables", []):
        table_name = table.get("name", "")
        
        for col in table.get("columns", []):
            if col.get("generator_type") == "faker":
                col_name = col.get("name", "")
                faker_provider = col.get("faker_provider", "")
                
                if not faker_provider:
                    errors.append(
                        f"Faker column '{table_name}.{col_name}' missing 'faker_provider'"
                    )
    
    return errors


def _validate_table_structure(plan: dict) -> list[str]:
    """Validate each table has required fields."""
    errors = []
    
    for i, table in enumerate(plan.get("tables", [])):
        if not isinstance(table, dict):
            errors.append(f"Table at index {i} must be a dictionary")
            continue
        
        if "name" not in table:
            errors.append(f"Table at index {i} missing 'name' field")
        
        if "columns" not in table:
            errors.append(f"Table '{table.get('name', f'index {i}')}' missing 'columns'")
        elif not isinstance(table.get("columns"), list):
            errors.append(f"Table '{table.get('name', f'index {i}')}' 'columns' must be list")
        elif len(table.get("columns", [])) == 0:
            errors.append(f"Table '{table.get('name', f'index {i}')}' has no columns")
        
        if "row_count" in table:
            row_count = table["row_count"]
            if not isinstance(row_count, int) or row_count < 1:
                errors.append(
                    f"Table '{table.get('name', f'index {i}')}' row_count must be positive int"
                )
    
    return errors


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def validate_generation_plan(plan: dict) -> ValidationResult:
    """
    Validate a Generation Plan JSON.
    
    This is the primary public interface for plan validation.
    Performs all validation rules (R1-R8) and returns structured result.
    
    Args:
        plan: Generation Plan dictionary to validate.
        
    Returns:
        ValidationResult with is_valid and list of errors.
    """
    all_errors: list[str] = []
    
    # Phase 1: Validate basic structure
    structure_errors = _validate_structure(plan)
    all_errors.extend(structure_errors)
    
    # If basic structure is invalid, can't proceed with deeper validation
    if structure_errors:
        return ValidationResult(is_valid=False, errors=all_errors)
    
    # Phase 2: Validate table structure
    all_errors.extend(_validate_table_structure(plan))
    
    # Phase 3: Validate execution order
    all_errors.extend(_validate_execution_order_matches_tables(plan))
    all_errors.extend(_validate_execution_order_dependencies(plan))
    
    # Phase 4: Validate column specifications
    all_errors.extend(_validate_generator_types(plan))
    all_errors.extend(_validate_pk_columns(plan))
    all_errors.extend(_validate_fk_columns(plan))
    all_errors.extend(_validate_faker_columns(plan))
    
    # Phase 5: Validate FK references (cross-table)
    all_errors.extend(_validate_fk_references(plan))
    
    return ValidationResult(
        is_valid=len(all_errors) == 0,
        errors=all_errors
    )


def validate_and_raise(plan: dict) -> None:
    """
    Validate a Generation Plan and raise if invalid.
    
    Convenience wrapper that raises PlanValidationError on failure.
    
    Args:
        plan: Generation Plan dictionary to validate.
        
    Raises:
        PlanValidationError: If plan is invalid.
    """
    result = validate_generation_plan(plan)
    if not result.is_valid:
        raise PlanValidationError(result.error_summary)

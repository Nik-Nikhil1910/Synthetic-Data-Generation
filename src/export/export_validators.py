"""
Export Validators
=================

Validates exported files for correctness.
Performs sanity checks on JSON, CSV, and SQL outputs.
"""

import json
import csv
import re
from pathlib import Path
from typing import Any


# =============================================================================
# EXCEPTIONS
# =============================================================================

class ExportValidationError(Exception):
    """Raised when export validation fails."""
    pass


# =============================================================================
# JSON VALIDATION
# =============================================================================

def validate_json_export(file_paths: list[str]) -> bool:
    """
    Validate JSON files are parseable and well-formed.
    
    Args:
        file_paths: List of JSON file paths to validate.
    
    Returns:
        True if all files are valid.
    
    Raises:
        ExportValidationError: If validation fails.
    """
    for path in file_paths:
        if not Path(path).exists():
            raise ExportValidationError(f"JSON file not found: {path}")
        
        try:
            with open(path, "r", encoding="utf-8") as f:
                json.load(f)
        except json.JSONDecodeError as e:
            raise ExportValidationError(f"Invalid JSON in {path}: {e}") from e
    
    return True


# =============================================================================
# CSV VALIDATION
# =============================================================================

def validate_csv_export(
    file_paths: list[str],
    expected_counts: dict[str, int] | None = None
) -> bool:
    """
    Validate CSV files have headers and expected row counts.
    
    Args:
        file_paths: List of CSV file paths to validate.
        expected_counts: Optional dict of table_name -> expected row count.
    
    Returns:
        True if all files are valid.
    
    Raises:
        ExportValidationError: If validation fails.
    """
    for path in file_paths:
        file_path = Path(path)
        if not file_path.exists():
            raise ExportValidationError(f"CSV file not found: {path}")
        
        try:
            with open(path, "r", encoding="utf-8", newline="") as f:
                reader = csv.reader(f)
                rows = list(reader)
        except Exception as e:
            raise ExportValidationError(f"Failed to read CSV {path}: {e}") from e
        
        if len(rows) == 0:
            raise ExportValidationError(f"CSV file is empty: {path}")
        
        # First row should be header
        header = rows[0]
        if not header:
            raise ExportValidationError(f"CSV has empty header: {path}")
        
        # Check row count if expected
        if expected_counts:
            table_name = file_path.stem
            expected = expected_counts.get(table_name)
            actual = len(rows) - 1  # Exclude header
            
            if expected is not None and actual != expected:
                raise ExportValidationError(
                    f"CSV {table_name} has {actual} rows, expected {expected}"
                )
    
    return True


# =============================================================================
# SQL VALIDATION
# =============================================================================

def validate_sql_export(file_path: str) -> bool:
    """
    Validate SQL file has basic structure and syntax.
    
    Performs lightweight checks:
    - File exists and is non-empty
    - Contains CREATE TABLE statements
    - Contains INSERT statements
    - No obvious syntax errors
    
    Args:
        file_path: Path to SQL file.
    
    Returns:
        True if file passes basic validation.
    
    Raises:
        ExportValidationError: If validation fails.
    """
    path = Path(file_path)
    
    if not path.exists():
        raise ExportValidationError(f"SQL file not found: {file_path}")
    
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            content = f.read()
    except Exception as e:
        raise ExportValidationError(f"Failed to read SQL {file_path}: {e}") from e
    
    if not content.strip():
        raise ExportValidationError(f"SQL file is empty: {file_path}")
    
    # Check for CREATE TABLE
    if "CREATE TABLE" not in content.upper():
        raise ExportValidationError(f"SQL file has no CREATE TABLE: {file_path}")
    
    # Check for INSERT
    if "INSERT INTO" not in content.upper():
        raise ExportValidationError(f"SQL file has no INSERT statements: {file_path}")
    
    # Basic syntax check: balanced parentheses in CREATE TABLE
    create_table_pattern = r"CREATE TABLE[^;]+;"
    matches = re.findall(create_table_pattern, content, re.IGNORECASE | re.DOTALL)
    
    for match in matches:
        open_parens = match.count("(")
        close_parens = match.count(")")
        if open_parens != close_parens:
            raise ExportValidationError(
                f"Unbalanced parentheses in CREATE TABLE: {file_path}"
            )
    
    return True

"""
CSV Exporter
============

Exports data_store tables to CSV files.
One file per table with header row.
"""

import csv
import os
from pathlib import Path
from typing import Any


# =============================================================================
# EXCEPTIONS
# =============================================================================

class CsvExportError(Exception):
    """Raised when CSV export fails."""
    pass


# =============================================================================
# CONSTANTS
# =============================================================================

# Representation of NULL values in CSV
NULL_REPRESENTATION = ""


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def export_to_csv(
    data_store: dict[str, list[dict]],
    output_dir: str,
    schema: dict | None = None
) -> list[str]:
    """
    Export each table in data_store to a CSV file.
    
    Args:
        data_store: dict of table_name -> list of records.
        output_dir: Directory to write files to.
        schema: Optional Stage 0 schema for column ordering.
    
    Returns:
        List of generated file paths.
    
    Raises:
        CsvExportError: If export fails.
    """
    if not data_store:
        raise CsvExportError("data_store is empty.")
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Build column order lookup from schema if provided
    column_order = _build_column_order(schema) if schema else {}
    
    generated_files = []
    
    for table_name, records in data_store.items():
        if not records:
            continue
        
        file_path = output_path / f"{table_name}.csv"
        _write_csv(table_name, records, file_path, column_order.get(table_name))
        generated_files.append(str(file_path))
    
    return generated_files


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _build_column_order(schema: dict) -> dict[str, list[str]]:
    """Build column order lookup from schema."""
    order = {}
    for table in schema.get("tables", []):
        table_name = table.get("name", "")
        columns = [col.get("name", "") for col in table.get("columns", [])]
        order[table_name] = columns
    return order


def _write_csv(
    table_name: str,
    records: list[dict],
    file_path: Path,
    column_order: list[str] | None
) -> None:
    """Write records to CSV file."""
    if not records:
        return
    
    # Determine column order
    if column_order:
        fieldnames = column_order
    else:
        # Use keys from first record, sorted for determinism
        fieldnames = sorted(records[0].keys())
    
    try:
        with open(file_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(
                f,
                fieldnames=fieldnames,
                extrasaction="ignore"  # Ignore extra fields not in fieldnames
            )
            writer.writeheader()
            
            for record in records:
                # Convert None to empty string
                row = {
                    k: (NULL_REPRESENTATION if v is None else v)
                    for k, v in record.items()
                }
                writer.writerow(row)
    
    except Exception as e:
        raise CsvExportError(f"Failed to write {file_path}: {e}") from e

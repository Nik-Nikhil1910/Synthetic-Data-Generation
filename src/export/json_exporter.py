"""
JSON Exporter
=============

Serializes data_store to JSON format.
Deterministic output with consistent key ordering.
"""

import json
import os
from typing import Any
from pathlib import Path


# =============================================================================
# EXCEPTIONS
# =============================================================================

class JsonExportError(Exception):
    """Raised when JSON export fails."""
    pass


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def export_to_json(
    data_store: dict[str, list[dict]],
    output_dir: str,
    single_file: bool = False
) -> list[str]:
    """
    Export data_store to JSON file(s).
    
    Args:
        data_store: dict of table_name -> list of records.
        output_dir: Directory to write files to.
        single_file: If True, export all tables to one file.
                     If False, one file per table.
    
    Returns:
        List of generated file paths.
    
    Raises:
        JsonExportError: If export fails.
    """
    if not data_store:
        raise JsonExportError("data_store is empty.")
    
    # Ensure output directory exists
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    generated_files = []
    
    if single_file:
        # Export all tables to one file
        file_path = output_path / "data_export.json"
        _write_json(data_store, file_path)
        generated_files.append(str(file_path))
    else:
        # Export each table to its own file
        for table_name, records in data_store.items():
            file_path = output_path / f"{table_name}.json"
            _write_json(records, file_path)
            generated_files.append(str(file_path))
    
    return generated_files


# =============================================================================
# INTERNAL HELPERS
# =============================================================================

def _write_json(data: Any, file_path: Path) -> None:
    """Write data to JSON file with consistent formatting."""
    try:
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(
                data,
                f,
                indent=2,
                ensure_ascii=False,
                sort_keys=True,  # Deterministic key order
                default=_json_serializer
            )
    except Exception as e:
        raise JsonExportError(f"Failed to write {file_path}: {e}") from e


def _json_serializer(obj: Any) -> Any:
    """Custom JSON serializer for types not natively supported."""
    # Handle datetime objects
    if hasattr(obj, "isoformat"):
        return obj.isoformat()
    # Handle other non-serializable types
    raise TypeError(f"Object of type {type(obj).__name__} is not JSON serializable")

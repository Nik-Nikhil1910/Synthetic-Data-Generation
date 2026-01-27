"""
Export Module
=============

Stage C of the Synthetic Data Generator pipeline.
Exports generated data to JSON, CSV, and SQL formats.

This is a deterministic export layer with NO AI, NO inference.
"""

from .json_exporter import (
    export_to_json,
    JsonExportError,
)

from .csv_exporter import (
    export_to_csv,
    CsvExportError,
)

from .sql_exporter import (
    export_to_sql,
    export_all_dialects,
    SqlExportError,
    VALID_DIALECTS,
    TYPE_MAPPING,
)

from .export_validators import (
    validate_json_export,
    validate_csv_export,
    validate_sql_export,
    ExportValidationError,
)

__all__ = [
    # JSON
    "export_to_json",
    "JsonExportError",
    
    # CSV
    "export_to_csv",
    "CsvExportError",
    
    # SQL
    "export_to_sql",
    "export_all_dialects",
    "SqlExportError",
    "VALID_DIALECTS",
    "TYPE_MAPPING",
    
    # Validation
    "validate_json_export",
    "validate_csv_export",
    "validate_sql_export",
    "ExportValidationError",
]

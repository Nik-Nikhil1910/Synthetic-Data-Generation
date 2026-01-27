"""
Generation Plan Generator
=========================

Purpose:
--------
Accepts a validated Schema JSON from Stage 0 (Schema Inference) and produces a
Generation Plan JSON conforming to the project contract.

Input Contract (Stage 0 Output):
--------------------------------
The input schema MUST match the Schema Inference Contract:
{
  "tables": [
    {
      "name": "string (snake_case)",
      "columns": [
        {"name": str, "type": str, "is_pk": bool, "is_nullable": bool}
      ],
      "foreign_keys": [
        {"column": str, "references_table": str, "references_column": str, "is_nullable": bool}
      ]
    }
  ]
}

Public Interface:
-----------------
    def generate_plan(schema: dict, row_count: int = 50) -> dict

Contract Compliance:
--------------------
- Input schema is treated as FINAL and AUTHORITATIVE
- No heuristic inference or schema recovery
- Any schema issues result in validation failure (not correction)
- Topological sorting of tables (parents before children)
- Explicit PK, FK, and Faker column mappings
- Deterministic and reproducible output
- No GenAI usage
"""

from typing import Any
from collections import defaultdict

from .validators import (
    GenerationPlan,
    PlanMeta,
    TablePlan,
    PrimaryKeyColumn,
    ForeignKeyColumn,
    ForeignKeyReference,
    FakerColumn,
)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class GenerationPlanError(Exception):
    """Base exception for generation plan failures."""
    pass


class CircularDependencyError(GenerationPlanError):
    """Raised when a circular dependency is detected in table relationships."""
    pass


class InvalidSchemaError(GenerationPlanError):
    """Raised when the input schema is invalid or incomplete."""
    pass


# =============================================================================
# CONSTANTS
# =============================================================================

DEFAULT_ROW_COUNT = 50
NULL_PROBABILITY = 0.2

# Faker provider mapping: column name patterns -> faker provider
# Order matters: more specific patterns should come first.
FAKER_COLUMN_MAPPING: dict[str, str] = {
    # Email patterns
    "email": "email",
    "email_address": "email",
    # Name patterns
    "first_name": "first_name",
    "firstname": "first_name",
    "last_name": "last_name",
    "lastname": "last_name",
    "name": "name",
    "full_name": "name",
    "fullname": "name",
    "username": "user_name",
    "user_name": "user_name",
    # Contact patterns
    "phone": "phone_number",
    "phone_number": "phone_number",
    "telephone": "phone_number",
    "address": "address",
    "street": "street_address",
    "city": "city",
    "state": "state",
    "country": "country",
    "zipcode": "zipcode",
    "zip_code": "zipcode",
    "postal_code": "postcode",
    # Text patterns
    "description": "paragraph",
    "bio": "paragraph",
    "content": "paragraph",
    "body": "paragraph",
    "title": "sentence",
    "subject": "sentence",
    "headline": "sentence",
    # Date/Time patterns
    "created_at": "date_time_this_year",
    "updated_at": "date_time_this_year",
    "date": "date_this_year",
    "birth_date": "date_of_birth",
    "birthdate": "date_of_birth",
    # Numeric patterns
    "price": "pyfloat",
    "amount": "pyfloat",
    "total": "pyfloat",
    "quantity": "pyint",
    "count": "pyint",
    "age": "pyint",
    # Boolean patterns
    "is_active": "pybool",
    "active": "pybool",
    "enabled": "pybool",
    "verified": "pybool",
    "is_verified": "pybool",
    # URL/Web patterns
    "url": "url",
    "website": "url",
    "image": "image_url",
    "image_url": "image_url",
    "avatar": "image_url",
    # Company patterns
    "company": "company",
    "company_name": "company",
    "job": "job",
    "job_title": "job",
}

# SQL Type -> Faker fallback mapping
SQL_TYPE_FALLBACK: dict[str, str] = {
    "INTEGER": "pyint",
    "FLOAT": "pyfloat",
    "VARCHAR": "lexify",
    "TEXT": "paragraph",
    "BOOLEAN": "pybool",
    "DATE": "date_this_year",
    "DATETIME": "date_time_this_year",
}


# =============================================================================
# INPUT SCHEMA VALIDATION
# =============================================================================

# Allowed SQL types per Stage 0 contract
ALLOWED_SQL_TYPES = frozenset({
    "INTEGER", "VARCHAR", "TEXT", "BOOLEAN", "DATE", "DATETIME", "FLOAT"
})


def _validate_input_schema(schema: dict) -> None:
    """
    Validate input schema matches Stage 0 contract.
    
    This is strict validation with NO RECOVERY. Any schema issues
    result in InvalidSchemaError being raised.
    
    Args:
        schema: Schema dict from Stage 0.
        
    Raises:
        InvalidSchemaError: If schema is invalid or incomplete.
    """
    if not isinstance(schema, dict):
        raise InvalidSchemaError("Schema must be a dictionary.")
    
    if "tables" not in schema:
        raise InvalidSchemaError("Schema must contain a 'tables' key.")
    
    if not isinstance(schema["tables"], list):
        raise InvalidSchemaError("'tables' must be a list.")
    
    if len(schema["tables"]) == 0:
        raise InvalidSchemaError("Schema must contain at least one table.")
    
    all_table_names: set[str] = set()
    
    for i, table in enumerate(schema["tables"]):
        _validate_table(table, i, all_table_names)
    
    # Second pass: validate FK references
    for table in schema["tables"]:
        _validate_fk_references(table, all_table_names, schema)


def _validate_table(table: dict, index: int, all_table_names: set[str]) -> None:
    """Validate a single table structure."""
    if not isinstance(table, dict):
        raise InvalidSchemaError(f"Table at index {index} must be a dictionary.")
    
    # Validate table name
    if "name" not in table:
        raise InvalidSchemaError(f"Table at index {index} missing 'name' field.")
    
    table_name = table["name"]
    if not isinstance(table_name, str) or not table_name:
        raise InvalidSchemaError(f"Table at index {index} 'name' must be non-empty string.")
    
    if table_name in all_table_names:
        raise InvalidSchemaError(f"Duplicate table name: '{table_name}'.")
    all_table_names.add(table_name)
    
    # Validate columns
    if "columns" not in table:
        raise InvalidSchemaError(f"Table '{table_name}' missing 'columns' field.")
    
    if not isinstance(table["columns"], list):
        raise InvalidSchemaError(f"Table '{table_name}' 'columns' must be a list.")
    
    if len(table["columns"]) == 0:
        raise InvalidSchemaError(f"Table '{table_name}' must have at least one column.")
    
    column_names: set[str] = set()
    pk_count = 0
    
    for j, col in enumerate(table["columns"]):
        pk_count += _validate_column(col, j, table_name, column_names)
    
    if pk_count != 1:
        raise InvalidSchemaError(
            f"Table '{table_name}' must have exactly one primary key, found {pk_count}."
        )
    
    # Validate foreign_keys (optional field, but must be list if present)
    if "foreign_keys" in table:
        if not isinstance(table["foreign_keys"], list):
            raise InvalidSchemaError(f"Table '{table_name}' 'foreign_keys' must be a list.")
        
        for fk in table["foreign_keys"]:
            _validate_fk_structure(fk, table_name, column_names)


def _validate_column(col: dict, index: int, table_name: str, column_names: set[str]) -> int:
    """Validate a single column structure. Returns 1 if PK, 0 otherwise."""
    if not isinstance(col, dict):
        raise InvalidSchemaError(
            f"Column at index {index} in table '{table_name}' must be a dictionary."
        )
    
    # Required fields
    for field in ("name", "type", "is_pk", "is_nullable"):
        if field not in col:
            raise InvalidSchemaError(
                f"Column at index {index} in table '{table_name}' missing '{field}'."
            )
    
    col_name = col["name"]
    if not isinstance(col_name, str) or not col_name:
        raise InvalidSchemaError(
            f"Column at index {index} in table '{table_name}' 'name' must be non-empty string."
        )
    
    if col_name in column_names:
        raise InvalidSchemaError(f"Duplicate column '{col_name}' in table '{table_name}'.")
    column_names.add(col_name)
    
    # Validate type
    if col["type"] not in ALLOWED_SQL_TYPES:
        raise InvalidSchemaError(
            f"Column '{table_name}.{col_name}' has invalid type '{col['type']}'. "
            f"Allowed: {sorted(ALLOWED_SQL_TYPES)}."
        )
    
    # Validate booleans
    if not isinstance(col["is_pk"], bool):
        raise InvalidSchemaError(
            f"Column '{table_name}.{col_name}' 'is_pk' must be boolean."
        )
    if not isinstance(col["is_nullable"], bool):
        raise InvalidSchemaError(
            f"Column '{table_name}.{col_name}' 'is_nullable' must be boolean."
        )
    
    # PK constraints
    if col["is_pk"]:
        if col["is_nullable"]:
            raise InvalidSchemaError(
                f"Primary key '{table_name}.{col_name}' cannot be nullable."
            )
        if col["type"] not in ("INTEGER", "VARCHAR"):
            raise InvalidSchemaError(
                f"Primary key '{table_name}.{col_name}' must be INTEGER or VARCHAR."
            )
        return 1
    
    return 0


def _validate_fk_structure(fk: dict, table_name: str, column_names: set[str]) -> None:
    """Validate FK structure within a table."""
    if not isinstance(fk, dict):
        raise InvalidSchemaError(f"FK in table '{table_name}' must be a dictionary.")
    
    for field in ("column", "references_table", "references_column"):
        if field not in fk:
            raise InvalidSchemaError(f"FK in table '{table_name}' missing '{field}'.")
    
    if fk["column"] not in column_names:
        raise InvalidSchemaError(
            f"FK column '{fk['column']}' does not exist in table '{table_name}'."
        )


def _validate_fk_references(table: dict, all_table_names: set[str], schema: dict) -> None:
    """Validate FK references point to existing tables/columns."""
    table_name = table["name"]
    
    # Build column lookup for all tables
    table_columns: dict[str, set[str]] = {}
    for t in schema["tables"]:
        table_columns[t["name"]] = {c["name"] for c in t["columns"]}
    
    for fk in table.get("foreign_keys", []):
        ref_table = fk["references_table"]
        ref_column = fk["references_column"]
        
        if ref_table not in all_table_names:
            raise InvalidSchemaError(
                f"FK in table '{table_name}' references non-existent table '{ref_table}'."
            )
        
        if ref_column not in table_columns.get(ref_table, set()):
            raise InvalidSchemaError(
                f"FK in table '{table_name}' references non-existent column "
                f"'{ref_table}.{ref_column}'."
            )


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================


def _get_faker_provider(column_name: str, column_type: str) -> str:
    """
    Determine the Faker provider for a column based on its name and type.
    
    Strategy:
    1. Check if column name matches a known pattern.
    2. If not, fallback to type-based mapping.
    """
    # Normalize column name for matching
    normalized_name = column_name.lower().strip()
    
    # Check for exact or partial match in column mapping
    if normalized_name in FAKER_COLUMN_MAPPING:
        return FAKER_COLUMN_MAPPING[normalized_name]
    
    # Check for substring matches (e.g., "user_email" contains "email")
    # Sort by pattern length descending for determinism (longer patterns matched first)
    sorted_patterns = sorted(FAKER_COLUMN_MAPPING.keys(), key=len, reverse=True)
    for pattern in sorted_patterns:
        if pattern in normalized_name:
            return FAKER_COLUMN_MAPPING[pattern]
    
    # Fallback to SQL type mapping
    return SQL_TYPE_FALLBACK.get(column_type, "lexify")


def _build_dependency_graph(schema: dict) -> tuple[dict[str, set[str]], set[str]]:
    """
    Build a dependency graph from the schema.
    
    Returns:
        - Graph: dict mapping child table -> set of parent tables
        - Junction tables: set of table names that are junction tables
    """
    graph: dict[str, set[str]] = defaultdict(set)
    all_tables: set[str] = set()
    junction_tables: set[str] = set()
    
    for table in schema.get("tables", []):
        table_name = table["name"]
        all_tables.add(table_name)
        
        foreign_keys = table.get("foreign_keys", [])
        fk_parents = set()
        all_fks_non_nullable = True
        
        for fk in foreign_keys:
            parent_table = fk["references_table"]
            if parent_table != table_name:  # Avoid self-references in graph
                graph[table_name].add(parent_table)
                fk_parents.add(parent_table)
            # Check if this FK is nullable
            if fk.get("is_nullable", False):
                all_fks_non_nullable = False
        
        # Junction table: exactly 2 FKs AND both are non-nullable (per Project Plan)
        if len(foreign_keys) == 2 and len(fk_parents) == 2 and all_fks_non_nullable:
            junction_tables.add(table_name)
    
    # Ensure all tables are in the graph (even those with no dependencies)
    for table_name in all_tables:
        if table_name not in graph:
            graph[table_name] = set()
    
    return dict(graph), junction_tables


def _topological_sort(graph: dict[str, set[str]]) -> list[str]:
    """
    Perform topological sort using Kahn's algorithm.
    
    Raises:
        CircularDependencyError: If a cycle is detected.
    """
    # Calculate in-degree for each node
    in_degree: dict[str, int] = {node: 0 for node in graph}
    for node, parents in graph.items():
        for parent in parents:
            if parent in in_degree:
                # node depends on parent, so parent must come first
                pass
    
    # Build reverse graph (parent -> children)
    reverse_graph: dict[str, list[str]] = defaultdict(list)
    for child, parents in graph.items():
        for parent in parents:
            if parent in graph:
                reverse_graph[parent].append(child)
    
    # Calculate in-degrees (number of dependencies)
    for child, parents in graph.items():
        in_degree[child] = len([p for p in parents if p in graph])
    
    # Start with nodes that have no dependencies
    queue = sorted([node for node, degree in in_degree.items() if degree == 0])
    result: list[str] = []
    
    while queue:
        # Process in sorted order for determinism
        node = queue.pop(0)
        result.append(node)
        
        # Reduce in-degree for all children
        for child in sorted(reverse_graph[node]):
            in_degree[child] -= 1
            if in_degree[child] == 0:
                queue.append(child)
                queue.sort()  # Maintain deterministic order
    
    # If we haven't processed all nodes, there's a cycle
    if len(result) != len(graph):
        remaining = set(graph.keys()) - set(result)
        raise CircularDependencyError(
            f"Circular dependency detected involving tables: {sorted(remaining)}"
        )
    
    return result


def _determine_row_count(
    table_name: str,
    junction_tables: set[str],
    table_row_counts: dict[str, int],
    graph: dict[str, set[str]]
) -> int:
    """
    Determine row count for a table.
    
    - Standard tables: DEFAULT_ROW_COUNT
    - Junction tables: min(parent_a_rows, parent_b_rows) * 2
    """
    if table_name not in junction_tables:
        return DEFAULT_ROW_COUNT
    
    # Junction table: get parent row counts
    parents = list(graph.get(table_name, set()))
    if len(parents) != 2:
        return DEFAULT_ROW_COUNT
    
    parent_a_rows = table_row_counts.get(parents[0], DEFAULT_ROW_COUNT)
    parent_b_rows = table_row_counts.get(parents[1], DEFAULT_ROW_COUNT)
    
    return min(parent_a_rows, parent_b_rows) * 2


def _build_column_specs(
    table: dict,
    fk_columns: set[str],
    fk_mapping: dict[str, dict]
) -> list[dict]:
    """
    Build column specifications for the generation plan.
    """
    columns: list[dict] = []
    
    for col in table.get("columns", []):
        col_name = col["name"]
        col_type = col["type"]
        is_pk = col.get("is_pk", False)
        is_nullable = col.get("is_nullable", False)
        
        if is_pk:
            # Primary Key column
            columns.append({
                "name": col_name,
                "generator_type": "primary_key",
                "strategy": "sequential_integer"
            })
        elif col_name in fk_columns:
            # Foreign Key column
            fk_info = fk_mapping[col_name]
            columns.append({
                "name": col_name,
                "generator_type": "foreign_key",
                "references": {
                    "table": fk_info["references_table"],
                    "column": fk_info["references_column"]
                },
                "is_nullable": is_nullable
            })
        else:
            # Regular column using Faker
            faker_provider = _get_faker_provider(col_name, col_type)
            col_spec = {
                "name": col_name,
                "generator_type": "faker",
                "faker_provider": faker_provider,
                "is_nullable": is_nullable
            }
            if is_nullable:
                col_spec["null_probability"] = NULL_PROBABILITY
            columns.append(col_spec)
    
    return columns


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def generate_plan(schema: dict, row_count: int = DEFAULT_ROW_COUNT) -> dict:
    """
    Generate a Generation Plan from a validated Stage 0 Schema JSON.
    
    The input schema is treated as FINAL and AUTHORITATIVE. No heuristic
    inference or schema recovery is performed. Any schema issues result
    in InvalidSchemaError being raised.
    
    Args:
        schema: Validated Schema JSON from Stage 0 (Schema Inference).
        row_count: Default row count per table (default: 50).
    
    Returns:
        Generation Plan JSON conforming to the contract.
    
    Raises:
        CircularDependencyError: If tables have circular dependencies.
        InvalidSchemaError: If the schema is invalid or doesn't match contract.
    """
    # Step 0: Validate input schema (strict, no recovery)
    _validate_input_schema(schema)

    
    # Step 1: Build dependency graph and identify junction tables
    graph, junction_tables = _build_dependency_graph(schema)
    
    # Step 2: Topological sort
    execution_order = _topological_sort(graph)
    
    # Step 3: Build table plans
    # First pass: determine row counts (need to process in order for junction tables)
    table_row_counts: dict[str, int] = {}
    for table_name in execution_order:
        if table_name in junction_tables:
            # Junction tables: min(parent_a, parent_b) * 2
            table_row_counts[table_name] = _determine_row_count(
                table_name, junction_tables, table_row_counts, graph
            )
        else:
            # Standard tables: use the provided row_count
            table_row_counts[table_name] = row_count
    
    # Create lookup for tables by name
    table_lookup = {t["name"]: t for t in schema["tables"]}
    # Note: FK validation now handled by _validate_input_schema

    
    # Build table plans
    table_plans: list[dict] = []
    for table_name in execution_order:
        table = table_lookup[table_name]
        
        # Build FK mapping for this table
        fk_columns = {fk["column"] for fk in table.get("foreign_keys", [])}
        fk_mapping = {fk["column"]: fk for fk in table.get("foreign_keys", [])}
        
        # Build column specs
        column_specs = _build_column_specs(table, fk_columns, fk_mapping)
        
        table_plans.append({
            "name": table_name,
            "row_count": table_row_counts[table_name],
            "columns": column_specs
        })
    
    # Step 4: Assemble the plan
    plan_dict = {
        "meta": {
            "faker_seed": 0,
            "version": "0.1.0"
        },
        "execution_order": execution_order,
        "tables": table_plans
    }
    
    # Step 5: Validate using Pydantic
    validated_plan = GenerationPlan.model_validate(plan_dict)
    
    return validated_plan.model_dump()

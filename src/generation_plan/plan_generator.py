"""
Generation Plan Generator
=========================

Purpose:
--------
Accepts a Schema JSON (output of infer_schema.py) and produces a
Generation Plan JSON conforming to the contract in Project Plan §3.3.

Public Interface:
-----------------
    def generate_plan(schema: dict) -> dict

Contract Compliance:
--------------------
- Topological sorting of tables (parents before children).
- Explicit PK, FK, and Faker column mappings.
- Deterministic and reproducible output.
- No GenAI usage.
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
    Generate a Generation Plan from a Schema JSON.
    
    Args:
        schema: Schema JSON from infer_schema.py
        row_count: Default row count per table (default: 50)
    
    Returns:
        Generation Plan JSON conforming to the contract.
    
    Raises:
        CircularDependencyError: If tables have circular dependencies.
        InvalidSchemaError: If the schema is invalid.
    """
    if not schema or "tables" not in schema:
        raise InvalidSchemaError("Schema must contain a 'tables' key.")
    
    if not schema["tables"]:
        raise InvalidSchemaError("Schema must contain at least one table.")
    
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
    all_table_names = set(table_lookup.keys())
    
    # Validate FK references point to existing tables
    for table in schema["tables"]:
        for fk in table.get("foreign_keys", []):
            ref_table = fk["references_table"]
            if ref_table not in all_table_names:
                raise InvalidSchemaError(
                    f"Table '{table['name']}' has FK referencing non-existent table '{ref_table}'"
                )
    
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

"""
Schema Inference Module
=======================

Purpose:
--------
Accepts a raw business use case description (natural language) and infers
a normalized relational database schema conforming to the Schema Inference
Contract (Output of Stage 2).

Public Interface:
-----------------
    def infer_schema(business_description: str) -> dict

Contract Compliance:
--------------------
The returned dict MUST serialize to JSON matching this structure:
{
  "tables": [
    {
      "name": "string (snake_case)",
      "columns": [
        {
          "name": "string (snake_case)",
          "type": "string (SQL_TYPE_ENUM)",
          "is_pk": "boolean",
          "is_nullable": "boolean"
        }
      ],
      "foreign_keys": [
        {
          "column": "string (must exist in columns)",
          "references_table": "string (must exist in tables)",
          "references_column": "string (must exist in target table)",
          "is_nullable": "boolean"
        }
      ]
    }
  ]
}

Allowed Types: INTEGER, VARCHAR, TEXT, BOOLEAN, DATE, DATETIME, FLOAT

Design Assumptions (per Schema Inference Strategy):
---------------------------------------------------
- Primary Key: Surrogate INTEGER 'id' column (Strategy §3: "Default to Surrogate").
- FK Naming: '{parent_table}_id' (Strategy §3: "Nomenclature").
- Relationships: Only inferred when explicitly stated in input text.
"""

from typing import Any
import re


# =============================================================================
# EXCEPTIONS
# =============================================================================

class SchemaInferenceError(Exception):
    """Base exception for schema inference failures."""
    pass


class NoEntitiesFoundError(SchemaInferenceError):
    """Raised when no extractable entities are found in the input."""
    pass


class ContractViolationError(SchemaInferenceError):
    """Raised when the inferred schema violates the contract."""
    pass


class AmbiguousRelationshipError(SchemaInferenceError):
    """Raised when a relationship cannot be safely inferred."""
    pass


# =============================================================================
# CONSTANTS
# =============================================================================

ALLOWED_TYPES = frozenset({
    "INTEGER", "VARCHAR", "TEXT", "BOOLEAN", "DATE", "DATETIME", "FLOAT"
})

# Common English stop words to filter out during entity extraction.
# These are domain-agnostic function words, not business entities.
STOP_WORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "is", "are", "was", "were", "be",
    "been", "being", "have", "has", "had", "do", "does", "did", "will",
    "would", "could", "should", "may", "might", "must", "shall", "can",
    "need", "to", "of", "in", "for", "on", "with", "at", "by", "from",
    "as", "into", "through", "during", "before", "after", "above", "below",
    "between", "under", "again", "further", "then", "once", "here", "there",
    "when", "where", "why", "how", "all", "each", "every", "both", "few",
    "more", "most", "other", "some", "such", "no", "nor", "not", "only",
    "own", "same", "so", "than", "too", "very", "just", "also", "now",
    "about", "if", "because", "until", "while", "this", "that", "these",
    "those", "what", "which", "who", "whom", "its", "their", "them", "they",
    "it", "he", "she", "we", "you", "i", "me", "my", "your", "his", "her",
    "our", "us", "many", "one", "two", "three", "first", "second", "new",
    "old", "good", "bad", "high", "low", "large", "small", "long", "short",
    "system", "management", "platform", "application", "software", "service",
    "data", "information", "process", "workflow", "using", "based",
})

# Relationship indicator patterns (domain-agnostic linguistic cues).
# Format: (regex_pattern, parent_group_index, child_group_index, is_nullable)
# These detect explicit statements like "X has many Y" or "Y belongs to X".
RELATIONSHIP_PATTERNS = [
    # "X has/have many/multiple Y" -> X is parent, Y is child (1:N)
    (r"\b(\w+)\s+(?:has|have)\s+(?:many|multiple|several)\s+(\w+)", 0, 1, False),
    # "X contains Y" -> X is parent, Y is child
    (r"\b(\w+)\s+contains?\s+(\w+)", 0, 1, False),
    # "Y belongs to X" -> X is parent, Y is child
    (r"\b(\w+)\s+belongs?\s+to\s+(\w+)", 1, 0, False),
    # "Y is owned by X" -> X is parent, Y is child
    (r"\b(\w+)\s+(?:is|are)\s+owned\s+by\s+(\w+)", 1, 0, False),
    # "X owns Y" -> X is parent, Y is child
    (r"\b(\w+)\s+owns?\s+(\w+)", 0, 1, False),
    # "Y is part of X" -> X is parent, Y is child
    (r"\b(\w+)\s+(?:is|are)\s+part\s+of\s+(\w+)", 1, 0, False),
    # "X includes Y" -> X is parent, Y is child
    (r"\b(\w+)\s+includes?\s+(\w+)", 0, 1, False),
    # "Y references X" / "Y refers to X" -> X is parent, Y is child
    (r"\b(\w+)\s+(?:references?|refers?\s+to)\s+(\w+)", 1, 0, False),
    # "X creates Y" / "X makes Y" -> X is parent, Y is child
    (r"\b(\w+)\s+(?:creates?|makes?|places?)\s+(\w+)", 0, 1, False),
]

# M:N relationship indicators (require explicit "many-to-many" or bidirectional phrasing).
MN_RELATIONSHIP_PATTERNS = [
    # "X and Y have a many-to-many relationship"
    (r"\b(\w+)\s+and\s+(\w+)\s+have\s+a?\s*many[- ]to[- ]many", 0, 1),
    # "X can have multiple Y and Y can have multiple X"
    (r"\b(\w+)\s+can\s+have\s+multiple\s+(\w+)\s+and\s+\2\s+can\s+have\s+multiple\s+\1", 0, 1),
]


# =============================================================================
# HELPER FUNCTIONS
# =============================================================================

def _to_snake_case(text: str) -> str:
    """Convert a string to snake_case."""
    # Replace spaces and hyphens with underscores
    s = re.sub(r"[\s\-]+", "_", text.strip().lower())
    # Remove non-alphanumeric characters except underscores
    s = re.sub(r"[^a-z0-9_]", "", s)
    # Collapse multiple underscores
    s = re.sub(r"_+", "_", s)
    return s.strip("_")


def _singularize(word: str) -> str:
    """
    Naive singularization for common English plural forms.
    Domain-agnostic heuristic.
    """
    word = word.lower()
    if word.endswith("ies") and len(word) > 3:
        return word[:-3] + "y"
    if word.endswith("es") and len(word) > 2:
        # Handle cases like "classes" -> "class", "boxes" -> "box"
        if word.endswith("sses") or word.endswith("xes") or word.endswith("ches") or word.endswith("shes"):
            return word[:-2]
        return word[:-2] if word[-3] in "sxzh" else word[:-1]
    if word.endswith("s") and len(word) > 1 and not word.endswith("ss"):
        return word[:-1]
    return word


def _extract_entities(description: str) -> set[str]:
    """
    Extract candidate entities from the business description using
    domain-agnostic NLP heuristics.

    Strategy:
    1. Tokenize into words.
    2. Filter out stop words.
    3. Keep nouns (heuristic: words not matching common verb/adj patterns).
    4. Singularize and normalize to snake_case.
    5. Deduplicate.
    """
    # Extract all word tokens (alphanumeric sequences).
    words = re.findall(r"\b[a-zA-Z][a-zA-Z0-9]*\b", description.lower())

    candidates: set[str] = set()
    for word in words:
        # Skip stop words.
        if word in STOP_WORDS:
            continue

        # Skip very short words (likely not entities).
        if len(word) < 3:
            continue

        # Heuristic: skip words ending in common verb suffixes.
        if word.endswith(("ing", "ed", "tion", "sion", "ment", "ness", "able", "ible", "ful", "less", "ive", "ly")):
            continue

        # Singularize and add.
        singular = _singularize(word)
        if len(singular) >= 3 and singular not in STOP_WORDS:
            candidates.add(_to_snake_case(singular))

    return candidates


def _extract_relationships(
    description: str,
    entities: set[str]
) -> list[tuple[str, str, bool]]:
    """
    Extract explicit relationships from the description text.

    Returns:
        List of (parent_entity, child_entity, is_nullable) tuples.
        Only returns relationships where BOTH entities are in the extracted set.
    """
    relationships: list[tuple[str, str, bool]] = []
    seen: set[tuple[str, str]] = set()

    description_lower = description.lower()

    for pattern, parent_idx, child_idx, is_nullable in RELATIONSHIP_PATTERNS:
        for match in re.finditer(pattern, description_lower, re.IGNORECASE):
            groups = match.groups()
            if len(groups) >= 2:
                parent_raw = _singularize(groups[parent_idx])
                child_raw = _singularize(groups[child_idx])

                parent = _to_snake_case(parent_raw)
                child = _to_snake_case(child_raw)

                # Only add if both entities are recognized AND different.
                if parent in entities and child in entities and parent != child:
                    key = (parent, child)
                    if key not in seen:
                        seen.add(key)
                        relationships.append((parent, child, is_nullable))

    return relationships


def _extract_mn_relationships(
    description: str,
    entities: set[str]
) -> list[tuple[str, str]]:
    """
    Extract explicit many-to-many relationships.

    Returns:
        List of (entity_a, entity_b) tuples requiring a junction table.
        Only returns if BOTH entities are recognized AND explicitly stated as M:N.
    """
    mn_relationships: list[tuple[str, str]] = []
    seen: set[frozenset[str]] = set()

    description_lower = description.lower()

    for pattern, idx_a, idx_b in MN_RELATIONSHIP_PATTERNS:
        for match in re.finditer(pattern, description_lower, re.IGNORECASE):
            groups = match.groups()
            if len(groups) >= 2:
                entity_a = _to_snake_case(_singularize(groups[idx_a]))
                entity_b = _to_snake_case(_singularize(groups[idx_b]))

                if entity_a in entities and entity_b in entities and entity_a != entity_b:
                    key = frozenset({entity_a, entity_b})
                    if key not in seen:
                        seen.add(key)
                        mn_relationships.append((entity_a, entity_b))

    return mn_relationships


def _build_table(entity_name: str) -> dict[str, Any]:
    """
    Build a minimal table definition for a given entity.

    Per Schema Inference Strategy §3:
    - Default to surrogate INTEGER primary key named 'id'.
    - PK is non-nullable.
    """
    return {
        "name": entity_name,
        "columns": [
            {
                "name": "id",
                "type": "INTEGER",
                "is_pk": True,
                "is_nullable": False,
            }
        ],
        "foreign_keys": [],
    }


def _build_junction_table(entity_a: str, entity_b: str) -> dict[str, Any]:
    """
    Build a junction table for an M:N relationship.

    Per Schema Inference Strategy §3:
    - Junction table name: {entity_a}_{entity_b} (alphabetical order).
    - Contains its own surrogate PK.
    - Contains two FK columns referencing both parent tables.
    """
    # Use alphabetical order for consistent naming.
    if entity_a > entity_b:
        entity_a, entity_b = entity_b, entity_a

    junction_name = f"{entity_a}_{entity_b}"
    fk_a = f"{entity_a}_id"
    fk_b = f"{entity_b}_id"

    return {
        "name": junction_name,
        "columns": [
            {"name": "id", "type": "INTEGER", "is_pk": True, "is_nullable": False},
            {"name": fk_a, "type": "INTEGER", "is_pk": False, "is_nullable": False},
            {"name": fk_b, "type": "INTEGER", "is_pk": False, "is_nullable": False},
        ],
        "foreign_keys": [
            {
                "column": fk_a,
                "references_table": entity_a,
                "references_column": "id",
                "is_nullable": False,
            },
            {
                "column": fk_b,
                "references_table": entity_b,
                "references_column": "id",
                "is_nullable": False,
            },
        ],
    }


def _apply_relationships(
    tables: dict[str, dict[str, Any]],
    relationships: list[tuple[str, str, bool]]
) -> None:
    """
    Apply 1:N relationships to the table definitions.
    Adds FK columns and constraints to child tables.
    """
    for parent, child, is_nullable in relationships:
        if parent not in tables or child not in tables:
            # Safety: skip if entities are missing (should not happen).
            continue

        child_table = tables[child]
        fk_col_name = f"{parent}_id"

        # Check for duplicate FK column.
        existing_col_names = {col["name"] for col in child_table["columns"]}
        if fk_col_name in existing_col_names:
            # FK column already exists, skip to avoid duplicates.
            continue

        # Add FK column.
        child_table["columns"].append({
            "name": fk_col_name,
            "type": "INTEGER",  # Must match parent PK type.
            "is_pk": False,
            "is_nullable": is_nullable,
        })

        # Check for duplicate FK constraint.
        existing_fk_refs = {
            (fk["column"], fk["references_table"], fk["references_column"])
            for fk in child_table["foreign_keys"]
        }
        fk_key = (fk_col_name, parent, "id")
        if fk_key in existing_fk_refs:
            continue

        # Add FK constraint.
        child_table["foreign_keys"].append({
            "column": fk_col_name,
            "references_table": parent,
            "references_column": "id",
            "is_nullable": is_nullable,
        })


def _validate_schema(schema: dict[str, Any]) -> None:
    """
    Validate the inferred schema against the System Contract rules.
    Raises ContractViolationError on any violation.
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


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def infer_schema(business_description: str) -> dict[str, Any]:
    """
    Infer a normalized relational schema from a business use case description.

    This function uses domain-agnostic NLP heuristics to:
    1. Extract candidate entities from nouns in the text.
    2. Detect explicit relationships from linguistic patterns.
    3. Build contract-compliant table definitions.

    Args:
        business_description: A natural language description of the business domain.

    Returns:
        A dict conforming to the Schema Inference Contract.

    Raises:
        NoEntitiesFoundError: If no extractable entities are found.
        ContractViolationError: If the schema fails contract validation.
        AmbiguousRelationshipError: If a relationship cannot be safely inferred.
    """
    if not business_description or not business_description.strip():
        raise NoEntitiesFoundError("Empty or blank business description provided.")

    # Step 1: Extract entities from the description.
    entities = _extract_entities(business_description)

    if not entities:
        raise NoEntitiesFoundError(
            "Could not extract any recognizable entities from the input. "
            "Please provide a description with clear business object nouns."
        )

    # Step 2: Build table definitions for each entity.
    tables: dict[str, dict[str, Any]] = {}
    for entity in sorted(entities):  # Sorted for determinism.
        tables[entity] = _build_table(entity)

    # Step 3: Extract and apply explicit 1:N relationships.
    relationships = _extract_relationships(business_description, entities)
    _apply_relationships(tables, relationships)

    # Step 4: Extract and create junction tables for explicit M:N relationships.
    mn_relationships = _extract_mn_relationships(business_description, entities)
    for entity_a, entity_b in mn_relationships:
        junction = _build_junction_table(entity_a, entity_b)
        if junction["name"] not in tables:
            tables[junction["name"]] = junction

    # Step 5: Assemble the final schema.
    schema: dict[str, Any] = {
        "tables": list(tables.values())
    }

    # Step 6: Validate against the contract.
    _validate_schema(schema)

    return schema

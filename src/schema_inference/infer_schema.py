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
import os
import json
from pathlib import Path
from dataclasses import dataclass, field

# LangChain imports for GenAI-powered schema inference
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Import validation logic from dedicated validator module
from .infer_schema_validator import (
    ContractViolationError,
    TerminationError,
    ValidationFeedback,
    validate_schema_with_feedback,
    validate_and_handle_termination,
)


# =============================================================================
# EXCEPTIONS
# =============================================================================

class SchemaInferenceError(Exception):
    """Base exception for schema inference failures."""
    pass


class NoEntitiesFoundError(SchemaInferenceError):
    """Raised when no extractable entities are found in the input."""
    pass


# NOTE: ContractViolationError is now imported from infer_schema_validator.py


class AmbiguousRelationshipError(SchemaInferenceError):
    """Raised when a relationship cannot be safely inferred."""
    pass


class LLMUnavailableError(SchemaInferenceError):
    """Raised when the LLM (Gemini) is unavailable in interactive mode."""
    pass


# NOTE: TerminationError is now imported from infer_schema_validator.py


# =============================================================================
# DATA STRUCTURES FOR INTERACTIVE WORKFLOW
# =============================================================================

@dataclass
class RefinedUnderstanding:
    """
    Phase 1 output: Human-first clarification for user review.
    
    The `authoritative_description` is the SOURCE OF TRUTH that the user
    can edit. All other fields are ADVISORY ONLY (for UI hints) and are
    NEVER used by Phase 2.
    """
    authoritative_description: str
    
    # Advisory fields (not used by Phase 2, for UI display only)
    entities: list[str] = field(default_factory=list)
    relationships: list[dict] = field(default_factory=list)
    assumptions: list[str] = field(default_factory=list)
    ambiguities: list[str] = field(default_factory=list)


# NOTE: ValidationFeedback is now imported from infer_schema_validator.py

# =============================================================================
# CONSTANTS
# =============================================================================

# NOTE: ALLOWED_TYPES is now defined in infer_schema_validator.py

# NOTE: Legacy heuristic constants (STOP_WORDS, RELATIONSHIP_PATTERNS, 
# MN_RELATIONSHIP_PATTERNS) have been removed. Schema inference is now 
# handled exclusively by the GenAI-powered interactive workflow.



# =============================================================================
# SCHEMA VALIDATION
# =============================================================================
# NOTE: Schema validation logic has been moved to infer_schema_validator.py.
# This module now imports validate_schema_with_feedback and validate_and_handle_termination
# from that module.
#
# Legacy heuristic helper functions have also been removed.
# Schema inference is now handled by the GenAI-powered interactive workflow.


# =============================================================================
# PUBLIC INTERFACE
# =============================================================================

def infer_schema(business_description: str) -> dict[str, Any]:
    """
    [DEPRECATED] Legacy heuristic-based schema inference.
    
    This function has been deprecated. Schema inference is now handled
    exclusively by the GenAI-powered interactive workflow.
    
    Use the new interactive workflow instead:
        1. Call `infer_schema_interactive(description)` to get Phase 1 output
        2. User reviews/edits the `authoritative_description`
        3. Call `finalize_schema(approved_description)` to get the final schema
    
    Args:
        business_description: A natural language description of the business domain.
    
    Raises:
        DeprecationWarning: Always. This function is no longer supported.
    """
    raise DeprecationWarning(
        "infer_schema() is deprecated and has been removed. "
        "Use the interactive GenAI workflow instead:\n"
        "  1. understanding = infer_schema_interactive(description)\n"
        "  2. # User reviews/edits understanding.authoritative_description\n"
        "  3. schema = finalize_schema(edited_description)\n"
        "\n"
        "See documentation for details on the new multi-phase workflow."
    )


# =============================================================================
# LANGCHAIN CONFIGURATION
# =============================================================================

def _get_llm() -> ChatGoogleGenerativeAI:
    """
    Initialize the Gemini 2.5 Pro LLM via LangChain.
    
    Raises:
        LLMUnavailableError: If GEMINI_API_KEY is not set or connection fails.
    """
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise LLMUnavailableError(
            "GEMINI_API_KEY environment variable is not set. "
            "Set it before using the interactive schema inference workflow."
        )
    
    try:
        return ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",  # Valid, stable model
            google_api_key=api_key,
            temperature=0.1,  # Low temperature for deterministic outputs
        )
    except Exception as e:
        raise LLMUnavailableError(f"Failed to initialize Gemini LLM: {e}")


# =============================================================================
# PROMPT TEMPLATES (EXTERNALIZED)
# =============================================================================

# Load prompts from external JSON file for easy modification
_PROMPTS_FILE = Path(__file__).parent / "prompts_infer_schema.json"

def _load_prompts() -> dict[str, str]:
    """Load prompt templates from external JSON file."""
    if not _PROMPTS_FILE.exists():
        raise FileNotFoundError(
            f"Prompts file not found: {_PROMPTS_FILE}. "
            "Ensure prompts_infer_schema.json exists in the schema_inference directory."
        )
    with open(_PROMPTS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def _get_prompt(key: str) -> PromptTemplate:
    """Get a prompt template by key from the external JSON file."""
    prompts = _load_prompts()
    if key not in prompts:
        raise KeyError(f"Prompt key '{key}' not found in {_PROMPTS_FILE}")
    return PromptTemplate.from_template(prompts[key])


# =============================================================================
# PHASE 1: UNDERSTANDING REFINEMENT (LLM)
# =============================================================================

def refine_understanding(description: str) -> RefinedUnderstanding:
    """
    Phase 1: Use LLM to generate a human-readable clarification of the
    user's business description.
    
    The output is designed to be reviewed and freely edited by the user.
    The `authoritative_description` field is the ONLY SOURCE OF TRUTH.
    
    Advisory fields (entities, relationships, assumptions, ambiguities) are
    intentionally left empty. They exist for potential future UI enhancements
    but are NEVER used by Phase 2 and do not depend on parsing the text.
    
    Args:
        description: Raw business description from user.
        
    Returns:
        RefinedUnderstanding with editable text. Advisory fields are empty.
        
    Raises:
        LLMUnavailableError: If Gemini is not available.
    """
    if not description or not description.strip():
        raise NoEntitiesFoundError("Empty or blank business description provided.")
    
    llm = _get_llm()
    prompt = _get_prompt("phase_1_refinement")
    chain = prompt | llm | StrOutputParser()
    
    try:
        refined_text = chain.invoke({"description": description})
    except Exception as e:
        raise LLMUnavailableError(f"LLM call failed during Phase 1: {e}")
    
    # Advisory fields are intentionally empty.
    # Phase 1 is human-first: the text is the only authoritative output.
    # Users may freely edit the text without breaking any internal logic.
    return RefinedUnderstanding(
        authoritative_description=refined_text,
        # Advisory fields left empty - no parsing of free-form text
        entities=[],
        relationships=[],
        assumptions=[],
        ambiguities=[],
    )


def refine_understanding_with_feedback(
    description: str,
    feedback: "ValidationFeedback"
) -> RefinedUnderstanding:
    """
    Refine the description using validation feedback (for attempt 2).
    
    When schema validation fails, this function takes the original description
    and the validation errors, then uses the LLM to produce an improved
    description that addresses the errors.
    
    Args:
        description: The original refined description that failed validation.
        feedback: ValidationFeedback containing errors and suggestions.
        
    Returns:
        RefinedUnderstanding with improved text.
        
    Raises:
        LLMUnavailableError: If Gemini is not available.
        NoEntitiesFoundError: If description is empty.
    """
    if not description or not description.strip():
        raise NoEntitiesFoundError("Empty description provided for refinement.")
    
    llm = _get_llm()
    prompt = _get_prompt("phase_1_refinement_with_feedback")
    chain = prompt | llm | StrOutputParser()
    
    # Format errors for the prompt
    error_text = "\n".join([f"- {e}" for e in feedback.errors])
    
    try:
        refined_text = chain.invoke({
            "description": description,
            "feedback": error_text
        })
    except Exception as e:
        raise LLMUnavailableError(f"LLM call failed during refinement with feedback: {e}")
    
    return RefinedUnderstanding(
        authoritative_description=refined_text,
        entities=[],
        relationships=[],
        assumptions=[],
        ambiguities=[],
    )


# NOTE: Parsing functions (_parse_entities_from_text, _parse_relationships_from_text,
# _parse_section) have been removed. Phase 1 is human-first and does not attempt
# to extract structured data from free-form text. Advisory fields are left empty.


# =============================================================================
# PHASE 2: SCHEMA GENERATION (LLM - TEXT ONLY INPUT)
# =============================================================================

def generate_schema_from_text(description_text: str) -> dict:
    """
    Phase 2: Use LLM to generate a contract-compliant JSON schema.
    
    CRITICAL: This function takes ONLY the approved text as input.
    It does NOT use any structured fields from Phase 1.
    
    Args:
        description_text: User-approved description text (may be edited).
        
    Returns:
        Schema dict conforming to the Schema Inference Contract.
        
    Raises:
        LLMUnavailableError: If Gemini is not available.
        ContractViolationError: If LLM output is not valid JSON.
    """
    if not description_text or not description_text.strip():
        raise NoEntitiesFoundError("Empty description provided to schema generator.")
    
    llm = _get_llm()
    prompt = _get_prompt("phase_2_schema_generation")
    chain = prompt | llm | StrOutputParser()
    
    try:
        raw_output = chain.invoke({"description": description_text})
    except Exception as e:
        raise LLMUnavailableError(f"LLM call failed during Phase 2: {e}")
    
    # Parse JSON from LLM output
    try:
        # Clean the output - LLM may include markdown code blocks
        cleaned = raw_output.strip()
        if cleaned.startswith("```json"):
            cleaned = cleaned[7:]
        if cleaned.startswith("```"):
            cleaned = cleaned[3:]
        if cleaned.endswith("```"):
            cleaned = cleaned[:-3]
        cleaned = cleaned.strip()
        
        schema = json.loads(cleaned)
    except json.JSONDecodeError as e:
        raise ContractViolationError(
            f"LLM produced invalid JSON: {e}. Raw output: {raw_output[:500]}"
        )
    
    # Basic structure validation
    if not isinstance(schema, dict) or "tables" not in schema:
        raise ContractViolationError(
            "LLM output missing 'tables' key. Schema must have format: {\"tables\": [...]}"
        )
    
    return schema


# =============================================================================
# PHASE 3: DETERMINISTIC VALIDATION (NO LLM)
# =============================================================================
# NOTE: validate_schema_with_feedback and _generate_suggestions are now
# imported from infer_schema_validator.py


# =============================================================================
# INTERACTIVE WORKFLOW ENTRY POINTS
# =============================================================================

def infer_schema_interactive(description: str) -> RefinedUnderstanding:
    """
    Entry point for interactive schema inference workflow.
    
    Returns Phase 1 output for user review and editing.
    User should edit the `authoritative_description` field, then call
    `finalize_schema()` with the edited text.
    
    Args:
        description: Raw business description from user.
        
    Returns:
        RefinedUnderstanding for user review.
        
    Raises:
        LLMUnavailableError: If Gemini is unavailable (NO FALLBACK).
        NoEntitiesFoundError: If description is empty.
    """
    return refine_understanding(description)


def finalize_schema(
    approved_description: str,
    attempt_number: int = 1
) -> dict | ValidationFeedback:
    """
    Finalize schema after user approval (Phase 2 + Phase 3).
    
    Args:
        approved_description: User-approved (possibly edited) description text.
        attempt_number: Current attempt (1 or 2). Start at 1.
        
    Returns:
        - On success: Valid schema dict.
        - On attempt 1 failure: ValidationFeedback for user to revise.
        
    Raises:
        TerminationError: On attempt 2 failure (hard stop).
        LLMUnavailableError: If Gemini is unavailable.
    """
    if attempt_number not in (1, 2):
        raise ValueError("attempt_number must be 1 or 2")
    
    # Phase 2: Generate schema from approved text
    schema = generate_schema_from_text(approved_description)
    
    # Phase 3: Validate
    feedback = validate_schema_with_feedback(schema, attempt_number)
    
    if feedback.is_valid:
        return schema
    
    # Handle failure based on attempt number
    if attempt_number == 1:
        # First failure: return feedback for user to revise
        return feedback
    else:
        # Second failure: terminate
        raise TerminationError(feedback.errors)

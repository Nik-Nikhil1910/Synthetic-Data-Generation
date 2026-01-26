"""
Generation Plan Validators
==========================

Pydantic models defining and enforcing the Generation Plan Contract.

Contract Source: Project Plan §3.3
"""

from typing import Literal, Optional
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# COLUMN MODELS
# =============================================================================

class ForeignKeyReference(BaseModel):
    """Reference metadata for a foreign key column."""
    table: str = Field(..., description="Parent table name")
    column: str = Field(..., description="Parent column name (usually 'id')")


class PrimaryKeyColumn(BaseModel):
    """Column definition for a primary key."""
    name: str
    generator_type: Literal["primary_key"]
    strategy: Literal["sequential_integer"] = "sequential_integer"


class ForeignKeyColumn(BaseModel):
    """Column definition for a foreign key."""
    name: str
    generator_type: Literal["foreign_key"]
    references: ForeignKeyReference
    is_nullable: bool = False


class FakerColumn(BaseModel):
    """Column definition using Faker for data generation."""
    name: str
    generator_type: Literal["faker"]
    faker_provider: str = Field(..., description="Faker method name (e.g., 'email', 'name')")
    is_nullable: bool = False
    null_probability: Optional[float] = Field(
        default=None,
        ge=0.0,
        le=1.0,
        description="Probability of null value (only if is_nullable=True)"
    )

    @model_validator(mode="after")
    def validate_null_probability(self):
        """Ensure null_probability is set correctly based on is_nullable."""
        if self.is_nullable and self.null_probability is None:
            # Default null probability per Project Plan Rule 3
            self.null_probability = 0.2
        if not self.is_nullable and self.null_probability is not None:
            # Non-nullable columns should not have null_probability
            self.null_probability = None
        return self


# Union type for all column types
ColumnSpec = PrimaryKeyColumn | ForeignKeyColumn | FakerColumn


# =============================================================================
# TABLE MODEL
# =============================================================================

class TablePlan(BaseModel):
    """Plan for generating data for a single table."""
    name: str = Field(..., description="Table name (snake_case)")
    row_count: int = Field(default=50, ge=1, description="Number of rows to generate")
    columns: list[ColumnSpec] = Field(..., min_length=1)


# =============================================================================
# META MODEL
# =============================================================================

class PlanMeta(BaseModel):
    """Metadata for the generation plan."""
    faker_seed: int = Field(default=0, description="Seed for Faker reproducibility")
    version: str = Field(default="0.1.0", description="Plan schema version")


# =============================================================================
# ROOT MODEL: GENERATION PLAN
# =============================================================================

class GenerationPlan(BaseModel):
    """
    Root model for the complete Generation Plan.
    
    This is the output contract for plan_generator.py.
    """
    meta: PlanMeta = Field(default_factory=PlanMeta)
    execution_order: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of table names for generation sequence"
    )
    tables: list[TablePlan] = Field(..., min_length=1)

    @model_validator(mode="after")
    def validate_execution_order_matches_tables(self):
        """Ensure execution_order contains exactly the table names in tables."""
        table_names = {t.name for t in self.tables}
        order_names = set(self.execution_order)
        
        if table_names != order_names:
            missing_in_order = table_names - order_names
            extra_in_order = order_names - table_names
            errors = []
            if missing_in_order:
                errors.append(f"Tables missing in execution_order: {missing_in_order}")
            if extra_in_order:
                errors.append(f"Unknown tables in execution_order: {extra_in_order}")
            raise ValueError("; ".join(errors))
        
        return self

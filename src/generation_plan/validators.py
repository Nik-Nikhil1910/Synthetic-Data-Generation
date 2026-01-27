"""
Generation Plan Validators (Pydantic Models)
=============================================

This module defines the Pydantic models for validating and typing
the Generation Plan JSON structure.
"""

from typing import Literal, Union, Annotated
from pydantic import BaseModel, Field, model_validator


# =============================================================================
# COLUMN SPECIFICATIONS
# =============================================================================

class ForeignKeyReference(BaseModel):
    """Reference to a parent table/column for FK relationships."""
    table: str = Field(..., description="Parent table name")
    column: str = Field(default="id", description="Parent column name (usually 'id')")


class PrimaryKeyColumn(BaseModel):
    """Specification for a primary key column."""
    name: str
    generator_type: Literal["primary_key"]
    strategy: Literal["sequential_integer"] = "sequential_integer"


class ForeignKeyColumn(BaseModel):
    """Specification for a foreign key column."""
    name: str
    generator_type: Literal["foreign_key"]
    references: ForeignKeyReference
    is_nullable: bool = False
    null_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @model_validator(mode="after")
    def validate_null_probability(self):
        if self.is_nullable and self.null_probability == 0.0:
            self.null_probability = 0.2  # Default for nullable FKs
        if not self.is_nullable and self.null_probability > 0.0:
            self.null_probability = 0.0
        return self


class FakerColumn(BaseModel):
    """Specification for a Faker-generated column."""
    name: str
    generator_type: Literal["faker"]
    faker_provider: str
    is_nullable: bool = False
    null_probability: float = Field(default=0.0, ge=0.0, le=1.0)
    
    @model_validator(mode="after")
    def validate_null_probability(self):
        if self.is_nullable and self.null_probability == 0.0:
            self.null_probability = 0.2  # Default for nullable columns
        if not self.is_nullable and self.null_probability > 0.0:
            self.null_probability = 0.0
        return self


# Union type for column specifications
ColumnSpec = Annotated[
    Union[PrimaryKeyColumn, ForeignKeyColumn, FakerColumn],
    Field(discriminator="generator_type")
]


# =============================================================================
# TABLE PLAN
# =============================================================================

class TablePlan(BaseModel):
    """Plan for generating a single table."""
    name: str
    row_count: int = Field(default=50, ge=1)
    columns: list[ColumnSpec] = Field(..., min_length=1)
    
    @model_validator(mode="after")
    def validate_has_primary_key(self):
        pk_count = sum(
            1 for col in self.columns
            if isinstance(col, PrimaryKeyColumn) or 
               (hasattr(col, 'generator_type') and col.generator_type == "primary_key")
        )
        if pk_count != 1:
            raise ValueError(f"Table must have exactly 1 PK, found {pk_count}")
        return self


# =============================================================================
# PLAN METADATA
# =============================================================================

class PlanMeta(BaseModel):
    """Metadata for the generation plan."""
    faker_seed: int = Field(default=0, description="Seed for Faker reproducibility")
    version: str = Field(default="0.1.0")


# =============================================================================
# ROOT MODEL
# =============================================================================

class GenerationPlan(BaseModel):
    """Root model for the complete Generation Plan."""
    meta: PlanMeta = Field(default_factory=PlanMeta)
    execution_order: list[str] = Field(
        ...,
        min_length=1,
        description="Ordered list of table names for generation sequence"
    )
    tables: list[TablePlan] = Field(..., min_length=1)
    
    @model_validator(mode="after")
    def validate_execution_order_matches_tables(self):
        table_names = {t.name for t in self.tables}
        order_names = set(self.execution_order)
        
        if table_names != order_names:
            missing = table_names - order_names
            extra = order_names - table_names
            msg = []
            if missing:
                msg.append(f"Missing in order: {missing}")
            if extra:
                msg.append(f"Extra in order: {extra}")
            raise ValueError("; ".join(msg))
        
        return self

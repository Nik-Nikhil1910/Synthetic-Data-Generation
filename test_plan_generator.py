"""
Test script for Generation Plan Module.
"""
import sys
sys.path.insert(0, 'src')

from schema_inference.infer_schema import infer_schema
from generation_plan.plan_generator import generate_plan, CircularDependencyError
import json


def test_basic_pipeline():
    """Test basic schema -> plan pipeline."""
    print('=== TEST 1: Basic Pipeline ===')
    schema = infer_schema('A blog with users and posts. Users have many posts.')
    plan = generate_plan(schema)
    
    print('Execution Order:', plan['execution_order'])
    print('Tables:', [t['name'] for t in plan['tables']])
    print('Meta:', plan['meta'])
    
    # Verify user comes before post (parent before child)
    order = plan['execution_order']
    assert 'user' in order
    assert 'post' in order
    assert order.index('user') < order.index('post'), "Parent must come before child"
    print('✓ Dependency ordering correct')


def test_fk_structure():
    """Test FK columns have correct contract structure."""
    print('\n=== TEST 2: FK Structure ===')
    schema = infer_schema('A blog with users and posts. Users have many posts.')
    plan = generate_plan(schema)
    
    fk_found = False
    for table in plan['tables']:
        for col in table['columns']:
            if col['generator_type'] == 'foreign_key':
                fk_found = True
                print(f"Table '{table['name']}', FK column: {json.dumps(col, indent=2)}")
                # Verify FK structure
                assert 'references' in col, "FK must have 'references'"
                assert 'table' in col['references'], "FK references must have 'table'"
                assert 'column' in col['references'], "FK references must have 'column'"
                assert 'is_nullable' in col, "FK must have 'is_nullable'"
    
    assert fk_found, "Should have found at least one FK"
    print('✓ FK structure is correct')


def test_pk_structure():
    """Test PK columns have correct contract structure."""
    print('\n=== TEST 3: PK Structure ===')
    schema = infer_schema('A simple table of products.')
    plan = generate_plan(schema)
    
    pk_found = False
    for table in plan['tables']:
        for col in table['columns']:
            if col['generator_type'] == 'primary_key':
                pk_found = True
                print(f"Table '{table['name']}', PK column: {json.dumps(col, indent=2)}")
                assert col['strategy'] == 'sequential_integer', "PK strategy must be sequential_integer"
    
    assert pk_found, "Should have found at least one PK"
    print('✓ PK structure is correct')


def test_determinism():
    """Test that same schema always produces same plan."""
    print('\n=== TEST 4: Determinism ===')
    schema = infer_schema('A CRM with customers, orders, and products.')
    
    plan1 = generate_plan(schema)
    plan2 = generate_plan(schema)
    
    assert plan1 == plan2, "Plans should be identical"
    print('✓ Plans are deterministic')


def test_nullable_column():
    """Test nullable columns get null_probability."""
    print('\n=== TEST 5: Nullable Handling ===')
    # We need a schema with nullable columns
    schema = {
        "tables": [
            {
                "name": "users",
                "columns": [
                    {"name": "id", "type": "INTEGER", "is_pk": True, "is_nullable": False},
                    {"name": "nickname", "type": "VARCHAR", "is_pk": False, "is_nullable": True}
                ],
                "foreign_keys": []
            }
        ]
    }
    plan = generate_plan(schema)
    
    for table in plan['tables']:
        for col in table['columns']:
            if col['name'] == 'nickname':
                print(f"Nullable column: {json.dumps(col, indent=2)}")
                assert col['is_nullable'] == True
                assert col['null_probability'] == 0.2, "Nullable should have null_probability=0.2"
    
    print('✓ Nullable handling is correct')


if __name__ == '__main__':
    test_basic_pipeline()
    test_fk_structure()
    test_pk_structure()
    test_determinism()
    test_nullable_column()
    print('\n=== ALL TESTS PASSED ===')

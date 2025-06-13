# WEAC Unit Test Suite

This directory contains a comprehensive unit test suite for the refactored WEAC (Weak layer Anticrack) simulation package. The test suite is designed to ensure reliability, correctness, and maintainability of the codebase.

## Test Suite Overview

The test suite follows a modular structure that mirrors the package organization:

### 1. Component Tests (`test_components_*.py`)

#### `test_components_layer.py`
Tests the foundational `Layer` and `WeakLayer` classes:
- **Material property calculations**: Validates Young's modulus calculations using Bergfeld, Scapozza, and Gerling methods
- **Validation logic**: Tests Pydantic validation for density, thickness, Poisson's ratio constraints
- **Auto-calculation features**: Ensures E, G, kn, kt are correctly computed when not specified
- **Physical consistency**: Verifies density-modulus relationships and stiffness scaling
- **Edge cases**: Handles zero values, negative parameters, and boundary conditions

#### `test_components_configs.py`
Tests all configuration classes and model input validation:
- **Config validation**: Tests enum values for Young's modulus and failure envelope methods
- **ScenarioConfig**: Validates slope angles, system types, collapse factors, and surface loads
- **CriteriaConfig**: Tests failure mode interaction parameters
- **Segment validation**: Ensures positive lengths and masses
- **ModelInput integration**: Tests complete model assembly and JSON serialization
- **Physical consistency**: Validates layer ordering and segment configurations

### 2. Core Physics Tests (`test_core_*.py`)

#### `test_core_slab.py`
Tests the `Slab` class for multi-layer assembly:
- **Layer assembly**: Validates coordinate system, thickness calculations, and property arrays
- **Center of gravity**: Tests CoG calculations for uniform and gradient density distributions
- **Weight calculations**: Verifies weight load computations and mass conservation
- **Coordinate consistency**: Ensures layer positioning and boundary calculations
- **Inclined surfaces**: Tests vertical CoG calculations for avalanche slope applications

#### `test_core_eigensystem.py`
Tests the `Eigensystem` class for mathematical computations:
- **System matrices**: Validates 6×6 system matrix assembly and structure
- **Eigenvalue calculations**: Tests eigenvalue classification (real vs complex) and eigenvector dimensions
- **Solution methods**: Tests complementary and particular solution calculations
- **Physical scaling**: Verifies that material properties correctly influence system behavior
- **Numerical stability**: Tests eigenvalue shifts and solution continuity

#### `test_core_field_quantities.py`
Tests the `FieldQuantities` class for result interpretation:
- **Displacement calculations**: Tests u, w, ψ and their derivatives with proper unit conversions
- **Stress calculations**: Validates normal force N, moment M, shear force V calculations
- **Weak layer stresses**: Tests σ and τ calculations with correct sign conventions
- **Strain calculations**: Validates normal and shear strain computations
- **Energy release rates**: Tests Mode I and II ERR calculations with unit conversions
- **Physical consistency**: Ensures continuity, sign conventions, and positivity constraints

### 3. Utility Tests (`test_utils.py`)

Tests utility functions for force calculations:
- **Force decomposition**: Tests `decompose_to_normal_tangential` for various angles
- **Skier loads**: Validates `get_skier_point_load` calculations and scaling
- **Unit conversions**: Tests angle units (degrees/radians) and force units
- **Edge cases**: Handles zero forces, extreme angles, and boundary conditions
- **Physical reasonableness**: Ensures results are in expected ranges for typical applications

### 4. Integration Tests (`test_integration.py`)

Tests complete system integration and comparison with legacy implementation:
- **Old vs New comparison**: Validates that refactored code produces equivalent results
- **Tolerance testing**: Uses appropriate tolerances for numerical comparison
- **Real-world scenarios**: Tests with physically meaningful snow profiles and loads

### 5. System Model Tests (`test_system_model.py`)

Tests the main orchestrator class:
- **Caching behavior**: Validates that expensive calculations are cached appropriately
- **Update mechanisms**: Tests selective invalidation when properties change
- **State consistency**: Ensures system remains consistent during updates

## Test Categories

### Validation Tests
- **Input validation**: Ensures invalid inputs are properly rejected
- **Physical constraints**: Tests that physical laws are respected (positive energies, etc.)
- **Boundary conditions**: Validates behavior at extreme parameter values

### Numerical Tests
- **Accuracy**: Compares calculated values against analytical solutions where possible
- **Stability**: Tests numerical stability for various parameter ranges
- **Convergence**: Ensures iterative calculations converge appropriately

### Integration Tests
- **Component interaction**: Tests that different modules work together correctly
- **End-to-end workflows**: Validates complete simulation workflows
- **Legacy compatibility**: Ensures refactored code maintains compatibility

### Performance Tests
- **Caching efficiency**: Validates that caching improves performance
- **Memory usage**: Ensures reasonable memory consumption
- **Computational complexity**: Tests scaling with problem size

## Running the Tests

### Run All Tests
```bash
# From the project root
python -m pytest tests_2/ -v

# Or using the test runner
python tests_2/run_tests.py
```

### Run Specific Test Categories
```bash
# Component tests only
python -m pytest tests_2/test_components_*.py -v

# Core physics tests only
python -m pytest tests_2/test_core_*.py -v

# Integration tests only
python -m pytest tests_2/test_integration.py -v
```

### Run Individual Test Files
```bash
# Layer tests
python -m pytest tests_2/test_components_layer.py -v

# Eigensystem tests
python -m pytest tests_2/test_core_eigensystem.py -v
```

### Run with Coverage
```bash
pip install pytest-cov
python -m pytest tests_2/ --cov=weac_2 --cov-report=html
```

## Test Data Philosophy

### Realistic Parameters
Tests use physically meaningful parameter ranges:
- **Snow densities**: 50-500 kg/m³ (typical range for weak layers to dense slabs)
- **Layer thicknesses**: 10-200 mm (typical snowpack layer thicknesses)
- **Slope angles**: 25-45° (typical avalanche terrain)
- **Skier masses**: 50-120 kg (typical range)

### Known Solutions
Where possible, tests compare against:
- **Analytical solutions**: For simple cases with known mathematical solutions
- **Physical limits**: Boundary cases where behavior is predictable
- **Legacy results**: Comparison with validated previous implementation

### Edge Cases
Tests specifically target:
- **Zero values**: Ensures graceful handling of zero inputs
- **Extreme parameters**: Very light/heavy materials, steep slopes, etc.
- **Boundary conditions**: Values at validation limits

## Test Maintenance

### Adding New Tests
When adding new functionality:
1. **Create test file**: Follow naming convention `test_[module]_[class].py`
2. **Test all public methods**: Every public method should have at least one test
3. **Include edge cases**: Test boundary conditions and error cases
4. **Validate physics**: Ensure results are physically reasonable
5. **Document purpose**: Clear docstrings explaining what each test validates

### Updating Existing Tests
When modifying code:
1. **Update affected tests**: Ensure tests reflect new behavior
2. **Maintain coverage**: Don't remove tests without replacement
3. **Check integration**: Ensure changes don't break downstream tests
4. **Update tolerances**: Adjust numerical tolerances if algorithms change

### Performance Considerations
- **Fast unit tests**: Individual tests should complete in milliseconds
- **Isolated tests**: Each test should be independent and not rely on others
- **Minimal setup**: Use `setUp()` methods to minimize repeated initialization
- **Mock expensive operations**: Use test doubles for expensive calculations when testing logic

## Expected Test Results

A fully passing test suite indicates:
- ✅ All components validate inputs correctly
- ✅ Mathematical calculations are accurate
- ✅ Physical laws are respected
- ✅ Integration between components works
- ✅ Results match legacy implementation (within tolerances)
- ✅ Code handles edge cases gracefully
- ✅ Performance optimizations (caching) work correctly

## Troubleshooting

### Common Issues

**Import Errors**: Ensure the project root is in Python path
```bash
export PYTHONPATH="${PYTHONPATH}:/path/to/weac"
```

**Tolerance Failures**: May indicate:
- Algorithmic changes affecting numerical precision
- Platform-dependent floating-point differences
- Need to adjust test tolerances

**Integration Test Failures**: May indicate:
- Breaking changes in refactored code
- Different parameter interpretations
- Need to update test scenarios

### Debugging Failed Tests
```bash
# Run with verbose output and stop on first failure
python -m pytest tests_2/test_file.py::TestClass::test_method -v -x

# Run with detailed assertion output
python -m pytest tests_2/ -v --tb=long

# Run specific test with Python debugger
python -m pytest tests_2/test_file.py::TestClass::test_method -v -s --pdb
```

This comprehensive test suite ensures the reliability and correctness of the WEAC simulation package, providing confidence in both individual components and their integration. 
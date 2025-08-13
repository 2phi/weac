#!/usr/bin/env python
"""
Test runner script for the WEAC package.

This script discovers and runs all tests in the tests directory.
"""

import os
import sys
import unittest

# Ensure the parent directory is in the system path to find the 'weac' package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

from weac.logging_config import setup_logging

setup_logging(level="WARNING")


def run_tests():
    """Discover and run all tests in the tests directory and subdirectories."""
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))

    print(f"Discovering tests in: {test_dir}")
    print("Looking for test files matching pattern: test_*.py")
    print("Searching recursively in subdirectories...")
    print("-" * 60)

    # Discover all tests in the tests directory (recursive by default)
    test_suite = unittest.defaultTestLoader.discover(
        test_dir, pattern="test_*.py", top_level_dir=parent_dir
    )

    # Count and display discovered tests
    test_count = test_suite.countTestCases()
    print(f"Found {test_count} test cases")
    print("-" * 60)

    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests
    result = test_runner.run(test_suite)

    # Print summary
    print("\n" + "=" * 60)
    print(f"Tests run: {result.testsRun}")
    print(f"Failures: {len(result.failures)}")
    print(f"Errors: {len(result.errors)}")
    if result.testsRun > 0:
        success_rate = (
            (result.testsRun - len(result.failures) - len(result.errors))
            / result.testsRun
            * 100
        )
        print(f"Success rate: {success_rate:.1f}%")
    else:
        print("No tests were run")

    return result


if __name__ == "__main__":
    run_tests()

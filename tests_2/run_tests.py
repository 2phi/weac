#!/usr/bin/env python
"""
Test runner script for the WEAC package.

This script discovers and runs all tests in the tests directory.
"""

import os
import sys
import unittest

# Add the parent directory to Python path so weac_2 can be imported
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)


def run_tests():
    """Discover and run all tests in the tests directory."""
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Discover all tests in the tests directory
    test_suite = unittest.defaultTestLoader.discover(test_dir)

    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests
    result = test_runner.run(test_suite)

    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())

#!/usr/bin/env python
"""
Test runner script for the WEAC package.

This script discovers and runs all tests in the tests directory.
"""

import os
import sys
import unittest

# Ensure the parent directory is in the system path to find the 'weac_2' package
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
if parent_dir not in sys.path:
    sys.path.insert(0, parent_dir)

# Import all test modules from the current directory


def run_tests():
    """Discover and run all tests in the tests directory."""
    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Discover all tests in the tests directory
    test_suite = unittest.defaultTestLoader.discover(test_dir, pattern="test_*.py")

    # Create a test runner
    test_runner = unittest.TextTestRunner(verbosity=2)

    # Run the tests
    test_runner.run(test_suite)


if __name__ == "__main__":
    run_tests()

#!/usr/bin/env python
"""
Test runner script for the WEAC package.

This script discovers and runs all tests in the tests directory.
Provides a pytest-like output with detailed reporting.
"""

import os
import sys
import time
import unittest
from collections import defaultdict
from typing import Dict


class PytestLikeTextTestResult(unittest.TextTestResult):
    """A test result class that provides pytest-like output format."""

    PASS = "\033[92m"  # Green
    FAIL = "\033[91m"  # Red
    SKIP = "\033[93m"  # Yellow
    END = "\033[0m"  # Reset color
    BOLD = "\033[1m"  # Bold text

    def __init__(self, stream, descriptions, verbosity):
        """Initialize the test result object."""
        # Override descriptions to prevent unittest from printing the test docstring
        super().__init__(stream, False, verbosity)
        self.stream = stream
        self.verbosity = verbosity
        self.descriptions = (
            False  # Override to prevent unittest from printing docstrings
        )
        self.successes = []
        self.start_time = time.time()
        self.test_times: Dict[str, float] = {}
        self.module_counts: Dict[str, Dict[str, int]] = defaultdict(
            lambda: defaultdict(int)
        )
        self.total_tests = 0
        self.test_counter = 0

        # Print header
        self.stream.write(
            f"\n{self.BOLD}============================== test session starts =============================={self.END}\n"
        )
        self.stream.write(f"platform: {sys.platform}, Python {sys.version.split()[0]}\n")
        self.stream.write(
            f"rootdir: {os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))}\n"
        )
        self.stream.flush()

    def getDescription(self, test):
        """Override to return an empty description, preventing unittest from printing the docstring."""
        return ""

    def set_total_tests(self, count):
        """Set the total number of tests to be run."""
        self.total_tests = count

    def startTest(self, test):
        """Called when a test starts."""
        super().startTest(test)
        self.test_start_time = time.time()
        self.test_counter += 1

        if self.verbosity > 1:
            # Extract test name and module in a cleaner format
            test_id = test.id()
            module_name, class_name, test_name = test_id.split(".")[-3:]

            # Get test description
            doc = test._testMethodDoc or ""

            # Print the test name with progress indicator and description
            progress = f"[ {self.test_counter}/{self.total_tests} ]"
            self.stream.write(f"\n{progress} {module_name}.{class_name}.{test_name}\n")
            if doc:
                self.stream.write(f"    {doc}\n")

            # Indentation for the result
            self.stream.write("    ")
            self.stream.flush()

    def _get_module_name(self, test):
        """Extract module name from test."""
        return test.__class__.__module__.split(".")[-1]

    def addSuccess(self, test):
        """Called when a test succeeds."""
        super().addSuccess(test)
        self.successes.append(test)
        self.test_times[test.id()] = time.time() - self.test_start_time
        module_name = self._get_module_name(test)
        self.module_counts[module_name]["passed"] += 1

        if self.verbosity > 1:
            self.stream.write(f"  {self.PASS}✓ PASS{self.END}\n")
            self.stream.flush()

    def addError(self, test, err):
        """Called when a test raises an error."""
        super().addError(test, err)
        self.test_times[test.id()] = time.time() - self.test_start_time
        module_name = self._get_module_name(test)
        self.module_counts[module_name]["errors"] += 1

        if self.verbosity > 1:
            self.stream.write(f"  {self.FAIL}E ERROR{self.END}\n")
            self.stream.flush()

    def addFailure(self, test, err):
        """Called when a test fails."""
        super().addFailure(test, err)
        self.test_times[test.id()] = time.time() - self.test_start_time
        module_name = self._get_module_name(test)
        self.module_counts[module_name]["failures"] += 1

        if self.verbosity > 1:
            self.stream.write(f"  {self.FAIL}✗ FAIL{self.END}\n")
            self.stream.flush()

    def addSkip(self, test, reason):
        """Called when a test is skipped."""
        super().addSkip(test, reason)
        self.test_times[test.id()] = time.time() - self.test_start_time
        module_name = self._get_module_name(test)
        self.module_counts[module_name]["skipped"] += 1

        if self.verbosity > 1:
            self.stream.write(f"  {self.SKIP}s SKIP{self.END} [{reason}]\n")
            self.stream.flush()

    def printErrors(self):
        """Print a formatted report of errors and failures."""
        if self.errors or self.failures:
            self.stream.write(
                f"\n{self.BOLD}============================== FAILURES =============================={self.END}\n"
            )

            for test, err in self.errors + self.failures:
                test_id = test.id()
                module_name, class_name, test_name = test_id.split(".")[-3:]
                self.stream.write(
                    f"\n{self.BOLD}{self.FAIL}FAILED{self.END} {module_name}.{class_name}.{test_name}{self.END}\n"
                )
                self.stream.write(f"{err}\n")

    def printTotal(self):
        """Print a summary of all tests run."""
        total_time = time.time() - self.start_time
        total_tests = self.testsRun
        passed = len(self.successes)
        failures = len(self.failures)
        errors = len(self.errors)
        skipped = len(self.skipped)

        # Print per-module summary
        self.stream.write(
            f"\n{self.BOLD}============================== test summary info =============================={self.END}\n"
        )

        for module, counts in sorted(self.module_counts.items()):
            result_str = []
            if counts["passed"]:
                result_str.append(f"{self.PASS}{counts['passed']} passed{self.END}")
            if counts["failures"]:
                result_str.append(f"{self.FAIL}{counts['failures']} failed{self.END}")
            if counts["errors"]:
                result_str.append(f"{self.FAIL}{counts['errors']} errors{self.END}")
            if counts["skipped"]:
                result_str.append(f"{self.SKIP}{counts['skipped']} skipped{self.END}")

            self.stream.write(f"{module}: {', '.join(result_str)}\n")

        # Print overall summary
        self.stream.write(
            f"\n{self.BOLD}============================== {total_tests} tests ran in {total_time:.2f}s =============================={self.END}\n"
        )

        result_parts = []
        if passed:
            result_parts.append(f"{self.PASS}{passed} passed{self.END}")
        if failures:
            result_parts.append(f"{self.FAIL}{failures} failed{self.END}")
        if errors:
            result_parts.append(f"{self.FAIL}{errors} errors{self.END}")
        if skipped:
            result_parts.append(f"{self.SKIP}{skipped} skipped{self.END}")

        self.stream.write(", ".join(result_parts) + "\n")


class PytestLikeTextTestRunner(unittest.TextTestRunner):
    """A test runner that uses PytestLikeTextTestResult to display results."""

    def __init__(
        self,
        stream=None,
        descriptions=False,  # Override to prevent unittest from printing docstrings
        verbosity=1,
        failfast=False,
        buffer=False,
        warnings=None,
    ):
        """Initialize the runner."""
        super().__init__(stream, descriptions, verbosity, failfast, buffer, warnings)

    def _makeResult(self):
        """Create and return a test result object that will be used to store results."""
        return PytestLikeTextTestResult(self.stream, self.descriptions, self.verbosity)

    def run(self, test):
        """Run the given test case or test suite."""
        result = self._makeResult()
        result.set_total_tests(self._count_tests(test))

        self.stream.write(f"collecting ... {result.total_tests} items collected\n")

        # Run tests
        startTestRun = getattr(result, "startTestRun", None)
        if startTestRun is not None:
            startTestRun()
        try:
            test(result)
        finally:
            stopTestRun = getattr(result, "stopTestRun", None)
            if stopTestRun is not None:
                stopTestRun()

        result.printErrors()
        result.printTotal()
        return result

    def _count_tests(self, test):
        """Count the total number of tests in a test suite."""
        if hasattr(test, "_tests"):
            return sum(self._count_tests(t) for t in test._tests)
        else:
            return 1


class CustomTextTestRunner(unittest.TextTestRunner):
    """Hide default unittest output since we're using our custom runner."""

    def run(self, test):
        """Run the test suite with no output."""
        result = super().run(test)
        return result


def run_tests():
    """Discover and run all tests in the tests directory."""
    # Redirect both standard out and standard error to capture unittest output
    # This prevents duplicate output from the standard unittest runner
    import io
    from contextlib import redirect_stderr, redirect_stdout

    f = io.StringIO()

    # Get the directory containing this script
    test_dir = os.path.dirname(os.path.abspath(__file__))

    # Create a test runner with pytest-like output
    test_runner = PytestLikeTextTestRunner(verbosity=2)

    # Discover all tests in the tests directory
    with redirect_stdout(f), redirect_stderr(f):
        test_suite = unittest.defaultTestLoader.discover(test_dir)

    # Run the tests with our custom output
    result = test_runner.run(test_suite)

    # Return appropriate exit code
    return 0 if result.wasSuccessful() else 1


if __name__ == "__main__":
    sys.exit(run_tests())

"""Unit tests for JSON helpers."""

from __future__ import annotations

import json
import unittest

import numpy as np

from .json_helpers import json_default


class TestJsonHelpers(unittest.TestCase):
    """Test the JSON serialization helpers."""

    def test_json_default_numpy_array(self):
        """Verify numpy arrays are serialized to lists."""
        data = {"a": np.array([1, 2, 3])}
        result = json.dumps(data, default=json_default)
        self.assertEqual(result, '{"a": [1, 2, 3]}')

    def test_json_default_numpy_scalars(self):
        """Verify numpy scalar types are serialized to Python primitives."""
        cases = {
            "int64": np.int64(42),
            "float64": np.float64(3.14),
            "bool_true": np.bool_(True),
            "bool_false": np.bool_(False),
        }
        data = {k: v for k, v in cases.items()}
        result = json.dumps(data, default=json_default)
        expected = (
            '{"int64": 42, "float64": 3.14, "bool_true": true, "bool_false": false}'
        )
        self.assertEqual(result, expected)

    def test_json_default_mixed_types(self):
        """Verify mixed data including numpy and standard types serializes correctly."""
        data = {
            "np_array": np.arange(3),
            "np_float": np.float32(1.23),
            "py_int": 100,
            "py_str": "hello",
            "py_list": [1, "a", None],
        }
        result = json.dumps(data, default=json_default)
        # Note: np.float32 may have precision differences, test against its .item()
        expected_py_float = np.float32(1.23).item()
        self.assertAlmostEqual(
            json.loads(result)["np_float"], expected_py_float, places=6
        )
        # Check the rest of the dictionary
        loaded_result = json.loads(result)
        del loaded_result["np_float"]
        expected_dict = {
            "np_array": [0, 1, 2],
            "py_int": 100,
            "py_str": "hello",
            "py_list": [1, "a", None],
        }
        self.assertDictEqual(loaded_result, expected_dict)

    def test_json_default_unhandled_type(self):
        """Verify unhandled types are converted to their string representation."""

        class Unserializable:
            def __str__(self):
                return "UnserializableObject"

        data = {"key": Unserializable()}
        result = json.dumps(data, default=json_default)
        self.assertEqual(result, '{"key": "UnserializableObject"}')

    def test_various_inputs(self):
        """Test a variety of inputs for comprehensive coverage."""
        test_cases = [
            (np.int32(-5), "-5"),
            (np.float64(1e-9), "1e-09"),
            (np.array([1.0, 2.5]), "[1.0, 2.5]"),
            (True, "true"),
            (None, "null"),
        ]

        for value, expected in test_cases:
            with self.subTest(value=value):
                self.assertEqual(json.dumps(value, default=json_default), expected)


if __name__ == "__main__":
    unittest.main()

"""Constrained Python edit actions: validate, isolate, budget, protect."""
import unittest

import numpy as np

from machaon.action import (
    ActionExecutionError,
    ActionSpec,
    ActionValidationError,
    execute_action,
    validate_action_source,
)
from machaon.schema import Target


def spec(source, **kwargs):
    return ActionSpec(
        source=source,
        target=Target("model.layers.0.mlp.down_proj.weight", 0),
        intent=kwargs.pop("intent", "unit test action"),
        **kwargs,
    )


class TestValidation(unittest.TestCase):
    def test_accepts_a_single_pure_edit_function(self):
        tree = validate_action_source(
            '"""docstring"""\n'
            "def edit(W):\n"
            "    updated, delta = rank1_constrained(W, keys, values, "
            "preserve=preserve, max_rel_norm=max_rel_norm)\n"
            "    return delta\n"
        )
        self.assertEqual(tree.body[1].name, "edit")

    def test_rejects_imports(self):
        with self.assertRaises(ActionValidationError):
            validate_action_source(
                "def edit(W):\n"
                "    import os\n"
                "    return W\n"
            )

    def test_rejects_private_attributes_and_arbitrary_calls(self):
        with self.assertRaises(ActionValidationError):
            validate_action_source(
                "def edit(W):\n"
                "    return W.__class__\n"
            )
        with self.assertRaises(ActionValidationError):
            validate_action_source(
                "def edit(W):\n"
                "    return open('x')\n"
            )

    def test_rejects_loops_and_multiple_functions(self):
        with self.assertRaises(ActionValidationError):
            validate_action_source(
                "def edit(W):\n"
                "    for item in W:\n"
                "        pass\n"
                "    return W\n"
            )
        with self.assertRaises(ActionValidationError):
            validate_action_source(
                "def helper(W):\n"
                "    return W\n"
                "def edit(W):\n"
                "    return helper(W)\n"
            )

    def test_source_hash_is_stable_and_result_omits_the_array(self):
        source = "def edit(W):\n    return 0.001 * W\n"
        first = spec(source).validate()
        second = spec(source).validate()
        self.assertEqual(first.source_sha256, second.source_sha256)
        self.assertEqual(len(first.source_sha256), 64)


class TestExecution(unittest.TestCase):
    def setUp(self):
        self.W = np.arange(20, dtype=np.float64).reshape(5, 4)

    def test_scales_within_budget(self):
        result = execute_action(
            spec("def edit(W):\n    return 0.001 * W\n", max_rel_norm=0.01),
            self.W,
        )
        np.testing.assert_allclose(result.delta, 0.001 * self.W)
        self.assertAlmostEqual(result.relative_norm, 0.001)
        payload = result.to_dict()
        self.assertNotIn("delta", payload)
        self.assertEqual(payload["delta_shape"], [5, 4])
        self.assertEqual(payload["source_sha256"], spec(
            "def edit(W):\n    return 0.001 * W\n"
        ).source_sha256)

    def test_safe_primitive_and_protected_key_contract(self):
        rng = np.random.default_rng(7)
        W = rng.normal(size=(5, 4))
        keys = rng.normal(size=4)
        values = rng.normal(size=5)
        preserve = rng.normal(size=(2, 4))
        source = (
            "def edit(W):\n"
            "    updated, delta = rank1_constrained(W, keys, values, "
            "preserve=preserve, max_rel_norm=max_rel_norm)\n"
            "    return delta\n"
        )
        result = execute_action(
            spec(source, max_rel_norm=0.5),
            W,
            keys=keys,
            values=values,
            preserve=preserve,
        )
        self.assertTrue(result.protected["passed"])
        np.testing.assert_allclose(result.delta @ preserve.T, 0.0, atol=1e-12)

    def test_budget_overflow_is_refused(self):
        with self.assertRaises(ActionExecutionError):
            execute_action(
                spec("def edit(W):\n    return W\n", max_rel_norm=0.01),
                self.W,
            )

    def test_protected_drift_is_refused(self):
        preserve = np.ones((1, self.W.shape[1]))
        with self.assertRaises(ActionExecutionError):
            execute_action(
                spec(
                    "def edit(W):\n    return ones_like(W) * 0.0001\n",
                    max_rel_norm=0.01,
                ),
                self.W,
                preserve=preserve,
            )

    def test_wrong_shape_and_non_finite_results_are_refused(self):
        with self.assertRaises(ActionExecutionError):
            execute_action(
                spec("def edit(W):\n    return W.T\n", max_rel_norm=0.01),
                self.W,
            )
        with self.assertRaises(ActionExecutionError):
            execute_action(
                spec("def edit(W):\n    return W / 0.0\n", max_rel_norm=0.01),
                self.W,
            )


if __name__ == "__main__":
    unittest.main()

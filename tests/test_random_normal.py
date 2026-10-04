import unittest

import numpy as np

from nn.initializers.random import random_normal


class TestRandomInitializer(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(42)

        weights = random_normal(
            3,
            4,
            rng=rng,
        )

        self.assertEqual(weights.shape, (3, 4))

    def test_same_seed_is_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        weights1 = random_normal(
            3,
            4,
            rng=rng1,
        )

        weights2 = random_normal(
            3,
            4,
            rng=rng2,
        )

        np.testing.assert_allclose(
            weights1,
            weights2,
        )

    def test_scale_is_applied_correctly(self):
        scale = 0.05

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        weights = random_normal(
            3,
            4,
            scale=scale,
            rng=rng1,
        )

        expected = (
            rng2.standard_normal((3, 4))
            * scale
        )

        np.testing.assert_allclose(
            weights,
            expected,
        )


if __name__ == "__main__":
    unittest.main()

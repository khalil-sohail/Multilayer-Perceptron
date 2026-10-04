import unittest

import numpy as np

from nn.initializers.he import he_normal


class TestHeInitializer(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(42)

        weights = he_normal(
            3,
            4,
            rng=rng,
        )

        self.assertEqual(weights.shape, (3, 4))

    def test_same_seed_is_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        weights1 = he_normal(
            3,
            4,
            rng=rng1,
        )

        weights2 = he_normal(
            3,
            4,
            rng=rng2,
        )

        np.testing.assert_allclose(
            weights1,
            weights2,
        )

    def test_he_formula(self):
        input_size = 3
        output_size = 4

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        weights = he_normal(
            input_size,
            output_size,
            rng=rng1,
        )

        std = np.sqrt(2.0 / input_size)

        expected = (
            rng2.standard_normal(
                (input_size, output_size)
            )
            * std
        )

        np.testing.assert_allclose(
            weights,
            expected,
        )


if __name__ == "__main__":
    unittest.main()

import unittest

import numpy as np

from nn import xavier_normal

class TestXavierInitializer(unittest.TestCase):
    def test_shape(self):
        rng = np.random.default_rng(42)

        weights = xavier_normal(
            3,
            4,
            rng=rng,
        )

        self.assertEqual(weights.shape, (3, 4))

    def test_same_seed_is_reproducible(self):
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        weights1 = xavier_normal(
            3,
            4,
            rng=rng1,
        )

        weights2 = xavier_normal(
            3,
            4,
            rng=rng2,
        )

        np.testing.assert_allclose(
            weights1,
            weights2,
        )

    def test_xavier_formula(self):
        input_size = 3
        output_size = 4

        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)

        weights = xavier_normal(
            input_size,
            output_size,
            rng=rng1,
        )

        std = np.sqrt(
            2.0 / (input_size + output_size)
        )

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

import unittest

from src.input_validation import normalize_whole_number


class WholeNumberValidationTests(unittest.TestCase):
    def test_rounds_half_up(self):
        self.assertEqual(
            normalize_whole_number(1.4, minimum=0, maximum=20),
            1,
        )
        self.assertEqual(
            normalize_whole_number(1.5, minimum=0, maximum=20),
            2,
        )
        self.assertEqual(
            normalize_whole_number(2.5, minimum=0, maximum=20),
            3,
        )

    def test_clamps_to_allowed_range(self):
        self.assertEqual(
            normalize_whole_number(-2, minimum=0, maximum=20),
            0,
        )
        self.assertEqual(
            normalize_whole_number(99, minimum=0, maximum=20),
            20,
        )

    def test_rejects_non_finite_values(self):
        with self.assertRaisesRegex(ValueError, "finite"):
            normalize_whole_number(float("nan"), minimum=0, maximum=20)


if __name__ == "__main__":
    unittest.main()

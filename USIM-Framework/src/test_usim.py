import unittest
from usim_metrics import usim_formula

class TestUSIM(unittest.TestCase):
    def test_usim_formula(self):
        score = usim_formula(0.18, 0.71, 1.0, 0.0011, 0.15, 4)
        self.assertAlmostEqual(score, 0.0005, places=4)

if __name__ == "__main__":
    unittest.main()

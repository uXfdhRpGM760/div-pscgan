from visualisations.charts.mse_to_gt import get_buckets, normalized_histogram
import unittest
import numpy as np

class TestBuckets(unittest.TestCase):
    def test_1000(self):
        output = get_buckets()
        self.assertEqual(output, [1, 9, 20, 220, 500, 220, 20, 9, 1])

    def test_500(self):
        output = get_buckets(buckets_number=500)
        self.assertEqual(output, [1, 5, 10, 110, 248, 110, 10, 5, 1])

    def test_100(self):
        output = get_buckets(buckets_number=100)
        self.assertEqual(output, [1, 1, 2, 22, 48, 22, 2, 1, 1])


class TestNormalizedHistogram(unittest.TestCase):
    def test_range_1000(self):
        ar = range(1000)
        buckets = get_buckets(1000)
        border_values = normalized_histogram(ar, buckets=buckets)
        self.assertEqual(border_values, [0, 1, 10, 30, 250, 749, 969, 989, 998, 999])

    def test_range_100(self):
        ar = range(100)
        buckets = get_buckets(100)
        border_values = normalized_histogram(ar, buckets=buckets)

        self.assertEqual(border_values, [0, 1, 2, 4, 26, 73, 95, 97, 98, 99])

    def test_values(self):
        ar = np.arange(0, 10, 0.1)
        np.random.shuffle(ar)
        buckets = get_buckets(100)
        border_values = normalized_histogram(ar, buckets=buckets)
        for x, y in (zip(border_values, [0., .1, .2, .4, 2.6, 7.3, 9.5, 9.7, 9.8, 9.9])):
            self.assertAlmostEqual(x, y)


if __name__ == '__main__':
    unittest.main()
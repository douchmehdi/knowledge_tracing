import numpy as np
from knowledge_tracing.utils.our_queue import OurQueue


def test_our_queue():

    queue = OurQueue(window_lengths=[10, 20, 30, np.inf])
    queue.push(0)

    counters = queue.get_counters(5)
    expected_counters = [1, 1, 1, 1]
    np.testing.assert_array_equal(counters, expected_counters)

    counters = queue.get_counters(15)
    expected_counters = [0, 1, 1, 1]
    np.testing.assert_array_equal(counters, expected_counters)

    counters = queue.get_counters(25)
    expected_counters = [0, 0, 1, 1]
    np.testing.assert_array_equal(counters, expected_counters)

    counters = queue.get_counters(35)
    expected_counters = [0, 0, 0, 1]
    np.testing.assert_array_equal(counters, expected_counters)

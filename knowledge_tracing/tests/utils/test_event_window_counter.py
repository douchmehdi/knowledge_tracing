import numpy as np
from knowledge_tracing.utils.event_window_counter import EventWindowCounter


def test_our_queue():

    queue = EventWindowCounter(window_lengths=[10, 20, 30, np.inf])
    queue.push(0)

    counters = queue.get_counters(0)
    expected_counters = [0, 0, 0, 0]
    np.testing.assert_array_equal(counters, expected_counters)

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

    counters = queue.get_counters(15)
    expected_counters = [0, 1, 1, 1]
    np.testing.assert_array_equal(counters, expected_counters)

    queue.push(0)
    counters = queue.get_counters(15)
    expected_counters = [0, 2, 2, 2]
    np.testing.assert_array_equal(counters, expected_counters)

    counters = queue.get_counters(0)
    expected_counters = [0, 0, 0, 0]
    np.testing.assert_array_equal(counters, expected_counters)

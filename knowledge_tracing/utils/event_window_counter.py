from collections import deque
import numpy as np


class EventWindowCounter:
    """
    A queue for counting efficiently the number of events within time windows.
    It keeps event ordered, and move a pair of inf_cursor/sup_cursor to isolate
    events that correspond to a given time window.

    Complexity: All operators in amortized O(W) time where W is the number of windows.

    Parameters:
    -----------
    window_lengths: int[]
        durations of time windows

    """

    def __init__(
        self, window_lengths=[np.inf, 3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
    ):
        self.queue = deque([-np.inf])
        self.window_lengths = window_lengths
        self.nb_windows = len(window_lengths)
        self.cursors_inf = [0] * len(self.window_lengths)
        self.cursors_sup = [0] * len(self.window_lengths)

    def __len__(self):
        return len(self.queue) - 1

    def get_counters(self, time):
        """ Returns the counters of past events in the differents time window preceeding time """
        self.update_cursors(time)
        return [
            c_sup - c_inf for c_sup, c_inf in zip(self.cursors_sup, self.cursors_inf)
        ]

    def insert(self, time):
        i = len(self)
        while i > 0 and time <= self.queue[i - 1]:
            i -= 1
        self.queue.insert(i, time)

    def push(self, time):
        """ If time already ordered, append it for efficiency, else insert it """
        if time > self.queue[-1]:
            self.queue.append(time)
        else:
            self.insert(time)

    def _update_idx(self, time, idx):
        """ Moves idx, so that new_idx = min_idx(time > self.queue[idx]) """
        # move idx back
        while (idx > 0) and (time <= self.queue[idx]):
            idx -= 1
        # move idx forth
        while (idx + 1) < len(self.queue) and (time > self.queue[idx + 1]):
            idx += 1
        return idx

    def update_cursors(self, time):
        """ Updates cursors to isolate events in each time window preceeding time """
        for pos, length in enumerate(self.window_lengths):
            self.cursors_inf[pos] = self._update_idx(
                time - length, self.cursors_inf[pos]
            )
            self.cursors_sup[pos] = self._update_idx(time, self.cursors_sup[pos])

from collections import deque
import numpy as np


class OurQueue:
    """
    A queue for counting efficiently the number of events within time windows.
    Complexity: All operators in amortized O(W) time where W is the number of windows.

    Warning: Should be fed

    Parameters:
    -----------
    window_lengths: int[]
    include_all: bool

    """

    def __init__(
        self, window_lengths=[np.inf, 3600 * 24 * 30, 3600 * 24 * 7, 3600 * 24, 3600]
    ):
        self.queue = deque([-np.inf])
        self.window_lengths = window_lengths
        self.cursors = [0] * len(self.window_lengths)
        self.nb_windows = len(window_lengths)
        self.last_time = -np.inf

    def __len__(self):
        return len(self.queue) - 1

    def get_counters(self, time):
        self.update_cursors(time)
        return [len(self.queue) - cursor for cursor in self.cursors]

    def push(self, time):
        if time > self.queue[-1]:
            self.queue.append(time)
        else:
            raise ValueError("time should be greater than previously pushed times")

    def update_cursors(self, time):
        if time < self.last_time:
            raise ValueError()

        for pos, length in enumerate(self.window_lengths):
            while (
                self.cursors[pos] < len(self.queue)
                and time - self.queue[self.cursors[pos]] >= length
            ):
                self.cursors[pos] += 1

    # def insert(self, time):
    #     i = len(self)
    #     while(i > 0 and time <= self.queue[i - 1]):
    #         i -= 1
    #     self.queue.insert(i, time)

    # def update_cursors(self, time):
    #     for pos, length in enumerate(self.window_lengths):

    #         if (time - self.queue[self.cursors[pos]]) > length:

    #             while (
    #                 self.cursors[pos] < len(self.queue) and
    #                 time - self.queue[self.cursors[pos]] >= length
    #             ):
    #                 self.cursors[pos] += 1

    #         else:

    #         if (self.cursors[pos] < len(self.queue)) and (time - self.queue[self.cursors[pos]] >= length):
    #             while (self.cursors[pos] < len(self.queue) and
    #                    time - self.queue[self.cursors[pos]] >= length):
    #                 self.cursors[pos] += 1
    #         else:
    #             while (self.cursors[pos] > 0 and
    #                    time - self.queue[self.cursors[pos]] < length):
    #                 self.cursors[pos] -= 1

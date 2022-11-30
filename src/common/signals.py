"""
Extremely simple signals/slots implementation for multiprocessing
information sharing
"""

import torch.multiprocessing as mp


class Slot:
    def __init__(self, callback=None):
        self._queue = mp.Queue()

    def has_value(self) -> bool:
        return not self._queue.empty()

    def get_value(self):
        if not self.has_value():
            return None
        return self._queue.get()

    def _insert(self, value):
        self._queue.put(value)


class Signal:
    def __init__(self):
        self._slots = []

    def flush(self):
        warned = False
        for s in self._slots:
            while not s._queue.empty():
                if not warned:
                    print(
                        "Warning: Leftover Items in Queue. This may or may not be an issue.")
                    warned = True
                s._queue.get()

    def register(self) -> Slot:
        self._slots.append(Slot())
        return self._slots[-1]

    def emit(self, value):
        for s in self._slots:
            s._insert(value)

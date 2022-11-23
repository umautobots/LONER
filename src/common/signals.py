"""
Extremely simple signals/slots implementation for multiprocessing
information sharing
"""

import torch.multiprocessing as mp


class Slot:
    def __init__(self, callback=None):
        self._queue = mp.Queue()

    def HasValue(self) -> bool:
        return not self._queue.empty()

    def GetValue(self):
        if not self.HasValue():
            return None
        return self._queue.get()

    def insert(self, value):
        self._queue.put(value)

class Signal:
    def __init__(self):
        self._slots = []

    def Flush(self):
        warned = False
        for s in self._slots:
            while not s._queue.empty():
                if not warned:
                    print("Warning: Leftover Items in Queue. This may or may not be an issue.")
                    warned = True
                s._queue.get()

    def Register(self) -> Slot:
        self._slots.append(Slot())
        return self._slots[-1]

    def Emit(self, value):
        for s in self._slots:
            s.insert(value)
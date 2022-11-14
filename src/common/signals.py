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

    def Register(self) -> Slot:
        self._slots.append(Slot())
        return self._slots[-1]

    def Emit(self, value):
        for s in self._slots:
            s.insert(value)

    def Empty(self):
        for s in self._slots:
            if s.HasValue():
                return False
        return True
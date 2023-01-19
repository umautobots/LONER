"""
Extremely simple signals/slots implementation for multiprocessing
information sharing
"""

import time
import torch.multiprocessing as mp

class StopSignal:
    """ Dummy class used to signal processes to stop by inserting this into MP queues.    
    """
    pass


class Slot:
    """ A Slot is a listener which listens to data on a particular signal.

    This is analogous to a subscriber in ROS
    """

    # This should not be called directly. Instead, call Signal.register
    def __init__(self):
        # The use of Manager().Queue() instead of mp.Queue() here is quite important.
        # It ensures inserts from multiple processes maintain the correct order.
        # I don't understand the details.
        # If this changes in the future, you'll also likely need to add back in
        # a call to ClonerSlam.cleanup() at the end to prevent deadlock waiting for
        # the queue to empty.
        self._queue = mp.Manager().Queue()

    # Checks whether a value is available
    def has_value(self) -> bool:
        return not self._queue.empty()

    # Returns a value if available, and otherwise None
    def get_value(self):
        if not self.has_value():
            return None
        return self._queue.get()

    # Used by Signal to send data. Don't call directly.
    def _insert(self, value):
        self._queue.put(value)


class Signal:
    """ A Signal defines a channel for communication.

    A Signal object is analogous to a topic and publisher in ROS, all in one.

    Calling @m register returns a slot, which functions as a subscriber.
    """

    # Constructor: An empty signal is just an empty list of slots
    # If @p synchronous is True, then emit will block until each item has been removed
    def __init__(self, synchronous = False):

        # Stores Slot objects to write to when data is emitted
        self._slots = []

        self._synchronous = synchronous

    # Removes all leftover items from the queue.
    # This is important if you want your code to terminate properly.
    def flush(self):
        warned = False
        for s in self._slots:
            while not s._queue.empty():
                if not warned:
                    print(
                        "Warning: Leftover Items in Queue. This may or may not be an issue.")
                    warned = True
                s._queue.get()

    # Creates and returns a Slot which listens on the Signal
    # If synchronous is True, this 
    def register(self) -> Slot:
        self._slots.append(Slot())
        return self._slots[-1]

    # Sends the given value to all the registered Slots
    def emit(self, value) -> None:
        for s in self._slots:
            while self._synchronous and s.has_value():
                time.sleep(1e-5)
            s._insert(value)
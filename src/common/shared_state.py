import torch.multiprocessing as mp

class SharedState:
    def __init__(self):
        self.last_mapped_frame_time = mp.Value('d', 0.)

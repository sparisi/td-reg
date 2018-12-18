from collections import namedtuple
import random

# Taken from
# https://github.com/pytorch/tutorials/blob/master/Reinforcement%20(Q-)Learning%20with%20PyTorch.ipynb

Transition = namedtuple('Transition', ('state', 'action', 'mask', 'next_state',
                                       'reward'))


class Memory(object):
    def __init__(self, capacity=None):
        self.capacity = capacity
        self.memory = []

    def push(self, *args):
        """Saves a transition."""
        self.memory.append(Transition(*args))
        if self.capacity is not None and len(self.memory) > self.capacity:
            self.memory = self.memory[1:]

    def sample(self, batch_size=None):
        if batch_size is None:
            return Transition(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Transition(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory
        if self.capacity is not None and len(self.memory) > self.capacity:
            self.memory = self.memory[-self.capacity:]

    def __len__(self):
        return len(self.memory)

import numpy as np
from abc import abstractmethod


class Buffer:
    def __init__(self, observation_size, action_size, buffer_size):
        # Variables to define state of the buffer
        self.max_size = buffer_size
        self.current_size = 0

        # Create buffers to record transitions
        #
        # s         : Initial state of the transition
        # a         : Action taken
        # r         : reward from the action
        # s_prime   : Resulting state after the action
        # d         : If s' is a terminal state (d = 1 if done, 0 otherwise)
        self.buffer_names = ["s", "a", "r", "s_prime", "d"]
        self.s = np.zeros((buffer_size, observation_size))
        self.a = np.zeros((buffer_size, action_size))
        self.r = np.zeros(buffer_size)
        self.s_prime = np.zeros((buffer_size, observation_size))
        self.d = np.zeros(buffer_size)

    @abstractmethod
    def store(self, transition):
        raise NotImplementedError()

    def __iter__(self):
        for buffer_name in self.buffer_names:
            yield (buffer_name, getattr(self, buffer_name))

    def sample(self, batch_size):
        idxs = np.random.choice(self.current_size, size=batch_size, replace=False)
        return {k: v[idxs] for k, v in self}


class FifoBuffer(Buffer):
    """
    First in first out buffer
    """

    def __init__(self, observation_size, action_size, buffer_size):
        super().__init__(observation_size, action_size, buffer_size)
        self.buffer_index = 0

    def store(self, transition):
        if isinstance(transition, dict):
            # Store dictionary
            assert list(transition) == list(self.buffer_names)
            for buffer_name, buffer in self:
                buffer[self.buffer_index] = transition[buffer_name]
        elif isinstance(transition, (tuple, list)):
            # Store list, tuple
            assert len(transition) == len(self.buffer_names)
            for buffer_name, value in zip(self.buffer_names, transition):
                getattr(self, buffer_name)[self.buffer_index] = value
        self.buffer_index = (self.buffer_index + 1) % self.max_size
        self.current_size = min(self.current_size + 1, self.max_size)


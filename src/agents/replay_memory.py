#  Authors:
#  François Marelli
#  Angel Martínez-González
#  Bastian Schnell 

from collections import namedtuple

import random
import torch

"""
Base implementation from:
https://pytorch.org/tutorials/intermediate/reinforcement_q_learning.html
"""

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class BaseMemory:
    def push(self, *args):
        raise NotImplementedError()

    def sample(self, batch_size):
        raise NotImplementedError()

    def sample_batch(self, batch_size):
        raise NotImplementedError()

    def push_episode(self):
        pass

    def __len__(self):
        raise NotImplementedError()


class ReplayMemory(BaseMemory):
    """
    Experience replay memory which saves tuples that map (state, action) pairs to their (next_state, reward) result.
    """

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.position = 0

    def push(self, *args):
        """Saves a transition."""
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = Transition(*args)
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        sampled_episodes = random.sample(self.memory, batch_size)

        return sampled_episodes

    def sample_batch(self, batch_size):
        transitions = self.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        next_state_batch = torch.cat(batch.next_state)
        reward_batch = torch.cat(batch.reward)

        return Transition(state_batch, action_batch, next_state_batch, reward_batch)

    def __len__(self):
        return len(self.memory)


class ContextReplayMemory(BaseMemory):
    """
    Experience replay memory which saves tuples that map (state, action) pairs to their (next_state, reward) result.
    """

    def __init__(self, episode_capacity, default_trace_length):
        self.episode_capacity = episode_capacity
        self.memory = []
        self.current_episode = []
        self.current_episode_number = 0
        self.trace_length = default_trace_length

        #  HACK for verbose in agent
        self.capacity = self.trace_length

    def push_episode(self):
        """Saves an episode in memory."""
        if len(self.memory) < self.episode_capacity:
            self.memory.append(self.current_episode)
        else:
            self.memory[self.current_episode_number] = self.current_episode

        self.current_episode = []
        self.current_episode_number = (self.current_episode_number + 1) % self.episode_capacity

    def push(self, *args):
        """Saves a transition."""
        self.current_episode.append(Transition(*args))

    def sample(self, batch_size):
        batch = []

        for _ in range(batch_size):
            # randomly select one episode in the stored episodes
            episode = random.sample(self.memory, 1)
            episode = episode[0]
            # it should contain at least as many samples as the trace asked
            assert len(episode) >= self.trace_length

            # randomly select a sample and put its trace in the batch
            sample = random.randint(self.trace_length, len(episode))

            trace = Transition(*zip(*episode[sample - self.trace_length: sample]))

            batch.append(trace)

        return batch

    @staticmethod
    def make_batch(tensors):
        return torch.stack(tuple(map(torch.cat, tensors)))

    def sample_batch(self, batch_size):
        transitions = self.sample(batch_size)
        batch = Transition(*zip(*transitions))

        state_batch = self.make_batch(batch.state)
        action_batch = self.make_batch(batch.action)
        next_state_batch = self.make_batch(batch.next_state)
        reward_batch = self.make_batch(batch.reward)

        return Transition(state_batch, action_batch, next_state_batch, reward_batch)

    def __len__(self):
        if len(self.memory) > 0:
            lengths = map(len, self.memory)
            return sum(lengths)
        else:
            return 0

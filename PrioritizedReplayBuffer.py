from collections import deque
import numpy as np
import random

class PrioritizedReplayBuffer():
    def __init__(self, maxlen,n_actions):
        self.buffer = deque(maxlen=maxlen)
        self.priorities = deque(maxlen=maxlen)
        self.n_actions = n_actions


        self.mem_cntr = 0

    def add(self, experience):

        actions = np.zeros(self.n_actions)
        actions[experience[1]] = 1.0

        experience = list(experience)
        experience[1] = actions

        experience = tuple(experience)

        self.buffer.append(experience)
        self.priorities.append(max(self.priorities, default=1))

        self.mem_cntr += 1

        #print(self.priorities)

    def get_probabilities(self, priority_scale):
        scaled_priorities = np.array(self.priorities) ** priority_scale
        sample_probabilities = scaled_priorities / sum(scaled_priorities)
        return sample_probabilities

    def get_importance(self, probabilities):
        importance = 1/len(self.buffer) * 1/probabilities
        importance_normalized = importance / max(importance)
        return importance_normalized

    def sample(self, batch_size, priority_scale=1.0):
        sample_size = min(len(self.buffer), batch_size)
        sample_probs = self.get_probabilities(priority_scale)
        sample_indices = random.choices(range(len(self.buffer)), k=sample_size, weights=sample_probs)
        #samples = np.array(self.buffer)[sample_indices]
        samples = np.array(self.buffer, dtype=object)[sample_indices]

        importance = self.get_importance(sample_probs[sample_indices])


        return map(list, zip(*samples)), importance, sample_indices

    def set_priorities(self, indices, errors, offset=0.01):
        for i,e in zip(indices, errors):

            self.priorities[i] = abs(e) + offset


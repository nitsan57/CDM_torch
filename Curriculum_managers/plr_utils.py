import numpy as np

class PLR_REPAIRED(object):
    def __init__(self, size, env_vec_size) -> None:

        self.level_buffer = np.zeros((size, env_vec_size))
        self.scores = np.zeros(size)+1e-03
        self.curr_size = 0

    def sample_level(self):
        weights = self.scores / (self.scores.sum())
        # softmax_weights = np.exp(weights)
        # softmax_weights = softmax_weights / np.sum(softmax_weights)
        indices = list(range(len(self.scores)))
        chosen_index = np.random.choice(indices, p=weights)
        return self.level_buffer[chosen_index]


    def add_level(self, level, score_new):
        if self.curr_size < len(self.level_buffer):
            self.level_buffer[self.curr_size] = level
            self.scores[self.curr_size] = score_new
            self.curr_size +=1
        else:
            s_min_idx = np.argmin(self.scores)
            s_min = self.scores[s_min_idx]
            if s_min < score_new:
                self.scores[s_min_idx] = score_new

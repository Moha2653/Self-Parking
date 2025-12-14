import numpy as np

class EvoPolicy:
    def __init__(self, obs_dim, act_dim):
        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.w1 = None
        self.b1 = None
        self.w2 = None
        self.b2 = None

    def set_weights(self, genome):
        idx = 0

        self.w1 = genome[idx:idx + self.obs_dim * 32].reshape(self.obs_dim, 32)
        idx += self.obs_dim * 32

        self.b1 = genome[idx:idx + 32]
        idx += 32

        self.w2 = genome[idx:idx + 32 * self.act_dim].reshape(32, self.act_dim)
        idx += 32 * self.act_dim

        self.b2 = genome[idx:idx + self.act_dim]

    def act(self, obs):
        x = np.tanh(obs @ self.w1 + self.b1)
        logits = x @ self.w2 + self.b2
        return np.argmax(logits)

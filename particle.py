import numpy as np

class Particle:

    def __init__(self, pos, velo, alpha, beta, gamma, delta):
        self.pos = pos
        self.velo = velo
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.prevBest = pos # Not sure should put pos or None here
        self.informant = []

    def changeVelo(self, xloc, xbest, infbest, globbest):
        for i in range(len(self.velo)):
            # Generate rand num from beta, gamma and delta
            b = np.random.rand() * self.beta
            c = np.random.rand() * self.gamma
            d = np.random.rand() * self.delta
            self.velo[i] = self.alpha * self.velo[i] + b * (xbest[i] - xloc[i]) + c * (infbest[i] - xloc[i]) + d * (globbest[i] - xloc[i])
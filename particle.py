import numpy as np

class Particle:

    def __init__(self, alpha, beta, gamma, delta, epsilon, dimensions = 1, pos = 0, velo = 0):
        self.dimensions = len(pos)
        self.pos = pos
        self.velo = velo
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon

        if pos == 0:
            pos = [0] * dimensions

            for i in len(range(pos)):
                pos[i] = np.random.uniform()
        
        if velo == 0:
            velo = [0] * dimensions

            for i in range(dimensions):
                velo[i] = np.random.uniform()

    def changeVelo(self, xloc, xbest, infbest, globbest):

        # Generate rand num from beta, gamma and delta
        a = self.alpha
        b = self.beta
        c = self.gamma
        d = self.delta

        self.velo = a * self.velo + b * (xbest - xloc) + c * (infbest - xloc) + d * (globbest - xloc)

    def AssessFitness():
        return
class Particle:

    def __init__(self, pos, velo):
        self.pos = pos
        self.velo = velo

    def changeVelo(self, xloc, alpha, beta, gamma, delta, xbest, infbest, globbest):

        # Generate rand num from beta, gamma and delta
        b = beta
        c = gamma
        d = delta

        self.velo = alpha * self.velo + b * (xbest - xloc) + c * (infbest - xloc) + d * (globbest - xloc)
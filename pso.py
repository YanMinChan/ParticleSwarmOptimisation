# PSO class to train weight
# The inputLen is the num of attribute of the cement: 8
import random
import particle
import numpy as np

class PSO:
    def __init__(self, X, y, network, swarmsize, alpha, beta, gamma, delta, epsilon, nIter,prints=True):
        self.X = X
        self.y = y
        self.swarmsize = swarmsize
        self.network = network
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.epsilon = epsilon
        self.particles = None
        self.nIter = nIter
        self.prints = prints

    def particleDim(self):
        dim = 0
        inputLen = len(self.X.columns)
        for i in range(len(self.network.layers)):
            dim = dim + ((self.network.layers[i].numOfNodes * inputLen) + self.network.layers[i].numOfNodes) # plus 1 for bias
            inputLen = self.network.layers[i].numOfNodes
        return dim
    
    # Array of random particles
    def randParticle(self):
        self.particles = []
        for i in range(self.swarmsize):
            #np.random.seed(0) # fix the random for easier checking
            pos = np.random.rand(self.particleDim())
            velo = np.random.rand(self.particleDim())
            self.particles.append(particle.Particle(pos, velo, self.alpha, self.beta, self.gamma, self.delta)) # Try to add rand particle here
        return self.particles
    
    # Randomly assign informants to particle
    def randAssignInformant(self, numOfInformant):
        for particle in self.particles:
            other_particles = self.particles.copy()
            other_particles.remove(particle)
            for i in range(numOfInformant):
                particle.informant = random.sample(other_particles, numOfInformant)

    # Return the fittest location of all particles given
    def fittestLoc(self, someParticles):
        fittest_pos = None
        fittest_mae = None
        for p in someParticles: # supposed to be particle instead of list
            mae = self.assessFitness(p.pos, self.y)
            if fittest_mae == None or mae < fittest_mae:
                fittest_mae = mae
                fittest_pos = p.pos
        return fittest_pos

    # return the nice weight and bias matrix
    def assessFitness_helper(self, pos):
        # Put the weights into a nice matrix
        weights_arr = []
        inputLen = len(self.X.columns)
        temp_inputLen = inputLen
        j = 0
        for i in range(len(self.network.layers)):
            weight_mat = np.reshape(pos[j:j + (self.network.layers[i].numOfNodes * temp_inputLen)], (self.network.layers[i].numOfNodes, -1))
            weights_arr.append(weight_mat)
            j = j + (self.network.layers[i].numOfNodes * temp_inputLen)
            temp_inputLen = self.network.layers[i].numOfNodes

        # Put the bias into a nice matrix
        bias_arr = []
        for i in range(len(self.network.layers)):
            bias_arr.append(pos[j:j + self.network.layers[i].numOfNodes])
            j = j + self.network.layers[i].numOfNodes

        return weights_arr, bias_arr

    # Check fitness of the particle
    # Also saves new best to the particle
    def assessFitness(self, pos, y):
        # Calculate sse of current pos
        weights_arr, bias_arr = self.assessFitness_helper(pos)
        
        # Calculate the pred y
        yhat = self.X.apply(self.network.forwardCalculation, args = (weights_arr, bias_arr), axis = 1)
        mae = self.network.errorCalculation(yhat, y)

        return mae
    
    # (Additional) Calculate sse of prev best pos and compare
    def isNewBest(self, mae, particle):
        
        prevBest = particle.prevBest
        mae_prev = self.assessFitness(prevBest, self.y)

        if mae < mae_prev:
            particle.prevBest = particle.pos

    # Find a best particle (weight) and return it
    def optimise(self):
        inputLen = len(self.X.columns)
        # Initialize rand particle and global best
        self.randParticle()
        self.randAssignInformant(6)
        best_pos = None
        best_mae = None
        best_mae_arr = []

        # There should be another for loop here covering all the steps until we reach an optimal best
        #while (self.assessFitness(best.pos) >= 1000):
        for i in range(self.nIter):
            # Access fitness of each particle (set of weights)
            for particle in self.particles:
                mae = self.assessFitness(particle.pos, self.y)
                self.isNewBest(mae, particle)
                if best_mae == None or mae < best_mae:
                    best_pos = particle.pos
                    best_mae = mae
                    best_mae_arr.append(best_mae)
                    if self.prints == True:
                        print("Current best mae:", best_mae)
                
            # Gather information
            for particle in self.particles:
                x_star = particle.prevBest
                x_plus = self.fittestLoc(particle.informant)
                x_prime = self.fittestLoc(self.particles)

                # Do the change velo
                particle.changeVelo(particle.pos, x_star, x_plus, x_prime)
            
            for particle in self.particles:
                # Do the change pos
                particle.pos = particle.pos + (self.epsilon * particle.velo)
        if self.prints == True:        
            print("Final best mae", best_mae)
        return best_pos, best_mae_arr
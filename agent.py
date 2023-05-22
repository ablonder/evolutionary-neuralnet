# agent.py
# The agent that operates in the world. It decides which action to take using a continuous-time recurrent neural network
# Aviva Blonder

import numpy as np

class Agent:

    """
    Translates the genome (a tuple of numpy arrays) split into weights and neurons with set parameters.
    Initializes the random seed.
    Initializes locaiton and score to 0 for the benefit of the evolutionary algorithm.
    """
    def __init__(self, genome, seed):
        # save the seed
        self.seed = seed
        # an array of weight matricies alternating between connections within layers and between layers
        self.weights = genome[0]
        # an array of matricies of the time constants, gains, and biases of each layer (in that order)
        # all have to be the same size, so differences in the actual size of layers are represented by 0s on the end
        self.nparams = genome[1]
        # creates an array of arrays of the activations of each layer initialized to 0 (if this doesn't work, I can switch it to random values)
        self.activation = np.zeros((self.nparams.shape[0], self.nparams.shape[1]))
        # also saves the genome as a whole for use in the evolutionary algorithm
        self.genome = genome
        # initialize location and score
        self.location = 0 # center of the agent's sensors in the world
        self.score = 0


    """
    Run the sensor readings through the network and use softmax to choose an action.
    """
    def getAction(self, sensors):
        # loop through each layer to feed activation forward
        for l in range(self.nparams.shape[0]):
            # if this is the input layer, do things a little differently
            if l == 0:
                self.activation[0, :] += (self.nparams[0, :, 2] + np.dot(self.sigmoid(0), self.weights[0, :, :]) - self.activation[0, :])/abs(self.nparams[0, :, 0])
                self.activation[0, :len(sensors)] += sensors/self.nparams[0, :len(sensors), 0]
            else:
                self.activation[l, :] += np.nan_to_num((self.nparams[l, :, 2] + np.dot(self.sigmoid(l-1), self.weights[(l*2)-1, :, :]) +
                                          np.dot(self.sigmoid(l), self.weights[l*2, :, :]) - self.activation[l, :])/abs(self.nparams[l, :, 0]))
            # either way, make sure the activation is between -10 and 10
            self.activation[l, :] = np.minimum(self.activation[l, :], 10)
            self.activation[l, :] = np.maximum(self.activation[l, :], -10)
        # take a softmax over the activations of the first two neurons in the final layer
        softmax = np.exp(self.activation[-1, :2])/np.sum(np.exp(self.activation[-1, :2]))
        # use that to choose an action
        self.location += np.random.choice([-1, 1], p = softmax)


    """
    Helper function to more efficiently implement the sigmoid functon for a given layer.
    """
    def sigmoid(self, l):
        return np.nan_to_num(1/(1+np.exp(-self.activation[l, :]*abs(self.nparams[l, :, 1]))))


    """
    Returns the agent's score so I can print out the population list.
    """
    def __repr__(self):
        return str(self.score)

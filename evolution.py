# evolution.py
# An evolutionary algorithm for selecting neural network agents that perform well in an environment.
# Aviva Blonder

import numpy as np
import matplotlib.pyplot as plt
from agent import Agent
from world import World
import time

class Evol:

    """
    Initializes the population and parameters for the evolutionary algorithm.
    """
    def __init__(self, netstruct, popsize, mutation, survival, crossover, seed):
        self.population = [] # currently alive agents
        self.survivalRate = survival # number of agents in the population that contribute to the next generation
        self.mutationRate = mutation # number of mutations per genome
        self.crossoverRate = crossover # number of crossover sites per genome
        self.seed = seed # random seed
        # seed random
        np.random.seed(seed)

        # grab the size of the largest layer
        size = np.amax(netstruct)
        # generate the starting population of agents
        for i in range(popsize):
            # start with the baseline genome of zeros
            weights = np.zeros(((len(netstruct)*2)-1, size, size))
            params = np.zeros((len(netstruct), size, 3))
            # draw the appropriate number of random values for the parameters of each layer based on its size
            for l in range(len(netstruct)):
                weights[l*2, :netstruct[l], :netstruct[l]] = np.random.uniform(low = -1, size = (1, netstruct[l], netstruct[l]))
                if l > 0:
                    weights[(l*2)-1, :netstruct[l-1], :netstruct[l]] = np.random.uniform(low = -1, size = (1, netstruct[l-1], netstruct[l]))
                params[l, :netstruct[l], :] = np.random.uniform(low = -1, size = (1, netstruct[l], 3))
            # generate the agent
            a = Agent((weights, params), seed)
            # add it to the list of agents and the current population
            self.population.append(a)


    """
    Evolve the population for the provided number of generations in a world with the given parameters.
    """
    def evolve(self, objInt, ticks, generations, plot = False, display = False):
        np.random.seed(self.seed)

        scores = np.zeros((generations, len(self.population))) # scores of each agent in each generation
        
        for g in range(generations):
            print(g)
            # run the world for the given number of ticks on each agent
            for a in range(len(self.population)):
                w = World(self.population[a], objInt, self.seed)
                for t in range(ticks):
                    w.tick()
                # add the agent's score to the array of scores
                scores[g, a] = self.population[a].score

            # sort agents by score
            self.population.sort(key = lambda a: -a.score)
            if g < generations-1:
                # an empty list to contain the new agents
                newpop = []
                # create the same number of new agents
                for i in range(len(self.population)):
                    # choose two random agents to be the new agent's parents
                    parents = np.random.choice(self.population[:self.survivalRate], 2)
                    # grab their weights and flatten them
                    parentweights = [parents[0].weights.flatten(), parents[1].weights.flatten()]
                    parentparams = [parents[0].nparams.flatten(), parents[1].nparams.flatten()]
                    # create an empty set of weights and an empty set of parameters
                    weights = np.zeros(len(parentweights[1]))
                    params = np.zeros(len(parentparams[1]))
                    # starting indices for crossover
                    prevw = 0
                    prevp = 0
                    # fill them with the individuals' parents' traits
                    for c in range(self.crossoverRate):
                        # randomly choose the index for this crossover
                        w = np.random.randint(prevw, len(weights))
                        weights[prevw:w] = parentweights[c%2][prevw:w]
                        p = np.random.randint(prevp, len(params))
                        params[prevp:p] = parentparams[c%2][prevp:p]
                        # store these indices for next round
                        prevw = w
                        prevp = p
                    # end off each genome
                    weights[prevw:] = parentweights[self.crossoverRate%2][prevw:]
                    params[prevp:] = parentparams[self.crossoverRate%2][prevp:]
                    # introduce mutations
                    for m in range(self.mutationRate):
                        wdex = np.random.randint(len(weights))
                        # make sure mutations don't effect non-coding regions
                        if weights[wdex] != 0:
                            weights[wdex] += np.random.uniform(low = -.5, high = .5)
                        pdex = np.random.randint(len(params))
                        if params[pdex] != 0:
                            params[pdex] += np.random.uniform(low = -.5, high = .5)
                    # reshape into weight and parameter arrays
                    weights = weights.reshape(self.population[0].weights.shape)
                    params = params.reshape(self.population[0].nparams.shape)
                    # generate the offspring and store them
                    a = Agent((weights, params), self.seed)
                    newpop.append(a)
                # set the population to be the new population
                self.population = newpop

        # plot the change in score over generations
        if plot:
            plt.scatter(np.repeat(np.arange(generations), len(self.population)), scores.flatten())
            plt.plot(np.arange(generations), np.mean(scores, axis = 1))
            plt.title("Evolution in a Minimally Cognitive Task")
            plt.xlabel("Generation")
            plt.ylabel("Score")
            plt.show()

        # display one run with the agent that performed the best in the last generation
        if display:
            # create an agent with the same genome as the one that performed the best in the last generation
            a = Agent(self.population[0].genome, self.seed)
            # and put it in a new world
            w = World(a, objInt, self.seed)
            # create the image
            f = plt.figure()
            ax = f.gca()
            f.show()
            for i in range(display):
                ax.imshow(w.tick(True), cmap="gray")
                plt.title("Score: " + str(a.score))
                f.canvas.draw()
                time.sleep(.1)
        
        return np.mean(scores)


"""
Runs the evolutionary algorithm for 50 generations, with a population of 50 individuals and what I found to be the optimal evolutionary parameters.
It displays a plot of the agents' performance in each generation and a visual representation of 50 ticks of the best performing agent in the last generation. 
"""
def main(s):
    e = Evol([5, 3, 2], popsize = 50, mutation = 2, survival = 4, crossover = 5, seed = s)
    e.evolve(5, 50, 50, plot = True, display = 50)

main(1)

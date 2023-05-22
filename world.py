# world.py
# The grid world that the agent inhabits, consisting of the agent and objects falling from the sky.
# Aviva Blonder

import random
import numpy as np
import agent

class World:

    """
    Constructor to initialize class variables
    """
    def __init__(self, agent, objInt, seed, height = 15, width = 15):
        self.agent = agent
        self.height = height
        self.width = width
        self.step = 0
        self.newObjInt = objInt # how many steps between creating new objects
        self.grid = np.zeros((height, width)) # the grid world itself to be populated with objects and the agent
        

    """
    Run the world for one step.
    """
    def tick(self, display = False):
        # store the agent's location
        agentloc = ((self.agent.location-2)%self.width, (self.agent.location+2)%self.width)
        # check to see if anyting in the bottom row of the grid hits the agent and adjust its score accordingly
        if agentloc[0] > agentloc[1]:
            catch = np.append(self.grid[0, agentloc[0]:], self.grid[0, :agentloc[1]])
        else:
            catch = self.grid[0, agentloc[0]:agentloc[1]]
        # if there is an object there, determine how much of it is caught
        self.agent.score += np.sum(catch)

        # move the rest of the grid down one
        self.grid[:-1, :] = self.grid[1:, :]
        # empty out the top row
        self.grid[-1, :] = 0
        
        # if it's time to create a new object, do that
        if self.step%self.newObjInt == 0:
            # set the object's size
            size = 4
            # and its horizontal location
            hloc = 5
            # use its size and location to determine its full horizontal location and add it to the grid
            if hloc+size < self.width:
                self.grid[self.height-1, hloc:hloc+size] = 1
            # if it wraps around, take that into account too
            else:
                self.grid[self.height-1, hloc:] = 1
                self.grid[self.height-1, :(hloc+size)%self.width] = 1

        # give the agent its sensor values so it can move
        if agentloc[0] > agentloc[1]:
            sensors = np.append(np.sum(self.grid[:, agentloc[0]:], axis = 0), np.sum(self.grid[:, :agentloc[1]], axis = 0))
        else:
            sensors = np.sum(self.grid[:, agentloc[0]:agentloc[1]], axis = 0)
        self.agent.getAction(sensors)

        # increment step
        self.step += 1

        # display the grid and agent
        if display:
            # create a new line with the agent's location as 1s
            aline = np.zeros(self.width)
            if agentloc[0] > agentloc[1]:
                aline[agentloc[0]:] = 1
                aline[:agentloc[1]] = 1
            else:
                aline[agentloc[0]:agentloc[1]] = 1
            return np.append(aline.reshape(1, self.width), self.grid, axis = 0)

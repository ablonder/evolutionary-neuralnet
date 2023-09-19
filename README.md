<img src="https://github.com/ablonder/evolutionary-neuralnet/blob/master/Plots/demo_display.png" />

# Evolving to Play a Simple Game Isn't So Simple

You've probably played a game like this before: the extra bright bar at the top is the "player" and all it has to do is to move from side to side to catch the thinner bars that are falling (or in this case rising) toward it. The catch is that in this game, the "player" isn't a player at all but a neural network that recieves the state of its environment and uses that to determine a course of action.

To complicate things further, these neural networks can accumulate information over time (through recurrence), but they can't learn. Instead, I created an entire population of neural networks and evolved them over the course of several generations with selection based on how well they performed at this task (you can think of it like gathering food). Between generations, they also experienced mutation in the form of random changes to their weights, and crossing over to enable the most successful "players" to share their strategies.

This was enough to cause a substantial improvement in performance, but there's still a long ways to go, and this shows just how tricky it is to evolve a functioning neural network.

You can see a complete description of the model and the results in Writeup.pdf.

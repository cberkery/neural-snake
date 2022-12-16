# neural-snake
Framework for training neural networks using an evolutionary algorithm - applied to Snake.


# How to run
pip install -r requirements.txt (Or poetry install)

Run.py will initialise a population of Snake agents for a specified number of iterations. There are some optional commandline
arguments for setting the high-level parameters of the evolution:

usage: Run.py [-h] [-p PopulationSize] [-i Iterations] [-m MuatationRate] [-s SelectionProportion]

Defaults:
PopulationSize = 100
Iterations = 100 
MutationRate = 0.01
SelectionProportion = 0.01

# Snake
Like the one you played on some relative's Nokia in the backseat of a car.



# Module structure (Is definitely a WIP)
The base class is Snake_computation which handles initialisation and propogation of the neural network. In addition there are methods for 
converting the game state into the network input.


# Key areas for improvement and experimentation



# How to have fun
My (potentially) hot-take on Evolutionary and Reinforcement algorithms in the context of personal projects is to begin with a random and intuition based parameter search.
The most fascinating part of meta-heuristic algorithms like Evolutionary/Genetic algorithms for me is their derivation from and emergence in nature. 
Before you seek an optimal convergence rate or end solution, I recommend playing with the key evolutionary parameters like mutation_rate, selection_proportion etc, and reason about 
the consequences of how you choose parents or persist best solutions through generations.

# Planned updates, extensions and fixes
- Reduce move set from 5 -> 3 (UP, DOWN, RIGHT, LEFT or CONTINUE -> RIGHT, LEFT or CONTINUE)
- Introduce biases to the network nodes (Totally forgot this somehow)
- Better visualisation of gameplay
- Refactor for future layering or other algorithms
- Reinforcement learning class
- Additional and improved cost functions for learning
- Anything else you lovely people suggest in the issues







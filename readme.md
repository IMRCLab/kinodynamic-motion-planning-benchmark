

# idbA\* or KdbA\*

Which name do you prefer? Please vote.
idbA\*: Iterative Discontinuity bounded A*
KdbA\*: Kinodynamic Discontinuity bounded A*


# Dynamical Systems

We have implemented a bunch of dynamical system. 
Each robot model provides function and derivative of the dynamics (e.g step ), distance,interpolation, cost, cost lower bound, control limitis, state constraints, collisions...

- Unicycle first order
- Unicycle second order
- ...

The parameters (e.g. max speed of a car) are loaded throuhg a configuation file (yaml).


# Problems

The problems are defined using user-friendly yaml interface. Parsing the problem description takes 2 lines of python code.


# Trajectory Optimization

- MPC
- MPCC
- Fixed time Trajectory Optimization
- Free time Trajectory Optimization
- Binary/Linear search on time

All solver are built on top of [crocoddyl] [https://github.com/loco-3d/crocoddyl] (Differential Dynamic Programming)

# dbA\*

Explain: ...

# kdbA\*

KdbA\*: itertative combination of search and

# Baseline

- Geometric RRT\* + Trajectory Optimization 
- SBPL 
- SST\*

SST\* and Geometric RRT\*  require [ompl](https://github.com/ompl/ompl)


# How to run our planner?


# Motion Primitives

Here we provide very few motion primitives -- we do not want to blow up the size of the repository.

You can access all motion primitive from our website. 

They are near-optimal trajectories. Could be useful to train a policy with supervised learning, imitation learning or to boostrap reinforcement learning algorithm. 
Also, you can use them tolearn distance and reachibility functions.


# Understanding the code

- The dynamical models are implemented in c++ and depen only on Eigen (Note: in the future, some will depend on Pinocchio). They include collision checking with FCL.
- The solver depends on crocoddyl.
- There is a ompl wrapper to use the models in the OMPL planner.
- Yaml and boost are used for setting program options, data serialization.
- There are several high-level integratation tests. They provide a good example on how to use the code and good hyperparameters.
- Visualization utils are in python, built on top of Matplotlib. You can plot a robot in an environment, display trajectories and record videos. The c++ robot models and the python viewer are connected through YAML files (not ideal, but works fine for debugging.
- There are several high-level integratation tests. They provide a good example on how to use the code and good hyperparameters.

# How to benchmark your solver in our test set?  

- You can clone our repo and compile only the dynamical models using the flag ... The only dependencies are FCL, Eigen, Boost, and YAML. 
- You can easily parse the problem description and convert to your desired format (e.g. URDF).
- You can acces all the evaluation data of our algorihtm and benchmark here.


# How to add a new dynamical system?

(lets add a cart-pole) Step by step.


# Future

- Use Pinocchio for complex dynamical model (e.g. quadcopter with a pendulum).
- Optimize quaternions using the tanget space. 
- Research Ideas: 







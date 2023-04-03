

# idbA\* or KdbA\*

Which name do you prefer? Please vote!

idbA\*: Iterative Discontinuity bounded A*
KdbA\*: Kinodynamic Discontinuity bounded A*


# Dynamical Systems

We have implemented a bunch of dynamical system. 
Each robot model provides function and derivative of the dynamics (e.g step ), distance,interpolation, cost, cost lower bound, control limits, state constraints, collisions...

- Unicycle first order
- Unicycle second order
- ...

The parameters (e.g. max speed of a car) are loaded through a configuration file (yaml).


# Problems

The problems are defined using user-friendly yaml interface. Parsing the problem description takes 2 lines of python code.


# Trajectory Optimization

- MPC
- MPCC
- Fixed time Trajectory Optimization
- Free time Trajectory Optimization
- Binary/Linear search on time

All solver are built on top of (crocoddyl)[https://github.com/loco-3d/crocoddyl] (Differential Dynamic Programming)

# dbA\*

Discontinuity bounded search.

# kdbA\*

KdbA\*: iterative combination of search and trajectory optimization

# Baselines

- Geometric RRT\* + Trajectory Optimization 
- SBPL 
- SST\*

SST\* and Geometric RRT\*  require (ompl)[https://github.com/ompl/ompl]


# How to run our planner?

It is easy!


# Motion Primitives

Here we provide very few motion primitives -- we do not want to blow up the size of the repository.

You can access all motion primitive from our website. 

They are near-optimal trajectories. Could be useful to train a policy with supervised learning, imitation learning or to bootstrap reinforcement learning algorithm. 
Also, you can use them to learn distance and reachability functions.


# Understanding the code

- The dynamical models are implemented in c++ and depend only on Eigen (Note: in the future, some will depend on Pinocchio). They include collision checking with FCL.
- The solver depends on Crocoddyl.
- There is a Ompl wrapper to use the models in the OMPL planner.
- Yaml and boost are used for setting program options, data serialization.
- There are several high-level integration tests. They provide a good example on how to use the code and how to choose good hyperparameters. We will not add comments or documentation to the code, but we are happy to help and offer support to use the repository.
- Visualization utilities  are in python, built on top of Matplotlib. You can plot a robot in an environment, display trajectories and record videos. The c++ robot models and the python viewer are connected through YAML files (not ideal, but works fine for debugging).
- There are several high-level integration tests. They provide a good example on how to use the code and good hyperparameters. If you want to add an additional model or solver, please provide also similar tests.

# How to benchmark your solver in our test set?  

- You can clone our repo and compile only the dynamical models using the flag ... The only dependencies are FCL, Eigen, Boost, and YAML. 
- You can easily parse the problem description and convert to your desired format (e.g. URDF).
- If you use OMPL, please check how we use SST\* and geometric RRT\*.
- You can access all the evaluation data of our algorithm and benchmark here.


# How to add a new dynamical system?

(lets add a cart-pole) Step by step.

# Future

- Refactor Trajectory Optimization -- fairly well tested, but monollitic function.
- Kdb\*  without OMPL -- for speed. Keeping option to use OMPL nearest neighbour structures for fair comparisson.
- Refactor Trajectory Optimization -- fairly well tested, but monollitic function.
- Use Pinocchio for complex dynamical model (e.g. quadcopter with a pendulum).
- Optimize quaternions using the tangent space. 
- Research Ideas: (TOP secret), but open to collaborations.







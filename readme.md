

# idbA\* 


# Dependencies

- Eigen (maths)
- YAML (config files)
- BOOST (test, program options, serialization)
- FCL (collision checking)
- OMPL (Baseline Planners)
- Nigh or OMPL (nearest neighbour search in our planner)




# Dynamical Systems

We have implemented a bunch of dynamical system. 
Each robot model provides function and derivative of the dynamics (e.g step ), distance,interpolation, cost, cost lower bound, control limits, state constraints, collisions...

- Unicycle first order
- Unicycle second order
- Car with a Trailer
- Planar multirotror
- Planar multirotror with Pole
- Quadrotor
- Quadrotor with Total Thrust-Moment Control

The parameters (e.g. max speed of a car) are loaded through configurations files (yaml).


# Problems

The problems are defined using user-friendly yaml interface. Parsing the problem description takes few lines of python code.


# Trajectory Optimization

- MPC
- MPCC
- Fixed time Trajectory Optimization
- Free time Trajectory Optimization
- Binary/Linear search on time

All solver are built on top of (crocoddyl)[https://github.com/loco-3d/crocoddyl] (Differential Dynamic Programming)

# dbA\*

Discontinuity bounded search.

# idbA\*

Our kynodynamic Motion Planner

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

Ty

# Todo

- Clean the CMakeLists.txt


# Roadmap

- [ ] Remove Nigh dependency and use incremental and approximate nearest neighbours kd tree.
- [ ] Remove the OMPL dependency in our planner.
- [ ] Refactor Trajectory Optimization -- currently is a monollitic function.
- [ ] Use Pinocchio for complex dynamical model (e.g. quadcopter with a pendulum).
- [ ] Use Pinochio for all models. Change from FCL to HPP-FCL.
- [ ] Optimize quaternions using the tangent space. 
- [ ] Fix memory leaks in ompl structs

# HOW TO

## Run benchmark

To run the benchmark in your computer, use:


from the build directory run:

```
python3 ../scripts/new_benchmark.py -m bench -bc ../bench/compare.yaml 
```

Results from each run are stored in, e.g.,

../results_new/quad2d/quad_obs_column/idbastar_v0/2023-07-05--11-43-02/run_1_out.yaml

You will find the files:

run_1_out.yaml.cfg.yaml
run_1_out.yaml.traj-sol.yaml
run_1_out.yaml.trajopt-0.yaml
run_1_out.yaml.trajraw-0.yaml

The summary for different run on same problem with same parameters is stored in 

report.pdf
report.yaml

The stdout and stderr of each run is stored in files:

/tmp/dbastar/stderr/stderr-ba936d2.log (ba936d2 is a random ID)
/tmp/dbastar/stdout/stdout-ba936d2.log

a Plot with all the results in the current benchmark is stored in:

../results_new/plots/plot_2023-07-05--11-55-16.pdf

Numeric results are stored in: 

../results_new/summary/summary_2023-07-05--11-55-16.csv

The results in this file can be loaded latter for creating the summary table

To create the fancy

python3 ../scripts/new_benchmark.py -m fancy -bc ../bench/selected_results.yaml 



## Run Ablation study for optimization


## Run Ablation study for heurisitic


# Tests

There are a lot of test, mainly integration tests, that show how to use our code. 








# kinodynamic-motion-planning-benchmark
This repository aims to compare different motion planners for dynamical systems, namely search-based, sampling-based, and optimization-based

## Building

```
mkdir build
cd build
cmake ..
make
```

## Planners

### ompl_rrt

* Sampling-based using OMPL
* Currently hard-coded to the mountain car example, without collision checking
* Probabilistically complete, suboptimal

### ompl_aorrt

* Sampling-based using OMPL
* Currently hard-coded to the mountain car example, without collision checking
* Implements a simplified version of AO-RRT
  * This does not prune the tree, but executes a new search
  * Currently, the weight of the cost components in the state is set to zero
* Probabilistically complete, asymptotically optimal

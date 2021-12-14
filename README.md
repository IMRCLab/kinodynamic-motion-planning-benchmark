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

### main_ompl

* Sampling-based using OMPL (main asymptotic optimal planner: SST)

## Notes

* can we find the best motions automatically using 
    a) statistical inference? (bayes learning?) -OR-
    b) using dispersion optimization -OR-
    c) graph theory (finding critical motions)
      * find histogram of motions used
      * remove one kind of motion (of the used ones) -> 1 - old cost/new cost is importance (if infeasible: new cost = inf -> importance =1; if no change in cost importance is 0)
* 
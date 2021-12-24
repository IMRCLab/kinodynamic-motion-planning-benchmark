# kinodynamic-motion-planning-benchmark
This repository aims to compare different motion planners for dynamical systems, namely search-based, sampling-based, and optimization-based

## Building

Tested on Ubuntu 20.04.

```
mkdir build
cd build
cmake -DCMAKE_BUILD_TYPE=RelWithDebugInfo ..
make
```

## Running

The following will run all planners on all environments multiple times. Results are stored in the `results` folder
```
cd build
python3 ../scripts/benchmark.py
```

The results can be visualized using

```
cd build
python3 ../scripts/benchmark_stats.py
```

## Planners

### SBPL

```
python3 ../scripts/gen_sbpl_prim.py ../deps/sbpl/matlab/mprim/unicycle_noturninplace.mprim ../tuning/carFirstOrder/car_first_order_0.mprim ../tuning/carFirstOrder/car_first_order_0_mprim.yaml
```

### main_ompl

* Sampling-based using OMPL (main asymptotic optimal planner: SST)

### KOMO


Optimize Trajectories

```
./rai_dubins -model  \"../benchmark/carFirstOrder/parallelpark_0.g\" -waypoints \"../benchmark/carFirstOrder/initGuess/result_dbastar_parallelpark.yaml\"  -one_every 2 -display 1 -out out.yaml -animate 0
```

```
./rai_dubins -model  \" ../benchmark/carFirstOrder/parallelpark_0.g \" -waypoints \" ../benchmark/carFirstOrder/initGuess/result_dbastar_parallelpark.yaml \"  -one_every 2 -display 1 -out out.yaml -animate 0
 ```

```
./rai_dubins -model  \"../benchmark/carFirstOrder/bugtrap_0.g \" -waypoints \"../benchmark/carFirstOrder/initGuess/result_dbastar_bugtrap.yaml\"  -one_every 2 -display 0 -out out.yaml -animate 0
```

Translate yaml environments to g

```
python3 translate_g.py --fin ../benchmark/carFirstOrder/parallelpark_0.yaml  --fout ../benchmark/carFirstOrder/parallelpark_0.g
```

```
python3 translate_g.py --fin ../benchmark/carFirstOrder/kink_0.yaml  --fout ../benchmark/carFirstOrder/kink_0.g
```

```
python3 translate_g.py --fin ../benchmark/carFirstOrder/bugtrap_0.yaml  --fout ../benchmark/carFirstOrder/bugtrap_0.g
```

## Notes

* can we find the best motions automatically using 
    a) statistical inference? (bayes learning?) -OR-
    b) using dispersion optimization -OR-
    c) graph theory (finding critical motions)
      * find histogram of motions used
      * remove one kind of motion (of the used ones) -> 1 - old cost/new cost is importance (if infeasible: new cost = inf -> importance =1; if no change in cost importance is 0)
* 

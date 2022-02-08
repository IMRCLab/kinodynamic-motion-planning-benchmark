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

## Tests

```
cd build
pytest ../test
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
python3 ../scripts/gen_sbpl_prim.py ../deps/sbpl/matlab/mprim/unicycle_noturninplace.mprim ../tuning/car_first_order_0/car_first_order_0.mprim ../tuning/car_first_order_0/car_first_order_0_mprim.yaml
```

### main_ompl

* Sampling-based using OMPL (main asymptotic optimal planner: SST)

### KOMO


Optimize Trajectories

```
./main_rai -model  \"../benchmark/car_first_order_0/parallelpark_0.g\" -waypoints \"../benchmark/car_first_order_0/initGuess/result_dbastar_parallelpark.yaml\"  -one_every 2 -display 1 -out out.yaml -animate 0
```

```
./main_rai -model  \" ../benchmark/car_first_order_0/parallelpark_0.g \" -waypoints \" ../benchmark/car_first_order_0/initGuess/result_dbastar_parallelpark.yaml \"  -one_every 2 -display 1 -out out.yaml -animate 0
 ```

```
./main_rai -model  \"../benchmark/car_first_order_0/bugtrap_0.g \" -waypoints \"../benchmark/car_first_order_0/initGuess/result_dbastar_bugtrap.yaml\"  -one_every 2 -display 0 -out out.yaml -animate 0
```

Translate yaml environments to g

```
python3 translate_g.py --fin ../benchmark/car_first_order_0/parallelpark_0.yaml  --fout ../benchmark/car_first_order_0/parallelpark_0.g
```

```
python3 translate_g.py --fin ../benchmark/car_first_order_0/kink_0.yaml  --fout ../benchmark/car_first_order_0/kink_0.g
```

```
python3 translate_g.py --fin ../benchmark/car_first_order_0/bugtrap_0.yaml  --fout ../benchmark/car_first_order_0/bugtrap_0.g
```

#### Issues

```
cd build
python3 ../scripts/translate_g.py --fin ../benchmark/unicycle_first_order_0/bugtrap_0.yaml  --fout env.g
```

The following fails with high eq/ineq:

```
./main_rai -model "env.g" -waypoints \"../test/unicycle_first_order_0/guess_bugtrap_0_sol1.yaml\" -one_every 1 -display 0 -animate 0 -out "komo.yaml" -order 1
```

However, the following (using sol2.yaml from a pure geometric planner) works just fine:
```
./main_rai -model "env.g" -waypoints \"../test/unicycle_first_order_0/guess_bugtrap_0_sol2.yaml\" -one_every 1 -display 0 -animate 0 -out "komo.yaml" -order 1
```

Note that T of the first solution is significantly higher than the T of the second solution.

### Profiling

```
mkdir buildProfile
cd buildProfile
cmake -DCMAKE_BUILD_TYPE=RelWithDebInfo ..
make
cd ..
perf record --call-graph dwarf <test application>
perf report --no-inline
```
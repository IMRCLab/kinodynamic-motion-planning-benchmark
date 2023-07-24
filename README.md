# kinodynamic-motion-planning-benchmark

NOTE: This repository is now Deprecated. Move to [Dynoplan](https://github.com/quimortiz/dynoplan) and [Dynobench](https://github.com/quimortiz/dynobench).

This repository compares different motion planners for dynamical systems.

The following algorithms are currently supported:

* Search-based: ARA* (using SBPL http://www.sbpl.net/)
* Sampling-based: SST* (using OMPL https://ompl.kavrakilab.org/)
* Optimization-based: KOMO (using RAI https://github.com/MarcToussaint/rai)
* Hybrid: db-A*

The following dynamical systems are currently implemented:

* Unicycle (first order)
* Unicycle (second order)
* Car with trailer (first order)
* Quadrotor

## Academic Origin

This benchmark was initially started while working on db-A*, a hybrid kinodynamic motion planner for translation-invariant dynamical systems.
When using this work in an academic setting, please cite (this paper has been submitted to IROS 2022):

```
@online{hoenigDbADiscontinuityboundedSearch2022,
  title = {Db-A*: Discontinuity-Bounded Search for Kinodynamic Mobile Robot Motion Planning},
  author = {Hoenig, Wolfgang and Ortiz-Haro, Joaquim and Toussaint, Marc},
  year = {2022},
  eprint = {2203.11108},
  eprinttype = {arxiv},
  url = {http://arxiv.org/abs/2203.11108},
  archiveprefix = {arXiv}
}
```

## Building

Tested on Ubuntu 20.04.

```
mkdir buildRelease
cd buildRelease
cmake -DCMAKE_BUILD_TYPE=Release ..
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

Edit benchmark.py and benchmark_stats.py in order to run a subset (or a single) experiment, only.

Note that some algorithms require motion primitives. These are not stored in this repository, but can be obtained pre-computed from the authors, or re-computed using the commands listed in `runall.sh`.

## Tests

```
cd build
pytest ../test
```

## Some Implementation Notes

### db-A*

db-A* requires motion primitives to operate. Currently, these primitives are generated in preprocessing, by solving many small optimal motion planning problems in free space. The resulting primitives can be sorted to maximize their exploration capabilities.

```
python3 ../scripts/gen_motion_primitive_komo.py --N 1000 unicycle_first_order_0 | grep Generated
```

### SBPL

SBPL requires motion primitives, which can be generated using the following command

```
python3 ../scripts/gen_sbpl_prim.py ../deps/sbpl/matlab/mprim/unicycle_noturninplace.mprim ../tuning/car_first_order_0/car_first_order_0.mprim ../tuning/car_first_order_0/car_first_order_0_mprim.yaml
```

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

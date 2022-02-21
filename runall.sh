python3 ../scripts/gen_motion_primitive_komo.py --N 2000 unicycle_first_order_0 | grep Generated
python3 ../scripts/gen_motion_primitive_komo.py --N 2000 unicycle_first_order_1 | grep Generated
python3 ../scripts/gen_motion_primitive_komo.py --N 2000 unicycle_first_order_2 | grep Generated
python3 ../scripts/gen_motion_primitive_komo.py --N 2000 unicycle_second_order_0 | grep Generated
python3 ../scripts/gen_motion_primitive_komo.py --N 2000 car_first_order_with_1_trailers_0 | grep Generated
python3 ../scripts/benchmark.py
python3 ../scripts/benchmark_stats.py

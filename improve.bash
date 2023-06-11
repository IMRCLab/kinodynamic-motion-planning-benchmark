

m4 main_primitives && time   ./main_primitives --mode 1   --in_file ../build/acrobot_v0_all.bin     --max_num_primitives 20   --dynamics acrobot_v0 --solver_id 9 --linear_search true --weight_goal 400 --max_iter 100 --tsearch_max_rate 1 

m4 main_primitives && ./main_primitives --mode 1   --in_file "../cloud/motionsV2/good/car1_v0/car1_v0_all.bin.sp.bin"  --max_num_primitives 20   --dynamics car1_v0 --solver_id 9 --linear_search true --weight_goal 400 --max_iter 100 --tsearch_max_rate 1 

m4 main_primitives && time   ./main_primitives --mode 1   --in_file ../build/quad3d_v0_all.bin    --max_num_primitives 200   --dynamics quad3d_v0 --solver_id 9 --linear_search true --weight_goal 300 --max_iter 100 

m4 main_primitives && time   ./main_primitives --mode 1   --in_file ../build/quad3d_v0_all3.bin    --max_num_primitives 200   --dynamics quad3d_v0 --solver_id 9 --linear_search true --weight_goal 300 --max_iter 100 












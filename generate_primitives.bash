#/bin/bash
#


time_stamp=$(date +%Y_%m_%d__%H_%M_%S)
echo $time_stamp
make main_primitives 

# Car
./main_primitives --dynamics car1_v0 --out_file  "../cloud/motionsV2/tmp_car1.bin"  --mode_gen_id 0  --max_num_primitives 100 --max_iter 200 --ref_x0 1  --ref_time_steps 50 --weight_goal 200 

# Acrobot
./main_primitives --dynamics acrobot_v0  --out_file  "../cloud/motionsV2/tmp_car1.bin"  --mode_gen_id 0  --max_num_primitives 10  --max_iter 200 --ref_x0 1  --ref_time_steps 300  --weight_goal 500 

# Quad 2d
./main_primitives  --dynamics quad2d_v0 \
                                       --out_file  "../cloud/motionsV2/tmp_quad_2d.bin" \
                                       --mode_gen_id 0 \
                                       --max_num_primitives  10  --max_iter 200 --ref_x0 1  --ref_time_steps 300 --weight_goal 400 

# Quad 3d
./main_primitives --dynamics quad3d_v0 --out_file  "../cloud/motionsV2/tmp_quad_3d.bin"  --mode_gen_id 0  --max_num_primitives 10 --max_iter 200 --ref_x0 1  --ref_time_steps 300 --weight_goal 400 





# MAKE primitives for all systems



# 


# # generate primitives
#
#
#


# ./main_primitives  \
#   --dynamics unicycle1_v0 \
#   --out_file  "../cloud/motionsV2/unicycle1_v0__${time_stamp}" \
#   --mode_gen_id 0 \
#   --max_num_primitives  10000



# ./main_primitives  \
#   --dynamics unicycle1_v0 \
#   --out_file  "../cloud/motionsV2/unicycle1_v0__${time_stamp}" \
#   --mode_gen_id 0 \
#   --max_num_primitives  10000
#
#
#
#


#
# make main_primitives  && ./main_primitives  \
#   --dynamics unicycle2_v0 \
#   --out_file  "../cloud/motionsV2/unicycle2_v0__${time_stamp}" \
#   --mode_gen_id 0 \
#   --max_num_primitives  20000


# ref=../cloud/motionsV2/good/unicycle2_v0/unicycle2_v0__2023_04_03__15_36_01.bin
# out_i=../cloud/motionsV2/good/unicycle2_v0/unicycle2_v0__i__2023_04_03__15_36_01
#
# ./main_primitives  \
#   --dynamics unicycle2_v0 \
#   --in_file $ref  \
#   --out_file $out_i \
#   --mode_gen_id 1 \
#   --solver_id 14


#
# out_i=../cloud/motionsV2/good/unicycle2_v0/unicycle2_v0__i__2023_04_03__15_36_01
# ref_sp="../cloud/motionsV2/unicycle1_v0__isp__2023_04_03__14_56_57"

# make main_primitives && ./main_primitives  \
#   --dynamics unicycle2_v0 \
#   --in_file ../cloud/motionsV2/good/unicycle2_v0/unicycle2_v0__i__2023_04_03__15_36_01.bin \
#   --out_file ../cloud/motionsV2/good/unicycle2_v0/unicycle2_v0__isp__2023_04_03__15_36_01 \
#   --mode_gen_id 2 





# ref_so="../cloud/motionsV2/unicycle1_v0__ispso__2023_04_03__14_56_57n"

# ref_sp="../cloud/motionsV2/good/unicycle2_v0/unicycle2_v0__isp__2023_04_03__15_36_01.bin"
# ref_so="../cloud/motionsV2/good/unicycle2_v0/unicycle2_v0__ispso__2023_04_03__15_36_01.bin"
#
# ./main_primitives  \
#   --dynamics unicycle2_v0 \
#   --in_file $ref_sp \
#   --out_file $ref_so \
#   --mode_gen_id 3 


#/bin/bash
#

make main_primitives 

# Car: 1s=1primitive
car_primitive () {
num_primitives=$1
time_stamp=$(date +%Y_%m_%d__%H_%M_%S)
./main_primitives --dynamics car1_v0 --out_file  "data_4/tmp_car1_v0_${time_stamp}.bin"  --mode_gen_id 0  --max_num_primitives $num_primitives --max_iter 200 --ref_x0 1  --ref_time_steps 50 --weight_goal 200  --time_limit 1e8 > /dev/null
}

# Acrobot
# 3s=1primitive

acrobot_primitive() {
num_primitives=$1
time_stamp=$(date +%Y_%m_%d__%H_%M_%S)
./main_primitives --dynamics acrobot_v0  --out_file  "data_4/tmp_acrobot_vo${time_stamp}.bin"  --mode_gen_id 0  --max_num_primitives $num_primitives  --max_iter 200 --ref_x0 1  --ref_time_steps 300  --weight_goal 500  --time_limit 1e8 > /dev/null
}


# Quad 2d
# 3s=1primitive
quad_2d_primitive() {
num_primitives=$1
time_stamp=$(date +%Y_%m_%d__%H_%M_%S)
./main_primitives  --dynamics quad2d_v0 \
                                       --out_file  "data_5/tmp_quad_2d_${time_stamp}.bin" \
                                       --mode_gen_id 0 \
                                       --max_num_primitives $num_primitives  --max_iter 200 --ref_x0 1  --ref_time_steps 300 --weight_goal 400  --time_limit 1e8 --adapt_infeas_primitives true > /dev/null
}

# Quad 3d
# 4s= 1primitive
# quad_3d_primitive() {
# num_primitives=$1
# time_stamp=$(date +%Y_%m_%d__%H_%M_%S)
# ./main_primitives --dynamics quad3d_v0 --out_file  "data_5/tmp_quad_3d_${time_stamp}.bin"  --mode_gen_id 0  --max_num_primitives $num_primitives --max_iter 200 --ref_x0 1  --ref_time_steps 300 --weight_goal 400 --time_limit 1e8 --adapt_infeas_primitives true  > /dev/null
# }

quad_3d_primitive() {
num_primitives=$1
time_stamp=$(date +%Y_%m_%d__%H_%M_%S)
./main_primitives --dynamics quad3d_v0 --out_file  "data_6/tmp_quad_3d_${time_stamp}.bin"  --mode_gen_id 0  --max_num_primitives $num_primitives --max_iter 200 --ref_x0 1  --ref_time_steps 300 --weight_goal 400 --time_limit 1e8 > /dev/null
}




# i want 10.000 of each!

# quad_3d_primitive 10

num_primitives=5000
for i in {1..50}
do
  # car_primitive 100  &
  quad_3d_primitive $num_primitives  &
  # quad_2d_primitive $num_primitives &
  # acrobot_primitive $num_primitives &
sleep 2
done







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


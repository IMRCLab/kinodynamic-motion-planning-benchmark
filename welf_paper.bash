

# run in Terminal: bash welf_paper.bash from the build directory
# increase max_iter in harder problems
# increase weight_goal if solution is not reaching the goal
#


# ./croco_main  \
#   --welf_format 1 \
#   --env_file "../benchmark/quadrotor_0/loop.yaml" \
#   --init_guess "../data_welf_yaml/guess_loop.yaml" \
#   --out "../my_trajectory.yaml" \
#   --max_iter 200  \
#   --weight_goal 200  \


./croco_main  \
  --welf_format 1 \
  --env_file "../benchmark/quadrotor_0/flip.yaml" \
  --init_guess "../data_welf_yaml/guess_flip.yaml" \
  --out "../my_trajectory.yaml" \
  --max_iter 200  \
  --weight_goal 200  \


# Other files are: 
#benchmark/quadrotor_0/recovery_paper.yaml
#benchmark/quadrotor_0/obstacle_flight.yaml

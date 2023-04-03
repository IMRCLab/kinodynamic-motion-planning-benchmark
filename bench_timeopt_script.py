import subprocess
import yaml
import numpy as np
import matplotlib.pyplot as plt
import argparse
from pathlib import Path
import sys



import os
from datetime import datetime

base_solver_path = "../solvers_timeopt/"
base_problem_path = "../problems_timeopt/"
results_path = "../results_timeopt/"
log_path = "../results_timeopt/logs/"


# 3d working

# 3d with obstacles

# ⋊> ~/s/w/k/build on dev ⨯ make && ./croco_main   --env ../benchmark/quadrotor_0/quad_one_obs.yaml         --init_guess ../test/quadrotor_0/quadroto
# r_0_obs_0.yaml   --out out.yaml --solver_id 0 --control_bounds 1 --use_warmstart 1  --weight_goal 300. --max_iter 400  --noise_level 1e-5 --use_fin
# ite_diff 0

# nice recovery flight :)

# (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make && ./croco_main   --env ../benchmark/quadrotor_0/empty_test_recovery_welf.yaml        --init_guess ../benchmark/quadrotor_0/empty_easy_guess_new_long.yaml    --out out.yaml --solver_id 
# 0 --control_bounds 1 --use_warmstart 1  --weight_goal 300. --max_iter 400  --noise_level 1e-5 --use_finite_diff 0


# ⋊> ~/s/w/k/build on dev ⨯ make -j4 && time  ./croco_main   --env ../benchmark/quadrotor_0/empty_test_easy.yaml 
#        --init_guess ../benchmark/quadrotor_0/empty_easy_guess_new.yaml    --out out.yaml --solver_id 0 --contro
# l_bounds 1 --use_warmstart 1  --weight_goal 100. --max_iter 400  --noise_level 1e-6 --use_finite_diff 0

# ⋊> ~/s/w/k/build_debug on dev ⨯ make && ./croco_main   --env ../benchmark/quadrotor_0/empty_test_easy2.yaml        --init_guess ../benchmark/quadrotor_0/empty_easy_guess_new.yaml    --out out.yaml --solver_id 0 --control_bounds 1
#  --use_warmstart 1  --weight_goal 300. --max_iter 400  --noise_level 1e-5 --use_finite_diff 0

# (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make && ./croco_main   --env ../benchmark/quadrotor_0/empty_test_easy3.yaml        --init_guess ../benchmark/quadrotor_0/empty_easy_guess_new.yaml    --out out.yaml --solver_id 0 --control_b
# ounds 1 --use_warmstart 1  --weight_goal 300. --max_iter 400  --noise_level 1e-5 --use_finite_diff 0

## stuff that works

# (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make -j4 &&  ./croco_main  --env ../benchmark/quad2d/quad_obs_recove
# ry_very_easy.yaml      --init_guess ../test/quad2d/quad_obs_0/quad_obs_recovery_very_easy_guess_long.yaml  
# --out out.yaml --solver_id 0 --control_bounds 1 --use_warmstart 1  --weight_goal 100. --max_iter 400  --noi
# se_level 1e-6

# (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make -j4 &&  ./croco_main  --env ../benchmark/quad2d/empty_1.yaml   
#   --init_guess ../test/quad2d/empty_0/guess_2.yaml --out out.yaml --solver_id 0 --control_bounds 1 --use_wa
# rmstart 1  --noise_level 1e-5

# (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make -j4 &&  ./croco_main  --env ../benchmark/quad2d/quad_obs_0.yaml
#     --init_guess ../test/quad2d/quad_obs_0/quad_obs_0_guess_0.yaml --out out.yaml --solver_id 0 --control_b
# ounds 1 --use_warmstart 1  --noise_level 1e-5

# (opti) ⋊> ~/s/w/k/build_debug on dev ⨯ make -j4 &&  ./croco_main  --env ../benchmark/quad2d/quad_obs_recove
# ry_upside.yaml      --init_guess ../test/quad2d/quad_obs_0/quad_obs_recovery_very_easy_guess_long.yaml  --o
# ut out.yaml --solver_id 0 --control_bounds 1 --use_warmstart 1  --weight_goal 100. --max_iter 400  --noise_
# level 1e-3

## TODO: add two tests for good initial guess. 

solvers = [
    "mpc_v0.yaml",
    "freetime_v0.yaml",
    "mpcc_v0.yaml",
    "bsearch_v0.yaml",
    "trajopt_v0.yaml"
]

problems = [
    # "bug_1_bad.yaml",
    # "bug_1_good.yaml",
    # "bug_1_very_good.yaml",
    # "park_1_very_good.yaml",
    # "park_1_bad.yaml",
    # "kink_1_good.yaml",
    # "bug_2.yaml",
    # "kink_2.yaml",
    # "bug_trailer.yaml"
    "quad2d_obs.yaml",
    # "quad2d_recovery_very_easy.yaml",
    # "quad2d_recovery.yaml"
    # "quad2d_recovery_good_guess.yaml"
]

# Continue HERE!!
solvers = [base_solver_path + s for s in solvers]
problems = [base_problem_path + p for p in problems]

main_cpp = "./croco_main"
out_files = []

run_solvers = True
load_existing_results = False
write_csv_files = False 



results_path_load = results_path
if run_solvers:
    for i, s in enumerate(solvers):
        for j, p in enumerate(problems):
            id = f"s-{Path(s).stem}--p-{Path(p).stem}"
            print(f"id {id}")
            out_file = results_path + id + ".yaml"
            out_files.append(out_file)

            now = datetime.now() # current date and time
            date_time = now.strftime("%m-%d-%Y--%H-%M-%S")

            fd = open(log_path + id + date_time + '.log', 'w')
            fe = open(log_path + id + date_time + '.error.log', 'w')
            args = [main_cpp, "--yaml_solver_file" ,  s, "--yaml_problem_file",  p, "--out_bench", out_file]
            print("running command: ", *args)
            p = subprocess.run(args, stdout=fd, stderr=fe)
            if p.returncode != 0:
                print("warning, return code is bad")
            fd.close()
            fe.close()
if load_existing_results:
    print("searching for files ...")
    res = []

    # construct path object
    d = Path(results_path_load)

    # iterate directory
    for entry in d.iterdir():
        # check if it a file
        if entry.is_file() and entry.suffix == ".yaml":
            res.append(entry)
    print(res)
    print("warning, outfiles is rewritten")
    out_files = [  str(r) for r in res  ]


    # std::cout << 
    # out_files = [
    #     "s0-p0.yaml",
    #     "s0-p1.yaml",
    #     "s0-p2.yaml",
    #     "s1-p0.yaml",
    #     "s1-p1.yaml",
    #     "s1-p2.yaml"]
datas = []


print("out files are" )
print(out_files )

for file in out_files:
    try: 
        with open(file, "r") as f:
            data = yaml.safe_load(f)
            datas.append(data)
    except:
        print(f"error while loading the file:{file}" )


if write_csv_files:
        # first write down as csv
    for data,file in zip(datas,out_files):
        csv_file = file + ".csv"
        with open(csv_file, 'w') as ff:
            for d in data:
                ff.write(d)
                ff.write(',')
            ff.write('\n')
            for k, v in data.items():
                ff.write(str(v))
                ff.write(',')
            ff.write('\n')


# I want to partition by algorithm.


solvers = list(set([data["solver_name"] for data in datas]))
problems = list(set([data["problem_name"] for data in datas]))


problems_dict = dict((j, i) for i, j in enumerate(problems))
solvers_dict = dict((j, i) for i, j in enumerate(solvers))


for y in ["feasible", "cost", "time_total"]:

    for solver in solvers:
        print(f"current solver {solver}")
        d = [data for data in datas if data["solver_name"] == solver]
        problem_idx = [problems_dict[dd["problem_name"]] for dd in d]
        time_totals = [dd[y] for dd in d]
        plt.plot(problem_idx, time_totals, 'o', label=solver)

    plt.legend()
    plt.title(y)

    plt.xticks(range(len(problems_dict)), problems_dict.keys(), rotation=45)

    plt.show()




    


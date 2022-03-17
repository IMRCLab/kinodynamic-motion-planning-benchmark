import numpy as np
import yaml
import argparse
import tempfile
from pathlib import Path
import subprocess
import shutil
import time

import sys
import os
sys.path.append(os.getcwd())
from motionplanningutils import CollisionChecker
from croco_models import *

from scp import SCP
import robots
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir) 


def croc_problem_from_env(robot_node, ORDER):

    x0 = np.array(robot_node["start"])
    goal = np.array(robot_node["goal"])

    LB = np.array([-.5,-.5])
    UB = np.array([.5,.5])

    obs = []
    for obstacle in robot_node["environment"]["obstacles"]:
        obs.append(np.array(obstacle["center"]))

    print("obs are")
    print(obs)
    obs_radius = .25

    if ORDER == 1:
        ActionModel = ActionModelUnicycle2
    elif ORDER == 2:
        ActionModel = ActionModelUnicycleSecondOrder

    weight_goal =  10
    weight_control=  1
    weight_obstacles=  10

    T = len(robot_node["states"])
    print("T ", T)

    featss = [] 
    # Eq_feat = AuglagFeat
    Eq_feat = PenaltyFeat
    for i in range(T): 
        feats = FeatUniversal([Eq_feat(Feat_terminal(goal, weight_goal),len(goal),0), CostFeat(Feat_control(weight_control),2,1.) , Eq_feat(Feat_obstacles(obs, obs_radius , weight_obstacles),1,1.)])
        featss.append(feats)

    feats = FeatUniversal([Eq_feat(Feat_terminal(goal, weight_goal),len(goal),1.0), CostFeat(Feat_control(weight_control),2,0.0) , Eq_feat(Feat_obstacles(obs, obs_radius, weight_obstacles),1,1.)])
    featss.append(feats)

    rM_basic = []
    rM = []
    # create models and features
    for i in range(T):
        unicycle =  ActionModel(featss[i], featss[i].nr)
        unicycleIAM = crocoddyl.ActionModelNumDiff(unicycle, True)
        unicycleIAM.u_lb=LB
        unicycleIAM.u_ub=UB
        print(unicycleIAM.has_control_limits)
        rM_basic.append(unicycle)
        rM.append(unicycleIAM)

    # terminal
    rT_basic =  ActionModel(featss[T], featss[T].nr)
    rT = crocoddyl.ActionModelNumDiff(rT_basic, True)

    problem = crocoddyl.ShootingProblem(x0, rM, rT)
    return problem 








def run_croco(filename_env, filename_initial_guess, filename_result='result_scp.yaml', iterations=5):


    with open(filename_env) as f:
        env = yaml.safe_load(f)

    robot_node = env["robots"][0]
    robot = robots.create_robot(robot_node["type"])

    x0 = np.array(robot_node["start"])
    xf = np.array(robot_node["goal"])

    cc = CollisionChecker()
    cc.load(filename_env)
    print("DONE")

    # SOLVE THE PROBLEM
    import benchmark.unicycle_first_order_0.visualize as vis

    aa = vis.Animation(filename_env)
    aa.show()

    ## GENERATE THE PROBLEM IN CROCODYLE

    ORDER = 1
    problem =  croc_problem_from_env(robot_node, ORDER)


    ## SOLVE






    ## FIRST: add minimal TESTs for crocodyle

    ## DO it NOW :)

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	parser.add_argument("initial_guess", help="file containing the initial_guess (e.g., from db-A*) (YAML)")
	args = parser.parse_args()

	run_croco(args.env, args.initial_guess)


if __name__ == '__main__':
	main()


#!/usr/bin/env python3
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib import animation
import matplotlib.animation as manimation
import os
import sys

def draw_box_patch(ax, center, size, angle = 0, **kwargs):
  xy = np.asarray(center) - np.asarray(size) / 2
  rect = Rectangle(xy, size[0], size[1], **kwargs)
  t = matplotlib.transforms.Affine2D().rotate_around(
      center[0], center[1], angle)
  rect.set_transform(t + ax.transData)
  ax.add_patch(rect)
  return rect


class Animation:
  def __init__(self, filename_env, filename_result = None):
    with open(filename_env) as env_file:
      env = yaml.safe_load(env_file)

    # sys.path.append(os.getcwd())
    # from motionplanningutils import CollisionChecker
    # self.cc = CollisionChecker()
    # self.cc.load(filename_env)

    self.fig = plt.figure()  # frameon=False, figsize=(4 * aspect, 4))
    self.ax = self.fig.add_subplot(111, aspect='equal')
    self.ax.set_xlim(env["environment"]["min"][0], env["environment"]["max"][0])
    self.ax.set_ylim(env["environment"]["min"][1], env["environment"]["max"][1])

    for obstacle in env["environment"]["obstacles"]:
      if obstacle["type"] == "box" or obstacle["type"] == "sphere":
        draw_box_patch(
            self.ax, obstacle["center"], obstacle["size"], facecolor='gray', edgecolor='black')
      else:
        print("ERROR: unknown obstacle type")

    for robot in env["robots"]:
      if robot["type"] in ["unicycle_first_order_1"]:
        self.size = np.array([0.5, 0.25])
        draw_box_patch(self.ax, robot["start"][0:2], self.size,
                      robot["start"][2], facecolor='red')
        draw_box_patch(self.ax, robot["goal"][0:2], self.size,
                      robot["goal"][2], facecolor='none', edgecolor='red')
      else:
        raise Exception("Unknown robot type!")

    if filename_result is not None:
      with open(filename_result) as result_file:
        self.result = yaml.safe_load(result_file)

      T = 0
      for robot in self.result["result"]:
        T = max(T, len(robot["states"]))
      print("T", T)

      self.robot_patches = []
      for robot in self.result["result"]:
        state = robot["states"][0]
        patch = draw_box_patch(
            self.ax, state[0:2], self.size, state[2], facecolor='blue')
        self.robot_patches.append(patch)

      self.anim = animation.FuncAnimation(self.fig, self.animate_func,
                                frames=T,
                                interval=100,
                                blit=True)

  def save(self, file_name, speed):
    self.anim.save(
      file_name,
      "ffmpeg",
      fps=10 * speed,
      dpi=200),
      # savefig_kwargs={"pad_inches": 0, "bbox_inches": "tight"})

  def show(self):
    plt.show()

  def animate_func(self, i):
    print(i)
    for robot, patch in zip(self.result["result"], self.robot_patches):
      state = robot["states"][i]
      center = state[0:2]
      angle = state[2]
      xy = np.asarray(center) - np.asarray(self.size) / 2
      patch.set_xy(xy)
      t = matplotlib.transforms.Affine2D().rotate_around(
          center[0], center[1], angle)
      patch.set_transform(t + self.ax.transData)
    return self.robot_patches

def visualize(filename_env, filename_result = None, filename_video=None):
  anim = Animation(filename_env, filename_result)
  if filename_video is not None:
    anim.save(filename_video, 1)
  else:
    anim.show()
  # with open(filename_env) as env_file:
  #   env = yaml.safe_load(env_file)

  # sys.path.append(os.getcwd())
  # from motionplanningutils import CollisionChecker
  # cc = CollisionChecker()
  # cc.load(filename_env)

  # fig = plt.figure()  # frameon=False, figsize=(4 * aspect, 4))
  # ax = fig.add_subplot(111, aspect='equal')
  # ax.set_xlim(0, env["environment"]["max"][0])
  # ax.set_ylim(0, env["environment"]["max"][1])

  # for obstacle in env["environment"]["obstacles"]:
  #   if obstacle["type"] == "box":
  #     draw_box_patch(
  #         ax, obstacle["center"], obstacle["size"], facecolor='gray', edgecolor='black')
  #   else:
  #     print("ERROR: unknown obstacle type")

  # for robot in env["robots"]:
  #   if robot["type"] in ["dubins_0", "car_first_order_0", "car_second_order_0"]:
  #     size = np.array([0.5, 0.25])
  #     draw_box_patch(ax, robot["start"][0:2], size,
  #                    robot["start"][2], facecolor='red')
  #     draw_box_patch(ax, robot["goal"][0:2], size,
  #                    robot["goal"][2], facecolor='none', edgecolor='red')
  #   else:
  #     raise Exception("Unknown robot type!")

  # if filename_result is not None:
  #   with open(filename_result) as result_file:
  #     result = yaml.safe_load(result_file)

  #   for robot in result["result"]:
  #     p_obs_all = []
  #     p_robot_all = []
  #     for state in robot["states"]:
  #       draw_box_patch(ax, state[0:2], size, state[2], facecolor='blue')
  #       if cc:
  #         d, p_obs, p_robot = cc.distance(state)
  #         p_obs_all.append(p_obs)
  #         p_robot_all.append(p_robot)
  #     p_obs_all = np.array(p_obs_all)
  #     ax.scatter(p_obs_all[:, 0], p_obs_all[:, 1])
  #     p_robot_all = np.array(p_robot_all)
  #     ax.scatter(p_robot_all[:, 0], p_robot_all[:, 1])

  # plt.show()

def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("env", help="input file containing map")
  parser.add_argument("--result", help="output file containing solution")
  parser.add_argument("--video", help="output file for video")
  args = parser.parse_args()

  visualize(args.env, args.result, args.video)

if __name__ == "__main__":
  main()

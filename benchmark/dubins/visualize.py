#!/usr/bin/env python3
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, Rectangle, Arrow


def draw_box_patch(ax, center, size, angle = 0, **kwargs):
  xy = np.asarray(center) - np.asarray(size) / 2
  rect = Rectangle(xy, size[0], size[1], **kwargs)
  t = matplotlib.transforms.Affine2D().rotate_around(
      center[0], center[1], angle)
  rect.set_transform(t + ax.transData)
  ax.add_patch(rect)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("env", help="input file containing map")
  args = parser.parse_args()

  with open(args.env) as env_file:
    env = yaml.safe_load(env_file)

  fig = plt.figure()#frameon=False, figsize=(4 * aspect, 4))
  ax = fig.add_subplot(111, aspect='equal')
  ax.set_xlim(0, env["environment"]["dimensions"][0])
  ax.set_ylim(0, env["environment"]["dimensions"][1])

  for obstacle in env["environment"]["obstacles"]:
    if obstacle["type"] == "box":
      draw_box_patch(ax, obstacle["center"], obstacle["size"], facecolor='gray', edgecolor='black')
    else:
      print("ERROR: unknown obstacle type")

  for robot in env["robots"]:
    if robot["type"] == "dubins_0":
      size = np.array([0.5, 0.25])
      draw_box_patch(ax, robot["start"][0:2], size, robot["start"][2], facecolor='red')
      draw_box_patch(ax, robot["goal"][0:2], size, robot["goal"][2], facecolor='none', edgecolor='red')

  plt.show()

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
  parser.add_argument("motions", help="output file containing solution")
  args = parser.parse_args()

  fig = plt.figure()#frameon=False, figsize=(4 * aspect, 4))
  ax = fig.add_subplot(111, aspect='equal')
  ax.set_xlim(-5, 5)
  ax.set_ylim(-5, 5)

  with open(args.motions) as motions_file:
    motions = yaml.safe_load(motions_file)

  size = np.array([0.5, 0.25])

  for motion in motions:
    for state in motion["states"]:
      draw_box_patch(ax, state[0:2], size, state[2], facecolor='blue', alpha=0.1)

  plt.show()

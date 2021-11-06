#!/usr/bin/env python3
import argparse
import numpy as np
import re
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, Rectangle, Arrow


def draw_box_patch(ax, center, size, angle=0, **kwargs):
  xy = np.asarray(center) - np.asarray(size) / 2
  rect = Rectangle(xy, size[0], size[1], **kwargs)
  t = matplotlib.transforms.Affine2D().rotate_around(
      center[0], center[1], angle)
  rect.set_transform(t + ax.transData)
  ax.add_patch(rect)


if __name__ == "__main__":
  parser = argparse.ArgumentParser()
  parser.add_argument("mprim", help="SBPL MPRIM File")
  args = parser.parse_args()

  # parse the input file
  prims = []
  with open(args.mprim) as f:
    remaining_prims = 0
    for line in f:
        if remaining_prims > 0:
            data = [float(num) for num in line.split()]
            prim.append(data)
            remaining_prims -= 1
            if remaining_prims == 0:
                prims.append(np.array(prim))
        
        match = re.search("^intermediateposes: (?P<num>[0-9]+)$", line)
        if match:
            remaining_prims = int(match.group('num'))
            prim = []

  # try to find the actions that were used (assuming unicycle model)
  dt = 0.1
  for prim in prims:
      x_dot = np.diff(prim, axis=0) / dt
      ctrl_v_x = x_dot[:,0] / np.cos(prim[0:-1, 2])
      ctrl_v_y = x_dot[:,1] / np.sin(prim[0:-1, 2])
      ctrl_w = x_dot[:,2]
      print(ctrl_v_x, ctrl_v_y, ctrl_w)
  exit()

# x += ctrl[0] * cosf(yaw) * dt
# y += ctrl[0] * sinf(yaw) * dt
# yaw += ctrl[1] * dt

  fig = plt.figure()  # frameon=False, figsize=(4 * aspect, 4))
  ax = fig.add_subplot(111, aspect='equal')
  ax.set_xlim(-5, 5)
  ax.set_ylim(-5, 5)

  size = np.array([0.5, 0.25])

  for prim in prims[10:15]:
    for state in prim:
      draw_box_patch(ax, state[0:2], size, state[2],
                     facecolor='blue', alpha=0.1)

  plt.show()

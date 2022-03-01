#!/usr/bin/env python3
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, Rectangle, Arrow
import matplotlib.lines as mlines
import matplotlib.collections as mcoll

def draw_box_patch(ax, center, size, angle = 0, **kwargs):
  xy = np.asarray(center) - np.asarray(size) / 2
  rect = Rectangle(xy, size[0], size[1], **kwargs)
  t = matplotlib.transforms.Affine2D().rotate_around(
      center[0], center[1], angle)
  rect.set_transform(t + ax.transData)
  ax.add_patch(rect)
  return rect

# See https://stackoverflow.com/questions/36074455/python-matplotlib-with-a-line-color-gradient-and-colorbar
def colorline(
        x, y, z=None, cmap='copper', norm=plt.Normalize(0.0, 1.0),
        linewidth=3, alpha=1.0):
    """
    http://nbviewer.ipython.org/github/dpsanders/matplotlib-examples/blob/master/colorline.ipynb
    http://matplotlib.org/examples/pylab_examples/multicolored_line.html
    Plot a colored line with coordinates x and y
    Optionally specify colors in the array z
    Optionally specify a colormap, a norm function and a line width
    """

    # Default colors equally spaced on [0,1]:
    if z is None:
        z = np.linspace(0.0, 1.0, len(x))

    # Special case if a single number:
    # to check for numerical input -- this is a hack
    if not hasattr(z, "__iter__"):
        z = np.array([z])

    z = np.asarray(z)

    segments = make_segments(x, y)
    lc = mcoll.LineCollection(segments, array=z, cmap=cmap, norm=norm,
                              linewidth=linewidth, alpha=alpha)

    ax = plt.gca()
    ax.add_collection(lc)

    return lc

def make_segments(x, y):
    """
    Create list of line segments from x and y coordinates, in the correct format
    for LineCollection: an array of the form numlines x (points per line) x 2 (x
    and y) array
    """

    points = np.array([x, y]).T.reshape(-1, 1, 2)
    segments = np.concatenate([points[:-1], points[1:]], axis=1)
    return segments

class Vis:
  def __init__(self, filename_env, filename_result = None):
    with open(filename_env) as env_file:
      env = yaml.safe_load(env_file)

    self.fig = plt.figure()  # frameon=False, figsize=(4 * aspect, 4))
    self.ax = self.fig.add_subplot(111, aspect='equal')
    self.ax.set_xlim(env["environment"]["min"][0], env["environment"]["max"][0])
    self.ax.set_ylim(env["environment"]["min"][1], env["environment"]["max"][1])
    self.ax.get_xaxis().set_visible(False)
    self.ax.get_yaxis().set_visible(False)

    for obstacle in env["environment"]["obstacles"]:
      if obstacle["type"] == "box":
        draw_box_patch(
            self.ax, obstacle["center"], obstacle["size"], facecolor='gray')
      else:
        print("ERROR: unknown obstacle type")

    for robot in env["robots"]:
      if robot["type"] in ["car_first_order_with_1_trailers_0"]:
        self.size = [np.array([0.5, 0.25]), np.array([0.3, 0.25])]
        self.hitch_length = [0.5]
        self.draw_robot(robot["start"], facecolor='blue', alpha=0.6)
        self.draw_robot(robot["goal"], edgecolor='blue', facecolor='none', alpha=0.6, linewidth=2)
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
        # for t in range(0, len(robot["states"]), 5):
        #   state = robot["states"][t]
        #   patches = self.draw_robot(state, facecolor='blue', alpha=0.5)
        #   self.robot_patches.extend(patches)
        states = np.array(robot["states"])
        segments = make_segments(states[:,0], states[:,1])
        z = np.arange(0, T/10, 0.1) #np.linspace(0.0, 1.0, T)
        lc = mcoll.LineCollection(segments, array=z, norm=plt.Normalize(0.0, T/10), linewidth=5, capstyle='round')
        self.ax.add_collection(lc)
        # plt.colorbar(lc)

  def show(self):
    plt.show()

  def draw_robot(self, state, **kwargs):
    xy = state[0:2]
    theta0 = state[2]
    theta1 = state[3]
    patch1 = draw_box_patch(self.ax, xy, self.size[0], theta0, **kwargs)
    link1 = np.array([np.cos(theta1), np.sin(theta1)]) * self.hitch_length[0]
    patch2 = draw_box_patch(self.ax, xy-link1,
                   self.size[1], theta1, **kwargs)

    x, y = ([xy[0], xy[0]-link1[0]], [xy[1], xy[1]-link1[1]])
    line = mlines.Line2D(x, y, lw=2., color='black', alpha=0.3)
    self.ax.add_line(line)

    return [patch1, patch2, line]

def main():

  l = [
    {
    'env': "../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml",
    'result': "../paper/fig2/bugtrap.yaml",
    'out': "fig2c.svg",
    },
    {
    'env': "../benchmark/car_first_order_with_1_trailers_0/kink_0.yaml",
    'result': "../paper/fig2/kink.yaml",
    'out': "fig2b.svg",
    },
    {
    'env': "../benchmark/car_first_order_with_1_trailers_0/parallelpark_0.yaml",
    'result': "../paper/fig2/parallelpark.yaml",
    'out': "fig2a.svg",
    },
  ]

  for item in l:
    v = Vis(item['env'], item['result'])
    plt.savefig(item['out'], bbox_inches='tight')
    # inkscape fig2b.svg --export-pdf=fig2b.pdf
  # v.show()

if __name__ == "__main__":
  main()

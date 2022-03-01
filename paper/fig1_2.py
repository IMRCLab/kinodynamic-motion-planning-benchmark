#!/usr/bin/env python3
import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, Rectangle, Arrow
import matplotlib.lines as mlines
import matplotlib.collections as mcoll

import sys, os
sys.path.append(os.getcwd())
from motionplanningutils import RobotHelper

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
  def __init__(self, filename_env, filenames_result):
    with open(filename_env) as env_file:
      env = yaml.safe_load(env_file)

    self.fig, axs = plt.subplots(1, len(filenames_result), sharex='all', sharey='all', squeeze=False)

    axs[0,0].plot([1.1, 1.1], [0, 1], color='black', linestyle='dashed', lw=1, transform=axs[0,0].transAxes, clip_on=False)
    axs[0,0].plot([2.3, 2.3], [0, 1], color='black', linestyle='dashed', lw=1, transform=axs[0,0].transAxes, clip_on=False)
    # axes.plot([-1, 1.5], [1, 1], color='black', lw=1, transform=axes.transAxes, clip_on=False)

    max_delta = None
    plot_idx = "A)"
    for ax, filename_result in zip(axs[0,:], filenames_result):
        ax.set_aspect('equal')
        ax.axis('off')
        # ax.set_xlim(env["environment"]["min"][0], env["environment"]["max"][0])
        # ax.set_ylim(env["environment"]["min"][1], env["environment"]["max"][1])
        # ax.get_xaxis().set_visible(False)
        # ax.get_yaxis().set_visible(False)

        ax.set_xlim(0.4, 3.9)
        ax.set_ylim(1, 5.2)


        for obstacle in env["environment"]["obstacles"]:
            if obstacle["type"] == "box":
                draw_box_patch(
                    ax, obstacle["center"], obstacle["size"], facecolor='gray')
            else:
                print("ERROR: unknown obstacle type")

        for robot in env["robots"]:
            if "unicycle" in robot["type"]:
                self.size = np.array([0.5, 0.25])
                self.draw_robot(ax, robot["start"], facecolor='blue', alpha=0.6)
                self.draw_robot(ax, robot["goal"], edgecolor='blue', facecolor='none', alpha=0.6, linewidth=2)
            else:
                raise Exception("Unknown robot type!")

        rh = RobotHelper(env["robots"][0]["type"])

        with open(filename_result) as result_file:
            result = yaml.safe_load(result_file)

        T = 0
        for robot in result["result"]:
            T = max(T, len(robot["states"]))
        print("T", T)

        for robot in result["result"]:
            STEP = 10
            for t in range(STEP, len(robot["states"]), STEP):
                state = robot["states"][t]
                self.draw_robot(ax, state, facecolor='blue', alpha=0.15)
            
            states = np.array(robot["states"])
            segments = make_segments(states[:,0], states[:,1])

            dist = np.empty(T-1)
            for t in range(T-1):
                if t == 0:
                    s = env["robots"][0]["start"]
                else:
                    s = robot["states"][t]

                if t == T-2:
                    state_desired = env["robots"][0]["goal"]
                else:
                    state_desired = rh.step(s, robot["actions"][t], 0.1)
                state_actual = robot["states"][t+1]
                dist[t] = rh.distance(state_desired, state_actual)

            if max_delta is None:
                max_delta = max(dist)
            widths = dist / max_delta * 5 + 3
            colors = ["g" if x < 1e-3 else "r" for x in dist]
            # print(dist)
            from  matplotlib.colors import LinearSegmentedColormap
            cmap=LinearSegmentedColormap.from_list('rg',["g", "r"], N=3) 
            lc = mcoll.LineCollection(segments, array=dist, norm=plt.Normalize(0.0, max_delta), cmap=cmap, linewidth=3, capstyle='round')#, linewidth=widths, cmap=cmap)
            # lc = mcoll.LineCollection(segments, array=dist, linewidth=widths, colors=colors)
            ax.add_collection(lc)
            ax.text(0.3,1, r"{} $\delta = {:.2f} \quad T\Delta t = {:.1f} s$".format(plot_idx, max(dist), T/10))

            if plot_idx == "A)":
              plot_idx = "B)"
            elif plot_idx == "B)":
              plot_idx = "C)"
            # plt.colorbar(lc)

  def show(self):
    plt.show()

  def draw_robot(self, ax, state, **kwargs):
    xy = state[0:2]
    theta0 = state[2]
    draw_box_patch(ax, xy, self.size, theta0, **kwargs)

def main():

  # Using seaborn's style
  # plt.style.use('seaborn')
  # width = 345

  tex_fonts = {
      # Use LaTeX to write all text
      "text.usetex": True,
      "font.family": "serif",
      # Use 10pt font in plots, to match 10pt font in document
      "axes.labelsize": 10,
      "font.size": 10,
      # Make the legend/label fonts a little smaller
      "legend.fontsize": 8,
      "xtick.labelsize": 8,
      "ytick.labelsize": 8
  }

  plt.rcParams.update(tex_fonts)


  filename_env = "../benchmark/unicycle_first_order_2/wall_0.yaml"
#   filename_result = "../paper/result_dbastar_sol1.yaml"
  filenames_result = [
      "../paper/fig1/result_dbastar_sol0.yaml",
      "../paper/fig1/result_dbastar_sol1.yaml",
      "../paper/fig1/result_opt_sol1.yaml",
  ]
  # filename_result = "../paper/fig2/result_dbastar_sol2.yaml"

  # filename_env = "../benchmark/car_first_order_with_1_trailers_0/kink_0.yaml"
  # filename_result = "../paper/fig2/kink.yaml"

  # filename_env = "../benchmark/car_first_order_with_1_trailers_0/parallelpark_0.yaml"
  # filename_result = "../paper/fig2/parallelpark.yaml"

  v = Vis(filename_env, filenames_result)
  plt.savefig('fig1.pdf', format='pdf', bbox_inches='tight')
#   plt.savefig('fig2c.svg', bbox_inches='tight')
  v.show()

if __name__ == "__main__":
  main()

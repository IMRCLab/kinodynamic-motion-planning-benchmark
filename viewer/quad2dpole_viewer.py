import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))


print(sys.path)
import viewer_utils
import numpy as np
import matplotlib

import argparse
import numpy as np
import yaml
import matplotlib.pyplot as plt
import matplotlib
from matplotlib.patches import Circle, Rectangle, Arrow
from matplotlib import animation
import os
import sys
from robot_viewer import RobotViewer


r = 1


def compute_pole_pose(x, r):
    xp = x[0] + r * np.sin(x[2] + x[3])
    yp = x[1] - r * np.cos(x[2] + x[3])
    return np.array([xp, yp])


def compute_center_bar_pose(x, r):
    xp = x[0] + r / 2. * np.sin(x[2] + x[3])
    yp = x[1] - r / 2. * np.cos(x[2] + x[3])
    return np.array([xp, yp])


class Robot():

    size = np.array([0.5, 0.25])
    size_pendulumn = np.array([r, 0.02])

    def draw_traj_minimal(self, ax, Xs, **kwargs):
        xx = [x[0] for x in Xs]
        yy = [x[1] for x in Xs]

        # TODO: get the position of the pole!

        ax.plot(xx, yy, **kwargs)

    def draw_basic(
            self,
            ax,
            X,
            fill=None,
            color="k",
            l=.05,
            alpha=1.,
            **kwargs):
        self.tri = viewer_utils.draw_tri(ax, X[:3], l=.2, add_90=True)
        ax.add_patch(self.tri)
        self.point = ax.plot([X[0]], [X[1]], '.',
                             alpha=alpha, color=color, **kwargs)

        p = compute_pole_pose(X, r)

        self.point_pendulumn = ax.plot([p[0]], [p[1]], '.',
                                       alpha=alpha, color=color, **kwargs)

    def draw(self, ax, X, **kwargs):
        self.ax = ax
        center = X[:2]
        angle = X[2]
        self.o1 = viewer_utils.draw_box_patch(
            ax, center, self.size, angle, **kwargs)
        self.o2 = viewer_utils.draw_box_patch_front(
            ax, center, self.size, angle + np.pi / 2, **kwargs)

        p = compute_pole_pose(X, r)
        pc = compute_center_bar_pose(X, r)
        print(p)
        size = .1

        angle_pole = X[2] + X[3]
        self.o4 = viewer_utils.draw_box_patch(
            ax, pc, self.size_pendulumn, angle_pole + np.pi / 2, **kwargs)

        ax.add_patch(self.o4)
        o = Circle(p, radius=0.05, **kwargs)
        ax.add_patch(o)
        self.o3 = o

    def update(self, X):
        center = X[:2]
        angle = X[2]
        xy = np.asarray(center) - np.asarray(self.size) / 2
        self.o1.set_xy(xy)
        t = matplotlib.transforms.Affine2D().rotate_around(
            center[0], center[1], angle)
        self.o1.set_transform(t + self.ax.transData)

        p = .2 * np.array([np.cos(angle + np.pi / 2),
                           np.sin(angle + np.pi / 2)])
        self.o2.center = (p + center).tolist()

        p = compute_pole_pose(X, r)

        self.o3.center = p.tolist()

        pc = compute_center_bar_pose(X, r)

        self.o4.set_xy(pc - np.asarray(self.size_pendulumn) / 2)

        angle_pole = X[2] + X[3]

        t = matplotlib.transforms.Affine2D().rotate_around(
            pc[0], pc[1], angle_pole - np.pi / 2)
        self.o4.set_transform(t + self.ax.transData)

        return [self.o1, self.o2, self.o3, self.o4]


class Quad2dpoleViewer(RobotViewer):

    def __init__(self):
        super().__init__(Robot)
        self.labels_x = ["x", "z", "o", "q", "vx", "vz", "w", "vq"]
        self.labels_u = ["f1", "f2"]


if __name__ == "__main__":

    viewer = Quad2dpoleViewer()

    fig, ax = plt.subplots()

    filename_env = "../benchmark/quad2dpole/empty_0.yaml"
    with open(filename_env) as env_file:
        env = yaml.safe_load(env_file)
    viewer.view_problem(ax, env)

    plt.show()

    # viewer_utils.check_viewer(viewer)


import sys
from pathlib import Path

sys.path.append(Path(__file__).parent)


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
import viewer_utils
from robot_viewer import RobotViewer


class Robot():
    def __init__(self):
        self.l1 = 1.
        self.l2 = 1.
        self.size = [1, .05]
        self.lc1 = self.l1 / 2.
        self.lc2 = self.l2 / 2.

    def get_pivots(self, x):
        q1 = x[0]
        q2 = x[1]
        p1 = self.l1 * \
            np.array([np.cos(3 * np.pi / 2 + q1), np.sin(3 * np.pi / 2 + q1)])
        p2 = p1 + self.l2 * \
            np.array([np.cos(3 * np.pi / 2 + q1 + q2),
                      np.sin(3 * np.pi / 2 + q1 + q2)])
        return p1, p2

    def get_centers(self, x):
        q1 = x[0]
        q2 = x[1]
        c1 = self.lc1 * \
            np.array([np.cos(3 * np.pi / 2 + q1), np.sin(3 * np.pi / 2 + q1)])
        pivot2 = self.l1 * \
            np.array([np.cos(3 * np.pi / 2 + q1), np.sin(3 * np.pi / 2 + q1)])
        c2 = pivot2 + self.lc2 * \
            np.array([np.cos(3 * np.pi / 2 + q1 + q2),
                      np.sin(3 * np.pi / 2 + q1 + q2)])
        return c1, c2

    def draw(self, ax, x, **kwargs):
        self.ax = ax
        q1 = x[0]
        q2 = x[1]
        pivot2, pivot3 = self.get_pivots(x)
        c1, c2 = self.get_centers(x)
        self.p1 = viewer_utils.draw_box_patch(
            ax,
            c1,
            self.size,
            q1 + 3 * np.pi / 2,
            **kwargs,
            # color=kwargs.get(
            #     "color",
            #     ".7")
        )
        self.p2 = viewer_utils.draw_box_patch(
            ax,
            c2,
            self.size,
            q1 + q2 + 3 * np.pi / 2,
            **kwargs,
            # color=kwargs.get(
            #
            #     "color",
            #     ".6")
        )
        self.dot1, = ax.plot([0], [0], 'o', color="black")
        self.dot2, = ax.plot([pivot2[0]], [pivot2[1]], 'o', color="black")
        self.dot3, = ax.plot([pivot3[0]], [pivot3[1]], 'o', color="black")

    def get_ends(self, x):
        q1 = x[0]
        q2 = x[1]
        e1 = self.l1 * \
            np.array([np.cos(3 * np.pi / 2 + q1), np.sin(3 * np.pi / 2 + q1)])
        e2 = e1 + self.l2 * \
            np.array([np.cos(3 * np.pi / 2 + q1 + q2),
                      np.sin(3 * np.pi / 2 + q1 + q2)])
        return e1, e2

    def draw_basic(self, ax, x, **kwargs):
        self.draw(ax, x, **kwargs)

    def draw_traj_minimal(self, ax, X):
        Es = [self.get_ends(x) for x in X]
        E1s_x = [e[0][0] for e in Es]
        E1s_y = [e[0][1] for e in Es]
        E2s_x = [e[1][0] for e in Es]
        E2s_y = [e[1][1] for e in Es]
        ax.plot(E1s_x, E1s_y)
        ax.plot(E2s_x, E2s_y)

    def update(self, x):

        pivot2, pivot3 = self.get_pivots(x)
        c1, c2 = self.get_centers(x)

        xy_1 = np.asarray(c1) - np.asarray(self.size) / 2
        self.p1.set_xy(xy_1)
        t = matplotlib.transforms.Affine2D().rotate_around(
            c1[0], c1[1], 3 * np.pi / 2 + x[0])
        self.p1.set_transform(t + self.ax.transData)

        xy_2 = np.asarray(c2) - np.asarray(self.size) / 2
        self.p2.set_xy(xy_2)
        t = matplotlib.transforms.Affine2D().rotate_around(
            c2[0], c2[1], 3 * np.pi / 2 + x[0] + x[1])
        self.p2.set_transform(t + self.ax.transData)

        self.dot2.set_xdata([pivot2[0]])
        self.dot2.set_ydata([pivot2[1]])

        self.dot3.set_xdata([pivot3[0]])
        self.dot3.set_ydata([pivot3[1]])

        return [self.p1, self.p2, self.dot1, self.dot2, self.dot3]


class AcrobotViewer(RobotViewer):

    def __init__(self):
        super().__init__(Robot)
        self.labels_x = ["q1", "q2", "w1", "w2"]
        self.labels_u = ["f"]


if __name__ == "__main__":

    viewer = AcrobotViewer()
    viewer_utils.check_viewer(viewer)

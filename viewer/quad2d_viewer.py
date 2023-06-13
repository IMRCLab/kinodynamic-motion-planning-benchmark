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


class Robot():

    size = np.array([0.6, 0.2])

    def draw_traj_minimal(self, ax, Xs, **kwargs):
        xx = [x[0] for x in Xs]
        yy = [x[1] for x in Xs]
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

    def draw(self, ax, X, **kwargs):
        self.ax = ax
        center = X[:2]
        angle = X[2]
        self.o1 = viewer_utils.draw_box_patch(
            ax, center, self.size, angle, **kwargs)
        self.o2 = viewer_utils.draw_box_patch_front(
            ax, center, self.size, angle + np.pi / 2, **kwargs)

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
        return [self.o1, self.o2]


class Quad2dViewer(RobotViewer):

    def __init__(self):
        super().__init__(Robot)
        self.labels_x = ["x", "z", "o", "vx", "vz", "w"]
        self.labels_u = ["f1", "f2"]


if __name__ == "__main__":

    viewer = Quad2dViewer()
    viewer_utils.check_viewer(viewer)

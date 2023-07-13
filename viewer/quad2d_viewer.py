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
        print(kwargs)
        self.o1 = viewer_utils.draw_box_patch(
            ax, center, self.size, angle, **{'facecolor': 'none', 'edgecolor': 'gray', 'alpha': 0})
        # self.o2 = viewer_utils.draw_box_patch_front(
        #     ax, center, self.size, angle + np.pi / 2, **kwargs)

        self.size_internal = np.array([0.5, 0.09])
        self.offset = np.array([0, -.05])

        self.o3 = viewer_utils.draw_box_patch(
            ax, center + viewer_utils.rotate(self.offset, angle),
            self.size_internal, angle, **kwargs)

        # self.o4 = viewer_utils.draw_box_patch(
        #     ax, center, size_internal, angle, fill="black")

        self.offset_propeller_right = np.array([.1, .05])
        self.offset_propeller_left = np.array([-.1, .05])
        p = center + viewer_utils.rotate(self.offset_propeller_left, angle)
        self.p_left = Circle(p, radius=0.05, **kwargs)
        ax.add_patch(self.p_left)

        p = center + viewer_utils.rotate(self.offset_propeller_right, angle)
        self.p_right = Circle(p, radius=0.05, **kwargs)
        ax.add_patch(self.p_right)

    def update(self, X):
        center = X[:2]
        angle = X[2]
        xy = np.asarray(center) - np.asarray(self.size) / 2
        self.o1.set_xy(xy)
        t = matplotlib.transforms.Affine2D().rotate_around(
            center[0], center[1], angle)
        self.o1.set_transform(t + self.ax.transData)

        # same for o3

        xy = np.asarray(center) + self.offset - \
            np.asarray(self.size_internal) / 2
        self.o3.set_xy(xy)
        t = matplotlib.transforms.Affine2D().rotate_around(
            center[0], center[1], angle)
        self.o3.set_transform(t + self.ax.transData)

        # p =

        self.p_left.center = center + \
            viewer_utils.rotate(self.offset_propeller_left, angle)
        self.p_right.center = center + \
            viewer_utils.rotate(self.offset_propeller_right, angle)

        return [self.o1, self.o3, self.p_left, self.p_right]


class Quad2dViewer(RobotViewer):

    def __init__(self):
        super().__init__(Robot)
        self.labels_x = ["x", "z", "o", "vx", "vz", "w"]
        self.labels_u = ["f1", "f2"]

    def view_primitives(self, ax, result):
        assert ("primitives" in result)
        primitives = result["primitives"]
        r = Robot()
        states = result["states"]
        print("drawing primitives")
        for p in primitives:
            first_state = p["states"][0]
            last_state = p["states"][-1]
            r.draw(ax, first_state, **
                   {'facecolor': 'none', 'edgecolor': 'deepskyblue'})
            r.draw(ax, last_state, **
                   {'facecolor': 'none', 'edgecolor': 'magenta'})

        r.draw_traj_minimal(ax, states)

        for i in range(0, len(states), 20):
            r = Robot()
            r.draw_basic(ax, states[i])

    def view_primitive_line_and_end(self, ax, result):
        r = Robot()
        states = result["states"]
        last_state = states[-1]
        c = np.random.random(3)
        r.draw_traj_minimal(ax, states, **{'color': c})
        r.draw(ax, last_state, **{'edgecolor': c, 'facecolor': 'none'})

        # **{'facecolor':'none','edgecolor':'deepskyblue'})


        # print("done!")
if __name__ == "__main__":

    viewer = Quad2dViewer()
    viewer_utils.check_viewer(viewer)

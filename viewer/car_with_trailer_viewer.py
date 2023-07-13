import viewer_utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
from matplotlib import animation
from robot_viewer import RobotViewer


class Robot():
    def __init__(self):
        self.size = [np.array([0.5, 0.25]), np.array([0.3, 0.25])]
        self.hitch_length = [0.5]

    def get_trailer_center(self, X):
        theta1 = X[3]
        xy = X[0:2]
        link1 = np.array([np.cos(theta1), np.sin(theta1)]) * \
            self.hitch_length[0]
        return xy - link1

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
        self.tri = viewer_utils.draw_tri(ax, X[:3], l=.2, add_90=False)
        ax.add_patch(self.tri)
        self.point = ax.plot([X[0]], [X[1]], '.',
                             alpha=alpha, color=color, **kwargs)

        link1 = self.get_trailer_center(X)
        # self.tri2 = viewer_utils.draw_tri(
        #     ax, [link1[0], link1[1], X[3]], l=.2, add_90=False)
        self.point2 = ax.plot([link1[0]], [link1[1]], '*',
                              alpha=alpha, color=color, **kwargs)

    def draw(self, ax, state, **kwargs):
        """
        """
        xy = state[0:2]
        theta0 = state[2]
        theta1 = state[3]
        self.ax = ax
        self.patch1 = viewer_utils.draw_box_patch(
            ax, xy, self.size[0], theta0, **kwargs)
        link1 = np.array([np.cos(theta1), np.sin(theta1)]) * \
            self.hitch_length[0]
        self.patch2 = viewer_utils.draw_box_patch(
            ax, xy - link1, self.size[1], theta1, **kwargs)

        self.o2 = viewer_utils.draw_box_patch_front(
            ax, xy, self.size, theta0, color="black")

        return [self.patch1, self.patch2, self.o2]

    def update(self, state):
        """
        """
        pos0 = state[0:2]
        theta0 = state[2]
        theta1 = state[3]
        pos1 = pos0 - np.array([np.cos(theta1), np.sin(theta1)]
                               ) * self.hitch_length[0]

        xy = np.asarray(pos0) - np.asarray(self.size[0]) / 2
        self.patch1.set_xy(xy)

        t = matplotlib.transforms.Affine2D().rotate_around(
            pos0[0], pos0[1], theta0)

        self.patch1.set_transform(t + self.ax.transData)

        xy = np.asarray(pos1) - np.asarray(self.size[1]) / 2
        self.patch2.set_xy(xy)
        t = matplotlib.transforms.Affine2D().rotate_around(
            pos1[0], pos1[1], theta1)
        self.patch2.set_transform(t + self.ax.transData)

        p = .2 * np.array([np.cos(theta0), np.sin(theta0)])
        print(p + xy)
        self.o2.center = (p + pos0).tolist()

        return [self.patch1, self.patch2, self.o2]


class CarWithTrailerViewer(RobotViewer):
    def __init__(self):
        super().__init__(Robot)
        self.labels_x = ["x", "y", "o", "o2"]
        self.labels_u = ["u", "p"]


if __name__ == "__main__":

    viewer = CarWithTrailerViewer()
    viewer_utils.check_viewer(viewer)

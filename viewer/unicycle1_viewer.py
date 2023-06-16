import viewer_utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
from matplotlib import animation
from robot_viewer import RobotViewer


class Robot():
    size = np.array([0.5, 0.25])

    def __init__(self):
        pass

    def draw_basic(self, ax, X, fill=None, color="k",
                   l=.05, alpha=1., **kwargs):
        self.tri = viewer_utils.draw_tri(ax, X[:3], l=.2, add_90=False)
        ax.add_patch(self.tri)
        self.point = ax.plot([X[0]], [X[1]], '.',
                             alpha=alpha, color=color, **kwargs)

    def draw_traj_minimal(self, ax, Xs, **kwargs):
        xx = [x[0] for x in Xs]
        yy = [x[1] for x in Xs]
        ax.plot(xx, yy, **kwargs)

    def draw(self, ax, X, **kwargs):
        self.ax = ax
        center = X[:2]
        angle = X[2]
        self.o1 = viewer_utils.draw_box_patch(
            ax, center, self.size, angle, **kwargs)
        self.o2 = viewer_utils.draw_box_patch_front(
            ax, center, self.size, angle, color="black")

    def update(self, X):
        center = X[:2]
        angle = X[2]
        xy = np.asarray(center) - np.asarray(self.size) / 2
        self.o1.set_xy(xy)
        t = matplotlib.transforms.Affine2D().rotate_around(
            center[0], center[1], angle)
        self.o1.set_transform(t + self.ax.transData)

        p = .2 * np.array([np.cos(angle), np.sin(angle)])
        print(p + center)
        self.o2.center = (p + center).tolist()
        return [self.o1, self.o2]


class Unicycle1Viewer(RobotViewer):

    def __init__(self):
        super().__init__(Robot)
        self.labels_x = ["x", "y", "o"]
        self.labels_u = ["u", "w"]


if __name__ == "__main__":

    viewer = Unicycle1Viewer()
    # viewer_utils.check_viewer(viewer)
    x = [1.901227e+00, 3.081316e-01, -4.412384e-03]
    fig, ax = plt.subplots()
    env = "../benchmark/unicycle_first_order_0/parallelpark_0.yaml"

    viewer.view_problem(ax, env)
    viewer.view_state(ax, x)
    plt.show()

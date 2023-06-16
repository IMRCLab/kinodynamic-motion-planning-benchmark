import viewer_utils
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import yaml
from matplotlib import animation
from robot_viewer import RobotViewer

from unicycle1_viewer import Robot as Robot2


class Unicycle2Viewer(RobotViewer):

    def __init__(self):
        super().__init__(Robot2)
        self.labels_x = ["x", "y", "o", "v", "w"]
        self.labels_u = ["a", "aa"]


if __name__ == "__main__":

    viewer = Unicycle2Viewer()
    viewer_utils.check_viewer(viewer)

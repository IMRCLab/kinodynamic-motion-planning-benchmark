
import viewer_utils

    

class RobotViewer():

    def __init__(self,RobotDrawerClass=None):
        self.RobotDrawerClass = RobotDrawerClass
        self.labels_x = []
        self.labels_u = []

    def view_problem(self, ax, env, **kwargs):
        viewer_utils.draw_problem_2d(ax, env, self.RobotDrawerClass)

    def view_trajectory(self, ax, result, **kwargs):
        viewer_utils.draw_traj_default(ax, result, self.RobotDrawerClass, draw_basic_every=10)

    def view_static(self, ax, env , result, **kwargs):
        self.view_problem( ax, env, **kwargs)
        self.view_trajectory( ax, result,  **kwargs)

    def view_state(self, ax, X, **kwargs):
        self.RobotDrawerClass().draw(ax, X, **kwargs)

    def plot_traj(self, axs, result, **kwargs):
        viewer_utils.plot_traj_default(
            axs, result, self.labels_x, self.labels_u, **kwargs)

    def make_video(self, env, result, filename_video: str = ""):
        viewer_utils.make_video_default(env, result, lambda ax, env: self.view_problem(ax, env),
                                        self.RobotDrawerClass, filename_video)

    def is_3d(self) -> bool : 
        return False

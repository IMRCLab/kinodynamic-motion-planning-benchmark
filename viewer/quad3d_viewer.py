
import numpy as np
import viewer_utils
import yaml
import matplotlib.pyplot as plt
from matplotlib import animation
from robot_viewer import RobotViewer


class Robot():
    def __init__(self):
        pass

    def draw(self, ax, x):
        ls = viewer_utils.plot_frame(ax, x)
        self.lx = ls[0]
        self.ly = ls[1]
        self.lz = ls[2]
        self.h, = ax.plot([x[0]], [x[1]], [x[2]],
                          color=".5", linestyle="", marker=".")

    def update(self, x):
        ll = viewer_utils.update_frame([self.lx, self.ly, self.lz], x)
        self.h.set_xdata([x[0]])
        self.h.set_ydata([x[1]])
        self.h.set_3d_properties([x[2]])
        return ll + [self.h]

    def draw_traj_minimal(self, ax, Xs):
        xs = [p[0] for p in Xs]
        ys = [p[1] for p in Xs]

        # print("xs is")
        # print(xs)
        # ys =  []
        # for i,x in enumerate(xs):
        #     print(i,x)
        #     ys.append(x[1])
        #
        # ys = [p[1] for p in xs]
        zs = [p[2] for p in Xs]
        ax.plot3D(xs, ys, zs, 'gray')

    def draw_basic(self, ax, x):
        self.draw(ax, x)


class Quad3dViewer(RobotViewer):
    """
    """

    def __init__(self):

        super().__init__(Robot)
        self.labels_x = [
            "x",
            "y",
            "z",
            "qx",
            "qy",
            "qz",
            "qw",
            "vx",
            "vy",
            "vz",
            "wx",
            "wy",
            "wz"]
        self.labels_u = ["f1", "f2", "f3", "f4"]

    def is_3d(self) -> bool:
        return True

    def view_problem(self, ax, env, **kwargs):

        if isinstance(env, str):
            with open(env, "r") as f:
                env = yaml.safe_load(f)

        print(env)
        lb = env["environment"]["min"]
        ub = env["environment"]["max"]
        obstacles = env["environment"]["obstacles"]
        start = np.array(env["robots"][0]["start"])
        goal = np.array(env["robots"][0]["goal"])

        for o in obstacles:
            if o["type"] == "box":
                viewer_utils.draw_cube(ax, o["center"], o["size"])
            if o["type"] == "sphere":
                viewer_utils.plt_sphere(ax, [o["center"]], [o["size"]])

        r = Robot()
        r.draw(ax, start)

        r = Robot()
        r.draw(ax, goal)

        ele = 30
        azm = -40

        ax.view_init(elev=ele, azim=azm)  # Reproduce view
        ax.axes.set_xlim3d(left=lb[0], right=ub[0])
        ax.axes.set_ylim3d(bottom=lb[1], top=ub[1])
        ax.axes.set_zlim3d(bottom=lb[2], top=ub[2])

        # ax.set_box_aspect((1,1,1))
        ax.set_xlabel("x")
        ax.set_ylabel("y")
        ax.set_zlabel("z")

    # def view_trajectory(self,ax,result, **kwargs):
    #     """
    #     """
    #
    #     if isinstance(result, str):
    #         with open(result) as f:
    #             __result = yaml.safe_load(f)
    #             result = __result["result"][0]
    #
    #     states = result["states"]
    #     xs = [p[0] for p in states]
    #     ys = [p[1] for p in states]
    #     zs = [p[2] for p in states]
    #     ax.plot3D(xs, ys, zs, 'gray')
    #
    #     plot_orientation_every = 50
    #     for i in range(0, len(states) , plot_orientation_every):
    #         print(f"states[i] {states[i]}")
    #         r = Robot()
    #         r.draw(ax,states[i])

    # def view_state(self,ax,state, facecolor='none' , edgecolor='black', **kwargs):
    #     """
    #     """
    #     r = Robot()
    #     r.draw(ax,state)

    # def plot_traj(self,axs,result, **kwargs):
    #     """
    #     """
    #
    #     xs = result["states"]
    #     us = result["actions"]
    #
    #     for i,l in enumerate( labels_x) :
    #         xi = [ x[i] for x in xs]
    #         axs[0].plot(xi, label = l)
    #     axs[0].legend()
    #
    #     for i,l in enumerate( labels_u) :
    #         ui = [ u[i] for u in us]
    #         axs[1].plot(ui, label = l)
    #     axs[1].legend()

    def make_video(self, env, result, filename_video: str = ""):

        fig = plt.figure()
        ax = plt.axes(projection='3d')
        self.view_problem(ax, env)

        if isinstance(result, str):
            with open(result) as f:
                __result = yaml.safe_load(f)
                result = __result["result"][0]

        states = result["states"]

        r = Robot()
        r.draw(ax, states[0])
        print(r)

        states = result["states"]
        print("hello")
        print(f"states {states}")

        r.draw_traj_minimal(ax, states)

        def animate_func(i):
            """
            """
            state = states[i]
            return r.update(state)

        T = len(states)

        anim = animation.FuncAnimation(fig, animate_func,
                                       frames=T,
                                       interval=5,
                                       blit=True)

        if len(filename_video):
            speed = 10
            print(f"saving video: {filename_video}")
            anim.save(filename_video, "ffmpeg", fps=10 * speed, dpi=100)
            print(f"saving video: {filename_video} -- DONE")

        else:
            plt.show()


if __name__ == "__main__":

    viewer = Quad3dViewer()
    viewer_utils.check_viewer(viewer, is_3d=True)

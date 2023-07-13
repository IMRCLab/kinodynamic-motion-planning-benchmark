from .basic import Sphere, Line, Arrow, Sphere2

import numpy as np


class Uav:
    '''
    Draws a quadrotor at a given position, with a given attitude.
    '''

    def __init__(self, ax, arm_length):
        '''
        Initialize the quadrotr plotting parameters.

        Params:
            ax: (matplotlib axis) the axis where the sphere should be drawn
            arm_length: (float) length of the quadrotor arm

        Returns:
            None
        '''

        self.ax = ax
        self.arm_length = arm_length

        self.b1 = np.array([1.0, 0.0, 0.0]).T
        self.b2 = np.array([0.0, 1.0, 0.0]).T
        self.b3 = np.array([0.0, 0.0, 1.0]).T

        # Center of the quadrotor
        self.body = Sphere2(self.ax, 0.08, 'y')

        # Each motor
        self.motor1 = Sphere2(self.ax, 0.05, 'r')
        self.motor2 = Sphere2(self.ax, 0.05, 'g')
        self.motor3 = Sphere2(self.ax, 0.05, 'b')
        self.motor4 = Sphere2(self.ax, 0.05, 'b')

        # Arrows for the each body axis
        self.arrow_b1 = Arrow(ax, self.b1, 'r', length=0.5)
        self.arrow_b2 = Arrow(ax, self.b2, 'g', length=0.5)
        self.arrow_b3 = Arrow(ax, self.b3, 'b', length=0.5)

        # Quadrotor arms
        self.arm_b1 = Line(ax, 'k')
        self.arm_b2 = Line(ax, 'k')
        self.arm_b3 = Line(ax, 'k')
        self.arm_b4 = Line(ax, 'k')

    def delete(self):
        # Center of the quadrotor
        self.body.delete()

        # Each motor
        self.motor1.delete()
        self.motor2.delete()
        self.motor3.delete()
        self.motor4.delete()

        # Arrows for the each body axis
        self.arrow_b1.delete()
        self.arrow_b2.delete()
        self.arrow_b3.delete()

        # Quadrotor arms
        self.arm_b1.delete()
        self.arm_b2.delete()
        self.arm_b3.delete()
        self.arm_b4.delete()

    def draw_at(self, x=np.array([0.0, 0.0, 0.0]).T, R=np.eye(3), **kwargs):
        '''
        Draw the quadrotor at a given position, with a given direction

        Args:
            x: (3x1 numpy.ndarray) position of the center of the quadrotor,
                default = [0.0, 0.0, 0.0]
            R: (3x3 numpy.ndarray) attitude of the quadrotor in SO(3)
                default = eye(3)

        Returns:
            None
        '''

        # First, clear the axis of all the previous plots
        # self.ax.clear()

        # Center of the quadrotor
        self.body.draw_at(x, **kwargs)

        # Each motor
        self.motor1.draw_at(x + R.dot(self.b1) * self.arm_length, **kwargs)
        self.motor2.draw_at(x + R.dot(self.b2) * self.arm_length, **kwargs)
        self.motor3.draw_at(x + R.dot(-self.b1) * self.arm_length, **kwargs)
        self.motor4.draw_at(x + R.dot(-self.b2) * self.arm_length, **kwargs)

        # Arrows for the each body axis
        # self.arrow_b1.draw_from_to(x, R.dot(self.b1) * self.arm_length * 1.8)
        # self.arrow_b2.draw_from_to(x, R.dot(self.b2) * self.arm_length * 1.8)
        self.arrow_b3.draw_from_to(
            x,
            R.dot(
                self.b3) *
            self.arm_length *
            1.8,
            **kwargs)

        # Quadrotor arms
        self.arm_b1.draw_from_to(
            x, x + R.dot(-self.b1) * self.arm_length, **kwargs)
        self.arm_b2.draw_from_to(
            x, x + R.dot(-self.b2) * self.arm_length, **kwargs)

        self.arm_b3.draw_from_to(
            x,
            x +
            R.dot(
                self.b1) *
            self.arm_length,
            **kwargs)
        self.arm_b4.draw_from_to(
            x,
            x +
            R.dot(
                self.b2) *
            self.arm_length,
            **kwargs)


if __name__ == '__main__':
    from utils import ypr_to_R

    from matplotlib import animation
    from mpl_toolkits.mplot3d import Axes3D

    import matplotlib.pyplot as plt

    def update_plot(i, x, R):
        uav_plot.draw_at(x[:, i], R[:, :, i])

        # These limits must be set manually since we use
        # a different axis frame configuration than the
        # one matplotlib uses.
        xmin, xmax = -2, 2
        ymin, ymax = -2, 2
        zmin, zmax = -2, 2

        ax.set_xlim([xmin, xmax])
        ax.set_ylim([ymax, ymin])
        ax.set_zlim([zmax, zmin])

    # Initiate the plot
    plt.style.use('seaborn')

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    arm_length = 0.24  # in meters
    uav_plot = Uav(ax, arm_length)

    # Create some fake simulation data
    steps = 60
    t_end = 1

    x = np.zeros((3, steps))
    x[0, :] = np.arange(0, t_end, t_end / steps)
    x[1, :] = np.arange(0, t_end, t_end / steps) * 2

    R = np.zeros((3, 3, steps))
    for i in range(steps):
        ypr = np.array([i, 0.1 * i, 0.0])
        R[:, :, i] = ypr_to_R(ypr, degrees=True)

    # Run the simulation
    ani = animation.FuncAnimation(fig, update_plot, frames=steps,
                                  fargs=(x, R,))

    plt.show()

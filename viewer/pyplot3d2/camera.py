from .basic import Line, Sphere

import numpy as np


class Camera:
    '''
    Draws a line at a given position, with a given attitude.
    '''

    def __init__(self, ax, c='b', x=np.array([0.0, 0.0, 0.0]).T, R=np.eye(3)):
        '''
        Initialize the camera.
        Params:
            ax: (matplotlib axis) the axis where the line should be drawn
            direction: (3x1 numpy.ndarray) direction of the arrow
            c: (string) color of the arrow, default = 'b'
            x: (3x1 numpy.ndarray) origin of the camera,
                default = [0.0, 0.0, 0.0]
            R: (3x1 numpy.ndarray) attitude of the camera,
                default = eye(3)

        Returns:
            None
        '''

        self.ax = ax
        self.color = c
        self.x = x
        self.R = R

        d = 0.3
        w = 0.2
        h = 0.1
        p1 = np.array([d, w, h])
        p2 = np.array([d, -w, h])
        p3 = np.array([d, -w, -h])
        p4 = np.array([d, w, -h])
        self.l1 = Line(self.ax, 'b', x, p1)
        self.l2 = Line(self.ax, 'b', x, p2)
        self.l3 = Line(self.ax, 'b', x, p3)
        self.l4 = Line(self.ax, 'b', x, p4)
        self.l5 = Line(self.ax, 'r', p1, p2)
        self.l6 = Line(self.ax, 'r', p2, p3)
        self.l7 = Line(self.ax, 'r', p3, p4)
        self.l8 = Line(self.ax, 'r', p4, p1)

        self.origin = Sphere(self.ax, 0.02, 'y')

    def draw(self):
        '''
        Draw a camera with the initially defined parameter when the class was
        instantiated.
        Args:
            None

        Returns:
            None
        '''

        self.l1.draw()
        self.l2.draw()
        self.l3.draw()
        self.l4.draw()
        self.l5.draw()
        self.l6.draw()
        self.l7.draw()
        self.l8.draw()
        self.origin.draw()

    def draw_at(self, x=np.array([0.0, 0.0, 0.0]).T, R=np.eye(3)):
        '''
        Draw the camera at a given point and attitude.
        Args:
            x: (3x1 numpy.ndarray) position of camera,
                default = [0.0, 0.0, 0.0]
            R: (3x1 numpy.ndarray) attitude of the camera,
                default = eye(3)

        Returns:
            None
        '''

        d = 0.5
        w = 0.4
        h = 0.3
        p1 = x + R@np.array([d, w, h])
        p2 = x + R@np.array([d, -w, h])
        p3 = x + R@np.array([d, -w, -h])
        p4 = x + R@np.array([d, w, -h])

        self.l1.draw_from_to(x, p1)
        self.l2.draw_from_to(x, p2)
        self.l3.draw_from_to(x, p3)
        self.l4.draw_from_to(x, p4)
        self.l5.draw_from_to(p1, p2)
        self.l6.draw_from_to(p2, p3)
        self.l7.draw_from_to(p3, p4)
        self.l8.draw_from_to(p4, p1)
        self.origin.draw_at(x)


if __name__ == '__main__':
    from utils import ypr_to_R
    from mpl_toolkits.mplot3d import Axes3D

    import numpy as np
    import matplotlib.pyplot as plt

    plt.style.use('seaborn')
    fig = plt.figure()
    ax = fig.gca(projection='3d')

    camera = Camera(ax)
    camera.draw_at([1, 1, 3], ypr_to_R([0, np.pi / 4, 0]))

    plt.show()

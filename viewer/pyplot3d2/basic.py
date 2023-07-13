import matplotlib.pyplot as plt
import numpy as np

from mpl_toolkits.mplot3d import Axes3D

from .utils import ypr_to_R


class Sphere2:

    def __init__(self, ax, r, c='b', x0=np.array([0, 0, 0]).T):
        '''
        Initialize the sphere.

        Params:
            ax: (matplotlib axis) the axis where the sphere should be drawn
            r: (float) radius of the sphere
            c: (string) color of the sphere, default 'b'
            x0: (3x1 numpy.ndarray) initial position of the sphere, default
                is [0, 0, 0]
            resolution: (int) resolution of the plot, default 20

        Returns:
            None
        '''

        self.ax = ax
        self.r = r
        self.color = c
        self.x0 = x0
        self.surface = None

    def draw(self):
        self.draw_at()

    def draw_at(self, position=np.array([0.0, 0.0, 0.0]).T, **kwargs):
        '''
        Draw the sphere at a given position.

        Args:
            position: (3x1 numpy.ndarray) position of the sphere,
                default = [0.0, 0.0, 0.0]

        Returns:
            None
        '''

        color = kwargs.pop('color', self.color)
        self.surface = self.ax.scatter([position[0]], [position[1]], [
            position[2]], marker="o", color=color, **kwargs)

    def delete(self):
        '''
        Delete the sphere from the plot.

        Args:
            None

        Returns:
            None
        '''

        if self.surface is not None:
            self.surface.remove()


class Sphere:
    '''
    Draws a sphere at a given position.
    '''

    def __init__(self, ax, r, c='b', x0=np.array([0, 0, 0]).T, resolution=20):
        '''
        Initialize the sphere.

        Params:
            ax: (matplotlib axis) the axis where the sphere should be drawn
            r: (float) radius of the sphere
            c: (string) color of the sphere, default 'b'
            x0: (3x1 numpy.ndarray) initial position of the sphere, default
                is [0, 0, 0]
            resolution: (int) resolution of the plot, default 20

        Returns:
            None
        '''

        self.ax = ax
        self.r = r
        self.color = c
        self.x0 = x0
        self.reso = resolution
        self.surface = None

    def draw(self):
        self.draw_at()

    def draw_at(self, position=np.array([0.0, 0.0, 0.0]).T):
        '''
        Draw the sphere at a given position.

        Args:
            position: (3x1 numpy.ndarray) position of the sphere,
                default = [0.0, 0.0, 0.0]

        Returns:
            None
        '''

        vertices = np.linspace(0, 2 * np.pi, self.reso + 1)
        u, v = np.meshgrid(vertices, vertices)

        x = self.r * np.cos(u) * np.sin(v) + position[0]
        y = self.r * np.sin(u) * np.sin(v) + position[1]
        z = self.r * np.cos(v) + position[2]

        self.surface = self.ax.plot_surface(x, y, z, color=self.color)

    def delete(self):
        '''
        Delete the sphere from the plot.

        Args:
            None

        Returns:
            None
        '''

        if self.surface is not None:
            self.surface.remove()


class Arrow:
    '''
    Draws an arrow at a given position, with a given attitude.
    '''

    def __init__(self, ax, direction, c='b', x0=np.array([0.0, 0.0, 0.0]).T,
                 length=1.0):
        '''
        Initialize the arrow.

        Params:
            ax: (matplotlib axis) the axis where the arrow should be drawn
            direction: (3x1 numpy.ndarray) direction of the arrow
            c: (string) color of the arrow, default = 'b'
            x0: (3x1 numpy.ndarray) origin of the arrow,
                default = [0.0, 0.0, 0.0]
            length: (float) length of the arrow, default = 1.0

        Returns:
            None
        '''

        self.ax = ax
        self.u0 = direction
        self.color = c
        self.x0 = x0
        self.arrow_length = length
        self.arrow = None

    def draw(self):
        '''
        Draw the arrow with the initially defined parameter when the class was
        instantiated.

        Args:
            None

        Returns:
            None
        '''

        x = self.x0
        u = self.u0

        self.arrow = self.ax.quiver(x[0], x[1], x[1],
                                    u[0], u[1], u[2],
                                    color=self.color,
                                    length=self.arrow_length,
                                    normalize=False)

    def delete(self):
        if self.arrow is not None:
            self.arrow.remove()

    def draw_from_to(self, x=np.array([0.0, 0.0, 0.0]).T,
                     u=np.array([1.0, 0.0, 0.0]).T, **kwargs):
        '''
        Draw the arrow at a given position, with a given direction

        Args:
            x: (3x1 numpy.ndarray) origin of the arrow,
                default = [0.0, 0.0, 0.0]
            u: (3x1 numpy.ndarray) direction of the arrow,
                default = [1.0, 0.0, 0.0]

        Returns:
            None
        '''

        color = kwargs.pop('color', self.color)
        self.arrow = self.ax.quiver(x[0], x[1], x[2],
                                    u[0], u[1], u[2],
                                    color=color,
                                    length=self.arrow_length,
                                    normalize=False)


class Line:
    '''
    Draws a line at a given position, with a given attitude.
    '''

    def __init__(self, ax, c='b', x0=np.array([0.0, 0.0, 0.0]).T,
                 x1=np.array([1.0, 0.0, 0.0]).T):
        '''
        Initialize the line.
        Params:
            ax: (matplotlib axis) the axis where the line should be drawn
            c: (string) color of the arrow, default = 'b'
            x0: (3x1 numpy.ndarray) start of the line,
                default = [0.0, 0.0, 0.0]
            x1: (3x1 numpy.ndarray) end of the line,
                default = [1.0, 0.0, 0.0]

        Returns:
            None
        '''

        self.ax = ax
        self.color = c
        self.x0 = x0
        self.x1 = x1
        self.line = None

    def draw(self):
        '''
        Draw the line with the initially defined parameter when the class was
        instantiated.
        Args:
            None

        Returns:
            None
        '''

        self.line = self.ax.plot([self.x0[0], self.x1[0]],
                                 [self.x0[1], self.x1[1]],
                                 [self.x0[2], self.x1[2]],
                                 color=self.color)

    def delete(self):
        if self.line is not None:
            self.line.pop(0).remove()

    def draw_from_to(self, x0=np.array([0.0, 0.0, 0.0]).T,
                     x1=np.array([1.0, 0.0, 0.0]).T, **kwargs):
        '''
        Draw the line between two points.
        Args:
            x0: (3x1 numpy.ndarray) start of the line,
                default = [0.0, 0.0, 0.0]
            x1: (3x1 numpy.ndarray) end of the line,
                default = [1.0, 0.0, 0.0]

        Returns:
            None
        '''
        color = kwargs.pop('color', self.color)

        self.line = self.ax.plot([x0[0], x1[0]],
                                 [x0[1], x1[1]],
                                 [x0[2], x1[2]],
                                 color=color, **kwargs)


class Plane:
    '''
    Draws a plane at a given position.
    '''

    def __init__(self, ax, h, w, c='b', x=np.array([0, 0, 0]).T,
                 R=np.eye(3), resolution=1):
        '''
        Initialize the sphere.
        Params:
            ax: (matplotlib axis) the axis where the plane should be drawn
            h = (float): height of the plane (x axis)
            w = (float): width of the plane (y axis)
            c: (string) color of the plane, default 'b'
            x: (3x1 numpy.ndarray) initial position of the plane, default
                is [0, 0, 0]
            R: (3x1 numpy.ndarray) attitude of the plane,
                default = eye(3)
            resolution: (int) resolution of the plot, default 1

        Returns:
            None
        '''

        self.ax = ax
        self.h = h
        self.w = w
        self.color = c
        self.x = x
        self.R = R
        self.reso = resolution

        self.uvw = np.array([])
        self.mesh_shape = (1, 1)

    def draw(self):
        '''
        Draw the plane with the initially defined position when the class was
        instantiated.
        Args:
            None

        Returns:
            None
        '''

        if self.uvw.size == 0:
            reso = self.reso
            h = self.h / 2.0
            w = self.w / 2.0

            vertices_h = np.linspace(-h, h, reso + 1)
            vertices_w = np.linspace(-w, w, reso + 1)

            u, v = np.meshgrid(vertices_h, vertices_w)
            w = u * 0.0

            self.mesh_shape = np.shape(u)
            self.uvw = np.array([u.ravel(), v.ravel(), w.ravel()])

        # NOTE: for higher resolutions, raveling and reshpaing can be
        # expensive. Replace this with np.einsum.
        uvw = self.R @ self.uvw

        self.ax.plot_surface(
            uvw[0, :].reshape(self.mesh_shape) + float(self.x[0]),
            uvw[1, :].reshape(self.mesh_shape) + float(self.x[1]),
            uvw[2, :].reshape(self.mesh_shape) + float(self.x[2]),
            color=self.color)

    def draw_at(self, x=np.array([0.0, 0.0, 0.0]).T, R=np.eye(3), **kwargs):
        '''
        Draw the plane at a given position and attitude.

        Args:
            x: (3x1 numpy.ndarray) position of plane,
                default = [0.0, 0.0, 0.0]
            R: (3x1 numpy.ndarray) attitude of the plane,
                default = eye(3)

        Returns:
            None
        '''

        if self.uvw.size == 0:
            reso = self.reso
            h = self.h / 2.0
            w = self.w / 2.0

            vertices_h = np.linspace(-h, h, reso + 1)
            vertices_w = np.linspace(-w, w, reso + 1)

            u, v = np.meshgrid(vertices_h, vertices_w)
            w = u * 0.0

            self.mesh_shape = np.shape(u)
            self.uvw = np.array([u.ravel(), v.ravel(), w.ravel()])

        # NOTE: for higher resolutions, raveling and reshpaing can be
        # expensive. Replace this with np.einsum.
        uvw = R @ self.uvw

        self.ax.plot_surface(
            uvw[0, :].reshape(self.mesh_shape) + float(x[0]),
            uvw[1, :].reshape(self.mesh_shape) + float(x[1]),
            uvw[2, :].reshape(self.mesh_shape) + float(x[2]),
            color=self.color)


class Cube:
    '''
    Draws a cube at a given position.
    '''

    def __init__(self, ax, dimensions, c='b', x=np.array([0, 0, 0]).T,
                 R=np.eye(3), resolution=10):
        '''
        Initialize the cube.
        Params:
            ax: (matplotlib axis) the axis where the cube should be drawn
            dimensions = (3x1 numpy.ndarray): dimensions along each axis
            c: (string) color of the cube, default 'b'
            x: (3x1 numpy.ndarray) initial position of the cube, default
                is [0, 0, 0]
            R: (3x1 numpy.ndarray) attitude of the cube,
                default = eye(3)
            resolution: (int) resolution of the plot, default 10

        Returns:
            None
        '''

        self.ax = ax
        self.d1 = dimensions[0]
        self.d2 = dimensions[1]
        self.d3 = dimensions[2]
        self.color = c
        self.x = x
        self.R = R
        self.reso = resolution

        theta = np.pi / 2.0

        self.pt1 = np.array([self.d1 / 2.0, 0.0, 0.0])
        self.R1 = ypr_to_R([0.0, theta, 0.0])
        self.p1 = Plane(self.ax, self.d3, self.d2, 'r',
                        self.pt1, self.R1)

        self.pt2 = np.array([-self.d1 / 2.0, 0.0, 0.0])
        self.R2 = ypr_to_R([0.0, -theta, 0.0])
        self.p2 = Plane(self.ax, self.d3, self.d2, 'r',
                        self.pt2, self.R2)

        self.pt3 = np.array([0.0, self.d2 / 2.0, 0.0])
        self.R3 = ypr_to_R([0.0, 0.0, -theta])
        self.p3 = Plane(self.ax, self.d1, self.d3, 'r',
                        self.pt3, self.R3)

        self.pt4 = np.array([0.0, -self.d2 / 2.0, 0.0])
        self.R4 = ypr_to_R([0.0, 0.0, theta])
        self.p4 = Plane(self.ax, self.d1, self.d3, 'r',
                        self.pt4, self.R4)

        self.pt5 = np.array([0.0, 0.0, self.d3 / 2.0])
        self.R5 = np.eye(3)
        self.p5 = Plane(self.ax, self.d1, self.d2, 'r',
                        self.pt5, self.R5)

        self.pt6 = np.array([0.0, 0.0, -self.d3 / 2.0])
        self.R6 = np.eye(3)
        self.p6 = Plane(self.ax, self.d1, self.d2, 'r',
                        self.pt6, self.R6)

    def draw(self):
        '''
        Draw the cube with the initially defined position when the class was
        instantiated.
        Args:
            None

        Returns:
            None
        '''

        self.p1.draw()
        self.p2.draw()
        self.p3.draw()
        self.p4.draw()
        self.p5.draw()
        self.p6.draw()

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

        raise NotImplementedError('This has not been implemented correctly')
        # self.p1.draw_at(x + R@self.R1@self.pt1, R@self.R1)
        # self.p2.draw_at(x + R@self.R2@self.pt2, R@self.R2)
        # self.p3.draw_at(x + R@self.R3@self.pt3, R@self.R3)
        # self.p4.draw_at(x + R@self.R4@self.pt4, R@self.R4)
        # self.p5.draw_at(x + R@self.R5@self.pt5, R@self.R5)
        # self.p6.draw_at(x + R@self.R6@self.pt6, R@self.R6)

        self.p1.draw_at(x + self.pt1, R @ self.R1)
        self.p2.draw_at(x + self.pt2, R @ self.R2)
        self.p3.draw_at(x + self.pt3, R @ self.R3)
        self.p4.draw_at(x + self.pt4, R @ self.R4)
        self.p5.draw_at(x + self.pt5, R @ self.R5)
        self.p6.draw_at(x + self.pt6, R @ self.R6)


if __name__ == '__main__':

    # Initiate the plot
    plt.style.use('seaborn')

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # s1 = Sphere(ax, 1)
    # s1.draw()

    # R = ypr_to_R([0, 0, np.pi/2.0])
    # p1 = Plane(ax, 3, 2, 'r', [0, 0, 1], R)
    # p1.draw()

    c1 = Cube(ax, [3, 4, 5])
    # c1.draw_at([1,0,0], ypr_to_R([0,0,0]))
    c1.draw_at([0, 0, 0], ypr_to_R([np.pi / 4, 0, 0]))

    ax.set_xlim([-5, 5])
    ax.set_ylim([-5, 5])
    ax.set_zlim([-5, 5])

    plt.show()

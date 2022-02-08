import numpy as np
import subprocess
import time
import rowan
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
# import matplotlib.patches as mpatches
# from mpl_toolkits.mplot3d import Axes3D
from matplotlib.backends.backend_pdf import PdfPages

# visualization related
import meshcat
import meshcat.geometry as g
import meshcat.transformations as tf

# from quadrotor_autograd import QuadrotorAutograd

def animate(data):
    vis = meshcat.Visualizer()
    vis.open()

    vis["/Cameras/default"].set_transform(
        tf.translation_matrix([0, 0, 0]).dot(
        tf.euler_matrix(0, np.radians(-30), -np.pi/2)))

    vis["/Cameras/default/rotated/<object>"].set_transform(
        tf.translation_matrix([1, 0, 0]))

    vis["Quadrotor"].set_object(
        g.StlMeshGeometry.from_file('../benchmark/quadrotor/crazyflie2.stl'))

    while True:
        for row in data:
            vis["Quadrotor"].set_transform(
                tf.translation_matrix([row[0], row[1], row[2]]).dot(
                    tf.quaternion_matrix(row[6:10])))
            time.sleep(0.1)


def generatePDF(data, data_propagated, filename):
    pp = PdfPages(filename)

    fig, axs = plt.subplots(2, 3, sharex='all', sharey='row')
    for k, name in enumerate(['x', 'y', 'z']):
        axs[0, k].plot(data[:, k], label="KOMO")
        if data_propagated is not None:
            axs[0, k].plot(data_propagated[:, k], label="Dynamics")
        axs[0, k].set_ylabel(name + " [m]")
    axs[0,0].legend()

    for k, name in enumerate(['vx', 'vy', 'vz']):
        axs[1, k].plot(data[:, 3+k], '--', label="KOMO (estimated)")
        if data_propagated is not None:
            axs[1, k].plot(data_propagated[:, 3+k], label="Dynamics")
        axs[1, k].set_ylabel(name + " [m/s]")
        axs[1, k].set_xlabel("timestep")
    axs[1, 0].legend()

    pp.savefig(fig)
    plt.close(fig)

    fig, axs = plt.subplots(2, 3, sharex='all', sharey='row')
    rpy = np.degrees(rowan.to_euler(data[:, 6:10], 'xyz'))
    if data_propagated is not None:
        rpy_propagated = np.degrees(rowan.to_euler(data_propagated[:, 6:10], 'xyz'))
    for k, name in enumerate(['roll', 'pitch', 'yaw']):
        axs[0, k].plot(rpy[:, k], label="KOMO")
        if data_propagated is not None:
            axs[0, k].plot(rpy_propagated[:, k], label="Dynamics")
        axs[0, k].set_ylabel(name + " [deg]")
    axs[0, 0].legend()

    for k, name in enumerate(['wx', 'wy', 'wz']):
        axs[1, k].plot(np.degrees(data[:, 10+k]), '--', label="KOMO (estimated)")
        if data_propagated is not None:
            axs[1, k].plot(np.degrees(data_propagated[:, 10+k]), label="Dynamics")
        axs[1, k].set_ylabel(name + " [deg/s]")
        axs[1, k].set_xlabel("timestep")
    axs[1, 0].legend()

    pp.savefig(fig)
    plt.close(fig)

    fig, axs = plt.subplots(2, 2, sharex='all', sharey='all')

    for k, name in enumerate(['m1', 'm2', 'm3', 'm4']):
        axs[k//2,k%2].plot(data[:-1, 13 + k])
        axs[k//2, k % 2].set_ylabel(name + " [N]")
        axs[1, k % 2].set_xlabel("timestep")
        axs[k//2, k % 2].axhline(12./1000.*9.81)
    
    pp.savefig(fig)
    plt.close(fig)

    fig, ax = plt.subplots()
    ax.plot(np.sum(data[:-1, 13:17], axis=1))
    ax.set_title('actions sum')
    pp.savefig(fig)
    plt.close(fig)

    pp.close()
    subprocess.call(["xdg-open", filename])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("env", help="input file containing map")
    parser.add_argument("--result", help="output file containing solution")
    parser.add_argument("--video", help="output file for video")
    args = parser.parse_args()

    import yaml
    with open(args.result) as f:
        result = yaml.safe_load(f)

    # x,y,z, qx, qy, qz, qw, vx, vy, vz, wx, wy, wz
    states = np.array(result["result"][0]["states"])
    data = np.full((states.shape[0], 13+4), np.nan)
    data[:,0:3] = states[:,0:3]	# copy positions
    data[:,3:6] = states[:,7:10]	# copy velocity
    data[:,7:10] = states[:,3:6]	# copy quaternion
    data[:,6] = states[:,6]	# copy quaternion
    data[:,6:10] = rowan.normalize(data[:,6:10])
    data[:,10:13] = states[:,10:13]	# copy omega
    data[:-1,13:17] = np.array(result["result"][0]["actions"])

    generatePDF(data, None, str(Path(args.video).with_suffix(".pdf")))
    # generatePDF(data, None, 'output.pdf')
    # animate(data)

if __name__ == "__main__":
    main()

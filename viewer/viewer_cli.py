
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))

import argparse
import viewer_utils
import quad2d_viewer
import quad2dpole_viewer
import quad3d_viewer
import acrobot_viewer
import unicycle1_viewer
import unicycle2_viewer
import car_with_trailer_viewer
import robot_viewer
import sys


def get_robot_viewer(robot: str) -> robot_viewer.RobotViewer:
    if robot == "quad3d":
        viewer = quad3d_viewer.Quad3dViewer()
    elif robot == "unicycle1":
        viewer = unicycle1_viewer.Unicycle1Viewer()

    elif robot == "unicycle2":
        viewer = unicycle2_viewer.Unicycle2Viewer()

    elif robot == "quad2d":
        viewer = quad2d_viewer.Quad2dViewer()

    elif robot == "quad2dpole":
        viewer = quad2dpole_viewer.Quad2dpoleViewer()

    elif robot == "acrobot":
        viewer = acrobot_viewer.AcrobotViewer()

    elif robot == "car":
        viewer = car_with_trailer_viewer.CarWithTrailerViewer()
    else:
        raise NotImplementedError("unknown model " + robot)
    return viewer


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--env", help="input file containing map")
    parser.add_argument("--result", help="output file containing solution")
    parser.add_argument("--robot", help="output file containing solution")
    args = parser.parse_args()

    viewer = get_robot_viewer(args.robot)

    viewer_utils.check_viewer(
        viewer, ["--env", args.env, "--result", args.result])

import argparse
import math
import yaml
from typing import List
from translate_utils import Box

# example of input:
# environment:
#   dimensions: [3.5, 1.5]
#   obstacles:
#     - type: box
#       center: [0.7, 0.2]
#       size: [0.5, 0.25]
#     - type: box
#       center: [2.7, 0.2]
#       size: [0.5, 0.25]
# robots:
#   - type: car_first_order_with_1_trailers_0
#     start: [0.7,0.6,0,0] # x,y,theta0,theta1
#     goal: [1.9,0.2,0,0] # x,y,theta0,theta1


class Robot:
    def __init__(self, start: List[float], goal: List[float], ID: int):
        assert len(start) == 4
        assert len(goal) == 4
        self.start = start
        self.goal = goal
        # NOTE: size of car and trailer is hardcoded in the .g file
        self.ID = ID
        self.include = "Include: '../src/car_with_trailer_base.g'\n"
        self.name_robot = "R_"
        self.name_goal = "GOAL_"
        self.car_start = start[0:3] 
        # Conversion of theta1 between Quim and Valle
        # Check Quim Drawing!
        self.trailer_start = math.degrees(math.pi/2 - start[3] + start[2])

        self.car_goal = goal[0:3]
        self.trailer_goal = math.degrees(math.pi/2 - goal[3] + goal[2])

    def to_g(self) -> str:

        # robot
        out = "Prefix: \"{}\"\n".format(self.name_robot)
        out += self.include
        out += "Edit {}robot {{ q:[{}]}}\n".format(self.name_robot,
                                           ' '.join(map(str, self.car_start)))
        out += "Edit {}front_wheel {{ q:[{}] }}\n".format(
            self.name_robot, 0)
        out += "Edit {}arm {{ q:[{}] }}\n".format(self.name_robot, self.trailer_start * math.pi / 180)

        # goal
        out += "Prefix: \"{}\"\n".format(self.name_goal)
        out += self.include

        out += "Edit {}robot {{ joint:rigid, Q:<t({} 0) d({} 0 0 1)>  }}\n".format(
                self.name_goal, ' '.join(map(str, self.car_goal[:2])), math.degrees(self.car_goal[2]))
        # there is no goal in the from wheel
        out += "Edit {}front_wheel {{ joint:rigid }}\n".format(self.name_goal)
        out += "Edit {}arm {{ joint:rigid, Q:<t(0 0 0) d({} 0 0 1)>}}\n".format(
            self.name_goal, self.trailer_goal)

        return out


def write(file_in: str, file_out: str) -> None:

    with open(file_in) as env_file:
        env = yaml.safe_load(env_file)

    boxes: List[Box] = []
    robots: List[Robot] = []
    i = 0
    for obstacle in env["environment"]["obstacles"]:
        if obstacle["type"] == "box":
            boxes.append(Box(obstacle["center"], obstacle["size"], i))
            i += 1
    ID_robot = 0
    assert len(env["robots"]) == 1
    for robot in env["robots"]:
        robots.append(Robot(robot["start"], robot["goal"], ID_robot))
        ID_robot += 1

    with open(file_out, "w") as g_file:
        g_file.write("world{ X:<t(0 0 0.5)>}\n")
        for i in boxes:
            g_file.write(i.to_g())
        for i in robots:
            g_file.write(i.to_g())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin", required=True, help="yaml file")
    parser.add_argument("--fout", required=True, help="g file")
    args = parser.parse_args()

    write(args.fin, args.fout)


if __name__ == "__main__":

    main()

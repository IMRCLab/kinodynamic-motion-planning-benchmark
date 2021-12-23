import argparse
import math
import yaml
from typing import List


class Robot:
    def __init__(self, start: List[float], goal: List[float], ID: int):
        self.start = start
        self.goal = goal
        self.size = [0.5, .25]
        self.ID = ID

    def to_g(self) -> str:
        start_pos = self.start[:2] + [.5]
        goal_pos = self.goal[:2] + [.5]
        size3 = self.size + [1]

        out0 = ("start{} {{X:<t({}) d({} 0 0 1)>, shape:ssBox, size:"
                "[{} 0.05],  color:[1 0 0 .5]}}\n").format(
            self.ID, ' '.join(map(str, start_pos)),
            180 / math.pi * self.start[2], ' '.join(map(str, size3)))

        out0f = "fstart{}(start{}) {{shape:marker, size:[.2]}}\n".format(
            self.ID, self.ID)

        light0 = ("light_start{}(start{}) {{Q:<t(.3 0 0)>, shape:ssBox,"
                  "size:[.1 .25 1  0.05], color:[1 1 0 0.5]}}\n").format(
            self.ID, self.ID)

        out1 = ("robot{}(world) {{joint:transXYPhi, q:[{}], shape:ssBox,"
                "size:[{}  0.05], color:[0 0 1 .5], contact}}\n").format(
            self.ID, ' '.join(map(str, self.start)), ' '.join(map(str, size3)))

        light1 = ("light_robot{}(robot{}) {{Q:<t(.3 0 0)> shape:ssBox,"
                  "size:[.1 .25 1 0.05], color:[1 1 0 0.5]}}\n").format(
            self.ID, self.ID)

        out1f = "frobot{}(robot{}) {{shape:marker, size:[.2]}}\n".format(
            self.ID, self.ID)

        out2 = ("goal{} {{X: <t({}) d({} 0 0 1)>, shape:ssBox, size:"
                "[{} 0.05],  color:[0 1 0 .5]}}\n").format(
            self.ID, ' '.join(map(str, goal_pos)), 180 /
            math.pi * self.goal[2],
            ' '.join(map(str, size3)))
        out2f = "fgoal{}(goal{}) {{shape:marker, size:[.2]}}\n".format(
            self.ID, self.ID)

        light2 = ("light_goal{}(goal{}) {{Q:<t(.3 0 0)> shape:ssBox,"
                  "size:[.1 .25 1 0.05], color:[1 1 0 0.5]}}\n").format(
            self.ID, self.ID)

        return (out0 + out0f + out1 + out1f + out2 + out2f + light0 + light1
                + light2)


class Box:
    def __init__(self, center: List[float], size: List[float], ID: int):
        self.size = size
        self.center = center
        self.color = [.2, .2, .2]
        self.ID = ID

    def to_g(self) -> str:
        size3 = self.size + [1]
        center3 = self.center + [0.5]
        out = ("obs{} {{X: <t({})>, shape:ssBox, size:[{}  0.05],"
               "color:[{}], contact}}\n").format(
            self.ID,
            ' '.join(map(str, center3)),
            ' '.join(map(str, size3)),
            ' '.join(map(str, self.color)))

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

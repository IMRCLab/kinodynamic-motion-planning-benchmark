import yaml


from typing import *
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import argparse



parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file")
parser.add_argument("-o", "--out")
parser.add_argument("-d", "--dynamics")

args = parser.parse_args()


file = args.file
outfile = args.out
dynamics = args.dynamics

if outfile is None:
    outfile = file + ".pdf"

with open(file) as f:
    data = yaml.safe_load(f)



def get_desc(name: str) -> Tuple[List[str], List[str]]:
    if name.startswith("unicycle1"):
        return (["x", "y", "theta"], ["v", "w"])
    elif name.startswith("unicycle2"):
        return (["x", "y", "theta", "v", "w"], ["a", "aa"])
    elif name.startswith("quad2d"):
        return (["x", "y", "theta", "vx", "vy", "w"], ["f1", "f2"])
    elif name.startswith("acrobot"):
        return (["q1", "q2", "w1", "w2"], ["f"])
    elif name.startswith("quad3d"):
        return (["x", "y", "z" , "qx" , "qy" , "qz" , "qw" , "vx" , "vy" , "vz" , "wx" , "wy" , "wz"], ["f1", "f2", "f3" , "f4"])
    elif name.startswith("car1"):
        return (["x", "y", "q1", "q2"], ["v","phi"])
    else:
        raise NotImplementedError(f"unknown {name}")


robot_model = args.dynamics 
x_desc, u_desc = get_desc(robot_model)

# fig, ax = plt.subplots()
# ax.set_title("Num steps: ")
# ax.hist([len(m["actions"]) for m in motions])
# pp.savefig(fig)
# plt.close(fig)

print(f"writing pdf to {outfile}")
pp = PdfPages(outfile)


fig, ax = plt.subplots()
ax.set_title("Length: ")
action = data["bins_lengths"][0]
centers = [a["center"] for a in action][1:-1:]
xs = [a["x"] for a in action] [1:-1:]
ax.plot( centers, xs) 
pp.savefig(fig)
plt.show()


for k, a in enumerate(u_desc):
    fig, ax = plt.subplots()
    ax.set_title("Action: " + a)

    action = data["bins_actions"][k]
    centers = [a["center"] for a in action][1:-1:]
    xs = [a["x"] for a in action] [1:-1:]
    ax.plot( centers, xs) 
    pp.savefig(fig)
    plt.show()
    # pp.savefig(fig)
    # plt.close(fig)

for k, s in enumerate(x_desc):
    fig, ax = plt.subplots()
    ax.set_title("state: " + s)
    state = data["bins_states"][k]

    centers = [a["center"] for a in state][1:-1:]
    xs = [a["x"] for a in state] [1:-1:]
    ax.plot( centers, xs) 
    pp.savefig(fig)
    plt.show()


    # ax.hist(all_states[:, k])
    # pp.savefig(fig)
    # plt.close(fig)

for k, s in enumerate(x_desc):

    fig, ax = plt.subplots()
    ax.set_title("Start state: " + s)
    state = data["bins_start"][k]

    centers = [a["center"] for a in state][1:-1:]
    xs = [a["x"] for a in state] [1:-1:]
    ax.plot( centers, xs) 
    pp.savefig(fig)
    plt.show()


    # fig, ax = plt.subplots()
    # ax.set_title("start state: " + s)
    # ax.hist(all_start_states[:, k])
    # pp.savefig(fig)
    # plt.close(fig)

for k, s in enumerate(x_desc):

    fig, ax = plt.subplots()
    ax.set_title("Goal state: " + s)
    state = data["bins_goals"][k]

    centers = [a["center"] for a in state][1:-1:]
    xs = [a["x"] for a in state] [1:-1:]
    ax.plot( centers, xs) 
    pp.savefig(fig)
    plt.show()


    #
    # fig, ax = plt.subplots()
    # ax.set_title("end state: " + s)
    # ax.hist(all_end_states[:, k])
    # pp.savefig(fig)
    # plt.close(fig)

pp.close()


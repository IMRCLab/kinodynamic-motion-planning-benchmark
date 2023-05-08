

import yaml
import numpy as np
import matplotlib.pyplot as plt

import argparse

filename = "debug_file.yaml"
parser = argparse.ArgumentParser()
parser.add_argument("-f", "--file", default=filename)
parser.add_argument('-quad', '--quadcopter',
                    action='store_true')  # on/off flag

args = parser.parse_args()
filename = args.file
quadcopter = args.quadcopter




def draw_tri(X,fill=None,color="k", l=.05, alpha=1.):
    x = X[0]
    y = X[1]
    t = X[2]
    pi2 = 3.1416 / 2
    ratio = 4


    if quadcopter : 
        t += pi2

    vertices = np.array([[ x + l / ratio * np.cos(t + pi2) , y + l/ratio * np.sin(t+pi2) ],
        [x +  l * np.cos(t), y + l * np.sin(t)],
        [ x + l / ratio * np.cos(t - pi2)  , y + l / ratio * np.sin(t - pi2) ]])
    t1 = plt.Polygon(vertices,fill=fill,color=color, alpha=alpha)
    plt.gca().add_patch(t1)
    plt.plot( [x], [y], '.' , alpha=alpha, color=color) 



    if len(X) == 5:
        # add a line to represent the velocity.
        l2 = l * .5
        x0 = [ x , y]
        x1 = [ x + l2 * X[3] , y + l2 * X[4]]
        plt.plot( [ x0[0] , x1[0] ] , [ x0[1] , x1[1]] , 'o-', markersize=1)




with open(filename,"r") as f:
    data = yaml.safe_load(f)


start = data["start"]
goal = data["goal"]



# xalpha_opt = data["xalphaOPT"]
# draw_tri(xalpha_opt, color="r")

x0s = data["xs0"]



for x in x0s:
    draw_tri(x, color='.8')

if "xsOPT" in data:
    XsOPT = data["xsOPT"]
    for x in XsOPT:
        draw_tri(x, color="b")



draw_tri(start,fill="g", alpha=.5,  color="g")
draw_tri(goal, fill="g", alpha=.5 , color="g")

plt.axis("equal")

plt.show()

# plot each optimization

draw_tri(start, fill="g", alpha=.5,  color="g")
draw_tri(goal, fill="g", alpha=.5, color="g")
# draw_tri(start, color="g")
# draw_tri(goal, color="r")

if "opti" in data:

    for i,o in enumerate(data["opti"]):
        x0s = o["xs0"]
        print(i)
        if "xsOPT" in o:
            xsOPT = o["xsOPT"]
            for x in xsOPT:
                draw_tri(x,fill="b", color="b", alpha=.1)

        if "xs0" in o:
            xsOPT = o["xs0"]
            for x in xsOPT:
                draw_tri(x,fill="b", color=".6", alpha=.5)



        if "goal" in o:
            g = o["goal"]
            draw_tri(g, color="r")
        if "state_alpha" in o:
            g = o["state_alpha"]
            draw_tri(g, color="r")



        for x in x0s:
            draw_tri(x,fill='.8', color='.8', alpha=.2)



    # plt.title(f"iteration {i}")

if 'xsOPT' in data:
    for x in data['xsOPT']:
        draw_tri(x, color="y")


plt.axis("equal")
plt.show()



# plot the initial guess

nx = len(data["xs0"][0])
nu = len(data["us0"][0])


color_map = [
    'tab:blue',
        'tab:orange', 'tab:green', 'tab:red', 'tab:purple', 'tab:brown', 'tab:pink', 'tab:gray', 'tab:olive', 'tab:cyan',
"blue",
"green",
"red",
]

fig, axs = plt.subplots(2)


for i in range(nx):
    xi = [ x[i] for x in data["xs0"] ] 
    xi_opt = [ x[i] for x in data["xsOPT"] ] 
    axs[0].plot(xi, "-", label=f"x{i}", alpha=.5,   color=color_map[i])
    axs[0].plot(xi_opt ,  "-", label=f"x{i}*",  color=color_map[i])

axs[0].legend()
for i in range(nu):
    ui = [ u[i] for u in data["us0"] ] 
    ui_opt = [ u[i] for u in data["usOPT"] ] 
    axs[1].plot(ui, "-", label=f"u{i}", alpha=.5,   color=color_map[i])
    axs[1].plot(ui_opt ,  "-", label=f"u{i}*",  color=color_map[i])


axs[1].legend()

plt.show()





import car_with_trailer_viewer
import matplotlib.pyplot as plt


viewer = car_with_trailer_viewer.CarWithTrailerViewer()
env = "../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml"


fig, ax = plt.subplots()

viewer.view_problem(ax, env)

p1 = [4.4, 3.12436, 0.5]
p2 = [4.0502, 3.12396, 0.5]


ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')


p1 = [4.6, 2.60269, 0.5]
p2 = [5.06671, 2.60269, 0.5]

ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')


# distance: 0.669828

p1 = [3.48023, 1.4, 0.5]
p2 = [3.48023, 1.25255, 0.5]

x = [3.6, 1, 1.55, 1.55]


viewer.view_state(ax, x)
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')


x = [5.2, 3, 1.55, .3]

p1 = [4.6, 2.92733, 0.5]
p2 = [4.54209, 2.92733, 0.5]

viewer.view_state(ax, x)
ax.plot([p1[0], p2[0]], [p1[1], p2[1]], '-o')


plt.show()

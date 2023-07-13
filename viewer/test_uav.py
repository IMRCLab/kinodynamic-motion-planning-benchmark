
from pathlib import Path
import sys
sys.path.append(str(Path(__file__).parent))


from pyplot3d.uav import Uav
from pyplot3d.utils import ypr_to_R

import numpy as np
import matplotlib.pyplot as plt

plt.style.use('seaborn')

# initialize plot
fig = plt.figure()
ax = fig.gca(projection='3d')

arm_length = 0.24  # in meters
uav = Uav(ax, arm_length)

uav.draw_at([1, 0, 0], ypr_to_R([np.pi / 2.0, 0, 0]))

plt.show()

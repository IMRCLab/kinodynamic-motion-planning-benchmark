world{ X:<t(0 0 0.5)>}
obs0 {X: <t(0.3 0.2 0.5)>, shape:ssBox, size:[0.5 0.25 1  0.05],color:[0.2 0.2 0.2], contact}
obs1 {X: <t(1.1 0.2 0.5)>, shape:ssBox, size:[0.5 0.25 1  0.05],color:[0.2 0.2 0.2], contact}
obs2 {X: <t(2.7 0.2 0.5)>, shape:ssBox, size:[0.5 0.25 1  0.05],color:[0.2 0.2 0.2], contact}
start0 {X:<t(0.7 0.7 0.5) d(0.0 0 0 1)>, shape:ssBox, size:[0.5 0.25 1 0.05],  color:[1 0 0 .5]}
fstart0(start0) {shape:marker, size:[.2]}
robot0(world) {joint:transXYPhi, q:[0.7 0.7 0], shape:ssBox,size:[0.5 0.25 1  0.05], color:[0 0 1 .5], contact}
frobot0(robot0) {shape:marker, size:[.2]}
goal0 {X: <t(1.9 0.2 0.5) d(0.0 0 0 1)>, shape:ssBox, size:[0.5 0.25 1 0.05],  color:[0 1 0 .5]}
fgoal0(goal0) {shape:marker, size:[.2]}
light_start0(start0) {Q:<t(.3 0 0)>, shape:ssBox,size:[.1 .25 1  0.05], color:[1 1 0 0.5]}
light_robot0(robot0) {Q:<t(.3 0 0)> shape:ssBox,size:[.1 .25 1 0.05], color:[1 1 0 0.5]}
light_goal0(goal0) {Q:<t(.3 0 0)> shape:ssBox,size:[.1 .25 1 0.05], color:[1 1 0 0.5]}

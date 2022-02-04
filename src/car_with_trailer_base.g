robot(world) {joint:transXYPhi, shape:marker,size:[.2]}

pivot_wheel(robot){ Q:<t(.4 0 0)> }

front_wheel(pivot_wheel) {joint:hingeZ,  shape:ssBox,size:[0.2 0.05 .5  0.01], color:[1 0 0 .5 ], contact}

wheel_joint(front_wheel){shape:marker, size:[.2]}

arm_pivot(robot) {Q:<t(0 0 0) d(90 0 0 1)>, shape:marker, size:[.2]}

arm(arm_pivot) {joint:hingeZ, shape:marker, size:[.2]}

arm_(arm) { Q:<t(.25 0 0)> , shape:ssBox, size:[.5 .05 .5 0.01] color:[.3 .3 .3 .8] }

trailer(arm_) { Q:<t(.2 0 0)> , shape:ssBox, size:[.3 .25 .5 0.01] color:[.3 .3 .3 .5] }

pivot_arm2(arm){ Q:<t(.5 0 0)> }

arm2(pivot_arm2) { shape:marker, size:[.2] }

goal_forward{ shape:marker, size:[.3], X:<t(2 .2 0) d(0 0 0 1)> }

goal{ shape:marker, size:[.3], X:<t(2 1 0) d(45 0 0 1)> }

robot_shape(robot){ shape:ssBox, Q:<t(.2 0 0 )> , size:[.5 .25 .5 .005],color:[.1 .1 .2 .4]   }

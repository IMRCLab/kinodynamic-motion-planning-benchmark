world{ X:<t(0 0 0)>}

robot(world) {joint:transXYPhi,q:[.4 .4 .3], shape:marker,size:[.2]}

# q:[0 0 1 1 0 0 0], shape:ssBox,size:[0.5 0.25 .2  0.01], color:[0 0 1 .5], contact}

robot_shape(robot){ shape:ssBox, Q:<t(.1 0 0 )> , size:[.2 .2 .2 .005],color:[0 0 .8 .5]   }
pivot_wheel(robot){
# shape:marker,size:[.1]
Q:<t(.2 0 0)>
}

front_wheel(pivot_wheel) {joint:hingeZ, q:[.0], shape:ssBox,size:[0.2 0.05 .2  0.01], color:[1 0 0 .5 ], contact}


wheel_joint(front_wheel){shape:marker, size:[.2]}

arm_pivot(robot) {Q:<t(0 0 0) d(90 0 0 1)>, shape:marker, size:[.3]}

arm(arm_pivot) {joint:hingeZ, q:[1.8], shape:marker, size:[.2]}

arm_(arm) { Q:<t(.1 0 0)> , shape:ssBox, size:[.2 .05 0.05 0.01] color:[.3 .3 .3 .5] }

trailer(arm_) { Q:<t(.2 0 0)> , shape:ssBox, size:[.2 .2 0.05 0.01] color:[.3 .3 .3 .5] }


pivot_arm2(arm){ Q:<t(.35 0 0)> }

arm2(pivot_arm2) {
# joint:hingeZ, 
# q:[.3], 
shape:marker, size:[.2]}

goal_forward{ shape:marker, size:[.3], X:<t(2 .2 0) d(0 0 0 1)> }

goal_curve{ shape:marker, size:[.3], X:<t(2 1 0) d(45 0 0 1)> }


# shape:ssBox,size:[0.1 0.1 .2  0.01], color:[0 0 0 .5], contact}




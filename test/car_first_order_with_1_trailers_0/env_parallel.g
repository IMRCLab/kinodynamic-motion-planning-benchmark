world{ X:<t(0 0 0.5)>}
obs0 {X: <t(0.7 0.2 0.5)>, shape:ssBox, size:[0.5 0.25 1  0.05],color:[0.2 0.2 0.2], contact}
obs1 {X: <t(2.7 0.2 0.5)>, shape:ssBox, size:[0.5 0.25 1  0.05],color:[0.2 0.2 0.2], contact}
Prefix: "R_"
Include: '/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/src/car_with_trailer_base.g'
Edit R_robot { q:[0.7 0.6 0]}
Edit R_front_wheel { q:[0] }
Edit R_arm { q:[1.5707963267948966] }
Prefix: "GOAL_"
Include: '/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/src/car_with_trailer_base.g'
Edit GOAL_robot { joint:rigid, Q:<t(1.9 0.2 0) d(0.0 0 0 1)>  }
Edit GOAL_front_wheel { joint:rigid }
Edit GOAL_arm { joint:rigid, Q:<t(0 0 0) d(90.0 0 0 1)>}

world{ X:<t(0 0 0.5)>}
obs0 {X: <t(4.5 3 0.5)>, shape:ssBox, size:[0.1 3 1  0.05],color:[0.2 0.2 0.2], contact}
obs1 {X: <t(3 1.5 0.5)>, shape:ssBox, size:[3.1 0.1 1  0.05],color:[0.2 0.2 0.2], contact}
obs2 {X: <t(3 4.5 0.5)>, shape:ssBox, size:[3.1 0.1 1  0.05],color:[0.2 0.2 0.2], contact}
obs3 {X: <t(1.5 4.0 0.5)>, shape:ssBox, size:[0.1 1 1  0.05],color:[0.2 0.2 0.2], contact}
obs4 {X: <t(1.5 2.0 0.5)>, shape:ssBox, size:[0.1 1 1  0.05],color:[0.2 0.2 0.2], contact}
Prefix: "R_"
Include: '/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/src/car_with_trailer_base.g'
Edit R_robot { q:[3.5 3 3.14]}
Edit R_front_wheel { q:[0] }
Edit R_arm { q:[1.5707963267948966] }
Prefix: "GOAL_"
Include: '/home/quim/stg/wolfgang/kinodynamic-motion-planning-benchmark/src/car_with_trailer_base.g'
Edit GOAL_robot { joint:rigid, Q:<t(5 3 0) d(88.8084582452776 0 0 1)>  }
Edit GOAL_front_wheel { joint:rigid }
Edit GOAL_arm { joint:rigid, Q:<t(0 0 0) d(90.0 0 0 1)>}

world{ X:<t(0 0 0.5)>}

Prefix: "R_"
Include: 'car_with_trailer_base.g'
Edit R_robot { q:[.4 .4 .3]}
Edit R_front_wheel { q:[0] }
Edit R_arm { q:[1.3] }

obs0 { shape:ssBox, X:<t(1.3 1.2 0.5)> , size:[.3 .3 1 .05]>}
obs2 { shape:ssBox, X:<t(1.3 .2 0.5)> , size:[.3 .3 1 .05]>}
obs3 { shape:ssBox, X:<t(.6 0 0.5)> , size:[.3 .3 1 .05]>}



Prefix: "GOAL_"
Include: 'car_with_trailer_base.g'

Edit GOAL_robot { joint:rigid, Q:<t(2 .7 .0) d(0 0 0 1)>  }
Edit GOAL_front_wheel { joint:rigid}
Edit GOAL_arm {  joint:rigid, Q:<t(0 0 0) d(90 0 0 1)> }



# def simple_flight(self):
#
#     self.name = "simple_flight"
#
#     self.x0 = np.array([0., 0., 1.,
#                         0., 0., 0.,
#                         1., 0., 0., 0.,
#                         0., 0., 0.],
#                         dtype=np.float64)
#
#     self.xf = np.array([0., 1., 1.,
#                         0., 0., 0.,
#                         1., 0., 0., 0.,
#                         0., 0., 0.], dtype=np.float64)
#
#     self.tf_min = 2.1
#     self.tf_max = 2.1
#
#     self.t_steps_scvx = 30
#     self.t_steps_komo = 30
#     self.t_steps_croco = 30
#     self.t_steps_casadi = 30
#
#     self.noise = 0.01
#
#     self.t2w = 1.4
environment:
  min: [-2, -2, 0]
  max: [2, 2, 2]
  obstacles: 
    - type: "sphere"
      center: [-0.8, 0.4, 0.2]
      size: [0.3]
    - type: "sphere"
      center: [0., 0.6, 0.5]
      size: [0.3]
    - type: "sphere"
      center: [0.8, -0.3, 0.2]
      size: [0.3]
    - type: "sphere"
      center: [-0.5, 0.2, 1.1]
      size: [0.3]
    - type: "sphere"
      center: [0., -0.4, 1.]
      size: [0.3]
    - type: "sphere"
      center: [0.5, 1., 0.8]
      size: [0.3]
    - type: "sphere"
      center: [-0.8, -0.8, 1.1]
      size: [0.3]
    - type: "sphere"
      center: [0., 0.4, 1.5]
      size: [0.3]
    - type: "sphere"
      center: [0.8, 0., 1.8]
      size: [0.3]

robots:
  - type: quad3d_v1
    start: [-0.1, -1.3, 1.,
                            0., 0., 0.,
                            1., 
                            0., 0., 0.,
                            0., 0., 0.]
    goal: [ -0.1, 1.3, 1.,
                            0., 0., 0.,
                            1., 0., 0., 0.,
                            0., 0., 0.]






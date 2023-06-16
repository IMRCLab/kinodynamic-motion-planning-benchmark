import viewer_utils
import quad2d_viewer
import quad3d_viewer
import acrobot_viewer
import unicycle1_viewer
import unicycle2_viewer
import car_with_trailer_viewer

import unittest


class TestViewers(unittest.TestCase):

    def test_quad3d_viewer(self):
        argv = ["--env", "../benchmark/quadrotor_0/quad_one_obs.yaml",
                "--result", "../test/quadrotor_0/quadrotor_0_obs_0.yaml"]
        viewer = quad3d_viewer.Quad3dViewer()
        viewer_utils.check_viewer(
            viewer,
            argv=argv,
            show_single_state=True,
            is_3d=True)

    def test_unicycle1_viewer(self):
        argv = [
            "--env",
            "../benchmark/unicycle_first_order_0/bugtrap_0.yaml",
            "--result",
            "../test/unicycle_first_order_0/guess_bugtrap_0_sol0.yaml"]
        viewer = unicycle1_viewer.Unicycle1Viewer()
        viewer_utils.check_viewer(
            viewer,
            argv=argv,
            show_single_state=True,
            is_3d=False)

    def test_unicycle2_viewer(self):
        argv = ["--env", "../benchmark/unicycle_second_order_0/parallelpark_0.yaml",
                "--result", "../test/unicycle_second_order_0/guess_parallelpark_0_sol0.yaml"]
        viewer = unicycle2_viewer.Unicycle2Viewer()
        viewer_utils.check_viewer(
            viewer,
            argv=argv,
            show_single_state=True,
            is_3d=False)

    def test_quad2d_viewer(self):
        argv = ["--env", "../benchmark/quad2d/quad_obs_recovery.yaml",
                "--result", "../test/quad2d/quad2d_recovery_good_init_guess.yaml"]
        viewer = quad2d_viewer.Quad2dViewer()
        viewer_utils.check_viewer(
            viewer,
            argv=argv,
            show_single_state=True,
            is_3d=False)

    def test_acrobot_viewer(self):
        argv = [
            "--env",
            "../benchmark/acrobot/swing_up_empty.yaml",
            "--result",
            "../benchmark/acrobot/swing_up_empty_init_guess.yaml"]
        viewer = acrobot_viewer.AcrobotViewer()
        viewer_utils.check_viewer(
            viewer,
            argv=argv,
            show_single_state=True,
            is_3d=False)

    def test_car_with_trailer(self):
        argv = ["--env",
                "../benchmark/car_first_order_with_1_trailers_0/bugtrap_0.yaml",
                "--result", "../test/car_first_order_with_1_trailers_0/guess_bugtrap_0_sol0.yaml"]
        viewer = car_with_trailer_viewer.CarWithTrailerViewer()
        viewer_utils.check_viewer(
            viewer,
            argv=argv,
            show_single_state=True,
            is_3d=False)


if __name__ == "__main__":
    unittest.main()

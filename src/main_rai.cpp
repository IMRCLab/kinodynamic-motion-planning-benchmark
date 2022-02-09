#include <iostream>

#include "KOMO/komo.h"

extern int main_unicycle(float min_v, float max_v, float min_w, float max_w);
extern int main_trailer();

int main(int argn, char **argv) {

  rai::initCmdLine(argn, argv);
  rnd.clockSeed();

  rai::String robot_type = rai::getParameter<rai::String>("robot", STRING("none"));

  if (robot_type.contains("unicycle_first_order_0")) {
    return main_unicycle(-0.5, 0.5, -0.5, 0.5);
  } else if (robot_type.contains("unicycle_first_order_1")) {
    return main_unicycle(0.25, 0.5, -0.5, 0.5);
  } else if (robot_type.contains("unicycle_first_order_2")) {
    return main_unicycle(0.0, 0.5, -0.25, 0.5);
  } else if (robot_type.contains("unicycle_second_order_0")) {
    return main_unicycle(-0.5, 0.5, -0.5, 0.5);
  } else if (robot_type.contains("trailer")) {
    return main_trailer();
  } else {
    std::cerr << "Unknown robot type: " << robot_type << std::endl;
  }

  return 1;
}

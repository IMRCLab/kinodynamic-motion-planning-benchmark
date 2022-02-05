#include <iostream>

#include "KOMO/komo.h"

extern int main_unicycle();
extern int main_trailer();

int main(int argn, char **argv) {

  rai::initCmdLine(argn, argv);
  rnd.clockSeed();

  rai::String robot_type = rai::getParameter<rai::String>("robot", STRING("none"));

  if (robot_type.contains("unicycle")) {
    return main_unicycle();
  } else if (robot_type.contains("trailer")) {
    return main_trailer();
  } else {
    std::cerr << "Unknown robot type: " << robot_type << std::endl;
  }

  return 1;
}

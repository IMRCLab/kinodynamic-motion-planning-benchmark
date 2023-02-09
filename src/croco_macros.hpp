#pragma once

#include <iostream> 


#define NAMEOF(variable) #variable
#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#define STR(x, sep) #x << sep << x

#define CHECK(A, msg)                                                          \
  if (!A) {                                                                    \
    std::cout << "CHECK failed: '" << #A << " " << A << " '"                   \
              << " -- " << msg << std::endl;                                   \
    throw std::runtime_error(msg);                                             \
  }

#define CHECK_EQ(A, B, msg)                                                    \
  if (!(A == B)) {                                                             \
    std::cout << "CHECK_EQ failed: '" << #A << "'=" << A << " '" << #B         \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

#define CHECK_GEQ(A, B, msg)                                                   \
  if (!(A >= B)) {                                                             \
    std::cout << "CHECK_GEQ failed: '" << #A << "'=" << A << " '" << #B        \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

#define CHECK_SEQ(A, B, msg)                                                   \
  if (!(A <= B)) {                                                             \
    std::cout << "CHECK_SEQ failed: '" << #A << "'=" << A << " '" << #B        \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

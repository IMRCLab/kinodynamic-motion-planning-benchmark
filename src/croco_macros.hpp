#pragma once

#include <iostream>

#define NAMEOF(variable) #variable

#define VAR_WITH_NAME(variable) variable, #variable

#define STRINGIFY(x) #x
#define TOSTRING(x) STRINGIFY(x)
#define AT __FILE__ ":" TOSTRING(__LINE__)

#define STR(x, sep) #x << sep << x

#define STR_(x) #x << ": " << x

#define CSTR_(x) std::cout << #x << ": " << x << std::endl;

#define STRY(x, out, be, af)  out <<  be << #x << af << x << std::endl

#define FMT_E                                                                  \
  Eigen::IOFormat(6, Eigen::DontAlignCols, ",", ",", "", "", "[", "]")

#define STR_VV(x, af) #x << af << x.format(FMT_E)

#define STR_V(x) #x << ": " << x.format(FMT_E)

#define CSTR_V(x) std::cout << STR_V(x) << std::endl;

#define CHECK(A, msg)                                                          \
  if (!A) {                                                                    \
    std::cout << "CHECK failed: '" << #A << " " << A << " '"                   \
              << " -- " << msg << std::endl;                                   \
    throw std::runtime_error(msg);                                             \
  }

#define WARN(A, msg)                                                           \
  if (!A) {                                                                    \
    std::cout << "CHECK failed: '" << #A << " " << A << " '"                   \
              << " -- " << msg << std::endl;                                   \
  }

#define CHECK_EQ(A, B, msg)                                                    \
  if (!(A == B)) {                                                             \
    std::cout << "CHECK_EQ failed: '" << #A << "'=" << A << " '" << #B         \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

#define CHECK_NEQ(A, B, msg)                                                   \
  if (A == B) {                                                                \
    std::cout << "CHECK_NEQ failed: '" << #A << "'=" << A << " '" << #B        \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

#define CHECK_GEQ(A, B, msg)                                                   \
  if (!(A >= B)) {                                                             \
    std::cout << "CHECK_GEQ failed: '" << #A << "'=" << A << " '" << #B        \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

#define CHECK_LEQ(A, B, msg)                                                   \
  if (!(A <= B)) {                                                             \
    std::cout << "CHECK_LEQ failed: '" << #A << "'=" << A << " '" << #B        \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

#define CHECK_GE(A, B, msg)                                                    \
  if (!(A > B)) {                                                              \
    std::cout << "CHECK_GE failed: '" << #A << "'=" << A << " '" << #B         \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

#define CHECK_SEQ(A, B, msg)                                                   \
  if (!(A <= B)) {                                                             \
    std::cout << "CHECK_SEQ failed: '" << #A << "'=" << A << " '" << #B        \
              << "'=" << B << " -- " << msg << std::endl;                      \
    throw std::runtime_error(msg);                                             \
  }

#define ERROR_WITH_INFO(msg)                                                   \
  throw std::runtime_error(std::string("--ERROR-- ") + __FILE__ +              \
                           std::string(":") + std::to_string(__LINE__) +       \
                           " \"" + std::string(msg) + "\"\n");

#define WARN_WITH_INFO(msg)                                                    \
  std::cout << __FILE__ + std::string(":") + std::to_string(__LINE__) + "\"" + \
                   std::string(msg) + "\"\n"                                   \
            << std::endl;

#pragma once
#include <algorithm>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <random>
#include <regex>
#include <type_traits>

#include "Eigen/Core"
#include <boost/program_options.hpp>
#include <boost/smart_ptr.hpp>

#include "croco_macros.hpp"
#include "math_utils.hpp"
#include "yaml-cpp/yaml.h"
#include <chrono>
#include <filesystem>

template <class T> using ptr = boost::shared_ptr<T>;
template <class T> using ptrs = std::shared_ptr<T>;

template <typename T, typename... Args> auto mk(Args &&...args) {
  return boost::make_shared<T>(std::forward<Args>(args)...);
}

template <class T>
void print_matrix(std::ostream &out, const std::vector<std::vector<T>> &data) {

  for (auto &v : data) {
    for (auto &e : v)
      out << e << " ";
    out << std::endl;
  }
}

template <typename T, typename... Args> auto mks(Args &&...args) {
  return std::make_shared<T>(std::forward<Args>(args)...);
}

template <typename Fun, typename... Args>
auto timed_fun(Fun fun, Args &&...args) {
  auto tic = std::chrono::high_resolution_clock::now();
  auto out = fun(std::forward<Args>(args)...);
  auto tac = std::chrono::high_resolution_clock::now();
  return std::make_pair(
      out, std::chrono::duration<double, std::milli>(tac - tic).count());
}

namespace po = boost::program_options;

template <typename T> bool __in(const std::vector<T> &v, const T &val) {
  return std::find(v.cbegin(), v.cend(), val) != v.cend();
}

bool inline startsWith(const std::string &str, const std::string &prefix) {
  return str.size() >= prefix.size() &&
         str.compare(0, prefix.size(), prefix) == 0;
}

typedef Eigen::Matrix<double, 3, 4> Matrix34;
typedef Eigen::Matrix<double, 1, 1> V1d;

template <typename T1, typename Fun>
bool __in_if(const std::vector<T1> &v, Fun fun) {
  return std::find_if(v.cbegin(), v.cend(), fun) != v.cend();
}

template <typename T>
void set_from_yaml(const YAML::Node &node, T &var, const char *name) {

  if (YAML::Node parameter = node[name]) {
    var = parameter.as<T>();
  }
}

void inline set_from_yaml_v(YAML::Node &node, std::vector<double> &var,
                            const char *name) {
  if (YAML::Node parameter = node[name]) {
    var = parameter.as<std::vector<double>>();
  }
}

template <int T>
void inline set_from_yaml_eigen(YAML::Node &node,
                                Eigen::Matrix<double, T, 1> &v,
                                const char *name) {
  using Vd = Eigen::Matrix<double, T, 1>;
  std::vector<double> var;
  if (YAML::Node parameter = node[name]) {
    var = parameter.as<std::vector<double>>();
    assert(var.size() == T);
    v = Vd(var.data());
  }
}

void inline set_from_yaml_eigenx(YAML::Node &node, Eigen::VectorXd &v,
                                 const char *name) {
  std::vector<double> var;
  if (YAML::Node parameter = node[name]) {
    var = parameter.as<std::vector<double>>();
    v = Eigen::Map<Eigen::VectorXd>(var.data(), var.size());
  }
}

template <typename T>
void set_from_boostop(po::options_description &desc, T &var, const char *name) {
  // std::cout << var << std::endl;
  // std::cout << NAMEOF(var) << std::endl;
  desc.add_options()(name, po::value<T>(&var));
}

inline std::string get_time_stamp() {

  auto t = std::time(nullptr);
  auto tm = *std::localtime(&t);

  std::ostringstream oss;
  oss << std::put_time(&tm, "%d-%m-%Y--%H-%M-%S");
  auto str = oss.str();

  return str;
}

void inline from_quim_format(const Eigen::VectorXd &xin,
                             Eigen::VectorXd &xout) {

  xout = xin;
  xout.segment(7, 3) = xin.segment(3, 3);

  // xout(6) = xin(6);
  xout.segment(3, 3) = xin.segment(7, 3);
}

void inline from_welf_format(const Eigen::VectorXd &xin,
                             Eigen::VectorXd &xout) {

  xout = xin;
  xout.segment(3, 3) = xin.segment(7, 3);
  // xout(6) = xin(6);
  xout.segment(7, 3) = xin.segment(3, 3);
}

YAML::Node inline load_yaml_safe(const char *str) {
  if (!std::filesystem::exists(str)) {
    ERROR_WITH_INFO(std::string("Not found file ") + str);
  }
  return YAML::LoadFile(str);
}
YAML::Node inline load_yaml_safe(const std::string &s) {
  return load_yaml_safe(s.c_str());
}

std::vector<Eigen::VectorXd> inline yaml_node_to_xs(const YAML::Node &node) {

  std::vector<std::vector<double>> states;
  std::vector<Eigen::VectorXd> xs;

  for (const auto &state : node) {
    std::vector<double> p;
    for (const auto &elem : state) {
      p.push_back(elem.as<double>());
    }
    states.push_back(p);
  }

  xs.resize(states.size());

  std::transform(states.begin(), states.end(), xs.begin(), [](const auto &s) {
    return Eigen::VectorXd::Map(s.data(), s.size());
  });

  return xs;
}

template <typename T> void print_vec(const T *a, size_t n, bool eof = true) {
  for (size_t i = 0; i < n; i++) {
    std::cout << a[i] << " ";
  }
  if (eof)
    std::cout << std::endl;
}

template <typename T> double get_time_stamp_ms(const T &start) {
  return static_cast<double>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::steady_clock::now() - start)
          .count());
};

bool inline hasEnding(std::string const &fullString,
                      std::string const &ending) {
  if (fullString.length() >= ending.length()) {
    return (0 == fullString.compare(fullString.length() - ending.length(),
                                    ending.length(), ending));
  } else {
    return false;
  }
}
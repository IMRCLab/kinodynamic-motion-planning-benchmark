#pragma once
#include "Core/array.h"
#include "KOMO/komo.h"
#include "Kin/kin.h"
#include <algorithm>
#include <numeric>
#include <vector>

template <typename T> T reduce_max(const std::vector<T> &data) {
  return *std::max_element(data.begin(), data.end());
}

template <typename T> T reduce_mean(const std::vector<T> &data) {
  return *std::accumulate(data.begin(), data.end()) / data.size();
}

template <typename Point> struct EuclideanInteroplation {
  Point interp(double x, double xa, double xb, const Point &ya,
               const Point &yb) {
    double r = (x - xa) / (xb - xa);
    return r * yb + (1. - r) * ya;
  }
};

template <typename Point> class LinearInterpolator {
public:
  using Map = std::map<double, Point>;
  LinearInterpolator() {}

  void addDataset(double *x, Point *d, size_t n) {
    for (size_t i = 0; i < n; i++) {
      addDataPoint(x[i], d[i]);
    }
  }

  void addDataPoint(double x, const Point &d) { data[x] = d; }

  template <typename InterpOperator = EuclideanInteroplation<Point>>
  Point interpolate(double x, InterpOperator op = InterpOperator()) {
    typename Map::iterator it = data.begin();
    bool found = false;
    while (it != data.end() && !found) {
      if (it->first >= x) {
        found = true;
        break;
      }
      it++;
    }

    // check to see if we're outside the data range
    if (it == data.begin()) {
      return data.begin()->second;
    } else if (it == data.end()) {
      // move the point back one, as end() points past the list
      it--;
      return it->second;
    }
    // check to see if we landed on a given point
    else if (it->first == x) {
      return it->second;
    }

    // nope, we're in the range somewhere
    // collect some values
    double xb = it->first;
    Point yb = it->second;
    it--;
    double xa = it->first;
    Point ya = it->second;

    return op.interp(x, xa, xb, ya, yb);
  }

  Map data;
};

arrA getPath_qAll_with_prefix(KOMO &komo, int order);

void komo_setConfiguration_X_name(KOMO &komo, int t, const arr &X,
                                  const char *name);

arr komo_getConfiguration_X_name(KOMO &komo, int t, const char *name);

bool is_feasible(KOMO &komo);

// returns {feasible,waypoints}
std::pair<bool, arrA>
solve_with_time_trick(const arrA &waypoints, rai::Configuration &C, double dt,
                      int order,
                      std::function<void(KOMO &)> set_komo_without_vel,
                      std::function<arrA(const arrA &)> compute_time_rescaling,
                      std::function<void(KOMO &)> set_komo_with_vel);

void set_start(KOMO &komo, const arr &s);

void update_reg(KOMO &komo);

void set_goal(rai::Configuration &Cref, KOMO &komo, const arr &s, int horizon);

std::pair<bool, arrA> komo_binary_search_time(
    const arrA &waypoints, int min_waypoints, int max_waypoints, int increment,
    double dt, rai::Configuration C, std::function<void(KOMO &)> set_komo);

std::pair<bool, arrA> iterative_komo_solver(
    const arrA &waypoints, int horizon, KOMO &komo, KOMO &komo_hard,
    const arr &start, std::function<void(KOMO &, const arr &)> set_start,
    std::function<void(KOMO &, const arr &)> set_goal,
    std::function<double(const arr &, const arr &)> distance_fun =
        euclideanDistance<double>);

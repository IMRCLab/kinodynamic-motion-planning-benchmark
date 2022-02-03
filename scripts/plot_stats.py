#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages


class Report:
  def __init__(self, filename):
    self.pp = PdfPages(filename)
    self.fig = None

  def start_experiment(self, name, T, dt):
    self.experiment_name = name
    self.T = T
    self.dt = dt
    self.times = np.arange(0, self.T, self.dt)
    self.stats = dict()

  def load_stat_files(self, name, filenames):
    costs = []
    for filename in filenames:
      costs.append(load_data(filename, self.T, self.dt))

    # convert to 2D array
    costs = np.array(costs)
    self.stats[name] = costs

  def add_time_cost_plot(self):
    self._add_page()
    self.fig, self.ax = plt.subplots()
    self.ax.set_title(self.experiment_name)

    for name, costs in self.stats.items():
      #   mean = costs.mean(axis=0)
      mean = np.nanmean(costs, axis=0)
      #   std = costs.std(axis=0)
      std = np.nanstd(costs, axis=0)

      self.ax.plot(self.times, mean, label=name)
      self.ax.fill_between(self.times, mean+std, mean-std, alpha=0.5)
    self.ax.legend()
    self.ax.set_xlabel("time [s]")
    self.ax.set_ylabel("cost [s]")

  def add_initial_time_cost_plot(self):
    self._add_page()
    self.fig, self.ax = plt.subplots()
    self.ax.set_title(self.experiment_name)

    for name, costs in self.stats.items():
      initial_costs = []
      initial_times = []
      for k in range(costs.shape[0]):
        l = costs[k, np.isfinite(costs[k])]
        if len(l) > 0:
          initial_cost = l[0]
          initial_costs.append(initial_cost)
          initial_time = self.times[np.isfinite(costs[k])][0]
          initial_times.append(initial_time)
      print(initial_times, initial_costs)
      self.ax.scatter(initial_times, initial_costs, label=name)
    self.ax.legend()
    self.ax.set_xlabel("time for first solution [s]")
    self.ax.set_ylabel("cost of first solution [s]")


  def close(self):
    self._add_page()
    self.pp.close()

  def _add_page(self):
    if self.fig is not None:
      self.pp.savefig(self.fig)
      plt.close(self.fig)
      self.fig = None


def load_data(filename, T, dt):
  with open(filename) as f:
    stats = yaml.safe_load(f)

  costs = np.zeros(int(T / dt)) * np.nan
  if stats is not None and "stats" in stats and stats["stats"] is not None:
    for d in stats["stats"]:
        idx = int(d["t"] / dt)
        costs[idx:] = d["cost"]
  return costs


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("stats", nargs='*', help="yaml files with stats")
  args = parser.parse_args()

  fig, ax = plt.subplots()

  T = 61
  dt = 0.1
  times = np.arange(0, T, dt)
  
  costs = []
  for filename in args.stats:
    costs.append(load_data(filename, T, dt))

  # convert to 2D array
  costs = np.array(costs)
#   mean = costs.mean(axis=0)
  mean = np.nanmean(costs, axis=0)
#   std = costs.std(axis=0)
  std = np.nanstd(costs, axis=0)

  ax.plot(times, mean, color='blue')
  ax.fill_between(times, mean+std, mean-std, facecolor='blue', alpha=0.5)
  plt.show()


if __name__ == "__main__":
  main()

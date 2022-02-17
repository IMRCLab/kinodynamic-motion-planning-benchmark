#!/usr/bin/env python3
import argparse
from mimetypes import init
import numpy as np
import matplotlib.pyplot as plt
import yaml
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.cm import get_cmap
from collections import defaultdict

class Report:
  def __init__(self, filename):
    self.pp = PdfPages(filename)
    self.fig = None
    cmap = get_cmap("Dark2")
    self.color_dict = {
      'sst': cmap.colors[0],
      'sbpl': cmap.colors[1],
      'komo': cmap.colors[2],
      'dbAstar-komo': cmap.colors[3],
    }

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

      self.ax.plot(self.times, mean, label=name, color=self.color_dict[name])
      self.ax.fill_between(self.times, mean+std, mean-std, color=self.color_dict[name], alpha=0.5)
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
      self.ax.scatter(initial_times, initial_costs, label=name, color=self.color_dict[name])
    self.ax.legend()
    self.ax.set_xlabel("time for first solution [s]")
    self.ax.set_ylabel("cost of first solution [s]")

  def add_success_rate_plot(self):
    self._add_page()
    self.fig, self.ax = plt.subplots()
    self.ax.set_title(self.experiment_name)

    success_dict = defaultdict(int)
    for name, costs in self.stats.items():
      for k in range(costs.shape[0]):
        l = costs[k, np.isfinite(costs[k])]
        if len(l) > 0:
          success_dict[name] += 1
    names = self.color_dict.keys()
    y = []
    for name in names:
      y.append(success_dict[name])

    self.ax.bar(range(len(y)), y)
    self.ax.set_xticks(range(len(y)))
    self.ax.set_xticklabels(names)

  def add_boxplot_initial_time_plot(self):
    self._add_page()
    self.fig, self.ax = plt.subplots()
    self.ax.set_title(self.experiment_name)

    result = []
    names = self.color_dict.keys()
    result_dict = defaultdict(list)
    for name, costs in self.stats.items():
      initial_times = []
      for k in range(costs.shape[0]):
        l = costs[k, np.isfinite(costs[k])]
        if len(l) > 0:
          initial_time = self.times[np.isfinite(costs[k])][0]
          initial_times.append(initial_time)
      result_dict[name] = initial_times
    # print(result_dict.values())
    result = []
    for name in names:
      result.append(result_dict[name])
    self.ax.boxplot(result)
    self.ax.set_xticks(range(1, len(names)+1))
    self.ax.set_xticklabels(names)
    # self.ax.legend()
    self.ax.set_ylabel("time for first solution [s]")

  def add_boxplot_initial_cost_plot(self):
    self._add_page()
    self.fig, self.ax = plt.subplots()
    self.ax.set_title(self.experiment_name)

    result = []
    names = self.color_dict.keys()
    result_dict = defaultdict(list)
    for name, costs in self.stats.items():
      initial_costs = []
      for k in range(costs.shape[0]):
        l = costs[k, np.isfinite(costs[k])]
        if len(l) > 0:
          initial_cost = l[0]
          initial_costs.append(initial_cost)
      result_dict[name] = initial_costs
    # print(result_dict.values())
    result = []
    for name in names:
      result.append(result_dict[name])
    self.ax.boxplot(result)
    self.ax.set_xticks(range(1, len(names)+1))
    self.ax.set_xticklabels(names)
    # self.ax.legend()
    self.ax.set_ylabel("cost for first solution [s]")


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

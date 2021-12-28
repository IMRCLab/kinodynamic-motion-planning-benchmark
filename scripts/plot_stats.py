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

  def add(self, name):
    self._add_page()
    self.fig, self.ax = plt.subplots()
    self.ax.set_title(name)

  def load_stat_files(self, filenames, T, dt, name):

    times = np.arange(0, T, dt)

    costs = []
    for filename in filenames:
      costs.append(load_data(filename, T, dt))

    # convert to 2D array
    costs = np.array(costs)
  #   mean = costs.mean(axis=0)
    mean = np.nanmean(costs, axis=0)
  #   std = costs.std(axis=0)
    std = np.nanstd(costs, axis=0)

    self.ax.plot(times, mean, label=name)
    self.ax.fill_between(times, mean+std, mean-std, alpha=0.5)
    self.ax.legend()
    self.ax.set_xlabel("time [s]")
    self.ax.set_ylabel("cost [s]")

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
  if "stats" in stats and stats["stats"] is not None:
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

#!/usr/bin/env python3
import argparse
import numpy as np
import matplotlib.pyplot as plt
import yaml


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

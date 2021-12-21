import yaml
from pathlib import Path
import plot_stats


def main():
	benchmark_path = Path("../benchmark")
	results_path = Path("../results")
	tuning_path = Path("../tuning")

	instances = ["carFirstOrder/bugtrap_0", "carFirstOrder/kink_0", "carFirstOrder/parallelpark_0"]
	# instances = ["carFirstOrder/bugtrap_0"]
	algs = ["sst", "sbpl", "dbAstar"]
	# algs = ["sst","dbAstar"]

	report = plot_stats.Report(results_path / "stats.pdf")

	for instance in instances:
		report.add("{}".format(instance))
		for alg in algs:
			result_folder = results_path / instance / alg
			stats_file = result_folder / "stats.yaml"
			if stats_file.exists():
				report.load_stat_files([stats_file], 5*60, 0.1, alg)

	report.close()

if __name__ == '__main__':
	main()

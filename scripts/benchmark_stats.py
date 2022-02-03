import yaml
from pathlib import Path
import plot_stats


def main():
	benchmark_path = Path("../benchmark")
	results_path = Path("../results")
	tuning_path = Path("../tuning")

	instances = [
		"unicycleFirstOrder/parallelpark_0",
		"unicycleFirstOrder/kink_0",
		"unicycleFirstOrder/bugtrap_0",
		"unicycleSecondOrder/parallelpark_0",
		"unicycleSecondOrder/kink_0",
		"unicycleSecondOrder/bugtrap_0",
		# "carFirstOrderWithTrailers/parallelpark_0",
	]
	algs = [
		# "sst",
		# "sbpl",
		"komo",
		"dbAstar-komo",
		# "dbAstar-scp",
	]

	report = plot_stats.Report(results_path / "stats.pdf")

	for instance in instances:
		report.start_experiment("{}".format(instance), 5*60, 0.1)
		for alg in algs:
			result_folder = results_path / instance / alg
			stat_files = [str(p) for p in result_folder.glob("**/stats.yaml")]
			# stat_files = [str(p) for p in result_folder.glob("000/stats.yaml")]
			print(stat_files)
			if len(stat_files) > 0:
				report.load_stat_files(alg, stat_files)
		report.add_time_cost_plot()
		report.add_initial_time_cost_plot()

	report.close()

if __name__ == '__main__':
	main()

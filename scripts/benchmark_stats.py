import yaml
from pathlib import Path
import plot_stats


def main():
	benchmark_path = Path("../benchmark")
	results_path = Path("../results")
	tuning_path = Path("../tuning")

	instances = [
		"unicycle_first_order_0/parallelpark_0",
		"unicycle_first_order_0/kink_0",
		"unicycle_first_order_0/bugtrap_0",
		"unicycle_first_order_1/kink_0",
		"unicycle_first_order_2/wall_0",
		"unicycle_second_order_0/parallelpark_0",
		"unicycle_second_order_0/kink_0",
		"unicycle_second_order_0/bugtrap_0",
		"car_first_order_with_1_trailers_0/parallelpark_0",
		"car_first_order_with_1_trailers_0/kink_0",
		"car_first_order_with_1_trailers_0/bugtrap_0",
	]
	algs = [
		"sst",
		"sbpl",
		"komo",
		"dbAstar-komo",
		# "dbAstar-scp",
	]

	report = plot_stats.Report(results_path / "stats.pdf", T=5*60, dt=0.1)

	for instance in instances:
		for alg in algs:
			result_folder = results_path / instance / alg
			stat_files = [str(p) for p in result_folder.glob("**/stats.yaml")]
			# stat_files = [str(p) for p in result_folder.glob("000/stats.yaml")]
			if len(stat_files) > 0:
				report.load_stat_files(instance, alg, stat_files)

	report.add_barplot_initial_cost_plot(instances)
	for instance in instances:
		report.add_time_cost_plot(instance)
		report.add_initial_time_cost_plot(instance)
		report.add_success_rate_plot(instance)
		report.add_boxplot_initial_time_plot(instance)
		report.add_boxplot_initial_cost_plot([instance])

	report.close()

if __name__ == '__main__':
	main()

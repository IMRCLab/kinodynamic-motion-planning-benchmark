from pathlib import Path
import yaml
import numpy as np


def main():
	results_path = Path("../results")

	rows = [
		{
			"system": "unicycle_first_order_0",
			"instance": "parallelpark_0",
		},
		{
			"system": "unicycle_first_order_0",
			"instance": "kink_0",
		},
		{
			"system": "unicycle_first_order_0",
			"instance": "bugtrap_0",
		},
		
		{
			"system": "unicycle_first_order_1",
			"instance": "kink_0",
		},
		
		{
			"system": "unicycle_first_order_2",
			"instance": "wall_0",
		},

		{
			"system": "unicycle_second_order_0",
			"instance": "parallelpark_0",
		},
		{
			"system": "unicycle_second_order_0",
			"instance": "kink_0",
		},
		{
			"system": "unicycle_second_order_0",
			"instance": "bugtrap_0",
		},

		{
			"system": "car_first_order_with_1_trailers_0",
			"instance": "parallelpark_0",
		},
		{
			"system": "car_first_order_with_1_trailers_0",
			"instance": "kink_0",
		},
		{
			"system": "car_first_order_with_1_trailers_0",
			"instance": "bugtrap_0",
		},
	]
	algs = [
		"sst",
		"sbpl",
		"komo",
		"dbAstar-komo",
		# "dbAstar-scp",
	]

	system_names = {
		'unicycle_first_order_0': "unicycle $1^{\mathrm{st}}$ order, v0",
		'unicycle_first_order_1': "unicycle $1^{\mathrm{st}}$ order, v1",
		'unicycle_first_order_2': "unicycle $1^{\mathrm{st}}$ order, v2",
		'unicycle_second_order_0': "unicycle $2^{\mathrm{nd}}$ order",
		'car_first_order_with_1_trailers_0': "car with trailer",
	}

	instance_names = {
		'parallelpark_0': "park",
		'kink_0': "kink",
		'bugtrap_0': "bugtrap",
		'wall_0': "wall",
	}

	T = 5*60

	out = r"\begin{tabular}{c || c|c || c|c|c || c|c|c || c|c|c  || c|c|c}"
	print(out)
	print(r"\# & System & Instance & \multicolumn{3}{c||}{SST*} & \multicolumn{3}{c||}{SBPL} & \multicolumn{3}{c||}{geom. RRT*+KOMO} & \multicolumn{3}{c}{kMP-db-A*}\\")
	print(r"&&& $t^{\mathrm{st}} [s]$ & $J^{\mathrm{st}} [s]$ & $J^{f} [s]$& $t^{\mathrm{st}} [s]$ & $J^{\mathrm{st}} [s]$ & $J^{f} [s]$& $t^{\mathrm{st}} [s]$ & $J^{\mathrm{st}} [s]$ & $J^{f} [s]$& $t^{\mathrm{st}} [s]$ & $J^{\mathrm{st}} [s]$ & $J^{f} [s]$\\")
	print(r"\hline")

	last_system = ""

	for r_number, row in enumerate(rows):

		out = ""
		if last_system != row["system"]:
			out += r"\hline"
		out += "{} & ".format(r_number+1)
		if last_system != row["system"]:
			# check how many use the same
			num_rows = 1
			for row_next in rows[r_number+1:]:
				if row_next["system"] != row["system"]:
					break
				num_rows += 1

			# now start a new multirow
			out += r"\multirow{"
			out += "{}".format(num_rows)
			out += r"}{*}{"
			out += system_names[row["system"]]
			out += r"}"
		last_system = row["system"]
		out += " & {} ".format(instance_names[row["instance"]])

		result = dict()
		for alg in algs:
			result_folder = results_path / row["system"] / row["instance"] / alg
			stat_files = [str(p) for p in result_folder.glob("**/stats.yaml")]

			# load data
			initial_times = []
			initial_costs = []
			final_costs = []
			for stat_file in stat_files:
				with open(stat_file) as f:
					stats = yaml.safe_load(f)
				if stats is not None and "stats" in stats and stats["stats"] is not None:
					last_cost = None
					for k, d in enumerate(stats["stats"]):
						# skip results that were after our time horizon
						if d["t"] > T:
							break
						if k == 0:
							initial_times.append(d["t"])
							initial_costs.append(d["cost"])
						last_cost = d["cost"]
					if last_cost is not None:
						final_costs.append(last_cost)

			# write a result row
			# out += " & ${:.1f} \pm {:.1f}$".format(np.mean(initial_times), np.std(initial_times))
			# out += " & ${:.1f} \pm {:.1f}$".format(np.mean(initial_costs), np.std(initial_costs))
			# out += " & ${:.1f} \pm {:.1f}$".format(np.mean(final_costs), np.std(final_costs))

			result[alg] = {
				't^st_mean': np.mean(initial_times) if len(initial_times) > 0 else None,
				'J^st_mean': np.mean(initial_costs) if len(initial_costs) > 0 else None,
				'J^f_mean': np.mean(final_costs) if len(initial_costs) > 0 else None,
			}

		def print_and_highlight_best(out, key):
			out += " & $"
			is_best = False
			if result[alg][key] is not None:
				# we only look at one digit
				is_best = np.array([round(result[alg][key],1) <= round(result[other][key],1) for other in algs if result[other][key] is not None]).all()
			if is_best:
				out += r"\mathbf{"
			if result[alg][key] is not None:
				out += "{:.1f}".format(result[alg][key])
			else:
				out += "-"
			if is_best:
				out += r"}"
			out += r"$"
			return out

		for alg in algs:
			out = print_and_highlight_best(out, 't^st_mean')
			out = print_and_highlight_best(out, 'J^st_mean')
			out = print_and_highlight_best(out, 'J^f_mean')

		out += r"\\"
		print(out)

	print(r"\end{tabular}")








if __name__ == '__main__':
	main()

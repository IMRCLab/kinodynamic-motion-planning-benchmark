import argparse
import subprocess
import yaml
import main_scp
import main_komo



def run_sbpl(filename_env, folder):

	result = subprocess.run(["./main_sbpl", 
		"-i", filename_env,
		"-o", "{}/result_sbpl.yaml".format(folder),
		"-p", "../tuning/unicycle_first_order_0/unicycle_first_order_0_mprim.mprim",
		"--stats", "{}/stats.yaml".format(folder),
		])
	if result.returncode != 0:
		print("SBPL failed")
	else:
		# extract the action sequence
		with open("{}/result_sbpl.yaml".format(folder)) as f:
				result = yaml.safe_load(f)

		with open("../tuning/unicycle_first_order_0/unicycle_first_order_0_mprim.yaml") as f:
				motions = yaml.safe_load(f)

		actions = []
		for start_c, dX, dY, end_c in result["result"][0]["actions_mprim"]:
			found = False
			for m in motions:
				if m['prim']['start_c'] == start_c and (m['prim']['endpose_c'] == [dX, dY, end_c] or m['prim']['endpose_c'] == [dX, dY, end_c-16] or m['prim']['endpose_c'] == [dX, dY, end_c+16]):
					actions.extend(m['actions'])
					found = True
					break
			if not found:
				print("ERROR: couldn't find corresponding motion!", start_c, dX, dY, end_c)

		result["result"][0]["actions"] = actions
		with open("{}/result_sbpl.yaml".format(folder), 'w') as f:
			yaml.Dumper.ignore_aliases = lambda *args : True
			yaml.dump(result, f)

		print(len(actions), len(result["result"][0]["states"]))

		main_scp.run_scp(
			filename_env,
			"{}/result_sbpl.yaml".format(folder),
			"{}/result_scp.yaml".format(folder))
		main_komo.run_komo(
			filename_env,
			"{}/result_sbpl.yaml".format(folder),
			"{}/result_komo.yaml".format(folder))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	args = parser.parse_args()

	for i in range(1):
		run_sbpl(args.env, i)


if __name__ == '__main__':
	main()

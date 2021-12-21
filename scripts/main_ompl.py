import argparse
import subprocess
import main_scp

def run_ompl(filename_env, folder, timelimit, cfg):
	result = subprocess.run(["./main_ompl", 
		"-i", filename_env,
		"-o", "{}/result_ompl.yaml".format(folder),
		"--stats", "{}/stats.yaml".format(folder),
		"--timelimit", str(timelimit),
		"-p", "sst",
		"--goalregion", str(cfg['goal_epsilon'])])
	if result.returncode != 0:
		print("OMPL failed")
	else:
		# pass
		success = main_scp.run_scp(
			filename_env,
			"{}/result_ompl.yaml".format(folder),
			"{}/result_scp.yaml".format(folder))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	args = parser.parse_args()

	for i in range(1):
		run_ompl(args.env, i)


if __name__ == '__main__':
	main()

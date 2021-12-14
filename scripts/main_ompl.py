import argparse
import subprocess
import main_scp

def run_ompl(filename_env, prefix=""):

	result = subprocess.run(["./main_ompl", 
		"-i", filename_env,
		"-o", "result_ompl_{}.yaml".format(prefix),
		"--stats", "stats_ompl_{}.yaml".format(prefix),
		"--timelimit", str(60),
		"-p", "sst"])
	if result.returncode != 0:
		print("OMPL failed")
	else:
		pass
		# success = main_scp.run_scp(filename_env, "result_ompl_{}.yaml".format(prefix))

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	args = parser.parse_args()

	for i in range(3):
		run_ompl(args.env, i)


if __name__ == '__main__':
	main()

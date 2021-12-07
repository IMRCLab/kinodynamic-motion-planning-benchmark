import argparse
import subprocess
import main_scp

def run_dbastar(filename_env):

	result = subprocess.run(["./ompl_rrt", "-i", filename_env, "-o", "result_rrt.yaml", "-p", "sst"])
	if result.returncode != 0:
		print("RRT failed")

	else:
		success = main_scp.run_scp(filename_env, "result_rrt.yaml")
		if not success:
			print("Optimization failed; Reducing delta")

def main():
	parser = argparse.ArgumentParser()
	parser.add_argument("env", help="file containing the environment (YAML)")
	args = parser.parse_args()

	run_dbastar(args.env)


if __name__ == '__main__':
	main()

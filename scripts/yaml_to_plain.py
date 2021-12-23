import argparse
import yaml


def write(file_in: str, file_out: str):
    with open(file_in) as env_file:
        env = yaml.safe_load(env_file)

    with open(file_out, "w") as txt_file:
        print("bb")
        result = env["result"][0]  # get first result
        for state in result["states"]:
            print("state {}".format(state))
            txt_file.write(' '.join(map(str, state)))
            txt_file.write("\n")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fin", required=True, help="yaml file")
    parser.add_argument("--fout", required=True, help="txt file")
    args = parser.parse_args()

    write(args.fin, args.fout)


if __name__ == "__main__":
    main()

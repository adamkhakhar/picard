import json
import os
import argparse


def write_dict_to_json(data, filepath):
    with open(filepath, "w") as f:
        json.dump(data, f)


def read_json_file(filepath):
    with open(filepath) as f:
        return json.load(f)


def update_config_num_beam_num_return(
    num_beams,
    num_return_sequences,
    filepath=f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/configs/serve.json",
):
    data = read_json_file(filepath)
    data["num_beams"] = num_beams
    data["num_return_sequences"] = num_return_sequences
    write_dict_to_json(data, filepath)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Update configs/serve.json num beams and num return sequences")
    parser.add_argument("num_beams", type=int)
    parser.add_argument("num_return_sequences", type=int)
    args = parser.parse_args()
    update_config_num_beam_num_return(args.num_beams, args.num_return_sequences)

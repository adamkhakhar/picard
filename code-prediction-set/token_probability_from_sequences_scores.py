import pickle
import torch
import os
import argparse
import pdb

DIR_NAME = os.path.dirname(os.path.realpath(__file__))


def store_data(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def retrieve_list_token_prob(sequence, scores):
    l_token_to_prob = []
    for index in range(1, len(sequence)):
        l_token_to_prob.append(
            {"p": torch.nn.Softmax()(scores[index - 1][0])[sequence[index]].item(), "token": sequence[index]}
        )
    return l_token_to_prob


def main():
    parser = argparse.ArgumentParser("Get fst prediction probs for tokens")
    parser.add_argument("numbeam", type=int)
    parser.add_argument("numpred", type=int)
    parser.add_argument("--startindx", dest="startindx", type=int, default=0)
    parser.add_argument("--endindx", dest="endindx", type=int, default=195)
    parser.add_argument("--step", dest="step", type=int, default=5)
    token_probs = []
    args = parser.parse_args()
    for indx in range(args.startindx, args.endindx, args.step):
        try:
            cache_data = load_data(
                f"{DIR_NAME}/results_v2/beam_size_{args.numbeam}/num_predictions_{args.numpred}/startindx_{indx}"
            )
            for i in range(len(cache_data)):
                q_index = indx + args.startindx + i
                token_probs.append(
                    {
                        "indx": q_index,
                        "token_probs": retrieve_list_token_prob(cache_data[i]["sequences"][0], cache_data[i]["scores"]),
                    }
                )
        except Exception:
            print(
                "FILE NOT FOUND ",
                f"results_v2/beam_size_{args.numbeam}/num_predictions_{args.numpred}/startindx_{indx}",
            )
            pass
    print(token_probs)
    store_data(f"{DIR_NAME}/results_v2/token_probs_beam_size_{args.numbeam}_num_pred_{args.numpred}.pkl", token_probs)


if __name__ == "__main__":
    main()

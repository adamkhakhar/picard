import argparse
import pickle
import os
import sys
import spider_json_to_sexpr
import traceback
import importlib
import ipdb

PICARD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(PICARD_DIR)
sys.path.append(f"{PICARD_DIR}/code-prediction-set/sql_tree")

import process_sql
import process_sql_combined_token_prob

spider_to_sexpr = importlib.import_module("synth-sql.python.main")


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def store_data(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def get_spider_sexpr(db_id, input, trace, custom_tokenize=True, toks=None, probs=None, target=False):
    path_to_sql = f"{PICARD_DIR}/database/{db_id}/{db_id}.sqlite"
    schema = process_sql.Schema(process_sql.get_schema(path_to_sql))
    spider_output = "ERROR"
    sexpr = "ERROR"
    try:
        if custom_tokenize:
            # spider_output = process_sql.get_sql_with_probs(schema, query)
            if not target:
                spider_output = process_sql.get_sql_from_tokens(schema, input, toks, probs)
            else:
                toks = input.lower().split(" ")
                spider_output = process_sql.get_sql_from_tokens(schema, input, toks, [-1]*len(toks))
        else:
            spider_output = process_sql.get_sql(schema, input.lower())
        sexpr_input = {"db_id": db_id, "sql": spider_output}
        sexpr = spider_to_sexpr.foo(sexpr_input)
    except Exception:
        if trace:
            traceback.print_exc()
        pass
    return spider_output, sexpr


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--b", dest="beam", type=int, default=16)
    parser.add_argument("--i", dest="includeCorrect", type=int, default=1)
    parser.add_argument("--include_spider_tree", dest="include_spider_tree", type=int, default=1)
    parser.add_argument("--traceback", dest="traceback", type=int, default=0)
    parser.add_argument("--include_sexpr_err", dest="include_sexpr_err", type=int, default=0)
    parser.add_argument("--include_results", dest="include_results", type=int, default=0)
    parser.add_argument("--max_count", dest="max_count", type=int, default=100_000)
    parser.add_argument("--fst_pred_only", dest="fst_pred_only", type=int, default=1)
    parser.add_argument("--custom_tokenize", dest="custom_tokenize", type=int, default=1)
    parser.add_argument("--save_res", dest="save_res", type=int, default=1)
    args = parser.parse_args()
    assert args.beam in [16]
    num_pred = 8 if args.beam == 16 else -1

    data = load_data(
        f"{PICARD_DIR}/code-prediction-set/results_v2/t53b_with_prob_result_num_beam_{args.beam}__num_pred_{num_pred}__store_preds_{True}.pkl"
    )
    cnt = 0
    cnt_sexpr_failure = 0
    cnt_sexpr = 0

    token_prob_data = load_data(f"{PICARD_DIR}/code-prediction-set/results_v2/prob_with_token.pkl")

    results = []

    for sample in data["target_in_set"]:
        if cnt >= args.max_count:
            break
        if (sample["solution_in_set"] == -1) or (args.includeCorrect):
            curr_sample = {
                "db_id": sample["db_id"],
                "q": sample["question"],
                "target": sample["solution_query"],
                "preds": [],
            }
            print_str = ""
            print_str += f'DB\t{sample["db_id"]}\n'
            print_str += f'Question\t{sample["question"]}\n'
            print_str += f'Target\t{sample["solution_query"]}\n'
            spider, sexpr = get_spider_sexpr(
                sample["db_id"], sample["solution_query"], bool(args.traceback), custom_tokenize=bool(args.custom_tokenize), target=True
            )
            cnt_sexpr += 1
            if sexpr == "ERROR":
                print(print_str)
                cnt_sexpr_failure += 1
                if not args.include_sexpr_err:
                    continue
            if args.include_spider_tree:
                print_str += f"spider tree:\t{spider}\n"
            print_str += f"SQL Expr: {sexpr}\n"
            if args.include_results:
                print_str += "\tresult:\n"
                for res in sample["spider_solution"]:
                    print_str += "\t", res + "\n"
                print_str += "\n"
            if sexpr == "ERROR":
                cnt_sexpr_failure += 1
            cnt_sexpr += 1

            target_sexpr = sexpr

            for res in sample["picard_result"]:
                print_str += "\tPrediction:\n"
                print_str += f'\t{res["query"]}\n'
                # ipdb.set_trace()
                spider, sexpr = "ERROR", "ERROR"

                if sample["startindx"] in token_prob_data:
                    fst_pred_token_probs = token_prob_data[sample["startindx"] + sample["INDEX_CURRENT_HOST"]]
                    lst_tokens = [tok["token_decoded"] for tok in fst_pred_token_probs["token_probs"]]
                    lst_probs = [tok["p"] for tok in fst_pred_token_probs["token_probs"]]
                if args.custom_tokenize and sample["startindx"] in token_prob_data:
                    spider, sexpr = get_spider_sexpr(
                        sample["db_id"],
                        res["query"],
                        bool(args.traceback),
                        toks=lst_tokens,
                        custom_tokenize=bool(args.custom_tokenize),
                        probs=lst_probs,
                    )
                else:
                    spider, sexpr = get_spider_sexpr(
                        sample["db_id"], res["query"], bool(args.traceback), custom_tokenize=bool(args.custom_tokenize)
                    )
                if sexpr == "ERROR":
                    cnt_sexpr_failure += 1
                cnt_sexpr += 1
                if args.include_spider_tree:
                    print_str += f"\t\tspider tree:\t{spider}\n"
                print_str += f"\t\tSQL Expr:\t{sexpr}\n"
                if args.include_results:
                    print_str += "\t\t\tresult:\n"
                    for output in res["execution_results"]:
                        print_str += "\t\t\t" + output + "\n"
                    print_str += "\n"
                print_str += "\n"
                curr_sample["preds"].append(
                    {
                        "prediction": res["query"],
                        "pred_sexpr": sexpr,
                        "pred_spider": spider,
                        "target_sexpr": target_sexpr,
                        "lst_tokens": lst_tokens,
                        "lst_probs": lst_probs,
                        "sexpr_tokenize": process_sql.tokenize(res["query"]),
                    }
                )
                # only include first prediction (prediction with higest likelihood)
                if args.fst_pred_only:
                    break
            results.append(curr_sample)
            print(print_str + "\n\n")
            cnt += 1
    if args.save_res:
        store_data(f"{PICARD_DIR}/code-prediction-set/sql_tree/create_sql_tree_result.bin", results)
    print("SEXPR FAILURE CASES: ", cnt_sexpr_failure)
    print("TOTAL_SEXPR_CREATED", cnt_sexpr)

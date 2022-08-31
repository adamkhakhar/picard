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

spider_to_sexpr = importlib.import_module("synth-sql.python.main")


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def get_spider_sexpr(db_id, input, trace, custom_tokenize=True):
    path_to_sql = f"{PICARD_DIR}/database/{db_id}/{db_id}.sqlite"
    schema = process_sql.Schema(process_sql.get_schema(path_to_sql))
    spider_output = "ERROR"
    sexpr = "ERROR"
    try:
        if custom_tokenize:
            # spider_output = process_sql.get_sql_with_probs(schema, query)
            spider_output = process_sql.get_sql_from_tokens(schema, input)
        else:
            spider_output = process_sql.get_sql(schema, input)
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
    args = parser.parse_args()
    assert args.beam in [16]
    num_pred = 8 if args.beam == 16 else -1

    print("TRACABACK    ", bool(args.traceback))

    data = load_data(
        f"{PICARD_DIR}/code-prediction-set/results_v2/t53b_with_prob_result_num_beam_{args.beam}__num_pred_{num_pred}__store_preds_{True}.pkl"
    )
    cnt = 0
    cnt_sexpr_failure = 0
    cnt_sexpr = 0

    token_prob_data = load_data(f"{PICARD_DIR}/code-prediction-set/results_v2/prob_with_token.pkl")

    for sample in data["target_in_set"]:
        if cnt >= args.max_count:
            break
        if (sample["solution_in_set"] == -1) or (args.includeCorrect):
            print_str = ""
            print_str += f'DB\t{sample["db_id"]}\n'
            print_str += f'Question\t{sample["question"]}\n'
            print_str += f'Target\t{sample["solution_query"]}\n'
            spider, sexpr = get_spider_sexpr(
                sample["db_id"], sample["solution_query"], bool(args.traceback), custom_tokenize=False
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

            for res in sample["picard_result"]:
                print_str += "\tPrediction:\n"
                print_str += f'\t{res["query"]}\n'
                # ipdb.set_trace()
                spider, sexpr = "ERROR", "ERROR"
                if args.custom_tokenize:
                    if sample["startindx"] in token_prob_data:
                        fst_pred_token_probs = token_prob_data[sample["startindx"]]
                        lst_tokens = [tok["token_decoded"] for tok in fst_pred_token_probs["token_probs"]]
                        lst_probs = [tok["p"] for tok in fst_pred_token_probs["token_probs"]]
                        spider, sexpr = get_spider_sexpr(
                            sample["db_id"],
                            lst_tokens,
                            bool(args.traceback),
                            custom_tokenize=bool(args.custom_tokenize),
                        )
                    else:
                        print("index oftoken_prob_data != sample['startindx']")
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
                # only include first prediction (prediction with higest likelihood)
                if args.fst_pred_only:
                    continue
            print(print_str + "\n\n")
            cnt += 1

    print("SEXPR FAILURE CASES: ", cnt_sexpr_failure)
    print("TOTAL_SEXPR_CREATED", cnt_sexpr)
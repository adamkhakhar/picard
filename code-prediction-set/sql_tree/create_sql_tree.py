import argparse
import pickle
import os
import sys
import spider_json_to_sexpr
import traceback

PICARD_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

sys.path.append(PICARD_DIR)
sys.path.append(f"{PICARD_DIR}/code-prediction-set/sql_tree")

import process_sql


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-b", "--beam", type=int)
    parser.add_argument("-i", dest="includeCorrect", type=int)
    parser.set_defaults(includeCorrect=1)
    args = parser.parse_args()
    assert args.beam in [16, 50]
    num_pred = 8 if args.beam == 16 else 25
    data = load_data(
        f"{PICARD_DIR}/code-prediction-set/results/prev_results/result_num_beam_{args.beam}__num_pred_{num_pred}__store_preds_{True}.pkl"
    )
    cnt_sexpr_failure = 0
    cnt_sexpr = 0
    for sample in data["target_in_set"]:
        if (sample["solution_in_set"] == -1) or (not args.includeCorrect):
            print("db_id", sample["db_id"])
            print("question", sample["question"])
            print("target\n", sample["solution_query"])
            path_to_sql = f'{PICARD_DIR}/database/{sample["db_id"]}/{sample["db_id"]}.sqlite'

            schema = process_sql.Schema(process_sql.get_schema(path_to_sql))
            solution_tree = process_sql.get_sql(schema, sample["solution_query"])
            print("spider tree", solution_tree)
            sexpr = "ERROR"
            try:
                sexpr = spider_json_to_sexpr.spider_json_to_sexpr(solution_tree, sample["db_id"])
                cnt_sexpr += 1
            except AssertionError:
                cnt_sexpr_failure += 1
                pass
            except Exception:
                traceback.print_exc()
                assert False
                cnt_sexpr_failure += 1
                pass
            print("sexpr", sexpr)
            print("\tresult:")
            for res in sample["spider_solution"]:
                print("\t", res)
            print("\n")
            # s = set()
            for res in sample["picard_result"]:
                # s.add(res["query"])
                print("\tprediction")
                print("\t", res["query"])
                solution_tree = "ERROR"
                sexpr = "ERROR"
                try:
                    solution_tree = process_sql.get_sql(schema, res["query"])
                    sexpr = spider_json_to_sexpr.spider_json_to_sexpr(solution_tree, sample["db_id"])
                    cnt_sexpr += 1
                except Exception:
                    cnt_sexpr_failure += 1
                    traceback.print_exc()
                    pass
                print("\t\tspider tree", solution_tree)
                print("\t\tsexpr", sexpr)
                print("\t\tresult:")
                for output in res["execution_results"]:
                    print("\t\t", output)
                print("\n")
    print("SEXPR FAILURE CASES: ", cnt_sexpr_failure)
    print("TOTAL_SEXPR_CREATED", cnt_sexpr)

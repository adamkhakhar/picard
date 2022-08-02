import sys
import os
import json
import time
import pickle
import argparse

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/seq2seq")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from programmatic_serve_seq2seq import Picard
from programmatic_database import SpiderDB

DEBUG = True


def check_if_equivalent(picard_result, spider_result):
    if len(picard_result) != len(spider_result):
        return False
    for i in range(len(picard_result)):
        if len(picard_result[i]) != len(spider_result[i]):
            return False
        for j in range(len(picard_result[i])):
            if picard_result[i][j] != spider_result[i][j]:
                return False
    return True


def load_train_spider(path_to_train_spider):
    with open(path_to_train_spider, "r") as f:
        train = json.loads(f.read())
        return train


def evaluate(
    port=8000,
    path_to_train_spider=f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/database/dev.json",
    startindx=0,
    endindx=200,
    log_every=5,
    store_preds=True,
):
    start = time.time()
    train = load_train_spider(path_to_train_spider)
    if DEBUG:
        print("[Logging] Total time to load train json", int(time.time() - start))
    target_in_set = []
    model = Picard(port=port)
    db = SpiderDB()
    start = time.time()
    for sample_ind in range(startindx, min(endindx, len(train))):
        curr_sample_data = {}
        curr_sample_data["startindx"] = startindx
        curr_sample_data["INDEX_CURRENT_HOST"] = sample_ind - startindx
        curr_sample_data["question"] = train[sample_ind]["question"]
        curr_sample_data["sample_ind"] = sample_ind
        spider_result = db.query(train[sample_ind]["db_id"], train[sample_ind]["query"])
        picard_result = model.query_model(train[sample_ind]["db_id"], train[sample_ind]["question"])
        curr_sample_data["spider_solution"] = spider_result
        curr_sample_data["picard_result"] = picard_result
        curr_sample_data["db_id"] = train[sample_ind]["db_id"]
        curr_sample_data["solution_query"] = train[sample_ind]["query"]
        flag_target_in_set = False
        for picard_prediction_ind in range(len(picard_result)):
            if flag_target_in_set:
                continue
            if (
                type(picard_result) == list
                and picard_prediction_ind < len(picard_result)
                and check_if_equivalent(picard_result[picard_prediction_ind]["execution_results"], spider_result)
            ):
                flag_target_in_set = True
                curr_sample_data["solution_in_set"] = picard_prediction_ind
        if not flag_target_in_set:
            curr_sample_data["solution_in_set"] = -1
        target_in_set.append(curr_sample_data)
        if DEBUG and sample_ind % log_every == 0:
            print(f"[{sample_ind}/{min(endindx, len(train))}] time to evaluate", int(time.time() - start))
    if DEBUG:
        print("[Logging] Total time to evaluate", int(time.time() - start))
    return target_in_set


def store_data(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Evaluate Picard.")
    parser.add_argument("numbeam", type=int)
    parser.add_argument("numpred", type=int)
    parser.add_argument("--port", dest="port", type=int, default=8000)
    parser.add_argument("--startindx", dest="startindx", type=int, default=0)
    parser.add_argument("--endindx", dest="endindx", type=int, default=200)
    parser.add_argument("--logevery", dest="logevery", type=int, default=1)
    parser.add_argument("--debug", dest="debug", type=bool, default=True)
    parser.add_argument("--store_preds", dest="store_preds", type=bool, default=True)
    args = parser.parse_args()
    DEBUG = args.debug
    if DEBUG:
        print(f"[Logging] Beginning evaluation. PORT: {args.port} | NUMBER PREDICTIONS: {args.numpred}")
    start = time.time()
    if not os.path.isdir(f"{os.path.dirname(os.path.realpath(__file__))}/results"):
        os.mkdir(f"{os.path.dirname(os.path.realpath(__file__))}/results")
    path_to_cache = f"{os.path.dirname(os.path.realpath(__file__))}/results/beam_size_{args.numbeam}/num_predictions_{args.numpred}/startindx_{args.startindx}"
    result = load_data(path_to_cache) if os.path.exists(path_to_cache) else {}
    if "target_in_set" in result and args.startindx in [x["sample_ind"] for x in result["target_in_set"]]:
        print("already exists. exiting")
        exit()
    target_in_set = evaluate(
        port=args.port,
        log_every=args.logevery,
        startindx=args.startindx,
        endindx=args.endindx,
        store_preds=args.store_preds,
    )
    result["number_predictions"] = args.numpred
    if "total_exec_time" in result:
        result["total_exec_time"] += int(time.time() - start)
    else:
        result["total_exec_time"] = int(time.time() - start)
    if "target_in_set" in result:
        result["target_in_set"] += target_in_set
    else:
        result["target_in_set"] = target_in_set
    store_data(path_to_cache, result)

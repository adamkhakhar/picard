import sys
import os
import json
import time
import pickle

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
    path_to_train_spider=f"{os.path.dirname(os.path.dirname(os.path.realpath(__file__)))}/database/train_spider.json",
):
    start = time.time()
    train = load_train_spider(path_to_train_spider)
    if DEBUG:
        print("[Logging] Total time to load train json", int(time.time() - start))
    target_in_set = {}
    model = Picard(port=port)
    db = SpiderDB()
    start = time.time()
    for sample_ind in range(len(train)):
        spider_result = db.query(train[sample_ind]["db_id"], train[sample_ind]["query"])
        picard_result = model.query_model(train[sample_ind]["db_id"], train[sample_ind]["question"])
        flag_target_in_set = False
        for picard_prediction_ind in range(len(picard_result)):
            if flag_target_in_set:
                continue
            if check_if_equivalent(picard_result[picard_prediction_ind]["execution_results"], spider_result):
                flag_target_in_set = True
                target_in_set[sample_ind] = picard_prediction_ind
        if not flag_target_in_set:
            target_in_set[sample_ind] = -1
        if DEBUG and sample_ind % int(len(train) / 20) == 0:
            print(f"[{sample_ind}/{len(train)}] time to evaluate", int(time.time() - start))
    if DEBUG:
        print("[Logging] Total time to evaluate", int(time.time() - start))
    return target_in_set


def store_data(fname, data):
    with open(fname, "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    number_predictions = int(sys.argv[1])
    port = 8000
    if len(sys.argv) == 3:
        port = int(sys.argv[2])
    if DEBUG:
        print(f"[Logging] Beginning evaluation. PORT: {port} | NUMBER PREDICTIONS: {number_predictions}")
    start = time.time()
    target_in_set = evaluate(port=port)
    result = {}
    result["number_predictions"] = number_predictions
    result["total_exec_time"] = int(time.time() - start)
    store_data(f"{os.path.dirname(os.path.realpath(__file__))}/result_num_pred_{number_predictions}.pkl", result)

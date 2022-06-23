import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))) + "/seq2seq")
sys.path.append(os.path.dirname(os.path.realpath(__file__)))

from programmatic_serve_seq2seq import Picard
from programmatic_database import SpiderDB


def check_if_equivalend(picard_result, spider_result):
    if len(picard_result) != len(spider_result):
        return False
    for i in range(len(picard_result)):
        if len(picard_result[i]) != len(spider_result[i]):
            return False
        for j in range(len(picard_result[i])):
            if picard_result[i][j] != spider_result[i][j]:
                return False
    return True


print("picard")
model = Picard()
picard_result = model.query_model("department_management", "Everything in department")
print(picard_result[0]["execution_results"])

print("spider")
db = SpiderDB()
spider_result = db.query("department_management", "select * from department")
print(spider_result)

print("check if equivalent", check_if_equivalend(picard_result[0]["execution_results"], spider_result))

import pickle
import os
from matplotlib import pyplot as plt
import numpy as np


def load_data(fname):
    with open(fname, "rb") as f:
        data = pickle.load(f)
    return data


def create_percent_coverage_plot(
    beam_return_pred_sizes, path=f"{os.path.dirname(os.path.realpath(__file__))}/results/"
):
    percent_coverage = []
    for beam_size, return_size in beam_return_pred_sizes:
        data = load_data(f"{path}result_num_beam_{beam_size}__num_pred_{return_size}.pkl")
        percent_coverage.append(
            sum([data["target_in_set"][key] >= 0 for key in data["target_in_set"]]) / len(data["target_in_set"])
        )
    return percent_coverage


num_pred = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 20]
pc = create_percent_coverage_plot([(4 + x, x) for x in num_pred])
for i in range(len(pc)):
    print("Number of predictions", num_pred[i], "Percent Set Coverage", round(pc[i] * 100, 1))


fig, ax = plt.subplots()
ax.spines["right"].set_color((0.8, 0.8, 0.8))
ax.spines["top"].set_color((0.8, 0.8, 0.8))
plt.plot(num_pred, [x * 100 for x in pc], marker="o", linestyle="--")
plt.xlabel("Number of Predictions Per Question")
plt.ylabel("Percent Solution in Prediction Set")
plt.title("Prediction Set Coverage For Varying Prediction Set Size")

# add more ticks
ax.set_xticks(np.arange(1, 21))

# tweak the axis labels
xlab = ax.xaxis.get_label()
ylab = ax.yaxis.get_label()
xlab.set_style("italic")
xlab.set_size(10)
ylab.set_style("italic")
ylab.set_size(10)

# tweak the title
ttl = ax.title
ttl.set_weight("bold")

plt.grid(linestyle="--", alpha=0.25)
plt.show()
plt.savefig("picard-set-coverage.png")

""" Dirty script to plot some results files againts one another """

import os
import json
import matplotlib.pyplot as plt

from knowledge_tracing import path_algos


DATA_DIR = os.path.join(path_algos, "data")


with open(os.path.join(DATA_DIR, "results_2020-04-24_18-38-14.json")) as f:
    results_Qrand = json.load(f)
with open(os.path.join(DATA_DIR, "results_2020-04-24_18-46-05.json")) as f:
    results_Q1 = json.load(f)
with open(os.path.join(DATA_DIR, "results_2020-04-24_18-53-23.json")) as f:
    results_Q2 = json.load(f)
with open(os.path.join(DATA_DIR, "results_2020-04-24_18-57-32.json")) as f:
    results_Q3 = json.load(f)
with open(os.path.join(DATA_DIR, "results_2020-04-24_19-01-46.json")) as f:
    results_Q4 = json.load(f)


xticks = ["Qrand", "Q1", "Q2", "Q3", "Q4"]
results = [results_Qrand, results_Q1, results_Q2, results_Q3, results_Q4]
res_afm_bg = [r["afm_bg"] for r in results]
res_afm_bgt = [r["afm_bgt"] for r in results]
res_pfa = [r["pfa"] for r in results]
res_item_avg = [r["item_avg"] for r in results]
res_global_avg = [r["global_avg"] for r in results]

fig = plt.figure()
ax = plt.subplot(221)
plt.plot([r["RMSE"] for r in res_afm_bg], "b", marker="o", label="afm_bg")
plt.plot([r["RMSE"] for r in res_afm_bgt], "r", marker="o", label="afm_bgt")
plt.plot([r["RMSE"] for r in res_pfa], "g", marker="o", label="pfa")
plt.plot([r["RMSE"] for r in res_item_avg], "--", color="#a9a9a9", label="item_avg")
plt.plot([r["RMSE"] for r in res_global_avg], "k--", label="global_avg")
plt.xticks([0, 1, 2, 3, 4], xticks)
plt.title("RMSE")
plt.subplot(222)
plt.plot([r["auc"] for r in res_afm_bg], "b", marker="o", label="afm_bg")
plt.plot([r["auc"] for r in res_afm_bgt], "r", marker="o", label="afm_bgt")
plt.plot([r["auc"] for r in res_pfa], "g", marker="o", label="pfa")
plt.plot([r["auc"] for r in res_item_avg], "--", color="#a9a9a9", label="item_avg")
plt.plot([r["auc"] for r in res_global_avg], "k--", label="global_avg")
plt.xticks([0, 1, 2, 3, 4], xticks)
plt.title("AUC")
plt.subplot(223)
plt.plot([r["f1_score"] for r in res_afm_bg], "b", marker="o", label="afm_bg")
plt.plot([r["f1_score"] for r in res_afm_bgt], "r", marker="o", label="afm_bgt")
plt.plot([r["f1_score"] for r in res_pfa], "g", marker="o", label="pfa")
plt.plot([r["f1_score"] for r in res_item_avg], "--", color="#a9a9a9", label="item_avg")
plt.plot([r["f1_score"] for r in res_global_avg], "k--", label="global_avg")
plt.xticks([0, 1, 2, 3, 4], xticks)
plt.title("F1")
plt.subplot(224)
plt.plot([r["balanced_accuracy"] for r in res_afm_bg], "b", marker="o", label="afm_bg")
plt.plot(
    [r["balanced_accuracy"] for r in res_afm_bgt], "r", marker="o", label="afm_bgt"
)
plt.plot([r["balanced_accuracy"] for r in res_pfa], "g", marker="o", label="pfa")
plt.plot(
    [r["balanced_accuracy"] for r in res_item_avg],
    "--",
    color="#a9a9a9",
    label="item_avg",
)
plt.plot([r["balanced_accuracy"] for r in res_global_avg], "k--", label="global_avg")
plt.xticks([0, 1, 2, 3, 4], xticks)
plt.title("Balanced Accuracy")
handles, labels = ax.get_legend_handles_labels()
fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.01))
plt.show()

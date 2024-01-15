import pandas as pd
import os
import matplotlib.pyplot as plt
from tabulate import tabulate
import math
from scipy import stats

from lingdata import database
from lingdata.categorical import CategoricalData

import code.distances as distances
from code.distances import DistanceMatrixIO
import code.util as util

def get_bins(a, nbins):
    min_val = min(a)
    max_val = max(a)
    step = (max_val - min_val) / (nbins - 1)
    return [min_val + i * step for i in range(nbins)]


def plot_distribution(df, column, label):
    data = df[column]
    print("Maximum", column, str(max(data)))
    plt.hist(data, bins = get_bins(data, 20))
    plt.xlabel(label)
    plt.ylabel('Number of datasets')
    plt.savefig(os.path.join(plots_dir, "hist_" + column +  ".png"))
    plt.clf()



results_dir = "data/results"
plots_dir = os.path.join(results_dir, "plots")
if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

pd.set_option('display.max_rows', None)

config_path = "synonyms_lingdata_config.json"

database.read_config(config_path)
df = database.data()

print("Datasets with less than 2 different values")
print(df[df["max_values"] < 2]["ds_id"])
df = df[df["max_values"] >= 2]
print("Datasets with more than 64 different values")
print(df[df["max_values"] > 64]["ds_id"])
df = df[df["max_values"] <= 64]

distance_matrices = []
d_io = DistanceMatrixIO(["rf", "gq"], ["glottolog", "bin", "catg_bin", "catg_multi", "consensus"])
for (i, row) in df.iterrows():
    dm = d_io.read_matrix(util.dist_dir(results_dir, row))
    distance_matrices.append(dm)
df["distance_matrix"] = distance_matrices

for (i, row) in df.iterrows():
    df.at[i, "rf_bin_avg"] = row["distance_matrix"].avg_ref_tree_dist("bin", "rf")
    df.at[i, "rf_bin_max"] = row["distance_matrix"].max_ref_tree_dist("bin", "rf")
    df.at[i, "rf_sampled_avg"] = row["distance_matrix"].sampled_avg_avg_dist("rf")
    df.at[i, "rf_sampled_max"] = row["distance_matrix"].sampled_max_avg_dist("rf")
    df.at[i, "rf_bin_catg_bin"] = row["distance_matrix"].ref_tree_dist("bin", "catg_bin", "rf")
    df.at[i, "rf_catg_bin_catg_multi"] = row["distance_matrix"].ref_tree_dist("catg_multi", "catg_bin", "rf")
    df.at[i, "rf_bin_catg_multi"] = row["distance_matrix"].ref_tree_dist("bin", "catg_multi", "rf")
    df.at[i, "gqd_sampled_avg"] = row["distance_matrix"].avg_ref_tree_dist("glottolog", "gq")
    gqds = {}
    for type in ["bin", "catg_bin", "catg_multi"]:
        gqds[type] = row["distance_matrix"].ref_tree_dist("glottolog", type, "gq")
        df.at[i, "gqd_" + type] = gqds[type]
    m = min(gqds.values())
    for type, gqd in gqds.items():
        if gqd == m:
            df.at[i, type + "_best"] = True
        else:
            df.at[i, type + "_best"] = False

df["gqd_diff_bin_catg_bin"] = df["gqd_bin"] - df["gqd_catg_bin"]
df["gqd_diff_catg_bin_catg_multi"] = df["gqd_catg_bin"] - df["gqd_catg_multi"]
df["gqd_diff_bin_catg_multi"] = df["gqd_bin"] - df["gqd_catg_multi"]
df["gqd_diff_bin_sampled"] = df["gqd_bin"] - df["gqd_sampled_avg"]

print("Datasets for which GQ distance to glottolog tree cannot be determined")
print(df[df["gqd_bin"] != df["gqd_bin"]]["ds_id"])
df = df[df["gqd_bin"] == df["gqd_bin"]] #sometimes gq distance is nan if glottolog tree is small and so multifurcating, that it does noti contain butterflies

results_df = pd.read_csv(os.path.join(results_dir, "raxml_pythia_results.csv"), sep = ";")
df = pd.merge(df, results_df, how = 'left', left_on=["ds_id", "source", "ling_type", "family"], right_on = ["ds_id", "source", "ling_type", "family"])
print(df)

print("Datasets")
sources = set(df['source'].tolist())
r = [[source, len(df[df["source"] == source])] for source in sources]
print(tabulate(r, tablefmt="pipe", headers = ["source", "number of datasets"]))
print("")

print("Effects of Synonym Selection")
plot_distribution(df, "rf_bin_avg", r'$\bar{\delta}$')
plot_distribution(df, "rf_bin_max", r'$\delta_{\max}$')
plot_distribution(df, "gqd_diff_bin_sampled", r'$\rho_full - \rho_s$')
r = []
for column in ["multistate_ratio", "difficulty"]:
    mini_df = df[["rf_bin_avg", column]]
    mini_df = mini_df.dropna()
    pearson = stats.pearsonr(mini_df['rf_bin_avg'], mini_df[column])
    r.append([column, pearson[0], pearson[1]])
print("Correlation with rf_bin_avg")
print(tabulate(r, tablefmt="pipe", floatfmt=".5f", headers = ["metric", "pearson correlation", "p-value"]))
print("")

print("Modelling Data with Synonyms")
best_type_dfs = {}
best_type_dfs["bin"] =  df[df["bin_best"] & (df["catg_bin_best"] == False) & (df["catg_multi_best"] == False)]
best_type_dfs["catg_bin"] =  df[(df["bin_best"] == False) & df["catg_bin_best"] & (df["catg_multi_best"] == False)]
best_type_dfs["catg_multi"] =  df[(df["bin_best"] == False) & (df["catg_bin_best"] == False) & df["catg_multi_best"]]
best_type_dfs["bin&catg_bin"] =  df[df["bin_best"] & df["catg_bin_best"] & (df["catg_multi_best"] == False)]
best_type_dfs["bin&catg_multi"] =  df[df["bin_best"] & (df["catg_bin_best"] == False) & df["catg_multi_best"]]
best_type_dfs["catg_bin&catg_multi"] =  df[(df["bin_best"] == False) & df["catg_bin_best"] & df["catg_multi_best"]]
best_type_dfs["all"] =  df[df["bin_best"] & df["catg_bin_best"] & df["catg_multi_best"]]



print("Mean GQ distances to gold standard")
r = [[cm_type, df['gqd_' + cm_type].mean()] for cm_type in ["bin", "catg_bin", "catg_multi", "sampled_avg"]]
print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["Inference on ", "mean GQ distance"]))
print("")

print("Number of datasets for which the inference on the respective type leads to the tree closest to the gold standard:")
r = [[cm_type, len(best_type_dfs[cm_type])] for cm_type in ["bin", "catg_bin", "catg_multi", "bin&catg_bin", "bin&catg_multi", "catg_bin&catg_multi", "all"]]
print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["cm_type(s)", "best in x datasets"]))
print("(In the following, we group the datasets according to which cm_type leads to the tree closest to the gold standard)")
print("")

cm_types = ["bin", "catg_bin", "catg_multi"]
gqd_diffs = []
rf_distances = []
for k, reference_cm_type in enumerate(cm_types):
    cur_gqd_diffs = [[] for _ in cm_types]
    cur_rf_distances = [[] for _ in cm_types]
    for i, row in best_type_dfs[reference_cm_type].iterrows():
        best_type_gqd = row["distance_matrix"].ref_tree_dist(reference_cm_type, "glottolog", "gq")
        for j, other_cm_type in enumerate(cm_types):
            other_gqd = row["distance_matrix"].ref_tree_dist(other_cm_type, "glottolog", "gq")
            if other_gqd != other_gqd:
                print(row["ds_id"])
            cur_gqd_diffs[j].append(other_gqd - best_type_gqd)
            cur_rf_distances[j].append(row["distance_matrix"].ref_tree_dist(reference_cm_type, other_cm_type, "rf"))
    gqd_diffs.append([reference_cm_type] + [sum(cur_gqd_diffs[j]) / len(cur_gqd_diffs[j]) for j in range(len(cm_types))])
    rf_distances.append([reference_cm_type] + [sum(cur_rf_distances[j]) / len(cur_rf_distances[j]) for j in range(len(cm_types))])
print("Each row refers to the group of datasets corresponding to the given cm_type")
print("Each entry provides the result of the comparison of the tree resulting from the inference on the best-performing cm_type with the tree resulting from the inference on cm_type the respecitve column corresponds to")
print("Average differences of GQ distance to gold standard")
print(tabulate(gqd_diffs, tablefmt="pipe", floatfmt=".4f", headers = ["reference_cm_type"] + cm_types))
print("Average RF Distance of best scoring tree")
print(tabulate(rf_distances, tablefmt="pipe", floatfmt=".4f", headers = ["reference_cm_type"] + cm_types))
print("")

print("Means of metrics within dataset groups")
columns = ["alpha", "sites_per_char", "difficulty"]
r = [[cm_type] + [best_type_dfs[cm_type][column].mean() for column in columns] for cm_type in ["bin", "catg_bin", "catg_multi", "all"]]
print(tabulate(r, tablefmt="pipe", floatfmt=".4f", headers = ["cm_type group"] + columns))
print("")

print("Number of datasets with high rate heterogenity within dataset groups")
r = []
for cm_type in ["bin", "catg_bin", "catg_multi", "all"]:
    type_df = best_type_dfs[cm_type]
    num = len(type_df[type_df["heterogenity"] == True])
    r.append([cm_type, num])
print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["cm_type group", "num"]))
print("")

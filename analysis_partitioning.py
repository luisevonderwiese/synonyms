import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
from tabulate import tabulate
import math
from scipy import stats

from lingdata import database
from lingdata.categorical import CategoricalData

import code.distances as distances
from code.distances import DistanceMatrixIO
import code.util as util


results_dir = "data/results"
plots_dir = os.path.join(results_dir, "plots_partitioning")
if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

pd.set_option('display.max_rows', None)

config_path = "synonyms_lingdata_config_partitioning.json"

database.read_config(config_path)
df = database.data()

print("Datasets with less than 2 different values")
print(df[df["max_values"] < 2]["ds_id"])
df = df[df["max_values"] >= 2]
print("Datasets with more than 64 different values")
print(df[df["max_values"] > 64]["ds_id"])
df = df[df["max_values"] <= 64]

distance_matrices = []
d_io = DistanceMatrixIO(["rf", "gq"], ["glottolog", "bin", "catg_bin", "catg_multi", "bin_BIN+G_2", "bin_BIN+G_x"])
for (i, row) in df.iterrows():
    dm = d_io.read_matrix(util.dist_dir_partitioning(results_dir, row))
    distance_matrices.append(dm)
df["distance_matrix"] = distance_matrices

for (i, row) in df.iterrows():
    gqds = {}
    gqd_names = ["ds_id", "ling_type", "alpha", "sites_per_char", "difficulty"]
    for type in ["bin", "bin_BIN+G_2", "bin_BIN+G_x", "catg_bin", "catg_multi"]:
        gqds[type] = row["distance_matrix"].ref_tree_dist("glottolog", type, "gq")
        df.at[i, "gqd_" + type] = gqds[type]
        gqd_names.append("gqd_" + type)
    m = min(gqds.values())
    for type, gqd in gqds.items():
        if gqd == m:
            df.at[i, type + "_best"] = True
        else:
            df.at[i, type + "_best"] = False

print("Datasets for which GQ distance to glottolog tree cannot be determined")
print(df[df["gqd_bin"] != df["gqd_bin"]]["ds_id"])
df = df[df["gqd_bin"] == df["gqd_bin"]] #sometimes gq distance is nan if glottolog tree is small and so multifurcating, that it does noti contain butterflies

results_df = pd.read_csv(os.path.join(results_dir, "raxml_pythia_results_partitioning.csv"), sep = ";")
df = pd.merge(df, results_df, how = 'left', left_on=["ds_id", "source", "ling_type", "family"], right_on = ["ds_id", "source", "ling_type", "family"])
#print(df)
print(df[gqd_names])


print("Datasets")
sources = set(df['source'].tolist())
r = [[source, len(df[df["source"] == source])] for source in sources]
print(tabulate(r, tablefmt="pipe", headers = ["source", "number of datasets"]))
print("")

print("Modelling Data with Synonyms")
best_type_dfs = {}
best_type_dfs["bin"] =  df[df["bin_best"] & (df["bin_BIN+G_2_best"] == False) & (df["bin_BIN+G_x_best"] == False)]
best_type_dfs["bin_BIN+G_2"] =  df[(df["bin_best"] == False) & df["bin_BIN+G_2_best"] & (df["bin_BIN+G_x_best"] == False)]
best_type_dfs["bin_BIN+G_x"] =  df[(df["bin_best"] == False) & (df["bin_BIN+G_2_best"] == False) & df["bin_BIN+G_x_best"]]
best_type_dfs["bin&bin_BIN+G_2"] =  df[df["bin_best"] & df["bin_BIN+G_2_best"] & (df["bin_BIN+G_x_best"] == False)]
best_type_dfs["bin&bin_BIN+G_x"] =  df[df["bin_best"] & (df["bin_BIN+G_2_best"] == False) & df["bin_BIN+G_x_best"]]
best_type_dfs["bin_BIN+G_2&bin_BIN+G_x"] =  df[(df["bin_best"] == False) & df["bin_BIN+G_2_best"] & df["bin_BIN+G_x_best"]]
best_type_dfs["all"] =  df[df["bin_best"] & df["bin_BIN+G_2_best"] & df["bin_BIN+G_x_best"]]



print("Mean GQ distances to gold standard")
r = [[cm_type, df['gqd_' + cm_type].mean()] for cm_type in ["bin", "bin_BIN+G_2", "bin_BIN+G_x"]]
print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["Inference on ", "mean GQ distance"]))
print("")

print("Number of datasets for which the inference on the respective type leads to the tree closest to the gold standard:")
r = [[cm_type, len(best_type_dfs[cm_type])] for cm_type in ["bin", "bin_BIN+G_2", "bin_BIN+G_x", "bin&bin_BIN+G_2", "bin&bin_BIN+G_x", "bin_BIN+G_2&bin_BIN+G_x", "all"]]
print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["cm_type(s)", "best in x datasets"]))
print("(In the following, we group the datasets according to which cm_type leads to the tree closest to the gold standard)")
print("")

cm_types = ["bin", "bin_BIN+G_2", "bin_BIN+G_x"]
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
r = [[cm_type] + [best_type_dfs[cm_type][column].mean() for column in columns] for cm_type in ["bin", "bin_BIN+G_2", "bin_BIN+G_x", "all"]]
print(tabulate(r, tablefmt="pipe", floatfmt=".4f", headers = ["cm_type group"] + columns))
print("")

print("Number of datasets with high rate heterogenity within dataset groups")
r = []
for cm_type in ["bin", "bin_BIN+G_2", "bin_BIN+G_x", "all"]:
    type_df = best_type_dfs[cm_type]
    num = len(type_df[type_df["heterogenity"] == True])
    r.append([cm_type, num])
print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["cm_type group", "num"]))
print("")

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



setups = ["bin", "catg_bin", "catg_multi", "bin&catg_bin", "bin&catg_multi", "catg_bin&catg_multi", "all"]
relevant_setups = ["bin", "catg_bin", "catg_multi", "all"]

results_dir = "data/results"
plots_dir = os.path.join(results_dir, "plots")
if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

pd.set_option('display.max_rows', None)

config_path = "synonyms_lingdata_config.json"

database.read_config(config_path)
df = database.data()

df = df[df["max_values"] >= 2] #filters out datasets with only one value/missing data (last time I checked it affected only one dataset)

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
print("Correlation with $\bar{\delta}")
print(tabulate(r, tablefmt="pipe", floatfmt=".5f", headers = ["column", "pearson correlation", "p-value"]))
print("")

print("Modelling Data with Synonyms")
winner_dfs = {}
winner_dfs["bin"] =  df[df["bin_best"] & (df["catg_bin_best"] == False) & (df["catg_multi_best"] == False)]
winner_dfs["catg_bin"] =  df[(df["bin_best"] == False) & df["catg_bin_best"] & (df["catg_multi_best"] == False)]
winner_dfs["catg_multi"] =  df[(df["bin_best"] == False) & (df["catg_bin_best"] == False) & df["catg_multi_best"]]
winner_dfs["bin&catg_bin"] =  df[df["bin_best"] & df["catg_bin_best"] & (df["catg_multi_best"] == False)]
winner_dfs["bin&catg_multi"] =  df[df["bin_best"] & (df["catg_bin_best"] == False) & df["catg_multi_best"]]
winner_dfs["catg_bin&catg_multi"] =  df[(df["bin_best"] == False) & df["catg_bin_best"] & df["catg_multi_best"]]
winner_dfs["all"] =  df[df["bin_best"] & df["catg_bin_best"] & df["catg_multi_best"]]


print("Number of datasets for which the inference on the respective type leads to the tree closest to the gold standard:")
r = [[setup, len(winner_dfs[setup])] for setup in setups]
print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["setup", "best in x datasets"]))
print("")
print("Mean GQ distances to gold standard")
r = [[type, df['gqd_' + type].mean()] for type in ["bin", "catg_bin", "catg_multi", "sampled_avg"]]
print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["Inference on ", "mean GQ distance"]))
print("")
print("Mean GQ distances to gold standard - only among datasets for which the respective inference performs best")
r = [[setup, winner_dfs[setup]["gqd_" + setup].mean()] for setup in setups[:3]]
print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["setup", "best average gqd"]))
print("")

gqd_diffs = []
rf_distances = []
for k, reference_setup in enumerate(relevant_setups[:-1]):
    cur_gqd_diffs = [[] for _ in relevant_setups[:-1]]
    cur_rf_distances = [[] for _ in relevant_setups[:-1]]
    for i, row in winner_dfs[reference_setup].iterrows():
        winner_gqd = row["distance_matrix"].ref_tree_dist(reference_setup, "glottolog", "gq")
        for j, other_setup in enumerate(relevant_setups[:-1]):
            other_gqd = row["distance_matrix"].ref_tree_dist(other_setup, "glottolog", "gq")
            cur_gqd_diffs[j].append(other_gqd - winner_gqd)
            cur_rf_distances[j].append(row["distance_matrix"].ref_tree_dist(reference_setup, other_setup, "rf"))
    gqd_diffs.append([reference_setup] + [sum(cur_gqd_diffs[j]) / len(cur_gqd_diffs[j]) for j in range(len(relevant_setups) - 1)])
    rf_distances.append([reference_setup] + [sum(cur_rf_distances[j]) / len(cur_rf_distances[j]) for j in range(len(relevant_setups) - 1)])
print("Each row corresponds to to group of datasets fo which the given MSA type leads to the best tree")
print("In each column, there is the result of the comparison of this tree with the tree resulting from the respective other setup")
print("Average differences of gq distance")
print(tabulate(gqd_diffs, tablefmt="pipe", floatfmt=".4f", headers = ["reference_setup"] + relevant_setups[:-1]))
print("Average RF Distance of best scoring tree")
print(tabulate(rf_distances, tablefmt="pipe", floatfmt=".4f", headers = ["reference_setup"] + relevant_setups[:-1]))
print("")

print("Means in groups of datasets")
columns = ["alpha", "sites_per_char", "difficulty"]
r = [[setup] + [winner_dfs[setup][column].mean() for column in columns] for setup in relevant_setups]
print(tabulate(r, tablefmt="pipe", floatfmt=".4f", headers = ["setup"] + columns))
print("")

print("Number of datasets with high rate heterogenity")
r = []
for setup in relevant_setups:
    setup_df = winner_dfs[setup]
    num = len(setup_df[setup_df["heterogenity"] == True])
    r.append([setup, num])
print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["setup", "num"]))
print("")






#iecor = df[df["ds_id"] == "iecor"]

#print("Mean rf_bin_avg: " + str(df["rf_bin_avg"].mean()))
#print("Mean difficulty: " + str(df["difficulty"].mean()))

#print("iecor rf_bin_avg: " + str(iecor["rf_bin_avg"].mean()))
#print("iecor difficulty: " + str(iecor["difficulty"].mean()))

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


setups = ["bin", "catg_bin", "catg_multi", "bin&catg_bin", "bin&catg_multi", "catg_bin&catg_multi", "all"]
relevant_setups = ["bin", "catg_bin", "catg_multi", "all"]

def add_distance_matrices(df):
    distance_matrices = []
    metrics = ["rf", "gq"]
    ref_tree_names = ["glottolog", "bin", "catg_bin", "catg_multi", "consensus"]
    d_io = DistanceMatrixIO(metrics, ref_tree_names)
    for (i, row) in df.iterrows():
        dm = d_io.read_matrix(util.dist_dir(results_dir, row))
        distance_matrices.append(dm)
    df["distance_matrix"] = distance_matrices

def add_distances(df):
    winner_dict = {"bin" : [], "catg_bin" : [], "catg_multi" : []}
    for (i, row) in df.iterrows():
        df.at[i, "rf_bin_avg"] = row["distance_matrix"].avg_ref_tree_dist("bin", "rf")
        df.at[i, "rf_bin_max"] = row["distance_matrix"].max_ref_tree_dist("bin", "rf")
        df.at[i, "rf_sampled_avg"] = row["distance_matrix"].sampled_avg_avg_dist("rf")
        df.at[i, "rf_sampled_max"] = row["distance_matrix"].sampled_max_avg_dist("rf")
        df.at[i, "rf_bin_catg_bin"] = row["distance_matrix"].ref_tree_dist("bin", "catg_bin", "rf")
        df.at[i, "rf_catg_bin_catg_multi"] = row["distance_matrix"].ref_tree_dist("catg_multi", "catg_bin", "rf")
        df.at[i, "rf_bin_catg_multi"] = row["distance_matrix"].ref_tree_dist("bin", "catg_multi", "rf")
        gqd_bin = row["distance_matrix"].ref_tree_dist("glottolog", "bin", "gq")
        df.at[i, "gqd_bin"] = gqd_bin
        gqd_catg = row["distance_matrix"].ref_tree_dist("glottolog", "catg_bin", "gq")
        df.at[i, "gqd_catg_bin"] = gqd_catg
        gqd_catg_multi = row["distance_matrix"].ref_tree_dist("glottolog", "catg_multi", "gq")
        df.at[i, "gqd_catg_multi"] = gqd_catg_multi

        df.at[i, "gqd_sampled_avg"] = row["distance_matrix"].avg_ref_tree_dist("glottolog", "gq")
        winner = ["bin"]
        if gqd_catg < gqd_bin:
            winner = ["catg_bin"]
        if gqd_catg == gqd_bin:
            winner.append("catg_bin")
        if gqd_catg_multi == gqd_bin:
            if gqd_catg_multi <= gqd_catg:
                winner.append("catg_multi")
        if gqd_catg_multi < gqd_bin:
            if gqd_catg_multi < gqd_catg:
                winner = ["catg_multi"]
            if gqd_catg_multi == gqd_catg:
                winner.append("catg_multi")
        for setup, values in winner_dict.items():
            if setup in winner:
                winner_dict[setup].append(True)
            else:
                winner_dict[setup].append(False)

    df["gqd_sampled_diff"] = df["gqd_sampled_avg"] - df["gqd_bin"]
    df["gqd_diff_bin_catg_bin"] = df["gqd_bin"] - df["gqd_catg_bin"]
    df["gqd_diff_catg_bin_catg_multi"] = df["gqd_catg_bin"] - df["gqd_catg_multi"]
    df["gqd_diff_bin_catg_multi"] = df["gqd_bin"] - df["gqd_catg_multi"]
    df["bin_best"] = winner_dict["bin"]
    df["catg_bin_best"] = winner_dict["catg_bin"]
    df["catg_multi_best"] = winner_dict["catg_multi"]



def add_result_data(df):
    #converters={"sampled_difficulties": lambda x: [float(el) for el in x.strip("[]").split(", ")]}
    results_df = pd.read_csv(os.path.join(results_dir, "raxml_pythia_results.csv"), sep = ";")
    df = pd.merge(df, results_df, how = 'left', left_on=["ds_id", "source", "ling_type", "family"], right_on = ["ds_id", "source", "ling_type", "family"])
    return df


def get_bins(a, nbins):
    min_val = min(a)
    max_val = max(a)
    step = (max_val - min_val) / (nbins - 1)
    return [min_val + i * step for i in range(nbins)]





def get_winner_dfs(df):
    winner_dfs = {}
    winner_dfs["bin"] =  df[df["bin_best"] & (df["catg_bin_best"] == False) & (df["catg_multi_best"] == False)]
    winner_dfs["catg_bin"] =  df[(df["bin_best"] == False) & df["catg_bin_best"] & (df["catg_multi_best"] == False)]
    winner_dfs["catg_multi"] =  df[(df["bin_best"] == False) & (df["catg_bin_best"] == False) & df["catg_multi_best"]]
    winner_dfs["bin&catg_bin"] =  df[df["bin_best"] & df["catg_bin_best"] & (df["catg_multi_best"] == False)]
    winner_dfs["bin&catg_multi"] =  df[df["bin_best"] & (df["catg_bin_best"] == False) & df["catg_multi_best"]]
    winner_dfs["catg_bin&catg_multi"] =  df[(df["bin_best"] == False) & df["catg_bin_best"] & df["catg_multi_best"]]
    winner_dfs["all"] =  df[df["bin_best"] & df["catg_bin_best"] & df["catg_multi_best"]]
    return winner_dfs




def setup_comparison(df, winner_dfs):
    print("MSA\t\tgqd(median)")
    print("bin\t\t" + str(df['gqd_bin'].median()))
    print("catg_bin\t\t" + str(df['gqd_catg_bin'].median()))
    print("catg_multi\t" + str(df['gqd_catg_multi'].median()))
    print("")
    print("MSA\t\tgqd(mean)")
    print("bin\t\t" + str(df['gqd_bin'].mean()))
    print("catg_bin\t\t" + str(df['gqd_catg_bin'].mean()))
    print("catg_multi\t" + str(df['gqd_catg_multi'].mean()))
    print("")
    r = [[setup, len(winner_dfs[setup])] for setup in setups]
    print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["setup", "best in x datasets"]))
    print("")
    r = [[setup, winner_dfs[setup]["gqd_" + setup].mean()] for setup in setups[:3]]
    print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["setup", "best average gqd"]))
    print("")

def num_het(winner_dfs):
    print("Number of datasets with high rate heterogenity")
    r = []
    for setup in relevant_setups:
        setup_df = winner_dfs[setup]
        num = len(setup_df[setup_df["heterogenity"] == True])
        r.append([setup, num])
    print(tabulate(r, tablefmt="pipe", floatfmt=".2f", headers = ["setup", "num"]))
    print("")

def winner_comparison(winner_dfs, columns):
    r = [[setup] + [winner_dfs[setup][column].mean() for column in columns] for setup in relevant_setups]
    print(tabulate(r, tablefmt="pipe", floatfmt=".4f", headers = ["setup"] + columns))
    print("")

def winner_comparison_plots(winner_dfs, x_column, y_column):
    colors = ["blue", "red", "green", "black"]
    plt.xlabel(x_column)
    plt.ylabel(y_column)
    for i, setup in enumerate(relevant_setups):
        df = winner_dfs[setup]
        plt.scatter(df[x_column], df[y_column], label = setup, color = colors[i], s = 2)
    plt.legend()
    plt.savefig(os.path.join(plots_dir, x_column + "_" +  y_column + ".png"))
    plt.clf()

def plot_distribution(df, column, label):
    data = df[column]
    plt.hist(data, bins = get_bins(data, 20))
    plt.xlabel(label)
    plt.ylabel('Number of datasets')
    plt.savefig(os.path.join(plots_dir, "hist_" + column +  ".png"))
    plt.clf()


def stability_correlation(df, columns):
    r = []
    for column in columns:
        mini_df = df[["rf_bin_avg", column]]
        mini_df = mini_df.dropna()
        pearson = stats.pearsonr(mini_df['rf_bin_avg'], mini_df[column])
        r.append([column, pearson[0], pearson[1]])
    print(tabulate(r, tablefmt="pipe", floatfmt=".5f", headers = ["column", "pearson correlation", "p-value"]))
    print("")




def boxplots(winner_dfs, column):
    data = [winner_dfs[setup][column] for setup in relevant_setups]
    plt.boxplot(data)
    plt.xticks([1, 2, 3, 4], relevant_setups)
    plt.savefig(os.path.join(plots_dir, "box_" + column +  ".png"))
    plt.clf()


def another_analysis(winner_dfs):
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

def sources_analysis(df):
    sources = set(df['source'].tolist())
    r = [[source, len(df[df["source"] == source])] for source in sources]
    print(tabulate(r, tablefmt="pipe", headers = ["source", "number of datasets"]))


def ambig_analysis(df):
    types = ["ambig", "bin", "catg_bin", "catg_multi"]
    r = []
    for i, row in df.iterrows():
        gqd_ambig = row["distance_matrix"].ref_tree_dist("glottolog", "ambig", "gqd")
        if gqd_ambig != gqd_ambig:
            continue
        else:
            for type in types:
                r.append(row["distance_matrix"].ref_tree_dist("glottolog", type, "gqd"))
            best = min(r)
            winners = []
            for i, type in enumerate(types):
                if r[i] == best:
                    winners.append(type)
            r.append(winners)
    print("GQD to Glottolog")
    print(tabulate(r, tablefmt="pipe", headers = types + ["winners"]))






results_dir = "data/results"
plots_dir = os.path.join(results_dir, "plots")
if not os.path.isdir(plots_dir):
    os.makedirs(plots_dir)

pd.set_option('display.max_rows', None)

config_path = "synonyms_lingdata_config.json"

database.read_config(config_path)
df = database.data()

df = df[df["glottolog_tree_path"] == df["glottolog_tree_path"] ]
df = df[df["ling_type"] == "cognate"]
df = df[df["source"] != "correspondence-pattern-data"] #datasets are double in different versions

add_distance_matrices(df)
add_distances(df)
df = df[df["gqd_bin"] == df["gqd_bin"]] #sometimes gq distance is nan if glottolog tree is small and so multifurcating, that it does not contain butterflies
df = add_result_data(df)
print(df)

iecor = df[df["ds_id"] == "iecor"]
df = df[df["max_values"] <= 64]
df = df[df["max_values"] >= 2]

winner_dfs = get_winner_dfs(df)
setup_comparison(df, winner_dfs)

num_het(winner_dfs)
another_analysis(winner_dfs)

plot_distribution(df, "rf_bin_avg", r'$\bar{d}$')
plot_distribution(df, "rf_bin_max", r'$d_{\max}$')

columns_interesting = [
        "alpha",
        "sites_per_char",
        "difficulty"
]

columns_uninteresting = [
    "rf_bin_avg",
    "num_taxa",
    "num_chars",
    "zero_base_frequency_bin"
]

winner_comparison(winner_dfs, columns_interesting)
for column in columns_interesting:
    boxplots(winner_dfs, column)

columns_uninteresting = [
        "num_chars",
        "informative_char_ratio",
        "difficulty_variance",
]


columns_interesting = [
        "multistate_ratio",
        "difficulty"
        ]
columns_bootstrap = [
        "mean_norm_rf_distance",
        "mean_parsimony_support",
        "mean_parsimony_bootstrap_support",
        "mean_bootstrap_support",
        "zero_base_frequency_bin"
        ]

stability_correlation(df, columns_interesting)

winner_comparison_plots(winner_dfs, "gqd_bin", "gqd_catg_multi")

sources_analysis(df)

ambig_analysis(df)

# print("Mean rf_bin_avg: " + str(df["rf_bin_avg"].mean()))
# print("Mean rf_sampled_avg: " + str(df["rf_sampled_avg"].mean()))
# print("Mean difficulty: " + str(df["difficulty"].mean()))
#
# print("iecor rf_bin_avg: " + str(iecor["rf_bin_avg"].mean()))
# print("iecor rf_sampled_avg: " + str(iecor["rf_sampled_avg"].mean()))
# print("iecor difficulty: " + str(iecor["difficulty"].mean()))
# print(" ")

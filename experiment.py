import os
import pandas as pd
import numpy as np

import lingdata.database as database

import code.raxmlng as raxmlng
import code.pythia as pythia
import code.distances as distances
import code.util as util





def run_raxml_ng(df):
    for (i, row) in df.iterrows():
        raxmlng.run_inference(row["msa_paths"]["bin"], "BIN+G", util.prefix(row, "raxmlng", "bin"))
        raxmlng.run_inference(row["msa_paths"]["catg_bin"], "BIN+G", util.prefix(row, "raxmlng" , "catg_bin"), "--prob-msa on")
        raxmlng.run_inference(row["msa_paths"]["catg_multi"], row["MULTIx_MK"] + "+G", util.prefix(row, "raxmlng", "catg_multi_mk"), "--prob-msa on")
        for (i, msa_path) in enumerate(row["sampled_msa_paths"]):
            raxmlng.run_inference(msa_path, "BIN+G", util.prefix(row, "raxmlng", "sampled/sampled" + str(i)))


def run_pythia(df):
    for (i, row) in df.iterrows():
        pythia.run_with_padding(row["msa_paths"]["bin"], util.prefix(row, "pythia", "bin"))
        for (i, msa_path) in enumerate(row["sampled_msa_paths"]):
            pythia.run_with_padding(msa_path, util.prefix(row, "pythia", "sampled/sampled" + str(i)))

def consense_trees(df):
    for (i, row) in df.iterrows():
        util.prefixes = []
        for (i, msa_path) in enumerate(row["sampled_msa_paths"]):
            util.prefixes.append(util.prefix(row, "raxmlng", "sampled/sampled" + str(i)))
        raxmlng.consense_tree(util.prefixes, util.prefix(row, "raxmlng", "sampled_consensus"))

def calculate_distances(df):
    for (i, row) in df.iterrows():
        dist_dir = util.dist_dir(row)
        if os.path.isfile(os.path.join(dist_dir, "matrix_rf.csv")) and os.path.isfile(os.path.join(dist_dir, "matrix_gq.csv")):
            continue
        ref_tree_paths = {}
        ref_tree_paths["glottolog"] = row["glottolog_tree_path"]
        ref_tree_paths["bin"] = raxmlng.best_tree_path(util.prefix(row, "raxmlng", "bin"))
        ref_tree_paths["catg_bin"] = raxmlng.best_tree_path(util.prefix(row, "raxmlng", "catg"))
        ref_tree_paths["catg_multi_mk"] = raxmlng.best_tree_path(util.prefix(row, "raxmlng", "catg_multi_mk"))
        ref_tree_paths["consensus"] = raxmlng.consensus_tree_path(util.prefix(row, "raxmlng", "sampled_consensus"))
        sampled_tree_paths = []
        for (i, msa_path) in enumerate(row["sampled_msa_paths"]):
            sampled_tree_paths.append(raxmlng.best_tree_path(util.prefix(row, "raxmlng", "sampled/sampled" + str(i))))
        distances.generate_distances(dist_dir, sampled_tree_paths, ref_tree_paths)


def write_results_df(df):
    sampled_difficulties = []
    for i, row in df.iterrows():
        alpha = raxmlng.alpha(util.prefix(row, "raxmlng", "bin"))
        df.at[i, "alpha"] = alpha
        if alpha < 20:
            df.at[i, "heterogenity"] = 1
        else:
            df.at[i, "heterogenity"] = 0
        df.at[i, "difficulty"] = pythia.get_difficulty(util.prefix(row, "pythia", "bin"))
        sampled_d = []
        for (j, msa_path) in enumerate(row["sampled_msa_paths"]):
            sampled_d.append(pythia.get_difficulty(util.prefix(row, "pythia", "sampled/sampled" + str(j))))
        df.at[i, "difficulty_variance"] =  np.var(sampled_d)
        df.at[i, "difficulty_mean"] =  np.mean(sampled_d)
        df.at[i, "avg_ml_dist_bin"] =  raxmlng.avg_ml_tree_dist(util.prefix(row, "raxmlng", "bin"))
        supports = raxmlng.support_values(util.prefix(row, "raxmlng_support", "bin"))
        if len(supports) == 0:
            df.at[i, "mean_bootstrap_support"] = float("nan")
        else:
            df.at[i, "mean_bootstrap_support"] = sum(supports) / len(supports)
        df.at[i, "aic_bin"] = raxmlng.aic(util.prefix(row, "raxmlng", "bin"))[0]
        df.at[i, "aic_catg"] = raxmlng.aic(util.prefix(row, "raxmlng", "catg"))[0]
        df.at[i, "aic_catg_multi_mk"] = raxmlng.aic(util.prefix(row, "raxmlng", "catg_multi_mk"))[0]
    print_df = df[["ds_id", "source", "ling_type", "family", "alpha", "difficulty", "difficulty_mean", "difficulty_variance", "avg_ml_dist_bin", "mean_bootstrap_support",
        "aic_bin", "aic_catg", "aic_catg_multi_mk"]]
    print(print_df)
    print_df.to_csv(os.path.join(results_dir, "raxml_pythia_results.csv"), sep = ";")






raxmlng.exe_path = "./bin/raxml-ng"
pythia.raxmlng_path = "./bin/raxml-ng"
distances.exe_path = "./bin/qdist"
config_path = "synonyms_lingdata_config.json"
results_dir = "data/results"



database.read_config(config_path)
database.update_native()
database.generate_data()
df = database.data()
print(df)


pd.set_option('display.max_rows', None)
run_raxml_ng(df)
consense_trees(df)
calculate_distances(df)
run_pythia(df)
write_results_df(df)

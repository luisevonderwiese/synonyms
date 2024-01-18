import os
import pandas as pd
import numpy as np

import lingdata.database as database

import code.raxmlng as raxmlng
import code.pythia as pythia
import code.distances as distances
from code.distances import DistanceMatrixIO
import code.util as util





def run_raxml_ng(df):
    for (i, row) in df.iterrows():
        raxmlng.run_inference(row["msa_paths"]["bin"], "BIN+G", util.prefix(results_dir, row, "raxmlng", "bin"))
        partition_name = util.partition_name("bin", "BIN", True, "2")
        raxmlng.run_inference(row["msa_paths"]["bin"], row["partition_paths"][partition_name], util.prefix(results_dir, row, "raxmlng", partition_name))
        partition_name = util.partition_name("bin", "BIN", True, "x")
        raxmlng.run_inference(row["msa_paths"]["bin"], row["partition_paths"][partition_name], util.prefix(results_dir, row, "raxmlng", partition_name))


def calculate_distances(df):
    metrics = ["rf", "gq"]
    ref_tree_names = ["glottolog", "bin", "catg_bin", "catg_multi", "bin_BIN+G_2", "bin_BIN+G_x"]
    d_io = DistanceMatrixIO(metrics, ref_tree_names)
    for (i, row) in df.iterrows():
        dist_dir = util.dist_dir_partitioning(results_dir, row)
        if os.path.isfile(os.path.join(dist_dir, "matrix_rf.csv")) and os.path.isfile(os.path.join(dist_dir, "matrix_gq.csv")):
            continue
        ref_tree_paths = {}
        ref_tree_paths["glottolog"] = row["glottolog_tree_path"]
        ref_tree_paths["bin"] = raxmlng.best_tree_path(util.prefix(results_dir, row, "raxmlng", "bin"))
        ref_tree_paths["catg_bin"] = raxmlng.best_tree_path(util.prefix(results_dir, row, "raxmlng", "catg_bin"))
        ref_tree_paths["catg_multi"] = raxmlng.best_tree_path(util.prefix(results_dir, row, "raxmlng", "catg_multi"))
        partition_name = util.partition_name("bin", "BIN", True, "2")
        ref_tree_paths[partition_name] = raxmlng.best_tree_path(util.prefix(results_dir, row, "raxmlng", partition_name))
        partition_name = util.partition_name("bin", "BIN", True, "x")
        ref_tree_paths[partition_name] = raxmlng.best_tree_path(util.prefix(results_dir, row, "raxmlng", partition_name))
        d_io.write_matrix(dist_dir, sampled_tree_paths, ref_tree_paths)


def write_results_df(df):
    sampled_difficulties = []
    for i, row in df.iterrows():
        alpha = raxmlng.alpha(util.prefix(results_dir, row, "raxmlng", "bin"))
        df.at[i, "alpha"] = alpha
        if alpha < 20:
            df.at[i, "heterogenity"] = 1
        else:
            df.at[i, "heterogenity"] = 0
        df.at[i, "difficulty"] = pythia.get_difficulty(util.prefix(results_dir, row, "pythia", "bin"))
        df.at[i, "zero_base_frequency_bin"] = raxmlng.base_frequencies(util.prefix(results_dir, row, "raxmlng", "bin"))[0]
    print_df = df[["ds_id", "source", "ling_type", "family", "alpha", "heterogenity", "difficulty", "zero_base_frequency_bin"]]
    print(print_df)
    print_df.to_csv(os.path.join(results_dir, "raxml_pythia_results_partitioning.csv"), sep = ";")




raxmlng.exe_path = "./bin/raxml-ng"
pythia.raxmlng_path = "./bin/raxml-ng"
pythia.predictor_path = "predictors/latest.pckl"
distances.exe_path = "./bin/qdist"
config_path = "synonyms_lingdata_config_partitioning.json"
results_dir = "data/results"



database.read_config(config_path)
database.compile()
df = database.data()
pd.set_option('display.max_rows', None)
print(df)

run_raxml_ng(df)
calculate_distances(df)
write_results_df(df)

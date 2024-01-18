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
        partition_name = util.partition_name("bin", "BIN", True, "2")
        raxmlng.run_inference(row["msa_paths"]["bin"], row["partition_paths"][partition_name], util.prefix(results_dir, row, "raxmlng", partition_name))




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

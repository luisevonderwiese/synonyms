import os
import shutil

results_dir = "data/results/distances/"
toremove = []
for ds_name in os.listdir(results_dir):
    if not os.path.isdir(os.path.join("data/lingdata/msa", ds_name)):
        toremove.append(os.path.join(results_dir, ds_name))

for path in toremove:
    shutil.rmtree(path)

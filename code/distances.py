import os
from ete3 import Tree
import numpy as np
import traceback
import shutil
import copy

metrics = ["rf", "gq"]
ref_tree_names = ["glottolog", "bin", "catg_bin", "catg_multi", "consensus"]
num_ref_trees = len(ref_tree_names)
ref_trees = {}
for i, ref_tree_name in enumerate(ref_tree_names):
    ref_trees[ref_tree_name] = i - num_ref_trees
exe_path = ""


def rf_distance(t1, t2):
    if t1 is None or t2 is None:
        return float('nan')
    if t1 != t1 or t2 != t2:
        return float("nan")
    rf, max_rf, common_leaves, parts_t1, parts_t2,discard_t1, discart_t2 = t1.robinson_foulds(t2, unrooted_trees = True)
    if max_rf == 0:
        return float('nan')
    return rf/max_rf

def gq_distance(tree_name1, tree_name2):
    if tree_name1 is None or tree_name2 is None:
        return float('nan')
    if tree_name1 != tree_name1 or tree_name2 != tree_name2:
        return float("nan")
    os.system(exe_path + " " + tree_name1 + " " + tree_name2 + " >out.txt")
    lines = open("out.txt").readlines()
    if len(lines) < 2: #error occurred
        return float('nan')
    res_q = float(lines[1].split("\t")[-3])
    qdist = 1 - res_q
    os.remove("out.txt")
    return qdist

def write_matrix(matrix, path):
    with open(path, "w+") as dm_file:
        dm_file.write("\n".join([",".join([str(el) for el in row]) for row in matrix]))


def matrix(tree_paths, metric):
    distance_matrix = [[0.0 for _ in range(i + 1)] for i in range(len(tree_paths))]
    if metric ==  "rf":
        trees = []
        for tree_path in tree_paths:
            try:
                trees.append(Tree(tree_path))
            except Exception as e:
                print(e)
                trees.append(None)
        for i in range(len(trees)):
            for j in range(i, len(trees)):
                assert(j >= i)
                rfd = rf_distance(trees[i], trees[j])
                distance_matrix[j][i] = rfd
        return distance_matrix
    if metric == "gq":
        for i in range(len(tree_paths)):
            for j in range(i, len(tree_paths)):
                gqd = gq_distance(tree_paths[i], tree_paths[j])
                distance_matrix[j][i] = gqd
        return distance_matrix
    else:
        print("Metric " + metric + " not defined")



def generate_distances(dist_dir, sampled_tree_paths, ref_tree_paths):
    if not os.path.isdir(dist_dir):
        os.makedirs(dist_dir)
    sampled_tree_paths = copy.deepcopy(sampled_tree_paths)
    for ref_tree_name in ref_tree_names:
        sampled_tree_paths.append(ref_tree_paths[ref_tree_name])
    try:
        for metric in metrics:
            matrix_path = os.path.join(dist_dir, "matrix_" + metric + ".csv")
            m = matrix(sampled_tree_paths, metric)
            write_matrix(m, matrix_path)
    except Exception as e:
        traceback.print_exc()
        shutil.rmtree(dist_dir)
    return DistanceMatrix(dist_dir)



class DistanceMatrix:

    def __init__(self, dist_dir):
        self. matrices = {}
        for metric in metrics:
            path = os.path.join(dist_dir, "matrix_" + metric + ".csv")
            self.matrices[metric] = self.read_matrix(path)
        self.num_sampled = len(self.matrices[metrics[0]]) - num_ref_trees



    def read_matrix(self, path):
        return [[float(val) for val in row.split(',')] for row in open(path, "r").readlines()]


    def d(self, idx1, idx2, metric):
        return self.matrices[metric][max(idx1, idx2)][min(idx1, idx2)]


    def ref_tree_dist(self, tree1, tree2, metric):
        idx1 = ref_trees[tree1]
        idx2 = ref_trees[tree2]
        max_idx = max(idx1, idx2)
        second_idx = min(idx1, idx2) - (max_idx + 1)
        return self.matrices[metric][max_idx][second_idx]

    def ref_tree_dist_vector(self, tree, metric):
        idx = ref_trees[tree]
        offset = - idx - (num_ref_trees + 1)
        res = self.matrices[metric][idx][:offset]
        assert(len(res) == self.num_sampled)
        return res

    def avg_ref_tree_dist(self, tree, metric):
        dists = [x for x in self.ref_tree_dist_vector(tree, metric) if x==x]
        if len(dists) == 0:
            return float("nan")
        return sum(dists) / len(dists)

    def max_ref_tree_dist(self, tree, metric):
        dists = [x for x in self.ref_tree_dist_vector(tree, metric) if x==x]
        if len(dists) == 0:
            return float("nan")
        return max(dists)

    def sampled_avg_dists(self, metric):
        avg_dists = []
        for i in range(self.num_sampled):
            temp_dists = []
            for j in range(self.num_sampled):
                if i == j:
                    continue
                x = self.d(i, j, metric)
                if x == x:
                    temp_dists.append(x)
            if len(temp_dists) == 0:
                avg_dists.append(float("nan"))
            else:
                avg_dists.append(sum(temp_dists) / len(temp_dists))
        return avg_dists


    def sampled_avg_avg_dist(self, metric):
        avg_dists = [x for x in self.sampled_avg_dists(metric) if x==x]
        if len(avg_dists) == 0:
            return float("nan")
        return sum(avg_dists) / len(avg_dists)

    def sampled_max_avg_dist(self, metric):
        avg_dists = [x for x in self.sampled_avg_dists(metric) if x==x]
        if len(avg_dists) == 0:
            return float("nan")
        return max(avg_dists)

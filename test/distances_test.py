import sys
sys.path.append("../code")

from distances import DistanceMatrixIO
import distances as distances

from ete3 import Tree
import os

def test_distances():
    metrics = ["rf", "gq"]
    ref_tree_names = ["glottolog", "bin", "catg_bin", "catg_multi", "consensus"]
    d_io = DistanceMatrixIO(metrics, ref_tree_names)
    dist_dir = "../test_data/distances"
    if not os.path.isdir(dist_dir):
        os.mkdir(dist_dir)
    sampled_tree_paths = [
        "../test_data/trees/bodtkhobwa/sampled00_bin.raxml.bestTree",
        "../test_data/trees/bodtkhobwa/sampled01_bin.raxml.bestTree",
        "../test_data/trees/bodtkhobwa/sampled02_bin.raxml.bestTree",
        "../test_data/trees/bodtkhobwa/sampled03_bin.raxml.bestTree",
        "../test_data/trees/bodtkhobwa/sampled04_bin.raxml.bestTree"]
    ref_tree_paths = {}
    ref_tree_paths["glottolog"] = "../test_data/trees/bodtkhobwa/glottolog.tre"
    ref_tree_paths["bin"] = "../test_data/trees/bodtkhobwa/full_bin.raxml.bestTree"
    ref_tree_paths["catg_bin"] = "../test_data/trees/bodtkhobwa/full_catg.raxml.bestTree"
    ref_tree_paths["catg_multi"] = "../test_data/trees/bodtkhobwa/full_catg_multi.raxml.bestTree"
    ref_tree_paths["consensus"] = "../test_data/trees/bodtkhobwa/sampled_consensus.raxml.consensusTreeMR"
    d_io.write_matrix(dist_dir, sampled_tree_paths, ref_tree_paths)
    dm = d_io.read_matrix(dist_dir)
    ref_trees = {}
    for name, path in ref_tree_paths.items():
        ref_trees[name] = Tree(path)
    sampled_trees = [Tree(sampled_tree_path) for sampled_tree_path in sampled_tree_paths]
    for ref_tree_name1 in ref_tree_paths:
        for ref_tree_name2 in ref_tree_paths:
            if ref_tree_name1 ==  ref_tree_name2:
                continue
            rf1 = dm.ref_tree_dist(ref_tree_name1, ref_tree_name2, "rf")
            rf2 = distances.rf_distance(ref_trees[ref_tree_name1], ref_trees[ref_tree_name2])
            assert(rf1 == rf2)
            gq1 = dm.ref_tree_dist(ref_tree_name1, ref_tree_name2, "gq")
            gq2 = distances.gq_distance(ref_tree_paths[ref_tree_name1], ref_tree_paths[ref_tree_name2])
            assert(gq1 == gq2)
        rf_dists1 = dm.ref_tree_dist_vector(ref_tree_name1, "rf")
        rf_dists2 = []
        for sampled_tree in sampled_trees:
            rf_dists2.append(distances.rf_distance(sampled_tree, ref_trees[ref_tree_name1]))
        assert(rf_dists1 == rf_dists2)
        gq_dists1 = dm.ref_tree_dist_vector(ref_tree_name1, "gq")
        gq_dists2 = []
        for sampled_tree_path in sampled_tree_paths:
            gq_dists2.append(distances.gq_distance(sampled_tree_path, ref_tree_paths[ref_tree_name1]))
        assert(gq_dists1 == gq_dists2)
    rf_dists = []
    gq_dists = []
    for i in range(len(sampled_trees)):
        for j in range(i + 1, len(sampled_trees)):
            rf_dists.append(distances.rf_distance(sampled_trees[i], sampled_trees[j]))
            gq_dists.append(distances.gq_distance(sampled_tree_paths[i], sampled_tree_paths[j]))
    rf_avg = sum(rf_dists)/len(rf_dists)
    gq_avg = sum(gq_dists)/len(gq_dists)
    rf_avg2 = dm.sampled_avg_dists("rf")
    gq_avg2 = dm.sampled_avg_dists("gq")





distances.exe_path = "./../bin/qdist"
test_distances()

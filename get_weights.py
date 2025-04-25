# -*- coding: utf-8 -*-

from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
from collections import Counter, OrderedDict
from ete3 import Tree
from numpy.core.defchararray import index
from scipy import stats
from sklearn.feature_selection import SelectFromModel, chi2, VarianceThreshold
from sklearn.ensemble import ExtraTreesClassifier
import Bio
import numpy as np
import pandas as pd
import math
import os
import sys
import pickle
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
import itertools

name_dataset = "MDR" #"NonMDR"
folder =  "Data"
results_folder = "Results"
folder_res_main = "MDR AMR PA"

def get_weights(filename, ids, file_end):
    distance_matrix, samples = get_mash_distances(filename,ids)

    dist_mat = distance_matrix_modifier(distance_matrix)
    distance_matrix_to_phyloxml(samples, dist_mat, file_end)   
    phyloxml_to_newick(results_folder+"/"+folder_res_main+"/tree_xml"+file_end+".txt", file_end)
    #tree = Phylo.read(folder+"/"+results_folder+"/tree_xml.txt", "phyloxml")
    #Phylo.draw(tree)
    #input("continue_draw")
    weights = GSC_weights_from_newick(results_folder+"/"+folder_res_main+"/tree_newick"+file_end+".txt", normalize="mean1")
    df = pd.DataFrame.from_dict(weights,orient='index', columns=['weights'])
    df = df.reindex(samples)
    
    return df

def get_mash_distances(filename, ids):
    distance_matrix = pd.read_excel(filename, header=[0], index_col=[0])
    distance_matrix = distance_matrix.loc[ids,ids]
    return distance_matrix.values.tolist(), list(distance_matrix.columns)

def distance_matrix_modifier(distancematrix):
    # Modifies distance matrix to be suitable argument 
    # for Bio.Phylo.TreeConstruction._DistanceMatrix function
    for i in range(len(distancematrix)):
        for j in range(len(distancematrix[i])):
            distancematrix[i][j] = float(distancematrix[i][j])
    distance_matrix = []
    counter = 1
    for i in range(len(distancematrix)):
        data = distancematrix[i]
        distance_matrix.append(data[0:counter])
        counter += 1

    return(distance_matrix)

def distance_matrix_to_phyloxml(samples_order, distance_matrix, file_end):
    #Converting distance matrix to phyloxml
    dm = _DistanceMatrix(samples_order, distance_matrix)
    tree_xml = DistanceTreeConstructor().nj(dm)
    with open(results_folder+"/"+folder_res_main+"/tree_xml"+file_end+".txt", "w+") as f1:
        Bio.Phylo.write(tree_xml, f1, "phyloxml")

def phyloxml_to_newick(phyloxml, file_end):
    #Converting phyloxml to newick
    with open(results_folder+"/"+folder_res_main+"/tree_newick"+file_end+".txt", "w+") as f1:
        Bio.Phylo.convert(phyloxml, "phyloxml", f1, "newick")

def GSC_weights_from_newick(newick_tree, normalize=None):
    # Calculating Gerstein Sonnhammer Coathia weights from Newick 
    # string. Returns dictionary where sample names are keys and GSC 
    # weights are values.
    tree = Tree(newick_tree, format=1)
    tree = clip_branch_lengths(tree)
    set_branch_sum(tree)
    set_node_weight(tree)

    #tree.show()
    
    weights = {}
    for leaf in tree.iter_leaves():
        weights[leaf.name] = leaf.NodeWeight
    if normalize == "mean1":
        weights = {k: v*len(weights) for k, v in weights.items()}
    return(weights)

def clip_branch_lengths(tree, min_val=1e-9, max_val=1e9): 
    for branch in tree.traverse("levelorder"):
        if branch.dist > max_val:
            branch.dist = max_val
        elif branch.dist < min_val:
            branch.dist = min_val
    
    return tree

def set_branch_sum(tree):
    total = 0
    for child in tree.get_children():
        tree_child = set_branch_sum(child)
        total += tree_child.BranchSum
        total += tree_child.dist
        
    tree.BranchSum = total

    return tree

def set_node_weight(tree):
    parent = tree.up
    if parent is None:
        tree.NodeWeight = 1.0
    else:
        tree.NodeWeight = parent.NodeWeight * (tree.dist + tree.BranchSum)/parent.BranchSum

    for child in tree.get_children():
        tree = set_node_weight(child)

    return tree

def update_progress(progress):
    barLength = 100 # Modify this to change the length of the progress bar
    status = ""
    if isinstance(progress, int):
        progress = float(progress)
    if not isinstance(progress, float):
        progress = 0
        status = "error: progress var must be float\r\n"
    if progress < 0:
        progress = 0
        status = "Halt...\r\n"
    if progress >= 1:
        progress = 1
        status = "Done...\r\n"
    block = int(round(barLength*progress))
    text = "\rPercent: [{0}] {1}% {2}".format( "#"*block + "-"*(barLength-block), round(progress*100, 2), status)
    sys.stdout.write(text)
    sys.stdout.flush()  

if __name__ == '__main__':
    # Get Weights:
    if not os.path.exists(results_folder+'/'+folder_res_main):
        os.makedirs(results_folder+'/'+folder_res_main)

    # Load Metadata
    metadata_df = pd.read_csv(folder+"/"+name_dataset+"_metadata.csv", header=[0], index_col=[0], encoding='windows-1252')
    print(metadata_df.shape)

    # Load AMR Data
    amr_df = pd.read_csv(folder+"/"+name_dataset+"_AMR.csv", header=[0], index_col=[0])
    amr_df = amr_df.loc[metadata_df.index,:]
    print(metadata_df.shape)


    # Get target values
    array_source = np.array(["All"])

    for source_val in array_source:
        print("Source: {}".format(source_val))
        if source_val == "All":
            idx = np.arange(len(metadata_df))
        else:
            idx = np.where(metadata_df["Dataset"] == source_val)[0]

        samples_sel = metadata_df.index[idx]

        antibiotic_df = amr_df.loc[samples_sel,:]

        print(antibiotic_df.columns)
        for name_antibiotic in antibiotic_df.columns:
            print("Antibiotic: {}".format(name_antibiotic))

            target_str = np.array(antibiotic_df[name_antibiotic])
            
            target = np.zeros(len(target_str)).astype(int)
            idx_S = np.where(target_str == 'S')[0]
            idx_R = np.where(target_str == 'R')[0]
            idx_NaN = np.where((target_str != 'R') & (target_str != 'S'))[0]
            target[idx_R] = 1    

            if len(idx_NaN) > 0:
                target = np.delete(target,idx_NaN)
                ids = np.delete(samples_sel,idx_NaN)
            else:
                ids = samples_sel
                
            # Check minimum number of samples:
            count_class = Counter(target)
            print(count_class)
            if count_class[0] < 0.1*len(target) or count_class[1] < 0.1*len(target):
                continue
   
            filename = folder+"/"+name_dataset+'_distancematrix.xlsx'

            weights = get_weights(filename, ids, '_'+source_val+'_'+name_antibiotic)
            weights.to_csv(results_folder+'/'+folder_res_main+'/GSC_weights_'+name_dataset+'_'+source_val+"_"+name_antibiotic+'.csv', index_label=['ID'])
        
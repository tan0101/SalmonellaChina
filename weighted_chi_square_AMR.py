# -*- coding: utf-8 -*-

from Bio.Phylo.TreeConstruction import DistanceTreeConstructor, _DistanceMatrix
from collections import Counter
from ete3 import Tree
from scipy import stats
from sklearn.feature_selection import VarianceThreshold
import Bio
import numpy as np
import pandas as pd
import os
import sys
import pickle
from sklearn.preprocessing import MinMaxScaler

name_dataset = "MDR" #"NonMDR"
folder =  "Data"
results_folder = "Results"
folder_res_main = name_dataset + " AMR PA"
data_folder =  "ARGs" #"PlasmidARGs" #"MGEs"

def get_kmers_filtered(name, pvalues, test_output, kmer_limit=10000, type_cutoff="B", pvalue_cutoff=0.05):
    kmers_for_ML = set()
    # Filters the k-mers by their p-value achieved in statistical 
    # testing.
    nr_of_kmers_tested = float(len(pvalues))
    pvalue_cutoff = get_pvalue_cutoff(type_cutoff,pvalues, nr_of_kmers_tested, pvalue_cutoff)
    print(pvalue_cutoff)
    directory = results_folder+"/"+folder_res_main+"/"+data_folder+"/Chi Square Features"
    inputfile = open(directory+"/"+test_output,'r')
    outputfile = open(directory+"/"+"k-mers_filtered_by_pvalue_" + name + ".txt", "w")
    write_headerline(outputfile)

    counter = 0
    index_array = []
    for n, line in enumerate(inputfile):
        counter += 1
        line_to_list = line.split()
        if float(line_to_list[2]) < pvalue_cutoff:
            outputfile.write(line)
            kmers_for_ML.add(line_to_list[0])
            index_array.append(n)
            
    sys.stderr.write("\n")
    sys.stderr.flush()
    if len(kmers_for_ML) == 0:
        outputfile.write("\nNo k-mers passed the filtration by p-value.\n")
    inputfile.close()
    outputfile.close()
    np.savetxt(directory+"/"+"features_"+name_dataset+'_'+name+'.txt', index_array, fmt='%d')

    return index_array

def get_pvalue_cutoff(type_cutoff, pvalues, nr_of_kmers_tested, pvalue_cutoff):
    pvalues = sorted(pvalues)
    if type_cutoff == "B":
        pvalue_cutoff = (pvalue_cutoff/nr_of_kmers_tested)
    elif type_cutoff == "FDR":
        pvalue_cutoff_by_FDR = 0
        for index, pvalue in enumerate(pvalues):
            if  (pvalue  < (
                    (index+1) 
                    / nr_of_kmers_tested) * pvalue_cutoff
                    ):
                pvalue_cutoff_by_FDR = pvalue
            elif pvalue > pvalue_cutoff:
                break
        pvalue_cutoff = pvalue_cutoff_by_FDR

    return pvalue_cutoff

def conduct_chi_squared_test(
    kmer, kmer_presence, test_results_file,
    samples_df
    ):
    min_samples = int(np.round(0.05*len(samples_df)))
    max_samples = int(np.round(0.95*len(samples_df)))
    
    (w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer,
    no_samples_wo_kmer, samples_w_kmer) = get_samples_distribution_for_chisquared(kmer_presence, samples_df)

    no_samples_w_kmer = len(samples_w_kmer)

    if no_samples_w_kmer < min_samples or no_samples_wo_kmer < 2 or no_samples_w_kmer > max_samples:
        test_results_file.write(
            kmer + "\t" + str(np.nan) + "\t" + str(np.nan) + "\t"
            + str(no_samples_w_kmer) + "\n"
            ) #+"\t| " + " ".join(samples_w_kmer) 
        return None

    (w_pheno, wo_pheno, w_kmer, wo_kmer, total) = get_totals_in_classes(
        w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer)

    (w_pheno_w_kmer_expected, w_pheno_wo_kmer_expected, wo_pheno_w_kmer_expected,
     wo_pheno_wo_kmer_expected) = get_expected_distribution(w_pheno, wo_pheno, w_kmer, wo_kmer, total)

    chisquare_results = stats.chisquare(
        [w_pheno_w_kmer, w_pheno_wo_kmer,
        wo_pheno_w_kmer, wo_pheno_wo_kmer],
        [w_pheno_w_kmer_expected, w_pheno_wo_kmer_expected, 
        wo_pheno_w_kmer_expected, wo_pheno_wo_kmer_expected],
        1)

    test_results_file.write(
        kmer + "\t%.2f\t%.2E\t" % chisquare_results 
        + str(no_samples_w_kmer) + "\n"
        ) #+"\t| " + " ".join(samples_w_kmer) 

    pvalue = chisquare_results[1]

    return pvalue

def get_samples_distribution_for_chisquared(kmers_presence_vector, samples_df):
    samples_w_kmer = []
    no_samples_wo_kmer = 0
    with_pheno_with_kmer = 0
    with_pheno_without_kmer = 0
    without_pheno_with_kmer = 0
    without_pheno_without_kmer = 0
    labels = np.array(samples_df['Label'])
    ids = np.array(samples_df['ID'])
    weights = np.array(samples_df['weights'])
    for count, label in enumerate(labels):
        if label == 1:
            if (kmers_presence_vector[count] != 0):
                with_pheno_with_kmer += weights[count] 
                samples_w_kmer.append(ids[count])
            else:
                with_pheno_without_kmer += weights[count]
                no_samples_wo_kmer += 1
        elif label == 0:
            if (kmers_presence_vector[count] != 0):
                without_pheno_with_kmer += weights[count]
                samples_w_kmer.append(ids[count])
            else:
                without_pheno_without_kmer += weights[count]
                no_samples_wo_kmer += 1
    return(
        with_pheno_with_kmer, with_pheno_without_kmer,
        without_pheno_with_kmer, without_pheno_without_kmer,
        no_samples_wo_kmer, samples_w_kmer
        )

def get_totals_in_classes(w_pheno_w_kmer, w_pheno_wo_kmer, wo_pheno_w_kmer, wo_pheno_wo_kmer):
    w_pheno = (w_pheno_w_kmer + w_pheno_wo_kmer)
    wo_pheno = (wo_pheno_w_kmer + wo_pheno_wo_kmer)
    w_kmer = (w_pheno_w_kmer + wo_pheno_w_kmer)
    wo_kmer = (w_pheno_wo_kmer + wo_pheno_wo_kmer)
    total = w_pheno + wo_pheno

    return w_pheno, wo_pheno, w_kmer, wo_kmer, total

def get_expected_distribution(w_pheno, wo_pheno, w_kmer, wo_kmer, total):
    w_pheno_w_kmer_expected = ((w_pheno * w_kmer) / float(total))
    w_pheno_wo_kmer_expected = ((w_pheno * wo_kmer) / float(total))
    wo_pheno_w_kmer_expected  = ((wo_pheno * w_kmer) / float(total))
    wo_pheno_wo_kmer_expected = ((wo_pheno * wo_kmer) / float(total))

    return(w_pheno_w_kmer_expected, w_pheno_wo_kmer_expected,
        wo_pheno_w_kmer_expected, wo_pheno_wo_kmer_expected)

def write_headerline(outputfile):
    outputfile.write(
        "K-mer\tChi-square_statistic\tp-value\
        \tNo._of_samples_with_k-mer\n"
        )#\tSamples_with_k-mer

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

    # Load Data
    data_df = pd.read_csv(folder+"/"+name_dataset+'_'+data_folder+'.csv', header = [0], index_col=[0], low_memory=False) 
    data_df = data_df.loc[metadata_df.index,:]
    data_array_df = np.array(data_df)
    features_name = np.array(data_df.columns)
    print(data_df.shape)

    samples_sel = metadata_df.index
    data_txt = data_array_df
    antibiotic_df = amr_df.copy()

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
            data = np.delete(data_txt,idx_NaN,axis=0)
            ids = np.delete(samples_sel,idx_NaN)
        else:
            data = data_txt
            ids = samples_sel
            
        # Check minimum number of samples:
        count_class = Counter(target)
        print(count_class)
        if count_class[0] < 0.1*len(target) or count_class[1] < 0.1*len(target):
            continue

        weights = pd.read_csv(results_folder+'/'+folder_res_main+'/GSC_weights_'+name_dataset+'_'+name_antibiotic+'.csv', header=[0])

        samples_df = weights.copy()
        samples_df['Label'] = target

        # Check minimum number of samples:
        count_class = Counter(target)
        print(count_class)

        scaler = MinMaxScaler()
        data_orig = scaler.fit_transform(data)

        # Remove low variance:
        print("Before removing low variance:{}".format(data.shape))
        selector = VarianceThreshold(threshold=.999 * (1 - .999))
        selector.fit_transform(data)
        cols=selector.get_support(indices=True)
        data = data[:,cols]
        features_anti = features_name[cols]
        n_features = len(features_anti)
        print("After removing low variance:{}".format(data.shape))  

        directory = results_folder+'/'+folder_res_main+'/'+data_folder+'/Chi Square Features'
        if not os.path.exists(directory):
            os.makedirs(directory)

        # Chi-square analysis
        pvalues = []
        test_results_file_name = "chi-squared_test_results_"+name_antibiotic+".txt"
        test_results_file = open(directory+"/"+test_results_file_name,"w")
        
        update_progress(0)
        for n in range(len(cols)):
            kmer_name = features_anti[n]
            kmer_presence_vector = data[:,n]
        
            pvalue = conduct_chi_squared_test(
                kmer_name, kmer_presence_vector,
                test_results_file, samples_df)
            if pvalue:
                pvalues.append(pvalue)
            else:
                pvalues.append(np.nan)               
                
            update_progress((n+1)/n_features)

        test_results_file.close()
        
        index_array = get_kmers_filtered(name_antibiotic, pvalues, test_results_file_name)

        index_array = np.array(index_array)
        pvalues = np.array(pvalues)

        if len(index_array) == 0:
            continue

        concat_array = np.zeros((len(index_array),2))
        concat_array[:,0] = cols[index_array]
        concat_array[:,1] = pvalues[index_array]
        
        data_AMR = data[:,index_array]
        print(data_AMR.shape)
        
        with open(directory+"/data_"+name_dataset+"_"+name_antibiotic+".pickle", 'wb') as f:
            pickle.dump(data_AMR, f)
        
        df_concat = pd.DataFrame(concat_array, columns=["Index","weighted p-values"], index=features_anti[index_array])
        df_concat.to_csv(directory+"/"+name_dataset+"_model_pvalue_"+name_antibiotic+".csv")
    
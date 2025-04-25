import numpy as np
import pandas as pd
import os
import sys

from pathlib import Path

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

if __name__ == "__main__":

    # Dictionary with the best classifiers according to the Nemenyi test: 0 - logistic regression; 1 - linear SVM; 2 - RBF-SVM; 
    # 3 - Random Forest; 4 - Extratree; 5 - Adaboost; 6 - XGBosst
    dict_clf = {
        "MDR": 3,
    }

    folder =  "Data"

    # Load AMR Data
    antibiotic_class_df = pd.read_csv("Data/antibiotic_class.csv", header=[0], index_col=[0])

    name_dataset = "MDR"

    name_file = "Data/"+name_dataset+"_RSI_Class.csv"

    # Load Metadata
    metadata_df = pd.read_csv(folder+"/"+name_dataset+"_metadata.csv", header=[0], index_col=[0])

    # Load AMR Data
    amr_df = pd.read_csv(folder+"/"+name_dataset+"_AMR.csv", header=[0], index_col=[0])
    amr_df = amr_df.loc[metadata_df.index,:]

    # Load AMR Class Data
    class_RSI_df = pd.read_csv(name_file, header=[0], index_col=[0])

    # Load Data ARGs:
    data_args_df = pd.read_csv(folder+"/"+name_dataset+'_ARGs.csv', header = [0], index_col=[0])
    data_args_df = data_args_df.loc[metadata_df.index,:]
    data_args_df[data_args_df>0]=1
    
    # Load Data MGEs:
    data_mge_df = pd.read_csv(folder+"/"+name_dataset+'_MGEs.csv', header = [0], index_col=[0])
    data_mge_df = data_mge_df.loc[metadata_df.index,:]
    data_mge_df[data_mge_df>0]=1

    # Load Data Plasmid with ARGs:
    data_plasmid_df = pd.read_csv(folder+"/"+name_dataset+'_PlasmidARGs.csv', header = [0], index_col=[0])
    data_plasmid_df = data_plasmid_df.loc[metadata_df.index,:]
    data_plasmid_df[data_plasmid_df>0]=1  
    
    # Concatenate Data:
    data_comb_df = pd.concat([data_args_df, data_mge_df, data_plasmid_df], axis=1)

    res_df = pd.DataFrame()

    k = 0
    
    year_unique = np.unique(metadata_df["year"])
    update_progress(0)
    for count, year in enumerate(year_unique):
        res_df.loc[k, "year"] = year

        idx_year = np.where(metadata_df["year"] == year)[0]
        n_isolates = len(idx_year)

        res_df.loc[k, "Num Isolates"] = n_isolates
        k+=1

        update_progress((count+1)/len(year_unique))

    if not os.path.exists("Results/Feature Rate"):
        os.makedirs("Results/Feature Rate")

    writer = pd.ExcelWriter("Results/Feature Rate/"+name_dataset+"_test.xlsx", engine='xlsxwriter')

    update_progress(0)
    for count, class_name in enumerate(class_RSI_df.columns):
        features_class = []
        for name_antibiotic in amr_df.columns:
            if antibiotic_class_df.loc[name_antibiotic,"Antibiotic Class"] != class_name:
                continue

            # Get features
            performance_file = "Results/"+name_dataset+" AMR PA/Population Correction/SMOTE_results_"+name_dataset+"_"+name_antibiotic+".csv"
                        
            my_file = Path(performance_file)

            try:
                my_abs_path = my_file.resolve(strict=True)
            except FileNotFoundError:
                continue

            df_performance = pd.read_csv(performance_file, header=[0], index_col=[0])
            performance_array = np.array(df_performance.loc["AUC_Mean",:])

            if performance_array[dict_clf[name_dataset]] < 0.9:
                continue

            # Get features
            features_file = "Results/"+name_dataset+" AMR PA/Population Correction/features_"+name_dataset+"_"+name_antibiotic+".csv"                
            df_features = pd.read_csv(features_file, header=[0], index_col=[0])

            for feat in df_features[df_features.columns[0]]:
                features_class.append(feat)

        features_unique = np.unique(features_class)

        if len(features_unique) == 0:
            continue
                    
        results_df = res_df.copy()

        for count_feat, feat in enumerate(features_unique):
            if count_feat == 0:
                for k in range(len(results_df)):
                    idx_year = np.where(metadata_df["year"] == results_df.loc[k,"year"])[0]

                    isolate_intersect = np.array(metadata_df.index[idx_year])

                    R_S_count = 0
                    R_rate_count = 0
                    for isolate in isolate_intersect:
                        if class_RSI_df.loc[isolate,class_name] == "R" or class_RSI_df.loc[isolate,class_name] == "S":
                            R_S_count+=1
                        
                        if class_RSI_df.loc[isolate,class_name] == "R":
                            R_rate_count+=1

                    if R_S_count == 0:
                        results_df.loc[k,"AMR Rate"] = ""
                    else:
                        results_df.loc[k,"AMR Rate"] = R_rate_count/R_S_count

            for k in range(len(results_df)):
                idx_year = np.where(metadata_df["year"] == results_df.loc[k,"year"])[0]

                isolate_intersect = np.array(metadata_df.index[idx_year])

                R_S_count = 0
                R_pres_count = 0
                R_rate_count = 0
                for isolate in isolate_intersect:
                    if class_RSI_df.loc[isolate,class_name] == "R" or class_RSI_df.loc[isolate,class_name] == "S":
                        R_S_count+=1
                    
                    if class_RSI_df.loc[isolate,class_name] == "R" and data_comb_df.loc[isolate,feat] == 1:
                        R_pres_count+=1

                    if class_RSI_df.loc[isolate,class_name] == "R":
                        R_rate_count+=1

                if R_S_count == 0:
                    results_df.loc[k,feat] = ""
                else:
                    results_df.loc[k,feat] = R_pres_count/R_S_count

        results_df.to_excel(writer, sheet_name = class_name, index = True)

        update_progress((count+1)/len(class_RSI_df.columns))

    writer.close() 




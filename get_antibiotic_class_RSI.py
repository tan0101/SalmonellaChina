import numpy as np
import pandas as pd
import os
import glob
import sys

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
    folder =  "Data"

    name_dataset = "MDR"

    metadata_df = pd.read_csv(folder+"/"+name_dataset+"_metadata.csv", header=[0], index_col=[0], encoding='windows-1252')

    # Load AMR Data
    antibiotic_class_df = pd.read_csv("Data/antibiotic_class.csv", header=[0], index_col=[0])

    name_file = "Data/"+name_dataset+"_AMR.csv"

    # Load AMR Data
    amr_df = pd.read_csv(name_file, header=[0], index_col=[0])
    amr_df = amr_df.loc[metadata_df.index,:]

    antibiotic_class_array = []
    for col in amr_df.columns:
        if col in antibiotic_class_df.index:
            antibiotic_class_array.append(antibiotic_class_df.loc[col,"Antibiotic Class"])

    res_df = pd.DataFrame()

    print(np.unique(antibiotic_class_array))
    
    update_progress(0)
    for count, class_name in enumerate(np.unique(antibiotic_class_array)):
        cols_array = []
        for col in amr_df.columns:
            if col in antibiotic_class_df.index:
                if antibiotic_class_df.loc[col,"Antibiotic Class"] == class_name:
                    cols_array.append(col)

        for isolate in amr_df.index:
            dummy_df = amr_df.loc[isolate,cols_array]

            if dummy_df.isin(["R"]).any().any() == True:
                res_df.loc[isolate,class_name] = "R"
            else:
                if dummy_df.isin(["S"]).any().any() == True:
                    res_df.loc[isolate,class_name] = "S"
                else:
                    res_df.loc[isolate,class_name] = ""
        
        update_progress((count+1)/len(np.unique(antibiotic_class_array)))


    res_df.to_csv("Data/"+name_dataset+"_RSI_Class.csv")

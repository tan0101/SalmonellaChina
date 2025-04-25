# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import sys
import os
import pickle

from sklearn.feature_selection import chi2
from collections import Counter
from sklearn.model_selection import StratifiedKFold, GridSearchCV,cross_validate
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, ExtraTreesClassifier,GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, accuracy_score, make_scorer, cohen_kappa_score
from sklearn.preprocessing import MinMaxScaler
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import SMOTE
from pathlib import Path

def tn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 0]
def fp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[0, 1]
def fn(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 0]
def tp(y_true, y_pred): return confusion_matrix(y_true, y_pred)[1, 1]
scoring = {'tp': make_scorer(tp), 'tn': make_scorer(tn),
           'fp': make_scorer(fp), 'fn': make_scorer(fn),
           'auc': 'roc_auc',
           'acc': make_scorer(accuracy_score),
           'kappa': make_scorer(cohen_kappa_score)}

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
    name_dataset = "MDR" #"NonMDR" 
    folder = "Data"
    results_folder = "Results"
    type_data = "combination"
    folder_res_main = name_dataset + " AMR PA"
     
    # Nested Cross Validation:
    inner_loop_cv = 3   
    outer_loop_cv = 5
    
    # Number of random trials:
    NUM_TRIALS = 30
    
    # Grid of Parameters:
    C_grid = {"clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    est_grid = {"clf__n_estimators": [2, 4, 8, 16, 32, 64]}
    MLP_grid = {"clf__alpha": [0.001, 0.01, 0.1, 1, 10, 100], "clf__learning_rate_init": [0.001, 0.01, 0.1, 1],
        "clf__hidden_layer_sizes": [10, 20, 40, 100, 200, 300, 400, 500]}
    SVC_grid = {"clf__gamma": [0.0001, 0.001, 0.01, 0.1], "clf__C": [0.001, 0.01, 0.1, 1, 10, 100, 1000]}
    DT_grid = {"clf__max_depth": [10, 20, 30, 50, 100]}
    XGBoost_grid = {"clf__n_estimators": [2, 4, 8, 16, 32, 64], "clf__learning_rate": [0.001, 0.01, 0.1, 1]}
        
    # Classifiers:
    names = ["Logistic Regression", "Linear SVM", "RBF SVM",
        "Extra Trees", "Random Forest", "AdaBoost", "XGBoost"]

    classifiers = [
        LogisticRegression(),
        LinearSVC(loss='hinge'),
        SVC(),
        ExtraTreesClassifier(),
        RandomForestClassifier(),
        AdaBoostClassifier(),
        GradientBoostingClassifier()
        ]
    
    # Load Metadata
    metadata_df = pd.read_csv(folder+"/"+name_dataset+"_metadata.csv", header=[0], index_col=[0], encoding='windows-1252')

    # Load Data ARGs:
    data_args_df = pd.read_csv(folder+"/"+name_dataset+'_ARGs.csv', header = [0], index_col=[0])
    data_args_df = data_args_df.loc[metadata_df.index,:]
    features_name_args = np.array(data_args_df.columns)
    sample_name_args = np.array(data_args_df.index)
    data_args = np.array(data_args_df)

    # Load Data MGEs:
    data_mge_df = pd.read_csv(folder+"/"+name_dataset+'_MGEs.csv', header = [0], index_col=[0])
    data_mge_df = data_mge_df.loc[metadata_df.index,:]
    features_name_mge = np.array(data_mge_df.columns)
    sample_name_mge = np.array(data_mge_df.index)
    data_mge = np.array(data_mge_df)

    # Load Data Plasmid with ARGs:
    data_plasmid_df = pd.read_csv(folder+"/"+name_dataset+'_PlasmidARGs.csv', header = [0], index_col=[0])
    data_plasmid_df = data_plasmid_df.loc[metadata_df.index,:]
    features_name_plasmid = np.array(data_plasmid_df.columns)
    sample_name_plasmid = np.array(data_plasmid_df.index)
    data_plasmid = np.array(data_plasmid_df)

    # Load AMR Data
    amr_df = pd.read_csv(folder+"/"+name_dataset+"_AMR.csv", header=[0], index_col=[0])
    amr_df = amr_df.loc[metadata_df.index,:]
    print(metadata_df.shape)
    
    
    # Concatenate Data:
    data_comb_df = pd.concat([data_args_df, data_mge_df, data_plasmid_df], axis=1)
    data_comb = np.array(data_comb_df)

    print(data_comb_df.shape)

    # Get target values
    samples_sel = metadata_df.index
    data_txt = data_comb
    antibiotic_df = amr_df.copy()

    for name_antibiotic in antibiotic_df.columns[19:26]:
        print("Antibiotic: {}".format(name_antibiotic))

        target_str = np.array(antibiotic_df[name_antibiotic])
        
        target = np.zeros(len(target_str)).astype(int)
        idx_S = np.where(target_str == 'S')[0]
        idx_R = np.where(target_str == 'R')[0]
        idx_NaN = np.where((target_str != 'R') & (target_str != 'S'))[0]
        target[idx_R] = 1    

        if len(idx_NaN) > 0:
            target = np.delete(target,idx_NaN)
            data_orig = np.delete(data_txt,idx_NaN,axis=0)
            ids = np.delete(samples_sel,idx_NaN)
        else:
            data_orig = data_txt
            ids = samples_sel
            
        # Check minimum number of samples:
        count_class = Counter(target)
        print(count_class)
        if count_class[0] < 0.1*len(target) or count_class[1] < 0.1*len(target):
            continue

        check_args = False
        check_mge = False
        check_plasmid = False
        
        # Combine data
        file_name_args = results_folder+"/"+folder_res_main+"/ARGs/Chi Square Features/"+name_dataset+"_model_pvalue_"+name_antibiotic+".csv"
        my_file = Path(file_name_args)
        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            check_args = False
        else:
            df_features_args = pd.read_csv(file_name_args, header=[0], index_col=[0])
            check_args = True

        file_name_mge = results_folder+"/"+folder_res_main+"/MGEs/Chi Square Features/"+name_dataset+"_model_pvalue_"+name_antibiotic+".csv"
        my_file = Path(file_name_mge)
        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            check_mge = False
        else:
            df_features_mge = pd.read_csv(file_name_mge, header=[0], index_col=[0])
            check_mge = True

        file_name_plasmid = results_folder+"/"+folder_res_main+"/PlasmidARGs/Chi Square Features/"+name_dataset+"_model_pvalue_"+name_antibiotic+".csv"
        my_file = Path(file_name_plasmid)
        try:
            my_abs_path = my_file.resolve(strict=True)
        except FileNotFoundError:
            check_plasmid = False
        else:
            df_features_plasmid = pd.read_csv(file_name_plasmid, header=[0], index_col=[0])
            check_plasmid = True


        features_df = []

        if check_args == True:
            features_df = df_features_args               

        if check_mge == True:
            features_df = pd.concat([features_df, df_features_mge], axis=0)

        if check_plasmid == True:
            features_df = pd.concat([features_df, df_features_plasmid], axis=0)

        if len(features_df) == 0:
            continue

        features_population_structure = features_df.index

        idx_cols = []
        features_anti = []
        for count, feat in enumerate(data_comb_df.columns):
            if feat in features_population_structure:
                idx_cols.append(count)
                features_anti.append(feat)

        features_anti = np.array(features_anti)

        data_orig = data_orig[:,idx_cols]
        scaler = MinMaxScaler()

        data_orig = scaler.fit_transform(data_orig)

        target_orig = target
        
        n_features = len(features_anti)

        features_sel = np.zeros(len(features_anti))
        features_sel_array = np.zeros((len(features_anti),1000))

        update_progress(0)
        for rs in range(0,1000):
            sm = SMOTE(random_state=rs) 
            data, target = sm.fit_resample(data_orig, target_orig)
            
            _, pvalue_chi2 = chi2(data, target)

            id_chi2 = np.where(pvalue_chi2 < 0.05)[0] 
        
            if len(id_chi2) == 0:
                continue

            features_sel[id_chi2]+=1
            features_sel_array[id_chi2,rs]=1

            update_progress((rs+1)/1000)

        id_sel = np.where(features_sel>=750)[0]

        rs_array = []
        update_progress(0)
        for rs in range(0,1000):
            feat_sel = np.where(features_sel_array[:,rs] == 1)[0]
            intersect = np.intersect1d(id_sel,feat_sel)

            if len(intersect) == len(id_sel):
                rs_array.append(rs)

            update_progress((rs+1)/1000)

        print(len(rs_array))

        sm = SMOTE(random_state=rs_array[0])
        data, target = sm.fit_resample(data_orig, target_orig)
        print('Resampled dataset shape %s' % Counter(target))

        # Preprocessing - Feature Selection
        idx = np.where(features_sel_array[:,rs_array[0]] == 1)[0]
        _, pvalue_sel = chi2(data, target)
        pvalue_sel = pvalue_sel[idx]

        print("Before select from chi2:{}".format(data.shape))

        data = data[:,idx]
        features_anti = features_anti[idx]
        n_features = len(features_anti)

        print("After select from chi2:{}".format(data.shape))

        if n_features == 0:
            continue

        directory = results_folder+"/"+folder_res_main+"/Population Correction"
        if not os.path.exists(directory):
            os.makedirs(directory)

        features_genes_df = pd.DataFrame(features_sel[idx], columns = ["selection times - rs "+str(rs_array[0])])
        features_genes_df["index"] = features_anti
        
        names_f = ["index", "selection times - rs "+str(rs_array[0])]
        features_genes_df = features_genes_df[names_f]
        features_genes_df["pvalue"] = pvalue_sel
        features_genes_df.to_csv(directory+"/features_"+name_dataset+"_"+name_antibiotic+".csv")
                        
        # Initialize Variables:
        scores_auc = np.zeros([NUM_TRIALS,len(classifiers)])
        scores_acc = np.zeros([NUM_TRIALS,len(classifiers)])
        scores_sens = np.zeros([NUM_TRIALS,len(classifiers)])
        scores_spec = np.zeros([NUM_TRIALS,len(classifiers)])
        scores_kappa = np.zeros([NUM_TRIALS,len(classifiers)])
        scores_prec = np.zeros([NUM_TRIALS,len(classifiers)])
        
        # Loop for each trial
        update_progress(0)
        for i in range(NUM_TRIALS):
            #print("Trial = {}".format(i))
        
            inner_cv = StratifiedKFold(n_splits=inner_loop_cv, shuffle=True, random_state=i)
            outer_cv = StratifiedKFold(n_splits=outer_loop_cv, shuffle=True, random_state=i)

            # Creating the Training and Test set from data
            k = 0
            for name, clf in zip(names, classifiers):
                model = Pipeline([('clf', clf)])

                if name == "QDA" or name == "LDA" or name == "Naive Bayes":
                    classif = model
                else:
                    if name == "RBF SVM":
                        grid = SVC_grid              
                    elif name == "Random Forest" or name == "AdaBoost" or name == "Extra Trees":
                        grid = est_grid
                    elif name == "Neural Net":
                        grid = MLP_grid
                    elif name == "Linear SVM":
                        grid = C_grid
                    elif name == "Decision Tree":
                        grid = DT_grid
                    elif name == "XGBoost":
                        grid = XGBoost_grid
                    else:
                        grid = C_grid

                    # Inner Search
                    classif = GridSearchCV(estimator=model, param_grid=grid, cv=inner_cv)
                    classif.fit(data, target)
                
                # Outer Search
                cv_results = cross_validate(classif, data, target, scoring=scoring, cv=outer_cv, return_estimator=True)

                tp = cv_results['test_tp']
                tn = cv_results['test_tn']
                fp = cv_results['test_fp']
                fn = cv_results['test_fn']
                
                sens = np.zeros(outer_loop_cv)
                spec = np.zeros(outer_loop_cv)
                prec = np.zeros(outer_loop_cv)
                
                for j in range(outer_loop_cv):
                    TP = tp[j]
                    TN = tn[j]
                    FP = fp[j]
                    FN = fn[j]
                    
                    # Sensitivity, hit rate, recall, or true positive rate
                    sens[j] = TP/(TP+FN)
                    
                    # Fall out or false positive rate
                    FPR = FP/(FP+TN)
                    spec[j] = 1 - FPR
                    if TP + FP > 0:
                        prec[j] = TP / (TP + FP)
    
                scores_sens[i,k] = sens.mean()
                scores_spec[i,k] = spec.mean()
                scores_prec[i,k] = prec.mean()
                scores_auc[i,k] = cv_results['test_auc'].mean()
                scores_acc[i,k] = cv_results['test_acc'].mean()
                scores_kappa[i,k] = cv_results['test_kappa'].mean()
                
                k = k + 1
                
            update_progress((i+1)/NUM_TRIALS)

        results = np.zeros((12,len(classifiers)))
        scores = [scores_auc, scores_acc, scores_sens, scores_spec, scores_kappa, scores_prec]
        for counter_scr, scr in enumerate(scores):
            results[2*counter_scr,:] = np.mean(scr,axis=0)
            results[2*counter_scr + 1,:] = np.std(scr,axis=0)
            
        names_scr = ["AUC_Mean", "AUC_Std", "Acc_Mean", "Acc_Std", 
            "Sens_Mean", "Sens_Std", "Spec_Mean", "Spec_Std", 
            "Kappa_Mean", "Kappa_Std", "Prec_Mean", "Prec_Std"]

        results_df=pd.DataFrame(results, columns=names, index=names_scr)
        
        results_df=pd.DataFrame(results, columns=names, index=names_scr)
        results_df.to_csv(directory+"/SMOTE_results_"+name_dataset+"_"+name_antibiotic+".csv")

        df_auc = pd.DataFrame(scores_auc, columns=names)
        df_auc.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_auc.csv")
        
        df_acc = pd.DataFrame(scores_acc, columns=names)
        df_acc.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_acc.csv")
        
        df_sens = pd.DataFrame(scores_sens, columns=names)
        df_sens.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_sens.csv")
        
        df_spec = pd.DataFrame(scores_spec, columns=names)
        df_spec.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_spec.csv")
        
        df_kappa = pd.DataFrame(scores_kappa, columns=names)
        df_kappa.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_kappa.csv")
        
        df_prec = pd.DataFrame(scores_prec, columns=names)
        df_prec.to_csv(directory+"/SMOTE_"+name_dataset+"_"+name_antibiotic+"_prec.csv")
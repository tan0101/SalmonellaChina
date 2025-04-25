import numpy as np
import pandas as pd
import os
import sys
import statsmodels.formula.api as smf

from scipy.stats import pearsonr

import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', pd.errors.PerformanceWarning)

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

    species_array = ["MDR"]

    # Load indicators
    indicators_df = pd.read_csv("Results/Indicators_China.csv", index_col=[0], header=[0])


    if not os.path.exists("Results/Forecasting Global/LinearRegression"):
        os.makedirs("Results/Forecasting Global/LinearRegression")

    for specie in species_array:
        print(specie)
        feature_rate_df = pd.read_excel("Results/Feature Rate/"+specie+".xlsx", sheet_name=None)

        writer = pd.ExcelWriter("Results/Forecasting Global/LinearRegression/LinearRegression_"+specie+".xlsx", engine='xlsxwriter')

        update_progress(0)
        for count, class_name in enumerate(feature_rate_df.keys()):
            anti_df = pd.read_excel("Results/Feature Rate/"+specie+".xlsx", sheet_name=class_name, header=[0], index_col=[0])
            
            # Drop rows with NaN values
            anti_df = anti_df.dropna(axis=0, how="any").reset_index(drop=True)

            # Keep columns
            idx_keep = np.where(anti_df["Num Isolates"] > 5)[0]
            anti_df = anti_df.loc[idx_keep,:].reset_index(drop=True)

            res_df = pd.DataFrame()
            k=0
            for indicator in indicators_df.columns[2:]:
                indicator_array = np.zeros(len(anti_df))
                for i in range(len(anti_df)):
                    year = anti_df.loc[i, "year"]

                    idx_year = np.where(indicators_df["Year"] == year)[0]

                    indicator_array[i] = indicators_df.loc[idx_year,indicator]

                check_nans = np.isnan(indicator_array)
                idx_nans = np.where(check_nans == True)[0]

                idx_not_nans = np.where(check_nans == False)[0]

                if len(idx_not_nans) < 5:
                    continue

                if len(idx_nans) > 0:
                    indicator_array = np.delete(indicator_array,idx_nans)

                if len(np.unique(indicator_array)) == 1:
                    continue
 
                year_array = anti_df["year"]
                year_array = np.delete(year_array,idx_nans)

                years = np.unique(year_array)
                n_years = len(years)
                first_year = np.min(years)
                last_year = np.max(years)
                years_str = [str(year_value) for year_value in years]

                if n_years < 5:
                    continue

                for feat in anti_df.columns[3:]:
                    rate_array = np.array(anti_df[feat]) 

                    if len(idx_nans) > 0:
                        rate_array = np.delete(rate_array,idx_nans)

                    if len(np.unique(rate_array)) == 1:
                        continue

                    stats_pearson, pvalue_pearson = pearsonr(rate_array, indicator_array)

                    if stats_pearson <= -0.5 or stats_pearson >= 0.5:
                        res_df.loc[k,"Indicator"] = indicator
                        res_df.loc[k,"Feature"] = feat
                        res_df.loc[k,"Pearson r"] = stats_pearson
                        res_df.loc[k,"Pearson p-value"] = pvalue_pearson

                        model_df = pd.DataFrame()
                        model_df["Rate"] = rate_array
                        model_df["indicator"] = indicator_array

                        model = smf.ols(formula='Rate ~ indicator', data=model_df).fit()
                        res_df.loc[k,"intercept"] = model.params[0]
                        res_df.loc[k,"intercept_STDerr"] = model.bse[0]
                        res_df.loc[k,"intercept_pvalue"] = model.pvalues[0]
                        res_df.loc[k,"slope"] = model.params[1]
                        res_df.loc[k,"slope_STDerr"] = model.bse[1]
                        res_df.loc[k,"slope_pvalue"] = model.pvalues[1]
                        res_df.loc[k,"Rsquared"] = model.rsquared
                        res_df.loc[k,"n"] = len(rate_array)
                        res_df.loc[k,"Years"] = ', '.join(years_str)
                        res_df.loc[k,"Num Years"] = n_years
                        res_df.loc[k,"First Year"] = first_year
                        res_df.loc[k,"Last Year"] = last_year
                        k+=1
        
            if len(res_df) > 0:
                res_df.to_excel(writer, sheet_name = class_name, index = True)

            update_progress((count+1)/len(feature_rate_df.keys()))
        
        writer.close()


                        


            

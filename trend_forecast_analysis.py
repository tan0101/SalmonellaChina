from tabnanny import check
import numpy as np
import pandas as pd
import os
import sys

from statsmodels.tsa.stattools import adfuller, kpss
from scipy.stats import linregress

import warnings
warnings.filterwarnings("ignore")

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

def find_csv_filenames( path_to_dir, suffix=".csv" ):
    filenames = os.listdir(path_to_dir)
    return [ filename for filename in filenames if filename.endswith( suffix ) ]

if __name__ == "__main__":
    directory = "Results/Forecasting Global/MonteCarloFeaturesForecasts"

    IndicatorBounds=pd.read_csv('Results/IndicatorForecastBounds.csv',index_col=[0], header=[0])

    files_array = find_csv_filenames(directory)
    
    trend_df = pd.DataFrame()
    update_progress(0)
    for count, filename in enumerate(files_array):
        split_filename = filename.split("_")

        specie = "MDR"
        class_name = split_filename[1]
        row = int(split_filename[-1].split(".csv")[0])

        RegressionParameters = pd.read_excel("Results/Forecasting Global/LinearRegression/LinearRegression_"+specie+".xlsx", sheet_name=class_name, header=[0], index_col=[0])

        indicator = RegressionParameters.loc[row,"Indicator"]

        if RegressionParameters.loc[row,"Pearson r"] > -0.5 and RegressionParameters.loc[row,"Pearson r"] < 0.5:
            continue
        
        trend_df.loc[count,"Filename"] = filename
        trend_df.loc[count,"Specie"] = specie
        trend_df.loc[count,"Antibiotic Class"] = class_name
        trend_df.loc[count,"Indicator"] = RegressionParameters.loc[row,"Indicator"]
        trend_df.loc[count,"Feature"] = RegressionParameters.loc[row,"Feature"]
        trend_df.loc[count,"Pearson r"] = RegressionParameters.loc[row,"Pearson r"]
        trend_df.loc[count,"Pearson p-value"] = RegressionParameters.loc[row,"Pearson p-value"]
        trend_df.loc[count,"n"] = RegressionParameters.loc[row,"n"]
        trend_df.loc[count,"Years"] = RegressionParameters.loc[row,"Years"]
        trend_df.loc[count,"Num Years"] = RegressionParameters.loc[row,"Num Years"]
        trend_df.loc[count,"First Year"] = RegressionParameters.loc[row,"First Year"]
        trend_df.loc[count,"Last Year"] = RegressionParameters.loc[row,"Last Year"]


        timeseries_df = pd.read_csv(directory+"/"+filename,header=[0], index_col=[0])

        df = pd.DataFrame()
        df["time"] = timeseries_df.columns
        df["data"] = np.array(timeseries_df.median(axis=0))
        check_unique = len(np.unique(df["data"]))
        if check_unique > 1:
            #print("ADF")
            dftest = adfuller(df.data, autolag='AIC')

            trend_df.loc[count,"Feature ADF Test statistic"] = "{:.3f}".format(dftest[0])
            trend_df.loc[count,"Feature ADF p-value"] = "{:.3f}".format(dftest[1])

            for k, v in dftest[4].items():
                if k not in ["1%", "5%", "10%"]:
                    continue

                trend_df.loc[count,"Feature ADF Confidence Level "+str(np.round(100-int(k[:-1]),0))+"%"] = v
                if v<dftest[0]:
                    trend_df.loc[count,"Feature ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = "No"
                else:
                    trend_df.loc[count,"Feature ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = "Yes"

            #print("KPSS")
            dftest_kpss = kpss(df.data, regression="c", nlags="auto")

            trend_df.loc[count,"Feature KPSS Test statistic"] = "{:.3f}".format(dftest_kpss[0])
            trend_df.loc[count,"Feature KPSS p-value"] = "{:.3f}".format(dftest_kpss[1])

            for k, v in dftest_kpss[3].items():
                if k not in ["1%", "5%", "10%"]:
                    continue

                trend_df.loc[count,"Feature KPSS Confidence Level "+str(np.round(100-int(k[:-1]),0))+"%"] = v
                if v<dftest_kpss[0]:
                    trend_df.loc[count,"Feature KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = "No"
                else:
                    trend_df.loc[count,"Feature KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = "Yes"


            for k in ["1%", "5%", "10%"]:
                if trend_df.loc[count,"Feature ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "No" and trend_df.loc[count,"Feature KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "No":
                    trend_df.loc[count,"Feature Result "+str(np.round(100-int(k[:-1]),0))+"%"] = "Not stationary"
                elif trend_df.loc[count,"Feature ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "No" and trend_df.loc[count,"Feature KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "Yes":
                    trend_df.loc[count,"Feature Result "+str(np.round(100-int(k[:-1]),0))+"%"] = "Trend stationary"
                elif trend_df.loc[count,"Feature ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "Yes" and trend_df.loc[count,"Feature KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "No":
                    trend_df.loc[count,"Feature Result "+str(np.round(100-int(k[:-1]),0))+"%"] = "Difference stationary"
                elif trend_df.loc[count,"Feature ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "Yes" and trend_df.loc[count,"Feature KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "Yes":
                    trend_df.loc[count,"Feature Result "+str(np.round(100-int(k[:-1]),0))+"%"] = "Stationary"

                
                if trend_df.loc[count,"Feature Result "+str(np.round(100-int(k[:-1]),0))+"%"] == "Not stationary" or trend_df.loc[count,"Feature Result "+str(np.round(100-int(k[:-1]),0))+"%"] == "Difference stationary" or trend_df.loc[count,"Feature Result "+str(np.round(100-int(k[:-1]),0))+"%"] == "Trend stationary":
                    res_linreg = linregress(np.arange(len(df["time"])), df["data"])
                    if res_linreg.slope < 0:
                        trend_df.loc[count,"Feature Trend "+str(np.round(100-int(k[:-1]),0))+"%"] = "Decreasing"
                    elif res_linreg.slope > 0:
                        trend_df.loc[count,"Feature Trend "+str(np.round(100-int(k[:-1]),0))+"%"] = "Increasing"
                else:
                    trend_df.loc[count,"Feature Trend "+str(np.round(100-int(k[:-1]),0))+"%"] = "Stationary"

        else:
            trend_df.loc[count,"Feature ADF Test statistic"] = ""
            trend_df.loc[count,"Feature ADF p-value"] = ""
            for k in ["1%", "5%", "10%"]:
                trend_df.loc[count,"Feature ADF Confidence Level "+str(np.round(100-int(k[:-1]),0))+"%"] = ""
                trend_df.loc[count,"Feature ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = ""

            trend_df.loc[count,"Feature KPSS Test statistic"] = ""
            trend_df.loc[count,"Feature KPSS p-value"] = ""
            for k in ["1%", "5%", "10%"]:
                trend_df.loc[count,"Feature KPSS Confidence Level "+str(np.round(100-int(k[:-1]),0))+"%"] = ""
                trend_df.loc[count,"Feature KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = ""
                

            for k in ["1%", "5%", "10%"]:
                trend_df.loc[count,"Feature Result "+str(np.round(100-int(k[:-1]),0))+"%"] = "Constant"
                trend_df.loc[count,"Feature Trend "+str(np.round(100-int(k[:-1]),0))+"%"] = ""

        
        timeseries_df = pd.read_csv("Results/Forecasting Global/MonteCarloIndicatorForecasts/Montecarlo_"+class_name+"_Indicator_"+str(IndicatorBounds.loc[indicator,"Code"])+".csv",header=[0], index_col=[0])
        timeseries_df = timeseries_df.transpose()

        df = pd.DataFrame()
        df["time"] = timeseries_df.columns
        df["data"] = np.array(timeseries_df.median(axis=0))
        check_unique = len(np.unique(df["data"]))
        if check_unique > 1:
            #print("ADF")
            dftest = adfuller(df.data, autolag='AIC')

            trend_df.loc[count,"Indicator ADF Test statistic"] = "{:.3f}".format(dftest[0])
            trend_df.loc[count,"Indicator ADF p-value"] = "{:.3f}".format(dftest[1])

            for k, v in dftest[4].items():
                if k not in ["1%", "5%", "10%"]:
                    continue

                trend_df.loc[count,"Indicator ADF Confidence Level "+str(np.round(100-int(k[:-1]),0))+"%"] = v
                if v<dftest[0]:
                    trend_df.loc[count,"Indicator ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = "No"
                else:
                    trend_df.loc[count,"Indicator ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = "Yes"

            #print("KPSS")
            dftest_kpss = kpss(df.data, regression="c", nlags="auto")

            trend_df.loc[count,"Indicator KPSS Test statistic"] = "{:.3f}".format(dftest_kpss[0])
            trend_df.loc[count,"Indicator KPSS p-value"] = "{:.3f}".format(dftest_kpss[1])

            for k, v in dftest_kpss[3].items():
                if k not in ["1%", "5%", "10%"]:
                    continue

                trend_df.loc[count,"Indicator KPSS Confidence Level "+str(np.round(100-int(k[:-1]),0))+"%"] = v
                if v<dftest_kpss[0]:
                    trend_df.loc[count,"Indicator KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = "No"
                else:
                    trend_df.loc[count,"Indicator KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = "Yes"


            for k in ["1%", "5%", "10%"]:
                if trend_df.loc[count,"Indicator ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "No" and trend_df.loc[count,"Indicator KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "No":
                    trend_df.loc[count,"Indicator Result "+str(np.round(100-int(k[:-1]),0))+"%"] = "Not stationary"
                elif trend_df.loc[count,"Indicator ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "No" and trend_df.loc[count,"Indicator KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "Yes":
                    trend_df.loc[count,"Indicator Result "+str(np.round(100-int(k[:-1]),0))+"%"] = "Trend stationary"
                elif trend_df.loc[count,"Indicator ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "Yes" and trend_df.loc[count,"Indicator KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "No":
                    trend_df.loc[count,"Indicator Result "+str(np.round(100-int(k[:-1]),0))+"%"] = "Difference stationary"
                elif trend_df.loc[count,"Indicator ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "Yes" and trend_df.loc[count,"Indicator KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] == "Yes":
                    trend_df.loc[count,"Indicator Result "+str(np.round(100-int(k[:-1]),0))+"%"] = "Stationary"

                
                if trend_df.loc[count,"Indicator Result "+str(np.round(100-int(k[:-1]),0))+"%"] == "Not stationary" or trend_df.loc[count,"Indicator Result "+str(np.round(100-int(k[:-1]),0))+"%"] == "Difference stationary" or trend_df.loc[count,"Indicator Result "+str(np.round(100-int(k[:-1]),0))+"%"] == "Trend stationary":
                    res_linreg = linregress(np.arange(len(df["time"])), df["data"])
                    if res_linreg.slope < 0:
                        trend_df.loc[count,"Indicator Trend "+str(np.round(100-int(k[:-1]),0))+"%"] = "Decreasing"
                    elif res_linreg.slope > 0:
                        trend_df.loc[count,"Indicator Trend "+str(np.round(100-int(k[:-1]),0))+"%"] = "Increasing"
                else:
                    trend_df.loc[count,"Indicator Trend "+str(np.round(100-int(k[:-1]),0))+"%"] = "Stationary"

        else:
            trend_df.loc[count,"Indicator ADF Test statistic"] = ""
            trend_df.loc[count,"Indicator ADF p-value"] = ""
            for k in ["1%", "5%", "10%"]:
                trend_df.loc[count,"Indicator ADF Confidence Level "+str(np.round(100-int(k[:-1]),0))+"%"] = ""
                trend_df.loc[count,"Indicator ADF Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = ""

            trend_df.loc[count,"Indicator KPSS Test statistic"] = ""
            trend_df.loc[count,"Indicator KPSS p-value"] = ""
            for k in ["1%", "5%", "10%"]:
                trend_df.loc[count,"Indicator KPSS Confidence Level "+str(np.round(100-int(k[:-1]),0))+"%"] = ""
                trend_df.loc[count,"Indicator KPSS Stationary "+str(np.round(100-int(k[:-1]),0))+"%"] = ""
                

            for k in ["1%", "5%", "10%"]:
                trend_df.loc[count,"Indicator Result "+str(np.round(100-int(k[:-1]),0))+"%"] = "Constant"
                trend_df.loc[count,"Indicator Trend "+str(np.round(100-int(k[:-1]),0))+"%"] = ""


        update_progress((count+1)/len(files_array))

    trend_df.to_csv("Results/Forecasting Global/Trend_Analysis.csv")



import numpy as np
import pandas as pd
import os
import sys

from pathlib import Path

import warnings
warnings.simplefilter('ignore', FutureWarning)

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
    #Number of simulations
    Sims=10000

    species_array = ["MDR"]

    if not os.path.exists("Results/Forecasting Global/MonteCarloFeaturesForecasts"):
        os.makedirs("Results/Forecasting Global/MonteCarloFeaturesForecasts")

    # Load indicators
    indicators_df = pd.read_csv("Results/Indicators_China.csv", index_col=[0], header=[0])

    #Import forcast bounds (ensures percentages remain 0-100, no unrealistic negative indicators)
    IndicatorBounds=pd.read_csv('Results/IndicatorForecastBounds.csv',index_col=[0], header=[0])

    for specie in species_array:
        print(specie)
        linear_regression_df = pd.read_excel("Results/Forecasting Global/LinearRegression/LinearRegression_"+specie+".xlsx", sheet_name=None)
        
        for count, class_name in enumerate(linear_regression_df.keys()):
            print(class_name)
            RegressionParameters = pd.read_excel("Results/Forecasting Global/LinearRegression/LinearRegression_"+specie+".xlsx", sheet_name=class_name, header=[0], index_col=[0])
            features_rate_df = pd.read_excel("Results/Feature Rate/"+specie+".xlsx", sheet_name=class_name, header=[0], index_col=[0])

            # Drop rows with NaN values
            features_rate_df = features_rate_df.dropna(axis=0, how="any").reset_index(drop=True)

            # Keep columns
            idx_keep = np.where(features_rate_df["Num Isolates"] > 5)[0]
            features_rate_df = features_rate_df.loc[idx_keep,:].reset_index(drop=True)
            
            update_progress(0)
            for k in range(len(RegressionParameters)):
                indicator = RegressionParameters.loc[k,"Indicator"]
                feature = RegressionParameters.loc[k,"Feature"]

                file_indicator = 'Results/Forecasting Global/MonteCarloIndicatorForecasts/Montecarlo_'+class_name+'_Indicator_'+ str(IndicatorBounds.loc[indicator,"Code"])+'.csv'

                my_file = Path(file_indicator)
                try:
                    my_abs_path = my_file.resolve(strict=True)
                except FileNotFoundError:
                    continue
                else:
                    IndicatorForecast = pd.read_csv(file_indicator,header=[0], index_col=[0])     

                indicator_array = np.zeros(len(features_rate_df))
                for i in range(len(features_rate_df)):
                    year = features_rate_df.loc[i, "year"]

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

                year_array = features_rate_df["year"]
                year_array = np.delete(year_array,idx_nans)

                years = np.unique(year_array)
                n_years = len(years)
                first_year = np.min(years)
                last_year = np.max(years)
                
                if n_years < 5:
                    continue
                
                years_forecast = np.arange(last_year+1,2051,1)
                years_total = np.concatenate([year_array,years_forecast])
                years_indicator_forecast = np.concatenate([years,years_forecast])
                n_index = np.where(years_indicator_forecast == last_year+1)[0]

                HistoricRates = np.array(features_rate_df[feature]) 

                if len(idx_nans) > 0:
                    HistoricRates = np.delete(HistoricRates,idx_nans)         
                                
                n=RegressionParameters.loc[k,'n']
                Intercept=RegressionParameters.loc[k,'intercept']
                Intercept_stderr=RegressionParameters.loc[k,'intercept_STDerr']
                Intercept_stddev=np.sqrt(float(n))*Intercept_stderr
                Slope=RegressionParameters.loc[k,'slope']
                Slope_stderr=RegressionParameters.loc[k,'slope_STDerr']
                Slope_stddev=np.sqrt(float(n))*Slope_stderr

                Ratesforecast_intercept=np.random.normal(Intercept,Intercept_stddev,Sims)
                Ratesforecast_slope=np.random.normal(Slope,Slope_stddev,Sims)

                #Simulations_HospitalBeds
                Rateslist=[None] * Sims
                for x in range(Sims):
                    
                    rates=IndicatorForecast[str(x)].values*Ratesforecast_slope[x]+Ratesforecast_intercept[x]
                    rates[rates>1]=1
                    rates[rates<0]=0
                    rates[0:n]=HistoricRates
                    Rateslist[x]=rates.tolist()

                RatesForecastSims=pd.DataFrame(data=Rateslist, columns=years_indicator_forecast)
                RatesForecastSims.to_csv('Results/Forecasting Global/MonteCarloFeaturesForecasts/FeatureForecast_'+class_name+'_LinearRegressionRow_'+str(k)+'.csv')

                update_progress((k+1)/len(RegressionParameters))

                
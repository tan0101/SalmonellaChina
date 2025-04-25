import numpy as np
import pandas as pd
import os
import sys
import math

import warnings
warnings.simplefilter('ignore', FutureWarning)
warnings.simplefilter('ignore', RuntimeWarning)

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

#Linear Growth Function
def func_lineargrowth(x, m,c):
    return m*x + c

#Nonlinear Growth Function
def func_nonlineargrowth(x, a, b, c,d):
    return a +b*(x**c)/(d**c+x**c)

if __name__ == "__main__":
    #Number of simulations
    Sims=10000

    species_array = ["MDR"]

    if not os.path.exists("Results/Forecasting Global/MonteCarloIndicatorForecasts"):
        os.makedirs("Results/Forecasting Global/MonteCarloIndicatorForecasts")

    # Load indicators
    indicators_df = pd.read_csv("Results/Indicators_China.csv", index_col=[0], header=[0])

    #Import forcast bounds (ensures percentages remain 0-100, no unrealistic negative indicators)
    IndicatorBounds=pd.read_csv('Results/IndicatorForecastBounds.csv',index_col=[0], header=[0])

    for specie in species_array:
        print(specie)
        curve_dataframe = pd.read_excel("Results/Forecasting Global/IndicatorParameters/Indicator_CurveParameters_"+specie+".xlsx", sheet_name=None)


        for count, class_name in enumerate(curve_dataframe.keys()):
            if class_name != "Tetracycline":
                continue

            print(class_name)
            curve_df = pd.read_excel("Results/Forecasting Global/IndicatorParameters/Indicator_CurveParameters_"+specie+".xlsx", sheet_name=class_name, header=[0], index_col=[0])
            anti_df = pd.read_excel("Results/Feature Rate/"+specie+".xlsx", sheet_name=class_name, header=[0], index_col=[0])

            curve_df=curve_df.dropna(how='any')
            curve_df=curve_df[curve_df.Result !='No curve fitted']
            curve_df = curve_df.reset_index(drop=True)

            # Drop rows with NaN values
            anti_df = anti_df.dropna(axis=0, how="any").reset_index(drop=True)

            # Keep columns
            idx_keep = np.where(anti_df["Num Isolates"] > 5)[0]
            anti_df = anti_df.loc[idx_keep,:].reset_index(drop=True)

            update_progress(0)
            for count_indicator, indicator in enumerate(indicators_df.columns[2:]):                
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
                
                if n_years < 5:
                    continue
                
                years_forecast = np.arange(last_year+1,2051,1)
                years_total = np.concatenate([years,years_forecast])

                Indicator_forecast=pd.DataFrame(columns=years_total)

                Years=np.arange(len(years_total))

                for row in curve_df.index:
                    if curve_df.loc[row, "Indicator"] != indicator:
                        continue

                    Parameters=[float(i) for i in curve_df.loc[row]['Parameters'].split(",")]
                    Parameters_sd=[float(i) for i in curve_df.loc[row]['Parameters_sd'].split(",")] 
                    if curve_df.loc[row]['Result'] == 'Linear curve fitted':
                        Forecastparam_m=np.random.normal(Parameters[0],Parameters_sd[0],Sims)
                        Forecastparam_c=np.random.normal(Parameters[1],Parameters_sd[1],Sims)

                        simulated_y={}
                        for x in range(Sims):
                            m=Forecastparam_m[x]
                            c=Forecastparam_c[x]
                            y=func_lineargrowth(Years, m,c)
                            y[y>IndicatorBounds.loc[indicator]['Max']]=IndicatorBounds.loc[indicator]['Max']
                            y[y<IndicatorBounds.loc[indicator]['Min']]=IndicatorBounds.loc[indicator]['Min']
                            simulated_y[x]=y.tolist()
                        df=pd.DataFrame(data=simulated_y, index=years_total)
                        df.to_csv('Results/Forecasting Global/MonteCarloIndicatorForecasts/Montecarlo_'+class_name+'_Indicator_'+str(IndicatorBounds.loc[indicator,"Code"])+ '.csv')
                    elif curve_df.loc[row]['Result'] == 'Nonlinear curve fitted':
                        Forecastparam_a=np.random.normal(Parameters[0],Parameters_sd[0],Sims)
                        Forecastparam_b=np.random.normal(Parameters[1],Parameters_sd[1],Sims)
                        Forecastparam_c=np.random.normal(Parameters[2],Parameters_sd[2],Sims)
                        Forecastparam_c[Forecastparam_c<0]=0
                        Forecastparam_d=np.random.normal(Parameters[3],Parameters_sd[3],Sims)
                        
                        simulated_y={}
                        flag_overflown = False
                        for x in range(Sims):
                            a=Forecastparam_a[x]
                            b=Forecastparam_b[x]
                            c=Forecastparam_c[x]
                            d=Forecastparam_d[x]
                            
                            if math.isinf(d**c) or math.isnan(d**c) or math.isinf(x**c) or math.isnan(x**c) or math.isinf(b*(x**c)) or math.isnan(b*(x**c)) or math.isinf(d**c + x**c) or math.isnan(d**c + x**c) or math.isinf(a +b*(x**c)/(d**c+x**c)) or math.isnan(a +b*(x**c)/(d**c+x**c)):     
                                flag_overflown = True

                            if flag_overflown == True:
                                continue

                            y=func_nonlineargrowth(Years, a,b,c,d)
                            y[y>IndicatorBounds.loc[indicator]['Max']]=IndicatorBounds.loc[indicator]['Max']
                            y[y<IndicatorBounds.loc[indicator]['Min']]=IndicatorBounds.loc[indicator]['Min']
                            simulated_y[x]=y.tolist()
                        
                        if flag_overflown == False:
                            df=pd.DataFrame(data=simulated_y, index=years_total)
                            df.to_csv('Results/Forecasting Global/MonteCarloIndicatorForecasts/Montecarlo_'+class_name+'_Indicator_'+ str(IndicatorBounds.loc[indicator,"Code"])+ '.csv')

                update_progress((count_indicator+1)/len(indicators_df.columns[2:]))
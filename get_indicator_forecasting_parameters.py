import numpy as np
import pandas as pd
import os
import sys
import scipy as sp

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

#Linear Growth Function
def func_lineargrowth(x, m,c):
    return m*x + c

#Nonlinear Growth Function
def func_nonlineargrowth(x, a, b, c,d):
    return a +b*(x**c)/(d**c+x**c)

if __name__ == "__main__":
    species_array = ["MDR"]

    if not os.path.exists("Results/Forecasting Global/IndicatorForecasts"):
        os.makedirs("Results/Forecasting Global/IndicatorForecasts")

    if not os.path.exists("Results/Forecasting Global/IndicatorParameters"):
        os.makedirs("Results/Forecasting Global/IndicatorParameters")

    # Load indicators
    indicators_df = pd.read_csv("Results/Indicators_China.csv", index_col=[0], header=[0])

    #Import forcast bounds (ensures percentages remain 0-100, no unrealistic negative indicators)
    IndicatorBounds=pd.read_csv('Results/IndicatorForecastBounds.csv',index_col=[0], header=[0])

    for specie in species_array:
        print(specie)
        feature_rate_df = pd.read_excel("Results/Feature Rate/"+specie+".xlsx", sheet_name=None)

        writer_curve = pd.ExcelWriter("Results/Forecasting Global/IndicatorParameters/Indicator_CurveParameters_"+specie+".xlsx", engine='xlsxwriter')

        update_progress(0)
        for count, class_name in enumerate(feature_rate_df.keys()):
            anti_df = pd.read_excel("Results/Feature Rate/"+specie+".xlsx", sheet_name=class_name, header=[0], index_col=[0])

            # Drop rows with NaN values
            anti_df = anti_df.dropna(axis=0, how="any").reset_index(drop=True)

            # Keep columns
            idx_keep = np.where(anti_df["Num Isolates"] > 5)[0]
            anti_df = anti_df.loc[idx_keep,:].reset_index(drop=True)

            k = 0
            curve_df = pd.DataFrame()
            Indicator_forecast=pd.DataFrame()#columns=years_total
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
                
                if n_years < 5:
                    continue
                
                years_forecast = np.arange(last_year+1,2051,1)
                years_total = np.concatenate([years,years_forecast])

                Indicator_forecast=pd.DataFrame(columns=years_total)

                Years=np.arange(len(years_total))

                #Produce forecasts
                d = {'x': year_array-first_year, 'y': indicator_array}
                data = pd.DataFrame(data=d).dropna()
                
                try:
                    popt_nonlinear, pcov_nonlinear=sp.optimize.curve_fit(func_nonlineargrowth,data['x'],data['y'].values,bounds=([-1000,-1000,0,0],[1000,1000,10,1000]))
                    ss_res = np.sum((data['y'].values- func_nonlineargrowth(data['x'], *popt_nonlinear))**2)
                    ss_tot = np.sum((data['y'].values-np.mean(data['y'].values))**2)
                    r_squared_nonlinear = 1 - (ss_res / ss_tot)
                    perr_nonlinear = np.sqrt(np.diag(pcov_nonlinear))
                except:
                    r_squared_nonlinear=0
                    continue

                try:
                    popt_linear, pcov_linear=sp.optimize.curve_fit(func_lineargrowth,data['x'],data['y'].values)
                    ss_res = np.sum((data['y'].values- func_lineargrowth(data['x'], *popt_linear))**2)
                    r_squared_linear = 1 - (ss_res / ss_tot)
                    perr_linear = np.sqrt(np.diag(pcov_linear))
                except:
                    r_squared_linear=0
                    continue
                
                if (r_squared_nonlinear>0.8)&(r_squared_nonlinear>r_squared_linear):
                    curve_df.loc[k,'Indicator']=indicator
                    curve_df.loc[k,'Result']='Nonlinear curve fitted'
                    curve_df.loc[k,'R-squared']=r_squared_nonlinear
                    curve_df.loc[k,'Parameters']=','.join(str(x) for x in popt_nonlinear)
                    curve_df.loc[k,'Parameters_sd']=','.join(str(x) for x in perr_nonlinear)
                    y=func_nonlineargrowth(Years,popt_nonlinear[0],popt_nonlinear[1],popt_nonlinear[2],popt_nonlinear[3])
                    y[y>IndicatorBounds.loc[indicator]['Max']]=IndicatorBounds.loc[indicator]['Max']
                    y[y<IndicatorBounds.loc[indicator]['Min']]=IndicatorBounds.loc[indicator]['Min']
                    Indicator_forecast.loc[indicator]=y
                    
                elif (r_squared_linear>0.8)&(r_squared_nonlinear<r_squared_linear): 
                    curve_df.loc[k,'Indicator']=indicator
                    curve_df.loc[k,'Result']='Linear curve fitted'
                    curve_df.loc[k,'R-squared']=r_squared_linear
                    curve_df.loc[k,'Parameters']=','.join(str(x) for x in popt_linear)
                    curve_df.loc[k,'Parameters_sd']=','.join(str(x) for x in perr_linear)
                    y=func_lineargrowth(Years,popt_linear[0],popt_linear[1])
                    y[y>IndicatorBounds.loc[indicator]['Max']]=IndicatorBounds.loc[indicator]['Max']
                    y[y<IndicatorBounds.loc[indicator]['Min']]=IndicatorBounds.loc[indicator]['Min']
                    Indicator_forecast.loc[indicator]=y
                    
                else:
                    curve_df.loc[k,'Indicator']=indicator
                    curve_df.loc[k,'Result']='No curve fitted'
                    curve_df.loc[k,'R-squared']='N/A'
                    curve_df.loc[k,'Parameters']='N/A'

                k+=1
            
                if len(Indicator_forecast) > 0:
                    Indicator_forecast.to_csv('Results/Forecasting Global/IndicatorForecasts/'+specie+'_'+class_name+"_indicator_"+str(IndicatorBounds.loc[indicator,"Code"])+'.csv')  
        
            if len(curve_df) > 0:
                curve_df.to_excel(writer_curve, sheet_name = class_name, index = True)

            update_progress((count+1)/len(feature_rate_df.keys()))
        
        writer_curve.close() 
import os
import sys
import shutil
import warnings
from pathlib import Path
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
from concurrent.futures import ProcessPoolExecutor, as_completed

warnings.simplefilter('ignore', (FutureWarning, UserWarning))

def update_progress(progress: float) -> None:
    bar_length = 50
    progress = min(1.0, max(0.0, progress))
    block = int(round(bar_length * progress))
    text = f"\rProgress: [{'#' * block + '-' * (bar_length - block)}] {round(progress * 100, 2)}%"
    sys.stdout.write(text)
    sys.stdout.flush()

def compute_and_save_forecast(
    file_indicator: str,
    out_path: str,
    Sims: int,
    intercept: float,
    intercept_stddev: float,
    slope: float,
    slope_stddev: float,
    is_diff: bool,
    last_amr: float,
    last_ind: float
) -> Tuple[str, Optional[str]]:
    try:
        indicator_forecast = pd.read_csv(file_indicator, header=0, index_col=0)
        IF_vals = indicator_forecast.values  

        ratesforecast_intercept = np.random.normal(intercept, intercept_stddev, Sims)
        ratesforecast_slope = np.random.normal(slope, slope_stddev, Sims)

        if is_diff:
            delta_ind = IF_vals - last_ind
            delta_amr = delta_ind * ratesforecast_slope[np.newaxis, :] + ratesforecast_intercept[np.newaxis, :]
            rates_matrix = last_amr + delta_amr
        else:
            rates_matrix = IF_vals * ratesforecast_slope[np.newaxis, :] + ratesforecast_intercept[np.newaxis, :]

        np.clip(rates_matrix, 0.0, 1.0, out=rates_matrix)
        df_out = pd.DataFrame(rates_matrix.T, columns=indicator_forecast.index.tolist())
        
        
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df_out.to_csv(out_path)

        return out_path, None
    except Exception as e:
        return out_path, str(e)

def main():
    specie = sys.argv[1] if len(sys.argv) > 1 else "Salmonella"
    base_drive = "ML"
    
    final_drive_dir = f"{base_drive}/Results/MonteCarloFeaturesForecasts"
    os.makedirs(final_drive_dir, exist_ok=True)
    
    lr_path = f"{base_drive}/Results/LinearRegression/LinearRegression_{specie}_permutationtest.xlsx"
    indicator_bounds = pd.read_csv(f"{base_drive}/Data/IndicatorForecastBounds.csv", index_col=0)
    linear_regression_all = pd.read_excel(lr_path, sheet_name=None)
    
    
    print("Loaded sheets:")
    for name, df in linear_regression_all.items():
        print("  ", name, df.shape)


    for class_name, reg_params in linear_regression_all.items():
        if reg_params.empty: 
            continue
        print(f"\nSimulating: {class_name}")
        
        reg_rows = reg_params.reset_index(drop=True)
        tasks = []

        
        with ProcessPoolExecutor(max_workers=os.cpu_count()) as executor:
            future_to_info = {}
            for idx, row in reg_rows.iterrows():
                indicator = row['Indicator']
                if indicator not in indicator_bounds.index: 
                    continue
                code = str(indicator_bounds.loc[indicator, 'Code'])
                
                file_indicator = f"{base_drive}/Results/MonteCarloIndicatorForecasts/IndicatorForecast_montecarlosimulations_{specie}_{class_name}_Indicator_{code}.csv"
                if not os.path.exists(file_indicator): 
                    continue

                n = float(row['n'])

                out_path = f"{final_drive_dir}/FeatureForecast_{specie}_{class_name}_Row_{idx}.csv"

                fut = executor.submit(
                    compute_and_save_forecast,
                    file_indicator, out_path, 10000,
                    float(row['intercept']), np.sqrt(n)*float(row['intercept_STDerr']),
                    float(row['slope']), np.sqrt(n)*float(row['slope_STDerr']),
                    bool(row['is_differenced']), float(row['last_AMR_value']), float(row['last_Indicator_value'])
                )
                future_to_info[fut] = idx
                print("TASK ADDED:", class_name, indicator, "→", out_path)
            
            completed = 0
            if future_to_info:
                for fut in as_completed(future_to_info):
                    completed += 1
                    update_progress(completed / len(future_to_info))

        
    print(f"\nAll tasks completed. Results have been successfully completed.")

if __name__ == '__main__':
    main()
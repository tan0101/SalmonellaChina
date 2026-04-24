
#!/usr/bin/env python3


import numpy as np
import pandas as pd
import os
import sys
from joblib import Parallel, delayed

# --- Growth functions ---
def func_lineargrowth(x, m, c):
    return m * x + c

def func_nonlineargrowth(x, a, b, c, d):
    return a + b * (x ** c) / (d ** c + x ** c)

def func_exponential(x, a, b, c):
    return a * np.exp(b * x) + c

def func_gompertz(x, a, b, c):
    return a * np.exp(-b * np.exp(-c * x))


def simulate_indicator(indicator, curve_df, IndicatorBounds, years_total, Years, Sims, specie, class_name):
    row = curve_df[curve_df['Indicator'] == indicator]
    if row.empty:
        return None

    Parameters = [float(i) for i in row.iloc[0]['Parameters'].split(",")]
    Parameters_sd = [float(i) for i in row.iloc[0]['Parameters_sd'].split(",")]
    result_type = row.iloc[0]['Result']

    simulated_y = None
    
    if result_type == 'Linear curve fitted':

        Forecastparam_m = np.random.normal(Parameters[0], Parameters_sd[0], Sims)
        Forecastparam_c = np.random.normal(Parameters[1], Parameters_sd[1], Sims)

        rates_matrix = np.outer(Forecastparam_m, Years) + Forecastparam_c[:, None]
        rates_matrix = np.clip(rates_matrix, IndicatorBounds.loc[indicator]['Min'], IndicatorBounds.loc[indicator]['Max'])

        simulated_y = pd.DataFrame(rates_matrix.T, index=years_total)

    elif result_type == 'Nonlinear curve fitted':

        Forecastparam_a = np.random.normal(Parameters[0], Parameters_sd[0], Sims)
        Forecastparam_b = np.random.normal(Parameters[1], Parameters_sd[1], Sims)
        Forecastparam_c = np.random.normal(Parameters[2], Parameters_sd[2], Sims)
        Forecastparam_d = np.random.normal(Parameters[3], Parameters_sd[3], Sims)

        Forecastparam_c[Forecastparam_c < 0] = 0  # nonlinear constraint

        a = Forecastparam_a[:, None]
        b = Forecastparam_b[:, None]
        c = Forecastparam_c[:, None]
        d = Forecastparam_d[:, None]
        x = Years[None, :]

        y_matrix = a + b * (x**c) / (d**c + x**c)
        y_matrix = np.clip(y_matrix, IndicatorBounds.loc[indicator]['Min'], IndicatorBounds.loc[indicator]['Max'])

        simulated_y = pd.DataFrame(y_matrix.T, index=years_total)

    elif result_type == 'Exponential curve fitted':

        Forecastparam_a = np.random.normal(Parameters[0], Parameters_sd[0], Sims)
        Forecastparam_b = np.random.normal(Parameters[1], Parameters_sd[1], Sims)
        Forecastparam_c = np.random.normal(Parameters[2], Parameters_sd[2], Sims)

        a = Forecastparam_a[:, None]
        b = Forecastparam_b[:, None]
        c = Forecastparam_c[:, None]
        x = Years[None, :]

        y_matrix = a * np.exp(b * x) + c
        y_matrix = np.clip(y_matrix, IndicatorBounds.loc[indicator]['Min'], IndicatorBounds.loc[indicator]['Max'])

        simulated_y = pd.DataFrame(y_matrix.T, index=years_total)

    elif result_type == 'Gompertz curve fitted':

        Forecastparam_a = np.random.normal(Parameters[0], Parameters_sd[0], Sims)
        Forecastparam_b = np.random.normal(Parameters[1], Parameters_sd[1], Sims)
        Forecastparam_c = np.random.normal(Parameters[2], Parameters_sd[2], Sims)

        a = Forecastparam_a[:, None]
        b = Forecastparam_b[:, None]
        c = Forecastparam_c[:, None]
        x = Years[None, :]

        y_matrix = a * np.exp(-b * np.exp(-c * x))
        y_matrix = np.clip(y_matrix, IndicatorBounds.loc[indicator]['Min'], IndicatorBounds.loc[indicator]['Max'])

        simulated_y = pd.DataFrame(y_matrix.T, index=years_total)

    else:
        return None


    out_file = f"Results/MonteCarloIndicatorForecasts/IndicatorForecast_montecarlosimulations_{specie}_{class_name}_Indicator_{IndicatorBounds.loc[indicator,'Code']}.csv"
    simulated_y.to_csv(out_file)
    return indicator

# --- Main ---
def main():
    species = sys.argv[1] if len(sys.argv) > 1 else "Example_Species"
    Sims = 10000

    # Paths
    curve_params_file = f"Results/IndicatorParameters/Indicator_CurveParameters_{species}.xlsx"
    indicator_bounds_file = "Data/IndicatorForecastBounds.csv"
    correlation_file = f"Results/Correlations/{species}.xlsx"
    output_dir = "Results/MonteCarloIndicatorForecasts"
    os.makedirs(output_dir, exist_ok=True)

    # Load data
    IndicatorBounds = pd.read_csv(indicator_bounds_file, index_col=0)
    curve_sheets = pd.read_excel(curve_params_file, sheet_name=None, engine="openpyxl")
    feature_rate_sheets = pd.read_excel(correlation_file, sheet_name=None, engine="openpyxl")

    for class_name, curve_df in curve_sheets.items():
        curve_df.columns = [str(c).strip() for c in curve_df.columns]
        if curve_df.empty or len(curve_df.columns) < 2 or 'Result' not in curve_df.columns:
            print(f"Skipping [{class_name}]: Step2 generated no valid fit data (empty or incomplete format).")
            continue
        curve_df = curve_df[curve_df['Result'] != 'No curve fitted'].reset_index(drop=True)
        if curve_df.empty:
            print(f"ℹ️ Skipping [{class_name}]: No successfully fitted indicators.")
            continue

        # Prepare years
        if class_name not in feature_rate_sheets:
            continue
        anti_df = feature_rate_sheets[class_name].dropna().reset_index(drop=True)
        if "Num Isolates" in anti_df.columns:
            anti_df = anti_df[anti_df["Num Isolates"] > 5].reset_index(drop=True)
        years = np.unique(anti_df["year"].values)
        if len(years) < 5:
            continue
        last_year = years.max()
        years_forecast = np.arange(last_year + 1, 2051)
        years_total = np.concatenate([years, years_forecast])
        Years = np.arange(len(years_total))

        indicators = curve_df['Indicator'].tolist()

        # Parallel processing of indicators
        Parallel(n_jobs=-1, backend="loky")(
            delayed(simulate_indicator)(indicator, curve_df, IndicatorBounds, years_total, Years, Sims, species, class_name)
            for indicator in indicators
        )

    print("Optimized Monte Carlo indicator forecasting completed successfully.")

if __name__ == "__main__":
    main()

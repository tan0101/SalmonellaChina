
#!/usr/bin/env python3


import os
import sys
import warnings
from typing import Tuple, Optional, Dict, Any

import numpy as np
import pandas as pd
import scipy.optimize as opt
from joblib import Parallel, delayed


warnings.simplefilter('ignore', FutureWarning)

# --- Growth functions ---
def func_lineargrowth(x: np.ndarray, m: float, c: float) -> np.ndarray:
    """Linear growth: y = m * x + c"""
    return m * x + c

def func_nonlineargrowth(x: np.ndarray, a: float, b: float, c: float, d: float) -> np.ndarray:
    """Nonlinear growth: y = a + b * x^c / (d^c + x^c)"""
    return a + b * (x ** c) / (d ** c + x ** c)

def func_exponential(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Exponential model: y = a * exp(b * x) + c"""
    return a * np.exp(b * x) + c


def func_gompertz(x: np.ndarray, a: float, b: float, c: float) -> np.ndarray:
    """Gompertz growth model: y = a * exp(-b * exp(-c * x))"""
    return a * np.exp(-b * np.exp(-c * x))

def _slope_signs(x, yhat, qs=(0.1, 0.5, 0.9)):
    order = np.argsort(x)
    x_sorted = x[order]; y_sorted = yhat[order]
    dydx = np.gradient(y_sorted, x_sorted)
    idxs = [int(np.clip(round(q*(len(x_sorted)-1)), 0, len(x_sorted)-1)) for q in qs]
    signs = []
    for i in idxs:
        val = dydx[i]
        signs.append(0.0 if abs(val) < 1e-12 else float(np.sign(val)) if np.isfinite(val) else np.nan)
    return signs

def _disagreement_index(yhats, y_obs):
    Y = np.vstack(yhats)
    sd_t = np.nanstd(Y, axis=0)
    di = float(np.nanmean(sd_t))
    rng = float(np.nanmax(y_obs) - np.nanmin(y_obs))
    return di / rng if rng > 0 else np.nan

def qualitative_consistency(x_obs, y_obs, accepted_yhats, agreement_threshold=0.8, di_threshold=0.10):
    if len(accepted_yhats) < 2:
        return None
    S = np.array([_slope_signs(x_obs, yhat) for yhat in accepted_yhats])  # m x 3
    agree_props = []
    for k in range(S.shape[1]):
        col = S[:, k]; col = col[np.isfinite(col)]
        if col.size == 0: agree_props.append(np.nan)
        else:
            _, counts = np.unique(col, return_counts=True)
            agree_props.append(np.max(counts)/counts.sum())
    slope_ok = all([(ap >= agreement_threshold) for ap in agree_props if np.isfinite(ap)])
    di = _disagreement_index(accepted_yhats, y_obs)
    di_ok = (np.isfinite(di) and di <= di_threshold)
    return bool(slope_ok and di_ok)



def fit_indicator(
    indicator: str,
    anti_df: pd.DataFrame,
    indicators_df: pd.DataFrame,
    bounds_df: pd.DataFrame,
    horizon_end: int = 2051
) -> Tuple[Optional[Dict[str, Any]], Optional[pd.Series]]:


    merged = anti_df.merge(
        indicators_df,
        left_on=["year"],
        right_on=["year"],
        how="left",
        suffixes=("", "_ind")
    )

    if indicator not in merged.columns:
        return None, None

    indicator_vals = merged[indicator].to_numpy()

  
    valid_mask = ~np.isnan(indicator_vals)
    indicator_vals = indicator_vals[valid_mask]

  
    if indicator_vals.size < 5 or np.unique(indicator_vals).size == 1:
        return None, None


    year_array = merged.loc[valid_mask, "year"].to_numpy()
    years = np.unique(year_array)
    if years.size < 5:
        return None, None

    first_year = years.min()
    last_year = years.max()


    years_forecast = np.arange(last_year + 1, horizon_end, 1)
    years_total = np.concatenate([years, years_forecast])
    Years = np.arange(years_total.size) 


    x = year_array - first_year 
    y = indicator_vals
    ss_tot = np.sum((y - y.mean()) ** 2)  
    accepted_yhats_obs: List[np.ndarray] = []

    r2_nl = 0.0
    popt_nl = None
    perr_nl = None
    try:
        popt_nl, pcov_nl = opt.curve_fit(
            func_nonlineargrowth, x, y,
            bounds=([-1000, -1000, 0, 0], [1000, 1000, 10, 1000]),
            maxfev=20000
        )
        yhat_nl = func_nonlineargrowth(x, *popt_nl)
        ss_res_nl = np.sum((y - yhat_nl) ** 2)
        r2_nl = 1.0 - (ss_res_nl / ss_tot) if ss_tot > 0 else 0.0
        perr_nl = np.sqrt(np.diag(pcov_nl))
    except Exception:
        pass  # keep r2_nl at 0
    
    if r2_nl >= 0.8: accepted_yhats_obs.append(yhat_nl)
    

    r2_lin = 0.0
    popt_lin = None
    perr_lin = None
    try:
        popt_lin, pcov_lin = opt.curve_fit(func_lineargrowth, x, y, maxfev=20000)
        yhat_lin = func_lineargrowth(x, *popt_lin)
        ss_res_lin = np.sum((y - yhat_lin) ** 2)
        r2_lin = 1.0 - (ss_res_lin / ss_tot) if ss_tot > 0 else 0.0
        perr_lin = np.sqrt(np.diag(pcov_lin))
    except Exception:
        pass  # keep r2_lin at 0
    
    if r2_lin >= 0.8: accepted_yhats_obs.append(yhat_lin)
    

    r2_exp = 0.0
    popt_exp = None
    perr_exp = None
    try:
        popt_exp, pcov_exp = opt.curve_fit(
            func_exponential, x, y,
            bounds=([-1e6, -10, -1e6], [1e6, 10, 1e6]),
            maxfev=20000
        )
        yhat_exp = func_exponential(x, *popt_exp)
        ss_res_exp = np.sum((y - yhat_exp)**2)
        r2_exp = 1.0 - (ss_res_exp / ss_tot)
        perr_exp = np.sqrt(np.diag(pcov_exp))
    except Exception:
        pass
    
    if r2_exp >= 0.8: accepted_yhats_obs.append(yhat_exp)
    

    
    r2_gomp = 0.0; popt_gomp = perr_gomp = None
    try:
        a_low = max(np.max(y), 1e-6)               
        a_high = max(a_low * 10.0, a_low + 1.0)
        bounds_low = [a_low, 1e-6, 1e-6]             
        bounds_high = [a_high, 100.0, 5.0]
        a0 = min(a_high, a_low * 1.1); b0 = 1.0
        span = max(x.max() - x.min(), 1.0); c0 = min(0.2, 2.0 / span)
        popt_gomp, pcov_gomp = opt.curve_fit(
            func_gompertz, x, y, p0=[a0, b0, c0],
            bounds=(bounds_low, bounds_high), maxfev=40000
        )
        yhat_gomp = func_gompertz(x, *popt_gomp)
        ss_res_gomp = np.sum((y - yhat_gomp) ** 2)
        r2_gomp = 1.0 - (ss_res_gomp / ss_tot) if ss_tot > 0 else 0.0
        perr_gomp = np.sqrt(np.diag(pcov_gomp))
    except Exception:
        pass
    
    if r2_gomp >= 0.8: accepted_yhats_obs.append(yhat_gomp)
       

    model_r2 = {
        "Nonlinear": r2_nl,
        "Linear": r2_lin,
        "Exponential": r2_exp,
        "Gompertz": r2_gomp
    }

    best_model = max(model_r2, key=model_r2.get)
    best_r2 = model_r2[best_model]

    params = None
    params_sd = None

    if best_r2 > 0.8:
        if best_model == "Nonlinear":
            params = popt_nl; params_sd = perr_nl
            y_forecast = func_nonlineargrowth(Years, *popt_nl)

        elif best_model == "Linear":
            params = popt_lin; params_sd = perr_lin
            y_forecast = func_lineargrowth(Years, *popt_lin)

        elif best_model == "Exponential":
            params = popt_exp; params_sd = perr_exp
            y_forecast = func_exponential(Years, *popt_exp)

        elif best_model == "Gompertz":
            params = popt_gomp; params_sd = perr_gomp
            y_forecast = func_gompertz(Years, *popt_gomp)

        result_text = f"{best_model} curve fitted"
        r2_final = best_r2

    else:
        result_text = "No curve fitted"
        r2_final = "N/A"
        y_forecast = np.zeros_like(Years)


    try:
        min_val = float(bounds_df.loc[indicator, "Min"])
        max_val = float(bounds_df.loc[indicator, "Max"])
        y_forecast = np.clip(y_forecast, min_val, max_val)
    except Exception:

        pass

    forecast_series = pd.Series(y_forecast, index=years_total, name=indicator)

    row = {
        "Indicator": indicator,
        "Result": result_text,
        "R-squared": r2_final,
        "Parameters": ",".join(map(str, params)) if params is not None else "N/A",
        "Parameters_sd": ",".join(map(str, params_sd)) if params_sd is not None else "N/A",
        "FirstYear": int(first_year),
        "LastYear": int(last_year),
        "NumYears": int(years.size),
        "NumPoints": int(y.size),
        "QualitativeConsistentObserved": (qualitative_consistency(x_obs=x, y_obs=y, accepted_yhats=accepted_yhats_obs,agreement_threshold=0.8, di_threshold=0.10) if len(accepted_yhats_obs) >= 2 else "N/A")}

    return row, forecast_series


def write_per_indicator_csvs(
    species: str,
    class_name: str,
    forecasts_df: pd.DataFrame,
    bounds_df: pd.DataFrame,
    out_dir: str,
    params_df: Optional[pd.DataFrame] = None,
    only_fitted: bool = True
) -> None:

    if forecasts_df is None or forecasts_df.empty:
        return

    os.makedirs(out_dir, exist_ok=True)


    fitted_set = None
    if only_fitted and params_df is not None and not params_df.empty and 'Result' in params_df.columns:
        fitted_set = set(params_df.loc[params_df['Result'].isin(['Linear curve fitted', 'Nonlinear curve fitted']),
                                       'Indicator'].astype(str))

    for indicator in forecasts_df.index:

        if fitted_set is not None and indicator not in fitted_set:
            continue

 
        try:
            code = str(bounds_df.loc[indicator, "Code"])
        except Exception:
            code = "unknown"

        out_csv = os.path.join(
            out_dir,
            f"{species}_{class_name}_indicator_{code}.csv"
        )

        series = forecasts_df.loc[indicator]

        series.to_frame().T.to_csv(out_csv)


# --- Main ---
def main():

    species = sys.argv[1] if len(sys.argv) > 1 else "Example_Species"

    indicators_csv = "Data/normalized_indicators.csv"
    bounds_csv = "Data/IndicatorForecastBounds.csv"
    correlations_xlsx = f"Results/Correlations/{species}.xlsx"
    params_xlsx_out = f"Results/IndicatorParameters/Indicator_CurveParameters_{species}.xlsx"
    forecasts_dir = "Results/IndicatorForecasts"

    os.makedirs(os.path.dirname(params_xlsx_out), exist_ok=True)
    os.makedirs(forecasts_dir, exist_ok=True)

    indicators_df = pd.read_csv(indicators_csv, index_col=0)
    bounds_df = pd.read_csv(bounds_csv, index_col=0)

    feature_rate_df = pd.read_excel(correlations_xlsx, sheet_name=None, engine="openpyxl")


    params_by_class: Dict[str, pd.DataFrame] = {}

    forecasts_by_class: Dict[str, pd.DataFrame] = {}

    for class_name, anti_df in feature_rate_df.items():

        anti_df = anti_df.dropna(axis=0, how="any").reset_index(drop=True)
        if "Num Isolates" in anti_df.columns:
            anti_df = anti_df[anti_df["Num Isolates"] > 5].reset_index(drop=True)

        indicator_cols = [c for c in indicators_df.columns if c not in ("year")]
        if not indicator_cols:

            indicator_cols = list(indicators_df.columns[2:])

        results = Parallel(n_jobs=-1, backend="loky")(
            delayed(fit_indicator)(ind, anti_df, indicators_df, bounds_df)
            for ind in indicator_cols
        )

        curve_rows = []
        forecast_rows = []
        forecast_index = []

        for row, series in results:
            if row is not None:
                curve_rows.append(row)
            if series is not None:
                forecast_rows.append(series)
                forecast_index.append(series.name)


        params_df = pd.DataFrame(curve_rows)
        params_by_class[class_name] = params_df

        if forecast_rows:
            forecasts_df = pd.DataFrame(forecast_rows)
            forecasts_df.index = forecast_index 
            forecasts_by_class[class_name] = forecasts_df
        else:
            forecasts_by_class[class_name] = pd.DataFrame()

    with pd.ExcelWriter(params_xlsx_out, engine="xlsxwriter") as writer:
        for class_name, df in params_by_class.items():
 
            df.to_excel(writer, sheet_name=class_name, index=False)

    for class_name, df in forecasts_by_class.items():
        out_csv = os.path.join(forecasts_dir, f"{species}_{class_name}_indicator_forecast.csv")
        df.to_csv(out_csv)

        write_per_indicator_csvs(
            species=species,
            class_name=class_name,
            forecasts_df=df,
            bounds_df=bounds_df,
            out_dir=os.path.join(forecasts_dir),  
            params_df=params_by_class.get(class_name, pd.DataFrame()),
            only_fitted=True  
        )


        print(f"Optimized indicator forecasting completed for species: {species}")

if __name__ == "__main__":
    main()

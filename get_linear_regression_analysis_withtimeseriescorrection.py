
import os
import sys
import warnings
warnings.filterwarnings("ignore")
warnings.simplefilter("ignore", category=FutureWarning)
import numpy as np
import pandas as pd
import statsmodels.api as sm
import statsmodels.formula.api as smf

from statsmodels.tsa.stattools import adfuller, coint   
from scipy.stats import pearsonr
from joblib import Parallel, delayed


from contextlib import contextmanager
from tqdm.auto import tqdm
import joblib

@contextmanager
def tqdm_joblib(tqdm_object):

    class TqdmBatchCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_callback
        tqdm_object.close()

species = sys.argv[1] if len(sys.argv) > 1 else "example_species"
output = f"Results/LinearRegression/LinearRegression_{species}_permutationtest.xlsx"
input_path = f"Results/Correlations/{species}.xlsx"
indicators_path = "Data/normalized_indicators.csv"

# Load data
indicators_df = pd.read_csv(indicators_path, index_col=[0], header=[0])
indicators_df = indicators_df.reset_index()
feature_rate_df = pd.read_excel(input_path, sheet_name=None, engine="openpyxl")

writer = pd.ExcelWriter(output, engine="xlsxwriter")

indicator_cols = indicators_df.columns[2:]


# ============================================================
#  FUNCTION: Time‑series aware regression logic
# ============================================================
def ts_regression(y_series, x_series):
    """
    Implements:
        - Pearson pre-screening
        - ADF stationarity tests
        - Differencing if needed
        - HAC-corrected OLS
    Returns:
        model_df, hac_model, obs_r2, is_differenced, last_value
    """

    model_df = pd.concat([y_series, x_series], axis=1).dropna()
    model_df.columns = ["Rate", "indicator"]
    
    if model_df["Rate"].nunique() <= 1:
        print(f"[DEBUG] Constant RATE series — skipping")
        return None

    if len(model_df) < 6:
        return None

    last_val = model_df["Rate"].iloc[-1]

   
    r_val, p_val = pearsonr(model_df["Rate"], model_df["indicator"])

    if p_val >= 0.05 or abs(r_val) <= 0.5:
        
        print(
            f"[DEBUG] Pearson failed: r={r_val:.3f}, p={p_val:.3f} for "
            f"Rate='{y_series.name}' × Indicator='{x_series.name}'"
        )

        return None


    # --- Stationarity tests ---
    
    try:
        p_y_adf = adfuller(model_df["Rate"])[1]
        p_x_adf = adfuller(model_df["indicator"])[1]
    except Exception as e:
        return None
    stationary = (p_y_adf < 0.05) and (p_x_adf < 0.05)

    is_differenced = False

    # --- If non-stationary, check cointegration ---
    if not stationary:
        p_coint = coint(model_df["Rate"], model_df["indicator"])[1]
        if p_coint >= 0.05:
            # Difference both
            model_df = model_df.diff().dropna()
            is_differenced = True

    if len(model_df) < 6:
        return None

    # === Calculate observed R² using standard OLS ===
    obs_r2 = smf.ols("Rate ~ indicator", data=model_df).fit().rsquared

    # === HAC‑corrected regression ===
    X = sm.add_constant(model_df["indicator"])
    hac_model = sm.OLS(model_df["Rate"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": 1}
    )

    return model_df, hac_model, obs_r2, is_differenced, last_val

def process_pair(indicator, feat, anti_df):
    # Merge indicator series

    merged_df = anti_df.merge(
        indicators_df[["year", indicator]],
        left_on="year",
        right_on="year",
        how="left"
    )



    indicator_arr = merged_df[indicator].values
    rate_arr = merged_df[feat].values
    year_arr = merged_df["year"].values  

    valid = ~np.isnan(indicator_arr)

    indicator_arr = indicator_arr[valid]
    rate_arr = rate_arr[valid]
    year_arr = year_arr[valid] 


    if len(indicator_arr) < 6:
        return None

    model_y = pd.Series(rate_arr)
    model_x = pd.Series(indicator_arr)

    ts_result = ts_regression(model_y, model_x)
    if ts_result is None:
        return None

    model_df, hac_model, obs_r2, is_diff, last_val = ts_result
    last_indicator_val = model_df["indicator"].iloc[-1]
    r_val, p_val = pearsonr(model_df["Rate"], model_df["indicator"])
    
    
    years_unique = np.unique(year_arr)

    years_str = ", ".join(map(str, years_unique))
    num_years = len(years_unique)
    first_year = int(np.min(years_unique))
    last_year = int(np.max(years_unique))

    n_permutations = 1000
    r2_null = []

    for _ in range(n_permutations):
        perm_df = model_df.copy()
        perm_df["Rate"] = np.random.permutation(model_df["Rate"])
        perm_model = smf.ols("Rate ~ indicator", data=perm_df).fit()
        r2_null.append(perm_model.rsquared)

    p_perm = np.mean(np.array(r2_null) >= obs_r2)

    return {
        "Indicator": indicator,
        "Feature": feat,
        "Pearson r": r_val,
        "Pearson p-value": p_val,
        "intercept": hac_model.params[0],
        "intercept_STDerr": hac_model.bse[0],
        "intercept_pvalue": hac_model.pvalues[0],
        "slope": hac_model.params[1],
        "slope_STDerr": hac_model.bse[1],
        "slope_pvalue": hac_model.pvalues[1],
        "Rsquared": obs_r2,
        "n": len(model_df),
        "is_differenced": is_diff,     
        "last_AMR_value": last_val,
        "last_Indicator_value": last_indicator_val,
        "Permutation_p_val": p_perm,
        "Years": years_str,
        "Num Years": num_years,
        "First Year": first_year,
        "Last Year": last_year

    }


for class_name, anti_df in feature_rate_df.items():

    anti_df = anti_df.dropna(axis=0, how="any").reset_index(drop=True)
    anti_df = anti_df.loc[anti_df["Num Isolates"] > 5].reset_index(drop=True)

    valid_features = [
        col for col in anti_df.columns[3:]
        if col not in ["Country Code", "year", "Num Isolates"]
        and anti_df[col].nunique() > 1
    ]
    

    tasks = [
        (indicator, feat)
        for indicator in indicator_cols
        for feat in valid_features
    ]

    
    total_pairs = len(tasks)
    desc = f"Sheet '{class_name}' ({len(valid_features)} features × 96 indicators)"
    with tqdm_joblib(tqdm(total=total_pairs, desc=desc, unit="pair")):
        results = Parallel(n_jobs=-1)(
            delayed(process_pair)(indicator, feat, anti_df)
            for indicator, feat in tasks
    )


    results = [r for r in results if r is not None]

    if results:
        pd.DataFrame(results).to_excel(writer, sheet_name=class_name, index=False)

writer.close()

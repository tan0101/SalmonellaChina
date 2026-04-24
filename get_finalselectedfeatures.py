import numpy as np
import pandas as pd
import sys
from statsmodels.stats.multitest import multipletests
from pathlib import Path
import xlsxwriter
import re

import warnings
warnings.filterwarnings('ignore')


odds_ratio_threshold=2


def _extract_main_number(x: str) -> float:
    if pd.isna(x):
        return float('nan')
    s = str(x)
    s = s.split('±')[0]
    m = re.search(r'[-+]?\d*\.?\d+(?:[eE][-+]?\d+)?', s)
    return float(m.group(0)) if m else float('nan')



name_dataset=sys.argv[1]

results_folder="Results"
classes=pd.read_csv(f"Data/antibiotic_class.csv", index_col=None)

models = pd.read_csv(
    f"{results_folder}/{name_dataset}_performance_summary.csv",
    index_col=None,                             
    converters={"AUC": _extract_main_number}    
)

models = models.rename(columns={models.columns[0]: "Dataset"})

models = models.set_index("Dataset")

selected_models = models[models["AUC"] > 0.7]


features=pd.read_csv(f"{results_folder}/{name_dataset}_all_oddsratio_summaries.csv", index_col=None)
features["Dataset_clean"] = features["Dataset"].str.replace(r"_FULL.*$", "", regex=True)
features = features[features['Dataset_clean'].isin(selected_models.index)]
m = features["Dataset_clean"].str.extract(r'^(?P<Species>[^_]+)_(?P<Antibiotic>.+)$')



features["Species"] = m["Species"]
features["Antibiotic"] = m["Antibiotic"]
features=features[features['OR_95CI_Lower']>1]
features=features[features['Odds_Ratio']>odds_ratio_threshold]


print(features)


features = pd.merge(features, classes, on='Antibiotic', how='inner')
features = features[['Species', 'Antibiotic Class', 'Feature','Antibiotic']]
selected_features = features[~features.duplicated(keep=False)]

print("selected_features columns (repr):", [repr(c) for c in selected_features.columns])
print("Duplicate column labels:", selected_features.columns[selected_features.columns.duplicated()].tolist())

blk = selected_features[['Species', 'Antibiotic Class', 'Feature']]
print("Slice shape (should be n×3):", blk.shape)
print("Slice column labels (repr):", [repr(c) for c in blk.columns])

selected_features['model-feature'] = selected_features[['Species', 'Antibiotic Class','Feature']].agg('_'.join, axis=1)
if name_dataset == 'Enterococcus_faecium' or  name_dataset == 'Staphylococcus_aureus' or name_dataset == 'Klebsiella_pneumoniae' or name_dataset == 'Acinetobacter_baumannii' or name_dataset == 'Pseudomonas_aeruginosa':
    selected_features['WHO/ESKAPE'] = "Yes"
else:
    selected_features['WHO/ESKAPE'] = "No"

third_gen_cephalosporins = [
    "cefixime", "cefoperazone", "cefotaxime",
    "cefpodoxime", "ceftazidime", "ceftiofur", "ceftriaxone"
]
if name_dataset=="Escherichia_coli" or name_dataset=="Neisseria_gonorrhoeae":
    mask = selected_features["Antibiotic"].str.strip().str.lower().isin(third_gen_cephalosporins)
    selected_features.loc[mask, "WHO/ESKAPE"] = "Yes"
    selected_features.loc[selected_features["Antibiotic Class"].str.strip().str.lower() == "fluoroquinolone", "WHO/ESKAPE"] = "Yes"

if name_dataset=="Shigella_flexneri" or name_dataset=="Shigella_sonnei":
    selected_features.loc[selected_features["Antibiotic Class"].str.strip().str.lower() == "fluoroquinolone", "WHO/ESKAPE"] = "Yes"

 
directory = Path('./')
file=f"Results/LinearRegression/LinearRegression_Salmonella_permutationtest.xlsx"
newfile=f"Results/LinearRegression_{name_dataset}_permutationtestFDRcorrected.xlsx"
sheet_names = pd.ExcelFile(file).sheet_names
existing_data = pd.read_excel(file, sheet_name=None)
for sheet in sheet_names:
    df_linear = pd.read_excel(file, sheet_name=sheet)
    df_linear['model-feature'] = name_dataset + '_' + sheet + "_" + df_linear['Feature'].astype(str)
    df_linear = df_linear[df_linear["model-feature"].isin(selected_features['model-feature'])]
    
    if df_linear.empty:
            existing_data[sheet] = pd.DataFrame(columns=list(df_linear.columns) + ['fdr_corrected_p', 'significant'])
            continue
    rejected, pvals_corrected, _, _ = multipletests(df_linear['Permutation_p_val'], alpha=0.1, method='fdr_bh')
    df_linear['fdr_corrected_p'] = pvals_corrected
    df_linear['significant'] = rejected
       
    existing_data[sheet] = df_linear
        

linear_df = pd.concat([df.assign(sheet_name=sn) for sn, df in existing_data.items()],ignore_index=True)

print("linear_df rows before FDR pvalue selection:", len(linear_df))
linear_df = linear_df[linear_df["fdr_corrected_p"] <= 0.1]
linear_df = pd.merge(linear_df, selected_features, left_on='model-feature', right_on='model-feature', how='inner')


print("linear_df columns:", list(linear_df.columns))
print("Duplicate column names:", linear_df.columns[linear_df.columns.duplicated()].tolist())


print("linear_df rows after FDR and significant==True:", len(linear_df))
print("Any significant rows?", linear_df['significant'].sum() if 'significant' in linear_df else 'no-col')


linear_df['model-feature-indicator'] = linear_df[['model-feature', 'Indicator']].agg('_'.join, axis=1)
  
with pd.ExcelWriter(newfile, engine='xlsxwriter') as writer:
    for sheet_name, df in existing_data.items():
        df.to_excel(writer, sheet_name=sheet_name, index=False)
writer.close()

forecasting=pd.read_csv(f"Results/Trend_Analysis.csv")


forecasting=forecasting[forecasting['Specie']==name_dataset]
forecasting=forecasting[["Specie", "Antibiotic Class", "Indicator", "Feature", "Pearson r", "Pearson p-value", "Years", "Num Years","First Year", "Last Year", "Feature ADF Test statistic", "Feature ADF p-value", "Feature KPSS Test statistic", "Feature KPSS p-value","Feature Result 95%", "Feature Trend 95%",	"Indicator ADF Test statistic", "Indicator ADF p-value", "Indicator KPSS Test statistic","Indicator KPSS p-value", "Indicator Result 95%", "Indicator Trend 95%"]]

print("forecasting columns:", list(forecasting.columns))
print("Duplicate column names:", forecasting.columns[forecasting.columns.duplicated()].tolist())

for col in ['Specie','Antibiotic Class','Feature','Indicator']:
    print(col, "count:", (forecasting.columns == col).sum())

sel = forecasting[['Specie','Antibiotic Class','Feature','Indicator']]
print("Selected block shape:", sel.shape)
print("Selected block columns:", list(sel.columns))


forecasting['model-feature-indicator'] = forecasting[['Specie','Antibiotic Class','Feature', 'Indicator']].agg('_'.join, axis=1)

to_merge = linear_df[["model-feature-indicator", "WHO/ESKAPE"]]
forecasting = forecasting.merge(to_merge, on="model-feature-indicator", how="inner")

selected_forecasting = forecasting[forecasting['model-feature-indicator'].isin(linear_df['model-feature-indicator'])]
selected_forecasting["Indicator Result 95%"] = selected_forecasting["Indicator Result 95%"].str.replace('Constant', 'Stationary')

selected_forecasting = selected_forecasting[selected_forecasting["Feature Trend 95%"]=='Increasing']

selected_forecasting.to_csv(f"{results_folder}/{name_dataset}_final_increasing_features.csv")

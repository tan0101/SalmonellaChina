# SalmonellaChina

"High-risk AMR genomic traits in MDR Salmonella across China's food chain: machine learning-based surveillance and projections to 2050" by Michelle Baker, Chengchang Luo, Xin Lu, Alexandre Maciel-Guerra, Komkiew Pinpimai, Ge Wu, Ming Luo, Mengyu Wang, Yuan Zhang, Weihua Meng, Baoli Zhu, Biao Kan, Tania Dottorini
 
Any questions should be made to the corresponding author Dr Tania Dottorini (Tania.Dottorini@kcl.ac.uk)

Ten scripts are available:

1.	Machine Learning: GLMM_MLscript.py, get_MLperformancemetrics.py and get_odds_ratios.py
2.	Prevalence rates: get_prevalance_rates.py
3.	Indicator normalization: get_normalized_indicators.py
4.	Feature and Indicator Associations: get_linear_regression_analysis _withtimeseriescorrection.py
5.	Indicator Monte Carlo Simulations: get_indicator_forecasting_parameters.py and get_indicator_forecasting_montecarlo.py
6.	Feature Monte Carlo Simulations: get_feature_forecasting_montecarlo.py
7.	Trend Analysis: get_trend_forecast_analysis.py
8.	Final increasing features selection and FDR correction: get_final_selected_features.py

# System Requirements

## Software requirements

The project was developed using the Conda v25.9.1 environment.

### OS Requirements

This package is supported for Linux. The package has been tested on the following system:
*	Linux: Red Hat Enterprise Linux v8.6 (Ootpa)



### Python Dependencies

```
python v3.14.0
numpy v2.3.4   
pandas v2.3.3
scikit-learn v1.7.2
scipy v1.16.2
networkx v2.8.4
matplotlib v3.6.2
statsmodels 0.14.5
joblib v1.5.2
xlsxwriter v3.2.9
```

# Installation Guide:

## Install from Github
```
git clone https://github.com/tan0101/SalmonellaChina
cd SalmonellaChina
python setup.py install
```

This takes 1-2 min to build

# Instructions for use

After installing the project, unzip any zipped files due to their size and run each available code using python code_name.py plus any required inputs (see below). All codes will automatically import the corresponding data from the Data folder and will produce the following output:

*	GLMM_MLscript.py: input format ‘python GLMM_MLscript.py <dataset_name> <antibiotic_name>’. Produces multiple csv files containing performance metrics, odds ratios, coefficient values, random effects and variance components, individually over each of 50 trials and summarised. This takes up to 6 hours to run. 
*	get_MLperformancemetrics.py: input format ‘python get_MLperformancemetrics.py <dataset_name> ‘. Produces a summary csv of the performance metrics, mean and standard deviation for each antibiotic over 50 runs. This takes up to 5 minutes to run.
*	get_odds_ratios.py and getindividualoddsratios.py: Produces a summary of the odds ratios of the final fitted model, and a summary of the odds ratios values for each feature in the cross-validation. This takes up to 5 minutes to run.
*	get_antibiotic_class_RSI.py: receives as input the AMR phenotypes for individual antibiotics and generates a csv file with Resistant and Susceptible phenotypes based on the antibiotic classes. This takes 1 min to run
*	get_prevalance_rates.py: generate the feature prevalence rate and outputs it as a csv file. The prevalence rate for each genomic feature, feature prevalence rate is calculated as the number of resistant isolates carrying the feature divided by the total number of resistant and susceptible isolates. Notably, prevalence rate was calculated considering each antibiotic class rather than each individual antibiotic; that is, by aggregating isolates resistant and susceptible to all the antibiotics of the same class. This takes 1 min to run.
*	get_normalized_indicators.py: received the raw data of the indicators and uses a min-max normalization approach for each individual country, considering all the years for each indicator. Outputs a csv file with the normalized indicators. This takes 10 min to run.
*	get_linear_regression_analysis_withtimeseriescorrection.py: It received as input the feature prevalence rate and the indicators rate to calculate the Pearson coefficient and a linear regression between them. It outputs a csv file containing the Pearson coefficient and p-value, and the linear regression mean and standard deviation intercept and slope and its corresponding R-square. The linear regression is permuted 1000 times to generate a null distribution of p values for FDR correction. This takes up to 3 hours to run.
*	get_indicator_forecasting_parameters.py: It receives as input the pearson correlation results and performs a curve fitting, considering either a linear function or one of three non-linear one, the final choice of function being driven by the highest R2 result (and as long as >0.8). It outputs a csv file with the parameters of the selected curve fitting and its respective R-square value. This takes 10 min to run
*	get_indicator_forecasting_montecarlo.py: It receives as input the parameters created in the previous step. The best fit model for each indicator, defined by its parameter’s values and associated confidence intervals, was then used to run a Monte Carlo simulation (10,000 iterations) to predict the value of the indicator until 2050. The forecast itself is returned as the mean of 10,000 simulations and 5th and 95th percentiles. It outputs a csv file for each species-feature-antibiotic_class-indicator with the values obtained for each of the 10,000 simulations. This takes 2 hours to run
*	get_feature_forecasting_montecarlo.py: It receives as input the forecast obtained for the indicators and the linear regression parameters between the feature prevalence rate and the indicators rate. Again, a Monte Carlo simulation is run (10,000) to forecast the genomic prevalence rate associated to the indicator over the same years (up to 2050). This takes 2 hours to run
*	get_trend_forecast_analysis.py: It receives as input the forecasts for both the features and the indicators. For each pair comprising genome prevalence rate and indicator processed, both datasets expressed as time series were also tested for stationarity, using the ADF (Augmented Dickey-Fuller) and KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test statistics. The ADF tests rejects the hypothesis that the series is nonstationary while the KPSS test rejects the hypothesis that the series is stationary. It outputs a csv file containing the results for both tests (ADF and KPSS) and if time series is increasing or decreasing. This takes 1 hour to run.
*	get_finalselectedfeatures.py: This receives as input the GLMM results file and linear regression results file and selects forecasting results for the GLMM features passing the performance thresholds set (the features with a model performance  > 0.7 AUC, an odds ratio ≥ 2, and a lower 95% confidence interval ≥1), conducts FDR correction selecting only those with FDR ≤ 0.1 and outputs the remaining features that are increasing over time. This takes 10 min to 

# License

This project is covered under the **AGPL-3.0 license**.

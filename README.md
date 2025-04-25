# SalmonellaChina

"Machine Learning Insights into Genomic Traits and Risk Factors Driving AMR in Salmonella Across the Food Chain: Global Health Implications by 2050" by Michelle Baker, Chengchang Luo, Xin Lu, Alexandre Maciel-Guerra, Komkiew Pinpimai, Ming Luo, Mengyu Wang, Yuan Zhang, Baoli Zhu, Biao Kan, Tania Dottorini
 
Any questions should be made to the corresponding author Dr Tania Dottorini (Tania.Dottorini@nottingham.ac.uk)

Thirteen scripts are available:

1. Population Correction: get_weights.py and weighted_chi_square_AMR.py
2. Machine Learning: AMR_Pipeline.py
3. Feature rate: get_antibiotic_class_RSI.py and get_feature_rate_Population_correction.py
4. Feature and Indicator Associations: pearson_correlation_and_linear_regression_analysis.py
5. Indicator Monte Carlo Simulations: get_indicator_forecasting_parameters.py and indicator_forecasting_montecarlo.py
6. Feature Monte Carlo Simulations: feature_forecasting_montecarlo.py
7. Trend Analysis: trend_forecast_analysis.py

# System Requirements

## Software requirements

The project was developed using the Conda v23.1.0 environment.

### OS Requirements

This package is supported for Windows. The package has been tested on the following system: 

* Windows: Windows 11 Pro version 23H2 OS build 22631.3296 Windows Feature Experience Pack 1000.22687.1000.0 


### Python Dependencies

```
python v3.9.15
numpy v1.21.5
pandas v1.4.4
scikit-learn v1.2.1
scipy v1.15.2
networkx v2.8.4
matplotlib v3.6.2
imblearn v0.13.0
biopython v1.81
ete3 v3.1.2
```

# Installation Guide:

## Install from Github
```
git clone https://github.com/tan0101/AMRGlobal
cd AMRGlobal
python setup.py install
```

This takes 1-2 min to build

# Instructions for use

After installing the project, unzip any zipped files due to their size and run each available code using _python code_name.py_. All codes will automatically import the corresponding data from the **Data** or **Results** folders and will produce the following output:

* get_weights.py: file used for the species that need population structure correction. It receives as input the distance matrix file based on MASH distance and outputs a csv file containing the weight of each isolate. This takes between 10 min and 2 hours to run, depending on the number of isolates
* weighted_chi_square_AMR.py: file used for the species that need population structure correction. It receives as input the weights, AMR phenotypes and genomic features (ARGs, MGEs and PlasmidARGs) files and outputs a csv file containing the features that passed the population structure correction based on a weighted chi-square test. This takes 10 min to run
* AMR_Pipeline.py:  produce multiple csv files containing the value for each run and the mean and standard deviation over 30 runs of the following performance metrics: AUC, accuracy, sensitivity, specificity, Cohen's Kappa score and precision. It also saves the pre-processed data in a pickle format and the selected features in a csv format. This takes between 30 min and 2 hours to run, depending on the number of isolates
* get_antibiotic_class_RSI.py: receives as input for each species the AMR phenotypes for individual antibiotics and generates a csv file with Resistant and Susceptible phenotypes based on the antibiotic classes. This takes 1 min to run
* get_feature_rate_Population_correction.py: generate the feature prevalence rate and outputs it as a csv file. The prevalence rate for each genomic feature selected by ML was calculated as the number of resistant isolates carrying the feature divided by the total number of resistant and susceptible isolates. Notably, prevalence rate was calculated considering each antibiotic class rather than each individual antibiotic; that is, by aggregating isolates resistant and susceptible to all the antibiotics of the same class. This takes 1 min to run
* pearson_correlation_and_linear_regression_analysis.py: It received as input the feature prevalence rate and the indicators rate to calculate the Pearson coefficient and a linear regression between them. It outputs a csv file containing the Pearson coefficient and p-value, and the linear regression mean and standard deviation, intercept and slope and its corresponding R-square. This takes 10 min to run
* get_indicator_forecasting_parameters.py: It receives as input the Pearson correlation results and performs a curve fitting, considering either a linear function or a non-linear one, the final choice of function being driven by the highest R2 result (and as long as >0.8). It outputs a csv file with the parameters of the selected curve fitting and its respective R-square value. This takes 10 min to run
* indicator_forecasting_montecarlo.py: It receives as input the parameters created in the previous step. The best fit model for each indicator, defined by its parameter’s values and associated confidence intervals, was then used to run a Monte Carlo simulation (10,000 iterations) to predict the value of the indicator until 2050. The forecast itself is returned as the mean of 10,000 simulations and 5th and 95th percentiles. It outputs a csv file for each feature-antibitioc_class-indicator with the values obtained for each of the 10,000 simulations. This takes 2 hours to run
* feature_forecasting_montecarlo.py: It receives as input the forecast obtained for the indicators and the linear regression parameters between the feature prevalence rate and the indicators rate. Again a Monte Carlo simulation is run (10,000) to forecast the genomic prevalence rate associated with the indicator over the same years (up to 2050). This takes 2 hours to run
* trend_forecast_analysis.py: It receives as input the forecasts for both the features and the indicators. For each pair comprising genome prevalence rate and indicator processed, both datasets expressed as time series were also tested for stationarity, using the ADF (Augmented Dickey-Fuller) and KPSS (Kwiatkowski-Phillips-Schmidt-Shin) test statistics. The ADF tests rejects the hypothesis that the series is nonstationary while the KPSS test rejects the hypothesis that the series is stationary. It outputs a csv file containing the results for both tests (ADF and KPSS) and if time series is increasing or decreasing. This takes 20 min to run
    

# Algorithm's Flow (ML_AMR_pipeline.py and ML_AMR_pipeline_Population_correction.py)

# License

This project is covered under the **AGPL-3.0 license**.

# Dynamic Duo

A data science project on the US cross border entry dataset from 1996 to 2019 that aims to:

- brainstorm explorative feature ideas and conduct feature engineering
- predict the border cross entry value using regression models
- apply time series analysis on the border crossing seasonal patterns
- detect anomalies in the dataset for case study and for other components
- do visualization and storytelling in various parts for data comprehension and reasoning

# Dataset used:

- US border-crossing entry dataset from 1996 to 2019 (from Kaggle)
- US historical average temperature by state from 1996 to 2019 (from noaa.gov)
- US quarterly GDP data by state from 2005 to 2019 (from bea.gov)

Border-crossing entry dataset overview

| Port Name  | State  | Port Code  | Border | Date  | Measure | Value  | Location |
| ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- | ------------- |
| Calexico East  | California  | 2507  | US-Mexico Border  | 03/01/2019 12:00:00 AM  | Trucks  | 34447  | POINT (-115.48433000000001 32.67524)  |
| Van Buren  | Maine  | 108  | US-Canada Border  | 03/01/2019 12:00:00 AM  | Rail Containers Full  | 428  | POINT (-67.94271 47.16207)  |
| Otay Mesa  | California  | 2506  | US-Mexico Border  | 03/01/2019 12:00:00 AM  | Trucks  | 81217  | POINT (-117.05333 32.57333)  |
| Trout River  | New York  | 715  | US-Canada Border  | 03/01/2019 12:00:00 AM  | Personal Vehicle Passengers  | 16377  | POINT (-73.44253 44.990010000000005)  |

# Dataset Visualization

This line graph displays the overall trend of traffic, ignoring port and transport information.

![total_trend](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/visualization/total_trend.png)

This graph does the same, but graphs each year on top each other in order to show the common yearly cycles.

![yearly_trend](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/visualization/yearly_trend.png)

Here, we graph each port separately.

![by_port](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/visualization/by_port.png)

Now, we graph each transport separately.

![by_transport](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/visualization/by_transport.png)

Finally, we attempt to graph both port and transport. Note each row has a share y-axis scale.

![by_both](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/visualization/by_both.png)

See the relevant notebook for more discussion on the graphs, and more visualizations of the data.

# Regression Analysis

1. Random Forest
- Parameters:
  - Trees per forest: 100
  - Max depth: 100
- Results:
  - R^2 value: 0.9893
  - Mean absolute error: 1875
  - Median absolute error: 15.1

2. XGBoost
- Parameters:
  - Trees per forest: 10
  - Max depth: 100
  - Learning rate: 0.3
  - Number of rounds: 10
- Results:
  - Mean absolute error: 2022
  - Median absolute error: 13.23

3. Neural Network
- Parameters:
  - Hidden layers: 6
  - Loss function: mean absolute error
  - Optimizer: Adam
- Results:
  - Mean absolute error: 4841
  - Median absolute error: 376.6


# Anomaly Detection

Applied and compared 2 anomaly detection algorithms: isolation forest and one class SVM

1. Isolation Forest, with 5% contamination, 200 estimators

detected anomalies on different subsets:

![anml_by_IF](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/breakdown_CA_IF_1.png)

visualize using t-SNE:

![IF_tSNE](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/tSNE_2d_CA_IF.png)

2. One Class SVM, with 5% contamination, RBF kernel

monthly distribution of anomalies:

![OCS_monthly](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/hist_monthly_value_us_mx_OCS.png)

3D interactive comparison with isolation forest:

[Click Me](https://htmlpreview.github.io/?https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/3D_comparison_US_MX.html)



# Time Series Analysis

1. Decomposition. Shows trend, seasonality and residual of various breakdown subsets.

![decomp_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/decomposition_alaska.png)
![decomp_trucks](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/decomposition_trucks.png)

2. ARIMA

- plot rolling mean & std and utilize AD-Fuller test for confirming stationarity.

![arima_rolling_mean_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/rolling_mean_std_alaska.png)

- determine p, d, q using autocorrelation & partial autocorrelation.

![ac_partial_ac_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/alaska_autocorr_partial_ac.png)

- verify hyperparameters using grid search

3. SARIMAX. Since a strong seasonal pattern has been observed, we used SARIMAX here provided by statsmodel.

- train SARIMAX to fit seasonal patterns

![sarimax_fit_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_fit_alaska.png)

- SARIMAX model fitting diagnostics

![sarimax_diagn_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_diagnostics_alaska.png)

- SARIMAX prediction (on Alaska subset)

![sarimax_predict_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_predict_alaska.png)

- SARIMAX prediction (on US-Canada border subset)

![sarimax_predict_us_ca](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_predict_us_ca.png)

4. Time series reasoning. Try to do case study and reasoning for time series patterns.

- strong correlation between border-crossing behavior with climate in northern states

![corr_minnesota_climate](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/compare_value_climate_minnesota.png)
![corr_miane_climate](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/compare_value_climate_maine.png)

- weak correlation between border-crossing behavior with climate in other states

![corr_arizona_climate](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/compare_value_climate_arizona.png)

- weak correlation between border-crossing behavior with GDP

![corr_NY_GDP](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/compare_value_gdp_NY.png)
![corr_trucks_GDP](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/compare_value_gdp_trucks.png)

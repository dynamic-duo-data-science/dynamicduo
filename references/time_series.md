# Time Series Analysis

1. Try decomposition, to display the decomposed trend, seasonality and residual patterns of various breakdown subsets.

![decomp_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/decomposition_alaska.png)
![decomp_trucks](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/decomposition_trucks.png)
![decomp_us_mx](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/decomposition_us_mx.png)


2. Use ARIMA for trend analysis & forecast.

- plot rolling mean & std and utilize AD-Fuller test for confirming stationarity. Both rolling mean & std charts and AD-Fuller tests indicates a stationary trend over time, so that differencing is not needed here for Alaska subset.

![arima_rolling_mean_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/rolling_mean_std_alaska.png)

- determine p, d, q using autocorrelation & partial autocorrelation.

![ac_partial_ac_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/alaska_autocorr_partial_ac.png)

Both ACF & PACF display a cut off, so we add parameters for AR and MA models. The subsequent p, d, q is set to 1, 0, 1 respectively

- verify hyperparameters using grid search. We self-implemented a grid search function to search for best parameters for ARIMA and SARIMA using SARIMAX and AIC criteria. The best parameter pairs corresponds with the theoretic one.

3. Use SARIMAX for seasonality analysis. Since a strong seasonal pattern has been observed, we used SARIMAX here provided by statsmodel.

- train SARIMAX to fit seasonal patterns. It shows the SARIMAX fits quite well on different breakdown subsets.

![sarimax_fit_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_fit_alaska.png)
![sarimax_fit_texas](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_fit_texas.png)
![sarimax_fit_us_ca](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_fit_us_ca.png)

- SARIMAX model fitting diagnostics

![sarimax_diagn_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_diagnostics_alaska.png)

The residuals are just oscillating around zero and conform to a normal distribution with a mean of zero, which is a strong indication of a good model fit.

- SARIMAX is used to predict the 15% data in the future for various breakdown subsets. The predictions by SARIMAX looks good, and the gray areas show the confidence interval. Below is for Alaska subset.

![sarimax_predict_AL](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_predict_alaska.png)


- For US-Canada border subset

![sarimax_predict_us_ca](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_predict_us_ca.png)

- For Texas border subset

![sarimax_predict_texas](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/SARIMAX_predict_texas.png)

4. Time series reasoning. Try to do case study and reasoning for time series patterns.

- strong correlation between border-crossing behavior with climate in northern states

![corr_minnesota_climate](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/compare_value_climate_minnesota.png)
![corr_miane_climate](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/compare_value_climate_maine.png)

- weak correlation between border-crossing behavior with climate in other states

![corr_arizona_climate](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/compare_value_climate_arizona.png)

- weak correlation between border-crossing behavior with GDP

![corr_NY_GDP](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/compare_value_gdp_NY.png)
![corr_trucks_GDP](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/time_series/compare_value_gdp_trucks.png)




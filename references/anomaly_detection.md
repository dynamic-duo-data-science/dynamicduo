# Anomaly Detection

Conduct anomaly detection to remove anomalies for other researches such as regression and time series, as well as to try case study and storytelling on the anomaly patterns. Applied and compared 2 anomaly detection algorithms: isolation forest and one class SVM

1. Built Isolation Forest models on various category breakdown subsets, with 5% contamination, 200 estimators

Anomaly samples of value column on different subsets by isolation forest:

![CA_by_IF_1](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/breakdown_CA_IF_1.png)
![CA_by_IF_2](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/breakdown_CA_IF_2.png)
![CA_by_IF_3](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/breakdown_CA_IF_3.png)

Try to visualize the anomalies generated. Since they're generated on the entire feature space, we need dimensionality reduction. First use PCA, and below is the explained variance chart on Alaska subset.
![PCA_explained](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/PCA_explained_variance.png)

Try to visualize 2d/3d patterns by PCA. The patterns does not look so organized since only 65% of the variance are preserved.
![PCA_2d](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/PCA_2d_CA_IF.png)
![PCA_3d](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/PCA_3d_CA_IF.png)

Try t-SNE to improve dimension reduction and visualization.
![tSNE_2d](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/tSNE_2d_CA_IF.png)
![tSNE-3d](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/tSNE_3d_entire_IF.png)

[Click Me for 3d interactive by t-SNE](http://htmlpreview.github.io/?https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/California_subset_by_Isolation_forest_3D.html)

Plot stacked chart for quarterly value distribution on anomalies:
![quarterly_distr_by_IF](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/hist_quarterly_value_us_mx_IF.png)

Plot histograms of various breakdowns by isolation forest:
![hist_measure_IF](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/hist_measure_CA_IF.png)
![hist_port_IF](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/hist_port_CA_IF.png)


2. Built One Class SVM models on various category breakdown subsets, with 5% contamination, RBF kernel

Anomaly samples of value column on different subsets by one class SVM:

![CA_by_OCS_1](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/breakdown_us_mx_OCS_1.png)
![CA_by_OCS_2](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/breakdown_us_mx_OCS_2.png)
![CA_by_OCS_3](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/breakdown_us_mx_OCS_3.png)


monthly distribution of anomalies:

![OCS_monthly](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/hist_monthly_value_us_mx_OCS.png)

Try t-SNE for 2d/3d visualization of anomalies by one class SVM:
![tSNE_2d_ocs](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/tSNE_2d_CA_OCS.png)
![tSNE_3d_ocs](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/tSNE_3d_entire_OCS.png)


3D interactive comparison with isolation forest:

[Click Me](https://htmlpreview.github.io/?https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/3D_comparison_US_MX.html)

Plot stacked chart for quarterly and monthly value distribution on anomalies:
![monthly_distr_by_IF](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/hist_monthly_value_us_mx_OCS.png)
![quarterly_distr_by_IF](https://github.com/dynamic-duo-data-science/dynamicduo/blob/master/reports/figures/anomaly_detection/hist_quarterly_value_us_mx_OCS.png)

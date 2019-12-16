import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.ensemble import IsolationForest
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from plotly.offline import download_plotlyjs, init_notebook_mode, iplot
import seaborn as sns


NORMAL_COLOR = '#9edae6'
NORMAL_COLOR_RGB = 'rgb(158,218,230)'
df = pd.read_csv("../data/processed/processed.csv", index_col=0)
df['Date'] = pd.to_datetime(df['Date'])
df['day_of_the_week'] = df['Date'].dt.dayofweek
df['is_week_day'] = (df['day_of_the_week'] < 5).astype(int)

df_alaska = df[(df['State'] == 'Alaska')]
df_texas = df[(df['State'] == 'Texas')]
df_CA = df[(df['State'] == 'California')]
df_maine = df[(df['State'] == 'Maine')]
df_us_mx = df[(df['Border'] == 'US-Mexico Border')]

def transfer_column(df_input):
    num_features = ['agg_month', 'day_of_the_week', 'Value', 'year', 'month']
    ctg_features = ['Port Name', 'State', 'Border', 'Measure', 'is_week_day']
    num_pipeline = Pipeline([
        ('imputer', SimpleImputer(strategy="median")),
        ('std_scaler', StandardScaler()),
    ])
    full_pipeline = ColumnTransformer([
        ("num", num_pipeline, num_features),
        ("cat", OneHotEncoder(), ctg_features)
    ])
    np_pipelined = full_pipeline.fit_transform(df_input)
    import scipy
    if isinstance(np_pipelined, scipy.sparse.csr.csr_matrix):
        np_pipelined = np_pipelined.toarray()
    df_pipelined = pd.DataFrame(np_pipelined, index=df_input.index)
    return df_pipelined


def train_isolation_forest(df_input, outliers_fraction=0.05):
    df_input_pipelined = transfer_column(df_input)
    df_input_pipelined = pd.DataFrame(df_input_pipelined)
    IF_model = IsolationForest(contamination=outliers_fraction, n_estimators=200)
    IF_model.fit(df_input_pipelined)
    print "Isolation forest trained, making predictions..."
    df_input_pipelined['is_anomaly'] = pd.Series(IF_model.predict(df_input_pipelined), index=df_input_pipelined.index)
    return df_input_pipelined

df_with_prediction = train_isolation_forest(df)
pca = PCA(n_components=20)
X2D = pca.fit_transform(transfer_column(df))

# print "pca.components_:{}".format(pca.components_)
print "pca.explained_variance_ratio_:{}".format(pca.explained_variance_ratio_)
#Plotting the Cumulative Summation of the Explained Variance
plt.figure()
plt.plot(np.cumsum(pca.explained_variance_ratio_))
plt.xlabel('Number of Components')
plt.ylabel('Explained variance ratio')
plt.title('Explained Variance')
plt.xticks(np.arange(0, 20, 1))
plt.show()


def plot_PCA_2d(df_input, df_input_with_prediction):
    pca_2d = PCA(n_components=2)
    X2D = pca_2d.fit_transform(transfer_column(df_input))
    X2D_df = pd.DataFrame(X2D, index=df_input.index)
    X2D_df['is_anomaly'] = df_input_with_prediction['is_anomaly']

    plt.figure(figsize=(8, 8))
    X2D_df_normal = X2D_df[X2D_df['is_anomaly'] == 1]
    X2D_df_anomaly = X2D_df[X2D_df['is_anomaly'] == -1]

    sns.set_style("whitegrid")
    plt.scatter(X2D_df_anomaly[0], X2D_df_anomaly[1], c='red', edgecolor='', marker='o', alpha=0.5, s=5)
    plt.scatter(X2D_df_normal[0], X2D_df_normal[1], c='#9edae6', edgecolor='', marker='o', alpha=0.5, s=5)


plot_PCA_2d(df_CA, df_CA_with_prediction)


def plot_pca_3d(df_input, title=None, need_transform=True, size=5):
    pca_3d = PCA(n_components=3)
    aa = df_input
    if need_transform:
        aa = transfer_column(df_input)
    print "Input transformation finished"
    X3D = pca_3d.fit_transform(aa)
    print "PCA model trained"

    X3D_df = pd.DataFrame(X3D, index=df_input.index)
    X3D_df['is_anomaly'] = df_input['is_anomaly']

    from mpl_toolkits.mplot3d import axes3d, Axes3D
    from matplotlib import cm
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig, azim=-60)
    X3D_df_normal = X3D_df[X3D_df['is_anomaly'] == 1]
    X3D_df_anomaly = X3D_df[X3D_df['is_anomaly'] == -1]

    # ax = fig.add_subplot(111, projection='3d')
    ax.scatter(X3D_df_anomaly[0], X3D_df_anomaly[1], X3D_df_anomaly[2], c='red', cmap=cm.coolwarm, edgecolor='', s=size)
    ax.scatter(X3D_df_normal[0], X3D_df_normal[1], X3D_df_normal[2], c='#9edae6', cmap=cm.coolwarm, edgecolor='', s=size)

    xAxisLine = ((min(X3D_df[0]), max(X3D_df[0])), (0, 0), (0, 0))
    ax.plot(xAxisLine[0], xAxisLine[1], xAxisLine[2], 'orange')
    yAxisLine = ((0, 0), (min(X3D_df[1]), max(X3D_df[1])), (0, 0))
    ax.plot(yAxisLine[0], yAxisLine[1], yAxisLine[2], 'orange')
    zAxisLine = ((0, 0), (0, 0), (min(X3D_df[2]), max(X3D_df[2])))
    ax.plot(zAxisLine[0], zAxisLine[1], zAxisLine[2], 'orange')

    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_zlabel("PC3")
    plt.gca().patch.set_facecolor('white')
    ax.set_title("PCA on the {} subset".format(title))


plot_pca_3d(df_with_prediction, need_transform=False)


def plot_anomaly_using_tsne(df_input, df_with_prediction, tsne_vectors=None, size=None, anomaly_marker=-1):
    print "Plotting predictions using tsne..."
    from sklearn.manifold import TSNE
    df_input_pipelined = transfer_column(df_input)
    if df_input_pipelined.shape[1] == 1:
        df_input_pipelined = df_input_pipelined
    if tsne_vectors is None:
        tsne_model = TSNE(n_components=dimension, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
        tsne_vectors = tsne_model.fit_transform(df_input_pipelined)

    df_tsne_vectors = pd.DataFrame(tsne_vectors)
    df_tsne_vectors['is_anomaly'] = df_with_prediction['is_anomaly']
    df_tsne_vectors['avg_temperature'] = df_input['avg_temperature']
    df_tsne_vectors['GDP'] = df_input['GDP']

    anomalies_indices = df_with_prediction[df_with_prediction['is_anomaly'] == anomaly_marker].index
    tsne_vectors_anomaly = [v for idx, v in enumerate(tsne_vectors) if idx in anomalies_indices]
    fig, ax = plt.subplots(figsize=(10, 10))
    COLOR_MAP = np.array(["#9edae6"])
    plt.scatter(x=tsne_vectors[:, 0], y=tsne_vectors[:, 1], c=df_tsne_vectors['avg_temperature'], marker='o', alpha=0.5, s=size)
    # plt.scatter(x=tsne_vectors[:, 0], y=tsne_vectors[:, 1], color=COLOR_MAP, marker='o', alpha=0.5, s=size)
    plt.scatter(x=[v[0] for v in tsne_vectors_anomaly], y=[v[1] for v in tsne_vectors_anomaly],
                color='red', marker='o', alpha=0.5, s=size)
    plt.show()


is_anomaly = df_with_prediction.is_anomaly.map({1:0, -1:1})
df['is_anomaly'] = is_anomaly
plot_anomaly_using_tsne(df, df, anomaly_marker=1)

df_CA_with_prediction_IF = train_isolation_forest(df_CA)
plot_pca_3d(df_CA_with_prediction_IF, title="CA", size=10, need_transform=False)

plot_anomaly_using_tsne(df_CA, df_CA_with_prediction)


def plot_anomaly_using_3d_tSNE(df_input, df_with_prediction, trained_tSNE_vectors=None, size=None, anomaly_marker=-1, title=None, azim=-60, elev=30):
    print "Plotting predictions using tsne..."
    df_input_pipelined = transfer_column(df_input)
    if df_input_pipelined.shape[1] == 1:
        df_input_pipelined = df_input_pipelined
    print "Input file transformation finished"
    if trained_tSNE_vectors is not None:
        tsne_vectors = trained_tSNE_vectors
    else:
        tsne_model = TSNE(n_components=3, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
        tsne_vectors = tsne_model.fit_transform(df_input_pipelined)
    print "t-SNE model trained"
    anomalies_indices = df_with_prediction[df_with_prediction['is_anomaly'] == anomaly_marker].index
    tsne_vectors_anomaly = [v for idx, v in enumerate(tsne_vectors) if idx in anomalies_indices]
    # fig, ax = plt.subplots(figsize=(10, 10))
    COLOR_MAP = np.array(["#9edae6"])
    print "anomalies differentiated"

    from mpl_toolkits.mplot3d import axes3d, Axes3D
    from matplotlib import cm
    fig = plt.figure(figsize=(10, 10))
    ax = Axes3D(fig, azim=azim, elev=elev)

    ax.scatter(tsne_vectors[:, 0], tsne_vectors[:, 1], tsne_vectors[:, 2], c='#9edae6', edgecolor='', s=10)
    ax.scatter([v[0] for v in tsne_vectors_anomaly], [v[1] for v in tsne_vectors_anomaly], [v[2] for v in tsne_vectors_anomaly], c='red', edgecolor='red', linewidths=3, s=10)

    ax.set_xlabel("t-SNE1")
    ax.set_ylabel("t-SNE2")
    ax.set_zlabel("t-SNE3")
    ax_title = "entire dataset"
    if title is not None:
        ax_title = "{} subset".format(title)
    ax.set_title("t-SNE on the {}".format(ax_title))

    # plt.scatter(x=tsne_vectors[:, 0], y=tsne_vectors[:, 1], color=COLOR_MAP, marker='o', alpha=0.5, s=size)
    # plt.scatter(x=[v[0] for v in tsne_vectors_anomaly], y=[v[1] for v in tsne_vectors_anomaly],
    #            color='red', marker='o', alpha=0.5, s=size)
    plt.show()


plot_anomaly_using_3d_tSNE(df_CA, df_CA_with_prediction, trained_tSNE_vectors=tsne_vectors)


def plot_3d_interactive(df_with_prediction, tsne_vectors, anomaly_marker=-1, normal_color=NORMAL_COLOR_RGB, anomaly_color='red', title=None):
    anomalies_indices = df_with_prediction[df_with_prediction['is_anomaly'] == anomaly_marker].index
    tsne_vectors_anomaly = [v for idx, v in enumerate(tsne_vectors) if idx in anomalies_indices]
    points = pd.DataFrame(tsne_vectors)
    df_tsne_vectors = pd.DataFrame(tsne_vectors, index=df_with_prediction.index)
    df_tsne_vectors['is_anomaly'] = df_with_prediction['is_anomaly']

    from plotly.graph_objs import *
    init_notebook_mode()
    trace0 = Scatter3d(x=df_tsne_vectors[0], y=df_tsne_vectors[1], z=df_tsne_vectors[2], mode='markers',
                       marker=dict(
                           size=2,
                           color=df_tsne_vectors['is_anomaly'],
                           colorscale=[[0, anomaly_color], [0.5, anomaly_color], [1.0, normal_color]],
                           symbol='circle',
                           opacity=0.5
                       )
                       )
    data = [trace0]
    layout = Layout(showlegend=False, height=600, width=600)
    fig = dict(data=data, layout=layout)
    plot_title = "3D plot by plotly"
    if title:
        plot_title = "_".join(title.split(" "))
    import plotly
    plotly.offline.plot(fig, filename='../reports/figures/anomaly_detection/{}_3D.html'.format(plot_title))
    iplot(fig)
    # fig.show()

df_CA_transformed = transfer_column(df_CA)
tsne_model = TSNE(n_components=3, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
tsne_vectors_CA = tsne_model.fit_transform(df_CA_transformed)
plot_3d_interactive(df_CA_with_prediction, tsne_vectors_CA, normal_color=NORMAL_COLOR, title="California subset by Isolation forest")


def plot_anomaly_hist(df_with_anomaly, title, col='Value', rotation=0, height=10, width=20, count=-1):
    normal = df_with_anomaly.loc[df_with_anomaly['is_anomaly'] == 1, col]
    anomaly = df_with_anomaly.loc[df_with_anomaly['is_anomaly'] == -1, col]

    if count > 0:
        normal = normal.head(count)
        anomaly = anomaly.head(count)

    plt.figure(figsize=(width, height))
    plt.hist([normal, anomaly], bins=32, stacked=True, color=[NORMAL_COLOR, 'red'], label=['normal', 'anomaly'])
    plt.legend()
    plt.xticks(rotation=rotation)
    plt.title("Distribution histogram of {} anomalies".format(title))
    plt.show()


plot_anomaly_hist(df_CA_copy, "measure column", col='Measure', rotation=75, height=4, width=9)
plot_anomaly_hist(df_CA_copy, "Port Name", col='Port Name', rotation=0, height=4, width=7)
plot_anomaly_hist(df_copy, col='Measure', rotation=45)
df_copy = df
df_copy['is_anomaly'] = df_with_prediction['is_anomaly']
plot_anomaly_hist(df_copy, col='Port Name', rotation=45)

def plot_anomalies_breakdown(df_anomaly, model_func, n_deep_dive_sample=3):
    print "Anomaly samples deep dive"
    from itertools import islice
    df_original_partial = df[df.index.isin(df_anomaly.index)]
    sorted_uniques = df_original_partial.groupby(['Port Name', 'Measure']).size().sort_values(ascending=False)
    print list(islice(sorted_uniques.iteritems(), 7))
    for i in range(n_deep_dive_sample):
        port_name, measure = list(islice(sorted_uniques.iteritems(), n_deep_dive_sample))[i][0]
        df_original_partial = df[(df['Port Name'] == port_name) & (df['Measure'] == measure)]
        df_with_prediction = model_func(df_original_partial)
        anml_sample_indices = df_with_prediction[df_with_prediction['is_anomaly'] == -1].index.to_list()

        plt.figure(figsize=(12, 5), dpi=100)
        plt.title("Anomalies in subset (port: {}, measure: {})".format(port_name, measure))
        plt.plot(df_original_partial['Date'], df_original_partial['Value'])
        for index in anml_sample_indices:
            plt.scatter(df_original_partial.loc[index, :]['Date'], df_original_partial.loc[index, :]['Value'], color='red')


plot_anomalies_breakdown(df_CA_with_prediction, train_isolation_forest)

df_us_mx_with_prediction_IF = train_isolation_forest(df_us_mx)
plot_anomaly_using_tsne(df_us_mx, df_us_mx_with_prediction_IF, size=5)

df_us_mx_transformed = transfer_column(df_us_mx)
tsne_model = TSNE(n_components=3, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
tsne_vectors_us_mx = tsne_model.fit_transform(df_us_mx_transformed)

plot_3d_interactive(df_us_mx_with_prediction_IF, tsne_vectors_us_mx, title="US-Mexico border subset by Isolation forest")

plot_value_anomaly_groupby_date(df_us_mx, df_us_mx_with_prediction_IF, 'US-Mexico border', freq='Q')

df_us_mx_copy = df_us_mx
df_us_mx_copy['is_anomaly'] = df_with_prediction['is_anomaly']
plot_amly_distr_by_month(df_us_mx, df_us_mx_with_prediction_IF, 'US-Mexico border')


def train_one_class_SVM(df_input, outliers_fraction=0.05, need_transform=True):
    from sklearn.svm import OneClassSVM
    df_input_pipelined = df_input
    if need_transform:
        df_input_pipelined = transfer_column(df_input)
        df_input_pipelined = pd.DataFrame(df_input_pipelined)

    OCS_model = OneClassSVM(nu=0.95 * outliers_fraction)
    OCS_model.fit(df_input_pipelined)

    print "One Class SVM trained, making predictions..."
    df_input_pipelined['is_anomaly'] = pd.Series(OCS_model.predict(df_input_pipelined), index=df_input_pipelined.index)
    print "Predictions made."
    return df_input_pipelined


df_CA_with_prediction = train_one_class_SVM(df_CA)
plot_anomaly_using_tsne(df_CA, df_CA_with_prediction)

anomalies_us_mx_indices = df_with_prediction_CA[df_with_prediction_CA['is_anomaly'] == -1].index
df_anomaly_us_mx = df_us_mx[df_us_mx.index.isin(anomalies_us_mx_indices)]
df_anomaly_us_mx.to_csv("../data/processed/anomalies_CA_by_OCS.csv")
plot_anomalies_breakdown(df_anomaly_us_mx, train_one_class_SVM)

df_us_mx_with_prediction_OCS = train_one_class_SVM(df_us_mx)
plot_anomaly_using_tsne(df_us_mx, df_us_mx_with_prediction_OCS, size=5)

plot_anomalies_breakdown(df_us_mx_with_prediction_OCS, train_one_class_SVM)


def plot_value_anomaly_groupby_date(df_input, df_input_with_prediction, title, freq='Q'):
    df_copy = df_input.copy()
    df_copy['is_anomaly'] = df_input_with_prediction['is_anomaly'].map({1: 0, -1: 1})

    anomaly_counts = df_copy.groupby(pd.PeriodIndex(df_copy['Date'], freq=freq))['is_anomaly'].sum()
    normal_counts = df_copy.groupby(pd.PeriodIndex(df_copy['Date'], freq=freq))['Value'].count()
    normal_list = normal_counts.values - anomaly_counts.values
    plt.figure(figsize=(20, 8))

    sns.set()
    sns.barplot(x=normal_counts.index, y=normal_list, color=NORMAL_COLOR)
    sns.barplot(x=anomaly_counts.index, y=anomaly_counts.values, color='red', bottom=normal_list)

    plt.xticks(rotation=90)
    plt.title("Quarterly anomaly distribution on {} subset".format(title))
    plt.show()


plot_value_anomaly_groupby_date(df_us_mx, df_us_mx_with_prediction_OCS, 'US-Mexico border', freq='Q')


def plot_amly_distr_by_month(df_input, df_input_with_prediction, title):
    df_copy = df_input.copy()
    df_copy['is_anomaly'] = df_input_with_prediction['is_anomaly'].map({1: 0, -1: 1})

    anomaly_counts = df_copy.groupby(['month'])['is_anomaly'].sum()
    normal_counts = df_copy.groupby(['month'])['Value'].count()
    normal_list = normal_counts.values - anomaly_counts.values
    plt.figure(figsize=(16, 8))

    sns.set()
    sns.barplot(x=normal_counts.index, y=normal_list, color=NORMAL_COLOR)
    sns.barplot(x=anomaly_counts.index, y=anomaly_counts.values, color='red', bottom=normal_list)

    plt.title("Anomaly distribution by month on {} subset".format(title))
    plt.show()


plot_amly_distr_by_month(df_us_mx, df_us_mx_with_prediction_OCS, 'US-Mexico border')

df_us_mx_transformed = transfer_column(df_us_mx)
tsne_model_us_mx = TSNE(n_components=3, perplexity=50, learning_rate=100, n_iter=2000, verbose=0, random_state=0, angle=0.75)
tsne_vectors_us_mx = tsne_model_us_mx.fit_transform(df_us_mx_transformed)


plot_anomaly_using_3d_tSNE(df_us_mx, df_us_mx_with_prediction_OCS, trained_tSNE_vectors=tsne_vectors_us_mx)

plot_3d_interactive(df_us_mx_with_prediction_OCS, tsne_vectors_us_mx, title="US-Mexico border subset by One Class SVM")


def plot_anomalies_with_other_features(df_input, df_input_with_prediction, feature_name, df_title):
    anml_indices = df_input_with_prediction[df_input_with_prediction['is_anomaly'] == -1].index.to_list()

    plt.figure(figsize=(20, 5), dpi=100)
    sns.set()
    plt.title("Anomalies in {} subset comparing with {}".format(df_title, feature_name))
    sns.lineplot(x=df_input['Date'], y=df_input['Value'], estimator=None)
    df_anml = df_input[df_input.index.isin(anml_indices)]
    sns.scatterplot(x=df_anml['Date'], y=df_anml['Value'], color='red', s=20)


df_CA_with_prediction_IF = train_isolation_forest(df_CA, outliers_fraction=0.01)
plot_anomalies_with_other_features(df_CA, df_CA_with_prediction_IF, "avg_temperature", 'California')





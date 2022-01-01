from math import ceil, log
from numpy.core.fromnumeric import argmax
import pandas as pd
import numpy as np
from scipy.fftpack import fft
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans, DBSCAN


# calculates the squared sum error of a cluster
def calc_sse(cluster_data, cluster_center):
    norms = [np.linalg.norm(data - cluster_center) for data in cluster_data]
    return np.sum(np.square(norms))


# calculates the entropy of a cluster
def calc_entropy(cluster_data):
    total_data_points = np.sum(cluster_data)
    return sum([-1*(d/total_data_points)*log(d/total_data_points, 2) if d != 0 else 0 for d in cluster_data])


# calculates the purity of a cluster
def calc_purity(cluster_data):
    total_data_points = np.sum(cluster_data)
    return max([d/total_data_points for d in cluster_data])


# get the clusters from dataset and corresponding cluster labels
def get_clusters(labels, data):

    no_of_clusters = len(set(labels)) if -1 not in labels else len(set(labels))-1

    # clusters[i] --> contains samples belonging to cluster i
    clusters = []
    for i in range(no_of_clusters):
        clusters.append(pd.DataFrame())

    for i in range(len(labels)):
        if labels[i] != -1:
            clusters[labels[i]] = clusters[labels[i]].append(data.iloc[i])
    
    return clusters


# split clusters obtained from dbscan into required number of clusters
def split_clusters(dbscan, meal_data, required_clusters):

    clusters = get_clusters(dbscan.labels_, meal_data)

    while(len(clusters) < required_clusters):
        cluster_centers = [cluster.mean(axis=0) for cluster in clusters]
        # get the sse for each cluster
        clusters_sse = [calc_sse(clusters[i].iloc[:, :-1].to_numpy(), cluster_centers[i][:-1].to_numpy()) for i in range(len(clusters))]
        # split cluster with max sse using bisecting kMeans
        index = argmax(clusters_sse)
        to_split_cluster = clusters[index]
        kMeans = KMeans(n_clusters=2).fit(to_split_cluster.iloc[:, :-1])
        new_clusters = get_clusters(kMeans.labels_, to_split_cluster)
        # delete the current cluster and add the 2 new clusters obtained
        del clusters[index]
        for c in new_clusters:  clusters.append(c)
        
        # repeat this process until we obtain the required number of clusters

    return clusters


# compute the metrics sse, entropy, purity
def compute_metrics(clusters, no_of_bins):

    # compute total sse
    sse_total = 0
    cluster_centers = [cluster.mean(axis=0) for cluster in clusters]
    clusters_sse = [calc_sse(clusters[i].iloc[:, :-1].to_numpy(), cluster_centers[i][:-1].to_numpy()) for i in range(len(clusters))]
    sse_total = np.sum(clusters_sse)

    # form the cluster bin matrix
    cluster_bin_matrix = np.empty([no_of_bins, no_of_bins])
    cluster_bin_matrix.fill(0)
    for idx, cluster in enumerate(clusters):
        for i in range(cluster.shape[0]):
            cluster_bin_matrix[idx][int(cluster.iloc[i,-1]) - 1] += 1
    
    # compute entropy
    entropy_cluster = [calc_entropy(cluster) for cluster in cluster_bin_matrix]
    entropy_total = sum([sum(cluster_bin_matrix[i])*e for i, e in enumerate(entropy_cluster)]) / np.sum(cluster_bin_matrix)

    # compute purity
    purity_cluster = [calc_purity(cluster) for cluster in cluster_bin_matrix]
    purity_total = sum([sum(cluster_bin_matrix[i])*p for i, p in enumerate(purity_cluster)]) / np.sum(cluster_bin_matrix)

    return sse_total, entropy_total, purity_total


def extract_meal_data_ground_truth(cgm_data, insulin_data):
    
    # sort the rows by date time stamp
    insulin_data = insulin_data.sort_values(by = 'date_time')
    # filter the column 'BWZ Carb Input (grams)' for non NAN non zero values. 
    insulin_data = insulin_data[insulin_data['BWZ Carb Input (grams)'] != 0.0].dropna()
    insulin_data = insulin_data.reset_index().drop(columns='index')

    # interpolate for missing data
    cgm_data['Sensor Glucose (mg/dL)'] = cgm_data['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    
    date_time = []
    carbs = []
    # get valid meal times and corresponding carbs value from insulin data
    for i in range(len(insulin_data['date_time'])-1):
        if (insulin_data['date_time'][i+1] - insulin_data['date_time'][i]).seconds//3600 >= 2:
            date_time.append(insulin_data['date_time'][i])
            carbs.append(insulin_data['BWZ Carb Input (grams)'][i])
    
    # find the number of bins
    min_carb, max_carb = min(carbs), max(carbs)
    no_of_bins = ceil((max_carb - min_carb) / 20)
    
    # filter the 'Sensor Glucose (mg/dL)' of the meal times from cgm data and the 'BWZ Carb Input (grams)' (ground truth)
    meal_data = []
    ground_truth = []
    for i in range(len(date_time)):
        start = date_time[i] - pd.Timedelta(minutes = 30)
        end = date_time[i] + pd.Timedelta(minutes = 120)
        # filter the cgm data between start and end times
        cgm_data_datetime_filter = cgm_data.loc[(cgm_data['date_time'] >= start) & (cgm_data['date_time'] <= end)]['Sensor Glucose (mg/dL)'].values.tolist()
        
        if len(cgm_data_datetime_filter) >= 30:
            meal_data.append(cgm_data_datetime_filter[:30])
            ground_truth.append(int(ceil((carbs[i]-min_carb)/20))) if carbs[i] != min_carb else ground_truth.append(1)
    
    return pd.DataFrame(meal_data), ground_truth, int(no_of_bins)


def get_zero_crossings(sample):
    
    zero_crossings = []
    slopes = np.diff(sample)

    if slopes[0] > 0:
 	    initSign = 1
    else:
 	    initSign = 0

    for i in range(1, len(slopes)):
        if slopes[i] > 0:
            newSign = 1
        else:
            newSign = 0
        if initSign != newSign:
            zero_crossings.append([slopes[i] - slopes[i-1], i])
            initSign = newSign

    zero_crossings.sort(reverse=True)
    for i in range(3):
        if i >= len(zero_crossings): zero_crossings.append([0, 0])
    
    return zero_crossings


# extracts 11 features from data and perform PCA on the features
def extract_features(data):

    features = []

    for i in range(0, data.shape[0]):

        sample = data.iloc[i].values.tolist()

        # fast fourier transformation feature
        fast_fourier = np.abs(fft(sample)).tolist()
        fast_fourier.sort(reverse=True)

        # slope features using zero crossings 
        zero_crossings = get_zero_crossings(sample)

        # append features for each sample
        features.append([
            np.amax(sample) - np.amin(sample),
            np.argmax(sample) - np.argmin(sample),
            zero_crossings[0][0], zero_crossings[0][1], zero_crossings[1][0], zero_crossings[1][1], zero_crossings[2][0], zero_crossings[2][1],
            fast_fourier[1], fast_fourier[2], fast_fourier[3]
        ])
    
    features = pd.DataFrame(features, dtype=float)

    # perform PCA on the features
    std = MinMaxScaler().fit_transform(features)
    features = pd.DataFrame(PCA(n_components=5).fit_transform(std))

    return features


# read the data from the respective csv files
cgm_data = pd.read_csv('CGMData.csv', low_memory = False, usecols = ['Date','Time','Sensor Glucose (mg/dL)'])
insulin_data = pd.read_csv('InsulinData.csv', low_memory = False, usecols = ['Date','Time','BWZ Carb Input (grams)'])

# introduce a new column date_time and calculate the timestamp using the Date and Time columns
cgm_data['date_time'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])
insulin_data['date_time'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])

# filter the meal data, ground truth
meal_data, ground_truth, no_of_bins = extract_meal_data_ground_truth(cgm_data, insulin_data)

# extract the features for the meal data
meal_features = extract_features(meal_data)
meal_features['ground_truth'] = ground_truth

# perform kMeans on meal features and calculate the metrics
kMeans = KMeans(n_clusters=no_of_bins).fit(meal_features.iloc[:, :-1])
kMeans_clusters = get_clusters(kMeans.labels_, meal_features)
sse_kmeans, entropy_kmeans, purity_kmeans = compute_metrics(kMeans_clusters, no_of_bins)

# perform dbscan on meal features and calculate the metrics
dbscan = DBSCAN(eps=0.3, min_samples=10).fit(meal_features.iloc[:, :-1])
dbscan_clusters = split_clusters(dbscan, meal_features, no_of_bins)
sse_dbscan, entropy_dbscan, purity_dbscan = compute_metrics(dbscan_clusters, no_of_bins)

# write output to Results.csv file
res = pd.DataFrame([[sse_kmeans, sse_dbscan, entropy_kmeans, entropy_dbscan, purity_kmeans, purity_dbscan]])
res.to_csv('Results.csv', header=False, index=False)

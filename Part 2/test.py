import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import pickle
from sklearn.decomposition import PCA
import pickle_compat
from scipy.fftpack import fft


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
    std = StandardScaler().fit_transform(features)
    features = pd.DataFrame(PCA(n_components=5).fit_transform(std))

    return features


pickle_compat.patch()

with open("model.pkl", 'rb') as file:
    model = pickle.load(file) 
    test_data = pd.read_csv('test.csv', header=None)

features = extract_features(test_data)
    
predictions = model.predict(features)
pd.DataFrame(predictions).to_csv("Results.csv", header=None, index=False)
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from scipy.fftpack import fft
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import pickle
from sklearn.svm.classes import SVC


# filters the meal or nomeal data using the data read from csv files
def filter_meal_nomeal_data(cgm_data, insulin_data, is_meal=True):
    
    # sort the rows by date time stamp
    insulin_data = insulin_data.sort_values(by = 'date_time')
    # filter the column 'BWZ Carb Input (grams)' for non NAN non zero values. 
    insulin_data = insulin_data[insulin_data['BWZ Carb Input (grams)'] != 0.0].dropna()
    insulin_data = insulin_data.reset_index().drop(columns='index')

    # interpolate for missing data
    cgm_data['Sensor Glucose (mg/dL)'] = cgm_data['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    
    # get valid meal or no meal times from insulin data
    date_time = []
    time_diff = 2 if is_meal else 4
    for i in range(len(insulin_data['date_time'])-1):
        if (insulin_data['date_time'][i+1] - insulin_data['date_time'][i]).seconds//3600 >= time_diff:
            date_time.append(insulin_data['date_time'][i])
    
    # filter the 'Sensor Glucose (mg/dL)' of the meal or no meal times from cgm data
    filtered_data = []
    cols = 30 if is_meal else 24
    n = len(date_time) if is_meal else len(date_time)-1
    for i in range(n):
        start = date_time[i] - pd.Timedelta(minutes = 30) if is_meal else date_time[i] + pd.Timedelta(minutes = 120)
        end = date_time[i] + pd.Timedelta(minutes = 120) if is_meal else date_time[i+1]
        # filter the cgm data between start and end times
        cgm_data_datetime_filter = cgm_data.loc[(cgm_data['date_time'] >= start) & (cgm_data['date_time'] <= end)]['Sensor Glucose (mg/dL)'].values.tolist()
        
        if len(cgm_data_datetime_filter) >= cols:
            filtered_data.append(cgm_data_datetime_filter[:cols])
    
    return pd.DataFrame(filtered_data)


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
    
    # features = features.apply(lambda x: (x-x.mean())/(x.max()-x.min()), axis=1)
    # features = (features-features.mean())/(features.max()-features.min())

    # perform PCA on the features
    std = StandardScaler().fit_transform(features)
    features = pd.DataFrame(PCA(n_components=5).fit_transform(std))

    return features


if __name__ == '__main__':

    # read the data from the respective csv files
    cgm_data = pd.read_csv('CGMData.csv', low_memory = False, usecols = ['Date','Time','Sensor Glucose (mg/dL)'])
    insulin_data = pd.read_csv('InsulinData.csv', low_memory = False, usecols = ['Date','Time','BWZ Carb Input (grams)'])

    # read the data for patient 2
    # cgm_data_patient = pd.read_excel("CGMData670GPatient3.xlsx", usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
    # insulin_data_patient = pd.read_excel("InsulinAndMealIntake670GPatient3.xlsx", usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])
    cgm_data_patient = pd.read_csv("CGM_patient2.csv", low_memory = False, usecols=['Date', 'Time', 'Sensor Glucose (mg/dL)'])
    insulin_data_patient = pd.read_csv("Insulin_patient2.csv", low_memory = False, usecols=['Date', 'Time', 'BWZ Carb Input (grams)'])

    # combine the data into one dataframe
    cgm_data = pd.concat([cgm_data, cgm_data_patient])
    insulin_data = pd.concat([insulin_data, insulin_data_patient])

    # introduce a new column date_time and calculate the timestamp using the Date and Time columns
    cgm_data['date_time'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])
    insulin_data['date_time'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])

    # filter the meal and no meal data
    meal_data = filter_meal_nomeal_data(cgm_data, insulin_data, True).dropna()
    no_meal_data = filter_meal_nomeal_data(cgm_data, insulin_data, False).dropna()

    # extract the features for the meal and no meal data
    meal_features = extract_features(meal_data)
    no_meal_features = extract_features(no_meal_data)

    # assign labels to the samples
    meal_features['label'] = 1
    no_meal_features['label'] = 0

    data = meal_features.append(no_meal_features)
    training_data = data.iloc[:, :-1]
    labels = data.iloc[:, -1]

    accuracies = []
    
    k = 5
    kfold = KFold(k, True, 1)
    # train the model and validate using kfold validation
    for train, test in kfold.split(training_data, labels):
        
        # split training, testing samples
        train_data, test_data = training_data.iloc[train], training_data.iloc[test]
        train_labels, test_labels = labels.iloc[train], labels.iloc[test]
        
        # choose the model to train
        model = SVC(kernel='rbf', gamma='scale', degree=3)
        
        # train the model and predict the labels for testing data
        model.fit(train_data, train_labels)
        predicted_labels = model.predict(test_data)

        accuracies.append(accuracy_score(test_labels, predicted_labels)) 
        
    print('Accuracies:', accuracies)
    
    # write model to pickle file
    with open('model.pkl', 'wb') as (file):
        pickle.dump(model, file)
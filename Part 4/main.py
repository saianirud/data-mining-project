from math import ceil
import pandas as pd
import numpy as np


def extract_meal_data(cgm_data, insulin_data):
    
    # sort the rows by date time stamp
    insulin_data = insulin_data.sort_values(by = 'date_time')
    # filter the column 'BWZ Carb Input (grams)' for non NAN non zero values. 
    insulin_data = insulin_data[insulin_data['BWZ Carb Input (grams)'] != 0.0].dropna()
    insulin_data = insulin_data.reset_index().drop(columns='index')

    # interpolate for missing data
    cgm_data['Sensor Glucose (mg/dL)'] = cgm_data['Sensor Glucose (mg/dL)'].interpolate(method='linear',limit_direction = 'both')
    
    date_time = []
    i_bolus = []
    # get valid meal times and corresponding bolus value from insulin data
    for i in range(len(insulin_data['date_time'])-1):
        if (insulin_data['date_time'][i+1] - insulin_data['date_time'][i]).seconds//3600 >= 2:
            date_time.append(insulin_data['date_time'][i])
            i_bolus.append(round(insulin_data['BWZ Estimate (U)'][i]))

    
    # filter the 'Sensor Glucose (mg/dL)' of the meal times from cgm data
    meal_data = []
    insulin_bolus = []
    for i in range(len(date_time)):
        start = date_time[i] - pd.Timedelta(minutes = 30)
        end = date_time[i] + pd.Timedelta(minutes = 120)
        # filter the cgm data between start and end times
        cgm_data_datetime_filter = cgm_data.loc[(cgm_data['date_time'] >= start) & (cgm_data['date_time'] <= end)]['Sensor Glucose (mg/dL)'].values.tolist()

        if len(cgm_data_datetime_filter) >= 30:
            meal_data.append(cgm_data_datetime_filter[:30])
            insulin_bolus.append(i_bolus[i])
    
    return meal_data, insulin_bolus


def get_itemsets(meal_data, insulin_bolus):
    
    meal_data = np.asarray(meal_data)

    cgm_min = meal_data.min()
    
    b_max = []
    b_meal = []

    for row in meal_data:
        cgm_max = row.max()
        b_max.append(ceil((cgm_max - cgm_min) / 20)) if cgm_max != cgm_min else b_max.append(1)
        cgm_meal = row[-6]
        b_meal.append(ceil((cgm_meal - cgm_min) / 20)) if cgm_meal != cgm_min else b_meal.append(1)

    return pd.DataFrame({'b_max': b_max, 'b_meal': b_meal, 'insulin_bolus': insulin_bolus})



# read the data from the respective csv files
cgm_data = pd.read_csv('CGMData.csv', low_memory = False, usecols = ['Date','Time','Sensor Glucose (mg/dL)'])
insulin_data = pd.read_csv('InsulinData.csv', low_memory = False, usecols = ['Date','Time','BWZ Carb Input (grams)', 'BWZ Estimate (U)'])

# introduce a new column date_time and calculate the timestamp using the Date and Time columns
cgm_data['date_time'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])
insulin_data['date_time'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])

# filter the meal data and insulin bolus
meal_data, insulin_bolus = extract_meal_data(cgm_data, insulin_data)
itemsets = get_itemsets(meal_data, insulin_bolus)

itemsets["b_max"] = pd.to_numeric(itemsets["b_max"], downcast='integer')
itemsets["b_meal"] = pd.to_numeric(itemsets["b_meal"], downcast='integer')
itemsets["insulin_bolus"] = pd.to_numeric(itemsets["insulin_bolus"], downcast='integer')

# get most frequent itemsets
freq_itemsets = itemsets.groupby(['b_max','b_meal','insulin_bolus']).size().reset_index(name='freq')
max_freq = freq_itemsets['freq'].max()
most_freq_itemsets = freq_itemsets.loc[freq_itemsets['freq'] == max_freq][['b_max','b_meal','insulin_bolus']]
most_freq_itemsets = most_freq_itemsets.apply(lambda x: (x[0], x[1], x[2]), axis=1)
most_freq_itemsets.to_csv('most_freq_itemsets.csv', header=False, index=False) 

# get largest confidence rules
freq_rules = itemsets.groupby(['b_max','b_meal']).size().reset_index(name='rule_freq')
freq_rules = pd.merge(freq_itemsets, freq_rules, on=['b_max','b_meal'])
freq_rules['confidence'] = freq_rules['freq'] / freq_rules['rule_freq']
largest_confidence = freq_rules['confidence'].max()
largest_confidence_rules = freq_rules.loc[freq_rules['confidence'] == largest_confidence][['b_max','b_meal','insulin_bolus']]
largest_confidence_rules = largest_confidence_rules.apply(lambda x: '{{{0},{1}}} -> {2}'.format(x[0], x[1], x[2]), axis=1)
largest_confidence_rules.to_csv('largest_confidence_rules.csv', header=False, index=False)

# get anomalous rules
anomalous_rules = freq_rules.loc[freq_rules['confidence'] < 0.15][['b_max','b_meal','insulin_bolus']]
anomalous_rules = anomalous_rules.apply(lambda x: '{{{0},{1}}} -> {2}'.format(x[0], x[1], x[2]), axis=1)
anomalous_rules.to_csv('anomalous_rules.csv', header=False, index=False)
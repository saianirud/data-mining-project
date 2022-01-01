import pandas as pd

# calculate the mean of percentages of a metric over all days
def get_percentage_mean(data, num_days):
    return ((data / 288) * 100).sum(axis=0) / num_days

# extract the metrics
def extract_metrics(data):
    res = []

    # each row represents a time interval -> [overnight, daytime, whole day]
    times_intervals = [['0:00:00', '05:59:59'], ['6:00:00', '23:59:59'], ['0:00:00', '23:59:59']]

    # total number of days available
    num_days = (data.groupby('Date').count()).count()[0]
    
    # for each time interval calculate the metrics and the mean of its percentages and append them to the result (res)
    for interval in times_intervals:

        # filter the data such that it is in between the interval
        interval_data = data.between_time(interval[0], interval[1])

        # extract the metric - percentage time in hyperglycemia (CGM > 180 mg/dL)
        count_data_180 = interval_data[interval_data['Sensor Glucose (mg/dL)'] > 180].groupby('Date')['Sensor Glucose (mg/dL)'].count()
        # calculate the mean of its percentages and append to res
        res.append(get_percentage_mean(count_data_180, num_days))

        # extract the metric - percentage of time in hyperglycemia critical (CGM > 250 mg/dL)
        count_data_250 = interval_data[interval_data['Sensor Glucose (mg/dL)'] > 250].groupby('Date')['Sensor Glucose (mg/dL)'].count()
        # calculate the mean of its percentages and append to res
        res.append(get_percentage_mean(count_data_250, num_days))

        # extract the metric - percentage time in range (CGM >= 70 mg/dL and CGM <= 180 mg/dL)
        count_data_70_180 = interval_data[(interval_data['Sensor Glucose (mg/dL)'] >= 70) & (interval_data['Sensor Glucose (mg/dL)'] <= 180)].groupby('Date')['Sensor Glucose (mg/dL)'].count()
        # calculate the mean of its percentages and append to res
        res.append(get_percentage_mean(count_data_70_180, num_days))

        # extract the metric - percentage time in range secondary (CGM >= 70 mg/dL and CGM <= 150 mg/dL)
        count_data_70_150 = interval_data[(interval_data['Sensor Glucose (mg/dL)'] >= 70) & (interval_data['Sensor Glucose (mg/dL)'] <= 150)].groupby('Date')['Sensor Glucose (mg/dL)'].count()
        # calculate the mean of its percentages and append to res
        res.append(get_percentage_mean(count_data_70_150, num_days))

        # extract the metric - percentage time in hypoglycemia level 1 (CGM < 70 mg/dL)
        count_data_70 = interval_data[interval_data['Sensor Glucose (mg/dL)'] < 70].groupby('Date')['Sensor Glucose (mg/dL)'].count()
        # calculate the mean of its percentages and append to res
        res.append(get_percentage_mean(count_data_70, num_days))

        # extract the metric - percentage time in hypoglycemia level 2 (CGM < 54 mg/dL)
        count_data_54 = interval_data[interval_data['Sensor Glucose (mg/dL)'] < 54].groupby('Date')['Sensor Glucose (mg/dL)'].count()
        # calculate the mean of its percentages and append to res
        res.append(get_percentage_mean(count_data_54, num_days))

    res.append(1.1)
    return res

# read the data from the respective csv files
cgm_data = pd.read_csv('CGMData.csv', low_memory = False, usecols = ['Date', 'Time', 'Sensor Glucose (mg/dL)'])
insulin_data = pd.read_csv('InsulinData.csv', low_memory = False, usecols = ['Date', 'Time', 'Alarm'])

# introduce a new column date_time and calculate the timestamp using the Date and Time columns
cgm_data['date_time'] = pd.to_datetime(cgm_data['Date'] + ' ' + cgm_data['Time'])
insulin_data['date_time'] = pd.to_datetime(insulin_data['Date'] + ' ' + insulin_data['Time'])

# sort the rows by date_time
cgm_data = cgm_data.sort_values(by = 'date_time')
insulin_data = insulin_data.sort_values(by = 'date_time')

# extract the row where the Auto Mode starts
auto_mode_start = insulin_data.loc[insulin_data['Alarm'] == 'AUTO MODE ACTIVE PLGM OFF'].iloc[0]

# interpolate the cgm_data for missing values
cgm_data['Sensor Glucose (mg/dL)'] = cgm_data['Sensor Glucose (mg/dL)'].interpolate(method = 'linear', limit_direction = 'both')

# seperate the manual and auto mode of cgm_data
cgm_manual_mode_data = cgm_data.loc[cgm_data['date_time'] < auto_mode_start['date_time']]
cgm_auto_mode_data = cgm_data.loc[cgm_data['date_time'] >= auto_mode_start['date_time']]

cgm_manual_mode_data = cgm_manual_mode_data.set_index('date_time')
cgm_auto_mode_data = cgm_auto_mode_data.set_index('date_time')

df = pd.DataFrame()
# extract the metric for manual mode data
df['manual'] = extract_metrics(cgm_manual_mode_data)
# extract the metric for auto mode data
df['auto'] = extract_metrics(cgm_auto_mode_data)
df = df.T
df = df.fillna(0)

# write the result to a csv file
df.to_csv('Results.csv', header = False, index = False)
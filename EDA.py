import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import zscore

"""
This script performs exploratory data analysis on the given dataset.
It prints out the column names, data overview, first few rows of the dataset,
summary statistics, missing values, negative values in GHI, DNI, DHI, and outliers
in the sensor columns.
"""
data = pd.read_csv(r'C:\Users\tesfaneshyisaiase\Documents\front\week-0\data\benin-malanville.csv')

print("Column Names:")
print(data.columns)
print("Data Overview:")
print(data.info())
print("\nFirst few rows of the dataset:")
print(data.head())


summary_statistics = data.describe()  
print("\nSummary Statistics:")
print(summary_statistics)


missing_values = data.isnull().sum()
print("\nMissing Values:")
print(missing_values[missing_values > 0])


negative_values = data[(data[['GHI', 'DNI', 'DHI']] < 0).any(axis=1)]
print("\nNegative Values in GHI, DNI, DHI:")
print(negative_values[['GHI', 'DNI', 'DHI']])


def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df[column] < lower_bound) | (df[column] > upper_bound)]

sensor_columns = ['ModA', 'ModB', 'WS', 'WSgust']
outliers = {}
for column in sensor_columns:
    outliers[column] = detect_outliers_iqr(data, column)

print("\nOutliers Detected:")
for column, outlier_data in outliers.items():
    if not outlier_data.empty:
        print(f"{column}:\n{outlier_data}\n")


data['Timestamp'] = pd.to_datetime(data['Timestamp'])  
# Set the 'Timestamp' column as the index to enable time series analysis
data.set_index('Timestamp', inplace=True)


plt.figure(figsize=(14, 7))
sns.lineplot(data=data[['GHI', 'DNI', 'DHI', 'Tamb']])
plt.title('Time Series of Solar Radiation and Temperature')
plt.xlabel('Timestamp')
plt.ylabel('Values')
plt.legend(['GHI', 'DNI', 'DHI', 'Tamb'])
plt.show()


cleaned_data = data[data['Cleaning'] == 1]  
plt.figure(figsize=(14, 7))
sns.lineplot(data=cleaned_data[['ModA', 'ModB']])
plt.title('Sensor Readings After Cleaning')
plt.xlabel('Timestamp')
plt.ylabel('Sensor Readings')
plt.legend(['ModA', 'ModB'])
plt.show()


correlation_matrix = data[['GHI', 'DNI', 'DHI', 'TModA', 'TModB']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm')
plt.title('Correlation Matrix for Solar Radiation and Temperature')
plt.show()


plt.figure(figsize=(14, 7))
sns.scatterplot(data=data, x='WS', y='WD', hue='WSgust', size='WSgust', sizes=(20, 200))
plt.title('Wind Speed vs Wind Direction')
plt.xlabel('Wind Speed (WS)')
plt.ylabel('Wind Direction (WD)')
plt.show()


plt.figure(figsize=(14, 7))
sns.scatterplot(data=data, x='RH', y='Tamb', hue='GHI')
plt.title('Temperature vs Relative Humidity')
plt.xlabel('Relative Humidity (RH)')
plt.ylabel('Temperature (Tamb)')
plt.show()


data[['GHI', 'DNI', 'DHI', 'WS', 'Tamb']].hist(bins=30, figsize=(14, 10))
plt.suptitle('Histograms of Key Variables')
plt.show()


z_scores = np.abs(zscore(data[['GHI', 'DNI', 'DHI', 'ModA', 'ModB', 'WS', 'WSgust']]))
print("\nZ-Scores:")
print(z_scores)


plt.figure(figsize=(14, 7))
plt.scatter(data['GHI'], data['Tamb'], s=data['RH']*10, alpha=0.5)
plt.title('Bubble Chart: GHI vs. Tamb vs. RH')
plt.xlabel('GHI')
plt.ylabel('Tamb')
plt.show()



data.dropna(subset=['Comments'], inplace=True)  
data['GHI'].clip(lower=0, inplace=True)  
data['DNI'].clip(lower=0, inplace=True)  
data['DHI'].clip(lower=0, inplace=True)  

print("\nCleaned Data Overview:")
print(data.info())
# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""
###LOAD AND INSPECT THE DATA###

# Install necessary libraries
!pip install pandas
!pip install seaborn
!pip install matplotlib
!pip install sklearn

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans

# Specify the file path and name
file_path_csv = r'C:\Users\efeme\Downloads\Online Retail.xlsx - Online Retail.csv'

# Load the .csv file
df = pd.read_csv(file_path_csv)

# Display the first few rows of the dataset
print(df.head())

# Check the structure of the dataframe
print(df.info())

# Check for missing values
print(df.isnull().sum())

###DATA CLEANİNG AND PREPERATİON###

# Remove rows with missing CustomerID
df_cleaned = df.dropna(subset=['CustomerID'])

# Remove rows with negative or zero quantity and unit price
df_cleaned = df_cleaned[(df_cleaned['Quantity'] > 0) & (df_cleaned['UnitPrice'] > 0)]

# Convert the InvoiceDate column to datetime format
df_cleaned['InvoiceDate'] = pd.to_datetime(df_cleaned['InvoiceDate'])

# Add a new column to calculate total spending per transaction
df_cleaned['TotalAmount'] = df_cleaned['Quantity'] * df_cleaned['UnitPrice']

# Display the first few rows of the cleaned dataset
print(df_cleaned.head())

###SEGMENTATION###

# Select relevant columns and group by CustomerID and Country to calculate total spending
X_extended = df_cleaned[['CustomerID', 'Country', 'TotalAmount']].groupby(['CustomerID', 'Country']).sum().reset_index()

# Standardize the TotalAmount feature
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_extended[['TotalAmount']])

# Apply K-means clustering with 3 clusters
kmeans_3 = KMeans(n_clusters=3, random_state=0)
X_extended['Segment'] = kmeans_3.fit_predict(X_scaled)

# Rename the segments based on spending levels
X_extended['Spending_Segment'] = X_extended['Segment'].map({
    0: 'Low Spenders',
    1: 'Moderate Spenders',
    2: 'High Spenders'
})

# Merge the segmented data back into the original cleaned dataframe
df_cleaned = df_cleaned.merge(X_extended[['CustomerID', 'Country', 'Spending_Segment']], on=['CustomerID', 'Country'], how='left')

# Display the first few rows of the dataframe with segmentation
print(df_cleaned.head())

###VISULATİON WITH PIE CHARTS###

# Identify unique countries in the dataset
unique_countries = df_cleaned['Country'].unique()

# Visualize the distribution of spending segments with pie charts for each country
for country in unique_countries:
    df_country = df_cleaned[df_cleaned['Country'] == country]
    segment_counts = df_country['Spending_Segment'].value_counts()
    
    plt.figure(figsize=(8,8))
    plt.pie(segment_counts, labels=segment_counts.index, autopct='%1.1f%%', startangle=140)
    plt.title(f'Spending Segment Distribution in {country}')
    plt.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
    plt.show()



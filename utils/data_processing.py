import pandas as pd
import numpy as np
from datetime import datetime, timedelta

# Load the data
df = pd.read_csv('yellow_tripdata_2020-06.csv')

# Convert datetime columns
df['tpep_pickup_datetime'] = pd.to_datetime(df['tpep_pickup_datetime'])
df['tpep_dropoff_datetime'] = pd.to_datetime(df['tpep_dropoff_datetime'])

# Extract time-based features
df['hour'] = df['tpep_pickup_datetime'].dt.hour
df['day_of_week'] = df['tpep_pickup_datetime'].dt.dayofweek
df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)

# Calculate trip duration and price per mile
df['trip_duration'] = (df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']).dt.total_seconds() / 60
df['price_per_mile'] = df['total_amount'] / df['trip_distance']

# Remove outliers and invalid data
df = df[(df['trip_distance'] > 0) & (df['fare_amount'] > 0) & (df['trip_duration'] > 0)]
df = df[df['price_per_mile'] < df['price_per_mile'].quantile(0.99)]

# Function to calculate demand
def calculate_demand(group):
    return len(group)

# Function to calculate competition (number of unique vendors)
def calculate_competition(group):
    return group['VendorID'].nunique()

# Function to calculate average price
def calculate_avg_price(group):
    return group['fare_amount'].mean()

# Group data by location, hour, and day of week
grouped = df.groupby(['PULocationID', 'DOLocationID', 'hour', 'day_of_week'])

# Calculate demand, competition, and average price
demand = grouped.apply(calculate_demand).reset_index(name='demand')
competition = grouped.apply(calculate_competition).reset_index(name='competition')
avg_price = grouped.apply(calculate_avg_price).reset_index(name='avg_price')

# Merge the calculated features
features = pd.merge(demand, competition, on=['PULocationID', 'DOLocationID', 'hour', 'day_of_week'])
features = pd.merge(features, avg_price, on=['PULocationID', 'DOLocationID', 'hour', 'day_of_week'])

# Calculate relative demand (demand / competition)
features['relative_demand'] = features['demand'] / features['competition']

# Normalize features
for col in ['demand', 'competition', 'avg_price', 'relative_demand']:
    features[f'{col}_normalized'] = (features[col] - features[col].min()) / (features[col].max() - features[col].min())

# Save processed features
features.to_csv('processed_features.csv', index=False)

print("Processed features saved to 'processed_features.csv'")
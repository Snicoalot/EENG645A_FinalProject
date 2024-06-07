''' 
This file holds all the architecture for the neural network. The preprocess.py file does all necesssary prerequisite training
to allow this script to operate quickly and efficiently.

In the 7 Step Chollet Process, this file covers steps 4 - 7.

STEP 4: CHOOSE AN EVALUATION METHOD

This network will be trained on a dataset that is split 65/17.5/17.5 into training, testing, and validation sets. The steps
to set this up are included here.
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
import matplotlib.pyplot as plt

# Load the DataFrame
df = pd.read_csv('/remote_home/EENG645A_FinalProject-1/processed_data/clean.csv')

# Split the data into training, validation, and test sets based on ship identifier
    # The identified split does a roughly 65/17/18 train/validate/test split of the data
unique_mmsi = df['mmsi'].unique()
train_val_mmsi, test_mmsi = train_test_split(unique_mmsi, test_size=0.10, random_state=42)
train_mmsi, val_mmsi = train_test_split(train_val_mmsi, test_size=0.21, random_state=42)

train_df = df[df['mmsi'].isin(train_mmsi)]
val_df = df[df['mmsi'].isin(val_mmsi)]
test_df = df[df['mmsi'].isin(test_mmsi)]

# Verify the split size
print(f"Training set size: {len(train_df)}")
print(f"Validation set size: {len(val_df)}")
print(f"Test set size: {len(test_df)}")

# Save the dataframes to GitHub in the split_data folder
train_df.to_csv('/remote_home/EENG645A_FinalProject-1/split_data/train_data.csv', index=False)
val_df.to_csv('/remote_home/EENG645A_FinalProject-1/split_data/val_data.csv', index=False)
test_df.to_csv('/remote_home/EENG645A_FinalProject-1/split_data/test_data.csv', index=False)

''' STEP 5: INITIAL MODEL '''
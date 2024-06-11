''' 
This script makes a confusion matrix for the specified model, for later analysis
'''

# Import necessary packages
import pandas as pd
from sklearn.metrics import confusion_matrix
import seaborn as sns
from tensorflow.keras.models import load_model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt
import numpy as np

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

# Define features and target
features = ['distance_from_shore', 'distance_from_port', 'speed', 'course', 'lat', 'lon']
target = 'is_fishing'

# Separate features and target
X_train = train_df[features]
y_train = train_df[target]
X_val = val_df[features]
y_val = val_df[target]
X_test = test_df[features]
y_test = test_df[target]

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# Load the best model
best_model = load_model('/remote_home/EENG645A_FinalProject-1/models/best_tuned_model.h5')

# Predict on the test set
y_pred = best_model.predict(X_test)
y_pred = (y_pred > 0.5)

# Create confusion matrix
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False)
plt.title('Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.savefig('/remote_home/EENG645A_FinalProject-1/figures/tuned_confusion_matrix3.png')
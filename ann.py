''' 
This file holds all the architecture for the neural network. The preprocess.py file does all necesssary prerequisite training
to allow this script to operate quickly and efficiently.

In the 7 Step Chollet Process, this file covers steps 4 - 7.

STEP 4: CHOOSE AN EVALUATION METHOD

This network will be trained on a dataset that is split 65/17/18 into training, validation, and testing sets. The steps
to set this up are included here.
'''

# Import necessary packages
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout
from tensorflow.keras.regularizers import l1_l2
from tensorflow.keras.callbacks import EarlyStopping
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

# # Save the dataframes to GitHub in the split_data folder
# train_df.to_csv('/remote_home/EENG645A_FinalProject-1/split_data/train_data.csv', index=False)
# val_df.to_csv('/remote_home/EENG645A_FinalProject-1/split_data/val_data.csv', index=False)
# test_df.to_csv('/remote_home/EENG645A_FinalProject-1/split_data/test_data.csv', index=False)

''' STEP 5: INITIAL MODEL '''

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

# # Build the initial neural network architecture
# model = Sequential([
#     Dense(16, input_dim=X_train.shape[1], activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# # Evaluate the model on the test set
# test_loss, test_accuracy = model.evaluate(X_test, y_test)
# print(f'Test Loss: {test_loss}')
# print(f'Test Accuracy: {test_accuracy}')

# # Plot the training history
# plt.figure(figsize=(12, 4))

# # Plot training & validation accuracy values
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# # Plot training & validation loss values
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# # Save the plots
# plt.savefig('/remote_home/EENG645A_FinalProject-1/figures/step5_training_history.png')

''' STEP 6: OVERFIT MODEL '''

# # Build the more complicated neural network model with more layers and more neurons per layer
# model2 = Sequential([
#     Dense(128, input_dim=X_train.shape[1], activation='relu'),
#     Dense(64, activation='relu'),
#     Dense(32, activation='relu'),
#     Dense(16, activation='relu'),
#     Dense(1, activation='sigmoid')
# ])

# # Compile the model
# model2.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# # Train the model
# history = model2.fit(X_train, y_train, epochs=20, batch_size=32, validation_data=(X_val, y_val))

# # Evaluate the model on the test set
# test_loss, test_accuracy = model2.evaluate(X_test, y_test)
# print(f'Test Loss: {test_loss}')
# print(f'Test Accuracy: {test_accuracy}')

# # Plot the training history
# plt.figure(figsize=(12, 4))

# # Plot training & validation accuracy values
# plt.subplot(1, 2, 1)
# plt.plot(history.history['accuracy'])
# plt.plot(history.history['val_accuracy'])
# plt.title('Model accuracy')
# plt.ylabel('Accuracy')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# # Plot training & validation loss values
# plt.subplot(1, 2, 2)
# plt.plot(history.history['loss'])
# plt.plot(history.history['val_loss'])
# plt.title('Model loss')
# plt.ylabel('Loss')
# plt.xlabel('Epoch')
# plt.legend(['Train', 'Validation'], loc='upper left')

# # Save the plots
# plt.savefig('/remote_home/EENG645A_FinalProject-1/figures/step6_training_history.png')

''' STEP 7: GENERALIZED MODEL '''

model3 = Sequential([
    Dense(128, input_dim=X_train.shape[1], activation='relu'),
    Dropout(0.5),
    Dense(64, activation='relu'),
    Dropout(0.5),
    Dense(32, activation='relu'),
    Dropout(0.5),
    Dense(16, activation='relu'),
    Dropout(0.5),
    Dense(1, activation='sigmoid')
])

# Compile the model
model3.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Define early stopping
early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

# Train the model
history = model3.fit(X_train, y_train, epochs=200, batch_size=32, validation_data=(X_val, y_val), callbacks=[early_stopping])

# Evaluate the model on the test set
test_loss, test_accuracy = model3.evaluate(X_test, y_test)
print(f'Test Loss: {test_loss}')
print(f'Test Accuracy: {test_accuracy}')

# Plot the training history
plt.figure(figsize=(12, 4))

# Plot training & validation accuracy values
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Plot training & validation loss values
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Validation'], loc='upper left')

# Save the plots
plt.savefig('/remote_home/EENG645A_FinalProject-1/figures/step7_training_history.png')
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import os
import glob
import random

# Function to create samples by concatenating the first 1003 rows with each 101-row segment
def create_samples(file_path, first_segment_length=1003, segment_length=101):
    """Creates samples by concatenating the first 1003 rows with each 101-row segment in sequence."""
    combined_df = pd.read_csv(file_path)
    columns_to_drop = ["Sequence_Name", "Position"]
    combined_df = combined_df.drop(columns=[col for col in columns_to_drop if col in combined_df.columns], errors='ignore')
    combined_df = combined_df.drop(combined_df.columns[[0, 24, 25, 26, 27, 28]], axis=1, errors='ignore')
    combined_df = combined_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    combined_matrix = combined_df.values
    first_segment = combined_matrix[:first_segment_length, :]
    samples = []

    for i in range(first_segment_length, len(combined_matrix), segment_length):
        end_index = i + segment_length
        if end_index > len(combined_matrix):
            break
        segment = combined_matrix[i:end_index, :]
        sample = np.vstack((first_segment, segment))
        samples.append(sample)
    
    return samples

# Generator to yield samples up to the specified limit across all datasets
def sample_generator(file_paths, sample_limit=100000, first_segment_length=1003, segment_length=101):
    sample_count = 0
    all_samples = []

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        samples = create_samples(file_path, first_segment_length, segment_length)
        all_samples.extend(samples)
        if len(all_samples) >= sample_limit:
            break
    
    random.shuffle(all_samples)  # Shuffle to ensure a random distribution
    all_samples = all_samples[:sample_limit]  # Limit to the specified number of samples

    for sample in all_samples:
        rna_binding_class = sample[-101:, -1]
        y_label = 1 if np.any(rna_binding_class == 1) else 0
        yield sample[:, :-1], y_label

# Define the base path to search for files
base_folder_path = '/Users/marcobenavides/repos/ML-4-FG/3D-RBP/datasets/clip'
file_pattern = os.path.join(base_folder_path, '**', 'Combined_RNA_Protein_Matrix.csv')
file_paths = glob.glob(file_pattern, recursive=True)

# Initialize counters
X, y = [], []
for X_sample, y_sample in sample_generator(file_paths, sample_limit=100000):
    X.append(X_sample)
    y.append(y_sample)

X = np.array(X, dtype=np.float32)
y = np.array(y, dtype=np.float32)

# Now split data into training and testing sets
split_index = int(0.8 * len(X))
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]

# Define and compile the CNN model
def create_cnn_model(input_shape):
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=5))
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

input_shape = X_train.shape[1:]
model = create_cnn_model(input_shape)
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

# Train and evaluate the model
batch_size = 32
epochs = 10
model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2)
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy:.2f}")

# Make predictions
predictions = model.predict(X_test)
print(predictions)

import os
import glob
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split

# Define paths
base_folder_path = '/Users/marcobenavides/repos/ML-4-FG/3D-RBP/datasets/clip'
file_pattern = os.path.join(base_folder_path, '**', 'Combined_RNA_Protein_Matrix.csv')
file_paths = glob.glob(file_pattern, recursive=True)

# Function to create samples by concatenating the first 1003 rows with each 101-row segment
def create_samples(file_path, first_segment_length=1003, segment_length=101):
    """Creates samples by concatenating the first 1003 rows with each 101-row segment in sequence."""
    print(f"Reading file: {file_path}")
    combined_df = pd.read_csv(file_path)
    print("Columns in file:", combined_df.columns)  # Debugging line to list all columns in the file
    
    # Drop unnecessary columns
    columns_to_drop = ["Sequence_Name", "Position"]
    combined_df = combined_df.drop(columns=[col for col in columns_to_drop if col in combined_df.columns], errors='ignore')
    combined_df = combined_df.drop(combined_df.columns[[0, 24, 25, 26, 27, 28]], axis=1, errors='ignore')
    combined_df = combined_df.apply(pd.to_numeric, errors='coerce').dropna(axis=1)
    combined_matrix = combined_df.values
    
    # Separate the first segment
    first_segment = combined_matrix[:first_segment_length, :]
    samples = []
    binding_samples = 0
    non_binding_samples = 0
    
    # Generate samples
    for i in range(first_segment_length, len(combined_matrix), segment_length):
        end_index = i + segment_length
        if end_index > len(combined_matrix):
            break
        segment = combined_matrix[i:end_index, :]
        sample = np.vstack((first_segment, segment))
        
        # Extract the Binding_Class column values for the last 101 rows
        rna_binding_class = combined_df["Binding_Class"].values[i:end_index]
        print(f"Binding labels in this sample segment: {rna_binding_class}")  # Debugging line
        
        # Determine if the sample is binding or non-binding
        y_label = 1 if np.any(rna_binding_class == 1) else 0
        
        # Count binding and non-binding samples
        if y_label == 1:
            binding_samples += 1
        else:
            non_binding_samples += 1
        
        samples.append((sample, y_label))
    
    # Print sample counts for this file
    print(f"Generated {len(samples)} samples from file: {file_path}")
    print(f"Binding samples: {binding_samples}, Non-binding samples: {non_binding_samples}")
    return samples

# Function to count total, binding, and non-binding samples across all files
def count_samples(file_paths, first_segment_length=1003, segment_length=101):
    """Counts total, binding, and non-binding samples across all files."""
    print("Counting total, binding, and non-binding samples...")
    total_samples = 0
    binding_samples = 0
    non_binding_samples = 0
    
    for file_path in file_paths:
        samples = create_samples(file_path, first_segment_length, segment_length)
        for _, y_label in samples:
            total_samples += 1
            if y_label == 1:
                binding_samples += 1
            else:
                non_binding_samples += 1
    
    # Print summary of samples
    print(f"Total samples: {total_samples}")
    print(f"Binding samples: {binding_samples}")
    print(f"Non-binding samples: {non_binding_samples}")

# Run the count function to display sample counts before training
count_samples(file_paths)

# Generator function to yield samples one at a time for training
def sample_generator(file_paths, first_segment_length=1003, segment_length=101):
    """Yields samples one at a time for training purposes."""
    for file_path in file_paths:
        print(f"Processing file in generator: {file_path}")
        samples = create_samples(file_path, first_segment_length, segment_length)
        for sample, y_label in samples:
            yield sample[:, :-1], y_label

# Function to wrap generator for tf.data compatibility
def tf_data_generator(file_paths, first_segment_length=1003, segment_length=101):
    """Wraps sample generator for TensorFlow compatibility."""
    return sample_generator(file_paths, first_segment_length, segment_length)

# Split data into training and testing sets
train_paths, test_paths = train_test_split(file_paths, test_size=0.2, random_state=42)
print(f"Training files: {len(train_paths)}, Testing files: {len(test_paths)}")

# Set up train and test datasets using tf.data
sample_input_shape = (1104, 21)  # Adjust shape based on final number of rows and columns

train_dataset = tf.data.Dataset.from_generator(
    lambda: tf_data_generator(train_paths),
    output_signature=(
        tf.TensorSpec(shape=(sample_input_shape), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
)

test_dataset = tf.data.Dataset.from_generator(
    lambda: tf_data_generator(test_paths),
    output_signature=(
        tf.TensorSpec(shape=(sample_input_shape), dtype=tf.float32),
        tf.TensorSpec(shape=(), dtype=tf.int32),
    )
)

# Batch and prefetch data
batch_size = 32
train_dataset = train_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
test_dataset = test_dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)

# Define the CNN model
def create_cnn_model(input_shape):
    """Creates and compiles the CNN model."""
    print("Creating CNN model...")
    model = models.Sequential()
    model.add(layers.Conv1D(filters=64, kernel_size=3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=128, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Conv1D(filters=256, kernel_size=3, activation='relu'))
    model.add(layers.MaxPooling1D(pool_size=2))
    model.add(layers.Flatten())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dropout(0.5))
    model.add(layers.Dense(64, activation='relu'))
    model.add(layers.Dense(1, activation='sigmoid'))
    print("CNN model created.")
    return model

# Create and compile the model
model = create_cnn_model((1104, 21))  # Adjust input shape to match sample_input_shape
model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
print("Model compiled.")

# Train the model
epochs = 10
print("Starting model training...")
model.fit(train_dataset, epochs=epochs)
print("Model training completed.")

# Evaluate the model on the test set
print("Evaluating model on test set...")
loss, accuracy = model.evaluate(test_dataset)
print(f"Test Accuracy: {accuracy:.2f}")

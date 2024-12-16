import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import random
import argparse
from sklearn.metrics import roc_auc_score
from tensorflow.keras.callbacks import Callback

# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
total_samples = 200000

# Function to create samples by concatenating the first 1003 rows with each 101-row segment
def create_samples(file_path, first_segment_length=1003, segment_length=101):
    """Creates samples by concatenating the first 1003 rows with each 101-row segment in sequence."""
    combined_df = pd.read_csv(file_path)
    columns_to_drop = ["Index","Protein_Label", "Sequence_Name", "Position"]
    combined_df = combined_df.drop(columns=[col for col in columns_to_drop if col in combined_df.columns], errors='ignore')
    combined_df = combined_df.drop(combined_df.columns[[25, 26, 27]], axis=1, errors='ignore')
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
def sample_generator(file_paths, sample_limit=total_samples, first_segment_length=1003, segment_length=101):
    sample_count = 0
    all_samples = []
    binding_counter = 0
    non_binding_counter = 0

    for file_path in file_paths:
        print(f"Processing file: {file_path}")
        samples = create_samples(file_path, first_segment_length, segment_length)
        all_samples.extend(samples)
        if len(all_samples) >= sample_limit:
            break
    
    random.shuffle(all_samples)  # Shuffle to ensure a random distribution
    all_samples = all_samples[:sample_limit]  # Limit to the specified number of samples

    for sample in all_samples:
        rna_binding_class = sample[-90:-20, -1]
        y_label = 1 if np.any(rna_binding_class == 1) else 0
        
        # Increment binding and non-binding counters
        if y_label == 1:
            binding_counter += 1
        else:
            non_binding_counter += 1
        
        yield sample[:, :-1], y_label

    # Print the counts
    print(f"[INFO] Total Binding Samples: {binding_counter}")
    print(f"[INFO] Total Non-Binding Samples: {non_binding_counter}")


class MetricsCallback(Callback):
    def __init__(self, validation_data):
            super().__init__()
            self.validation_data = validation_data
            self.epoch_metrics = []  # To store metrics for each epoch

    def on_epoch_end(self, epoch, logs=None):
            # Get validation data
            val_x, val_y = self.validation_data
            # Predict probabilities
            val_predictions = self.model.predict(val_x).flatten()
            # Compute AUC
            auc = roc_auc_score(val_y, val_predictions)
            # Retrieve accuracy from logs
            accuracy = logs.get("val_accuracy")
            # Store metrics
            self.epoch_metrics.append({"epoch": epoch + 1, "accuracy": accuracy, "auc": auc})
            print(f"Epoch {epoch + 1} - Validation AUC: {auc:.4f}, Validation Accuracy: {accuracy:.4f}")

    def get_epoch_metrics(self):
        return self.epoch_metrics
    
    

# Main execution
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run CNN model on RNA-protein data.")
    parser.add_argument(
        "--base_path",
        type=str,
        default=os.path.join(os.path.dirname(os.path.abspath(__file__)), "Data", "datasets", "clip"),
        help="Base path for RNA-protein data (default: relative path to 'Data/datasets/clip')."
    )
    args = parser.parse_args()

    base_folder_path = args.base_path
    file_pattern = os.path.join(base_folder_path, '**', 'Combined_RNA_Protein_Matrix.csv')
    file_paths = glob.glob(file_pattern, recursive=True)

    if not file_paths:
        print(f"No files found in {base_folder_path}. Please check the path and ensure the data exists.")
        exit(1)

    # Initialize counters
    X, y = [], []
    for X_sample, y_sample in sample_generator(file_paths, sample_limit=total_samples):
        X.append(X_sample)
        y.append(y_sample)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.float32)

    # Split data into training and testing sets
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
    

    # Create metrics callback
    metrics_callback = MetricsCallback(validation_data=(X_test, y_test))

    # Train and evaluate the model
    batch_size = 32
    epochs = 100
    model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size, validation_split=0.2, callbacks=[metrics_callback])
    test_loss, test_accuracy = model.evaluate(X_test, y_test)
    print(f"Test Accuracy: {test_accuracy:.2f}")
    epoch_metrics = metrics_callback.get_epoch_metrics()

    # Make predictions
    predictions = model.predict(X_test).flatten()

    # Calculate AUC
    auc_score = roc_auc_score(y_test, predictions)
    print(f"[INFO] AUC Score: {auc_score:.4f}")

    # Save results
    OUTPUT_DIR = "/Users/marcobenavides/repos/ML4FG/3D-RBP/Results"
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    results_file = os.path.join(OUTPUT_DIR, "CNN.txt")
    with open(results_file, "w") as f:
        for metrics in epoch_metrics:
            f.write(f"Epoch {metrics['epoch']}: Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}\n")
    print(f"[INFO] Epoch metrics saved to {results_file}")

import os
import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
import glob
import random
import argparse
from collections import defaultdict
from sklearn.metrics import roc_auc_score
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import EarlyStopping
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.callbacks import ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from collections import Counter

OUTPUT_DIR = "/Users/marcobenavides/repos/ML4FG/3D-RBP/Results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True
)

lr_scheduler = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.5,
    patience=5,
    verbose=1
)


# Set seed for reproducibility
random.seed(42)
np.random.seed(42)
tf.random.set_seed(42)
total_samples = 43000

# Function to create samples by concatenating the first 1003 rows with each 101-row segment
def create_samples(file_path, first_segment_length=1003, segment_length=101):
    """Creates samples by concatenating the first 1003 rows with each 101-row segment in sequence."""
    combined_df = pd.read_csv(file_path)
    columns_to_drop = ["Index","Protein_Label", "Sequence_Name", "Position", "Binding_Class"]
    combined_df = combined_df.drop(columns=[col for col in columns_to_drop if col in combined_df.columns], errors='ignore')
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

# Generator to yield balanced samples across 34 classes
def sample_generator(file_paths, total_samples, first_segment_length=1003, segment_length=101):
    samples = []
    max_per_class = total_samples // 17  # For 17 proteins (binding samples only)

    counts = defaultdict(int)  # Track counts for each protein (binding samples only)

    for file_path in file_paths:
        combined_df = pd.read_csv(file_path)
        binding_labels = combined_df['Binding_Class'].values

        print(f"File: {file_path}, Binding: {sum(binding_labels == 1)}")

        # Extract Protein_Label for the first segment only (first 1003 rows)
        protein_label = combined_df.loc[:first_segment_length - 1, 'Protein_Label'].values[5]

        # Create samples
        samples_in_file = create_samples(file_path, first_segment_length, segment_length)

        # Iterate over the `Binding_Class` labels and corresponding segments
        binding_labels = combined_df.loc[first_segment_length:, 'Binding_Class'].values  # Get all Binding_Class values after the first segment

        for i, sample in enumerate(samples_in_file):
            # Calculate the index for the 5th value in the current segment
            label_idx = first_segment_length + i * segment_length + 4  # Always take the 5th row of the current segment
            
            if label_idx >= len(binding_labels):  # Avoid out-of-bounds access
                break

            # Retrieve the label from the 5th row of the current segment
            binding_label = binding_labels[label_idx]

            # Only process binding samples (binding_label == 1)
            if binding_label == 1:
                label = protein_label - 1  # Binding classes for proteins (0-16)
                if counts[label] < max_per_class:
                    samples.append((sample[:, :-1], label))  # Exclude true label from features
                    counts[label] += 1

            # Stop if we've reached the total sample limit
            if len(samples) >= total_samples:
                break

    random.shuffle(samples)
    print(f"[INFO] Total binding samples collected: {len(samples)}")
    print(f"[INFO] Samples per protein: {dict(counts)}")
    return samples

def split_and_track_assignments(X, y):
    # Step 1: Split into training and testing
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    # Step 2: Further split training data into training and validation
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )

    # Step 3: Track assignments
    assignments = (
        ["train"] * len(X_train_final) +
        ["val"] * len(X_val) +
        ["test"] * len(X_test)
    )
    return X_train_final, X_val, X_test, y_train_final, y_val, y_test, assignments

# Helper function to interpret predicted class labels
def interpret_class_label(class_index):
    protein_number = class_index + 1
    return f"RNA binds to Protein {protein_number}"


class PositionalEmbedding(tf.keras.layers.Layer):
    def __init__(self, sequence_length, embedding_dim):
        super(PositionalEmbedding, self).__init__()
        self.sequence_length = sequence_length
        self.embedding_dim = embedding_dim
        self.positions = self.add_weight(
            shape=(sequence_length, embedding_dim),
            initializer='uniform',
            trainable=True,
            name='positional_embeddings'
        )

    def call(self, inputs):
        return inputs + self.positions

# Define Transformer block
def transformer_block(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    x = layers.MultiHeadAttention(key_dim=head_size, num_heads=num_heads)(inputs, inputs)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    res = x + inputs  # Residual connection

    x = layers.Dense(ff_dim, activation="relu")(res)
    x = layers.BatchNormalization()(x)
    x = layers.Dense(inputs.shape[-1])(x)
    x = layers.Dropout(dropout)(x)
    x = layers.LayerNormalization(epsilon=1e-6)(x)
    return x + res  # Residual connection

# Define model with Transformer and CNN
def create_transformer_cnn_model(input_shape, num_classes=17, transformer_params=None):
    if transformer_params is None:
        transformer_params = {
            "head_size": 128,
            "num_heads": 18,
            "ff_dim": 128,
            "num_transformer_blocks": 2,
            "dropout": 0.5,
        }

    inputs = layers.Input(shape=input_shape)

    # CNN layers
    x = layers.Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=512, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=1024, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    x = layers.Conv1D(filters=1024, kernel_size=3, activation='relu', padding='same')(x)
    x = layers.MaxPooling1D(pool_size=2)(x)

    # Positional Embedding AFTER CNN layers
    sequence_length = x.shape[1]  # Sequence length after CNN layers
    embedding_dim = x.shape[2]   # Embedding dimension (number of filters)
    positional_embedding = PositionalEmbedding(sequence_length=sequence_length, embedding_dim=embedding_dim)
    x = positional_embedding(x)



    # Transformer layers
    for _ in range(transformer_params["num_transformer_blocks"]):
        x = transformer_block(
            x,
            head_size=transformer_params["head_size"],
            num_heads=transformer_params["num_heads"],
            ff_dim=transformer_params["ff_dim"],
            dropout=transformer_params["dropout"],
        )

    # Classification layers
    x = layers.Flatten()(x)
    x = layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = layers.Dropout(0.3)(x)
    x = layers.BatchNormalization()(x)

    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation='softmax')(x)

    model = models.Model(inputs, outputs)
    return model


class MetricsCallback(Callback):
    def __init__(self, validation_data, num_classes):
        super().__init__()
        self.validation_data = validation_data
        self.num_classes = num_classes
        self.epoch_metrics = []

    def on_epoch_end(self, epoch, logs=None):
        val_x, val_y = self.validation_data
        val_predictions = self.model.predict(val_x)  # Shape: [samples, num_classes]
        
        # One-hot encode val_y if not already
        val_y_one_hot = tf.keras.utils.to_categorical(val_y, num_classes=self.num_classes)

        try:
            auc = roc_auc_score(val_y_one_hot, val_predictions, multi_class="ovr")
        except ValueError as e:
            print(f"[WARNING] AUC computation failed: {e}")
            auc = None  # Default to None if AUC can't be computed
        
        accuracy = logs.get("val_accuracy", None)
        self.epoch_metrics.append({"epoch": epoch + 1, "accuracy": accuracy, "auc": auc})
        print(f"Epoch {epoch + 1} - Validation AUC: {auc if auc else 'Unavailable'}, Validation Accuracy: {accuracy if accuracy else 'Unavailable'}")
    
    def get_epoch_metrics(self):
        return self.epoch_metrics


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

    # Prepare data: Only binding samples
    samples = sample_generator(file_paths, total_samples)
    X = np.array([sample[0] for sample in samples], dtype=np.float32)
    y = np.array([sample[1] for sample in samples], dtype=np.int32)


from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score

# Explicitly split the data into training, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

# Rescale the data using MinMaxScaler
scaler = MinMaxScaler()
X_train_scaled = scaler.fit_transform(X_train.reshape(-1, X_train.shape[-1])).reshape(X_train.shape)
X_val_scaled = scaler.transform(X_val.reshape(-1, X_val.shape[-1])).reshape(X_val.shape)
X_test_scaled = scaler.transform(X_test.reshape(-1, X_test.shape[-1])).reshape(X_test.shape)

# Compile the model
input_shape = X_train.shape[1:]
model = create_transformer_cnn_model(input_shape, num_classes=17)
model.compile(optimizer=Adam(learning_rate=0.0005),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

metrics_callback = MetricsCallback(validation_data=(X_val_scaled, y_val), num_classes=17)

# Train the model
model.fit(
    X_train_scaled, y_train,
    validation_data=(X_val_scaled, y_val),
    epochs=100,
    batch_size=128,
    callbacks=[early_stopping, lr_scheduler, metrics_callback]
)

# Evaluate the model on the test set
test_predictions = model.predict(X_test_scaled)
y_test_pred = np.argmax(test_predictions, axis=1)
test_loss, test_accuracy = model.evaluate(X_test_scaled, y_test)
test_auc = roc_auc_score(
    tf.keras.utils.to_categorical(y_test, num_classes=17),
    test_predictions, multi_class="ovr"
)
print(f"Test Accuracy: {test_accuracy:.4f}")
print(f"Test AUC: {test_auc:.4f}")

# Save predictions and ground truth
ground_truth_labels = np.concatenate([y_train, y_val, y_test])
data_assignments = (
    ["train"] * len(y_train) +
    ["val"] * len(y_val) +
    ["test"] * len(y_test)
)
predicted_classes = np.concatenate([
    np.argmax(model.predict(X_train_scaled), axis=1),
    np.argmax(model.predict(X_val_scaled), axis=1),
    y_test_pred,
])
output_file = os.path.join(OUTPUT_DIR, "predictions_ground_truth.csv")
pd.DataFrame({
    "Ground_Truth": ground_truth_labels,
    "Predicted_Class": predicted_classes,
    "Assignment": data_assignments,
}).to_csv(output_file, index=False)
print(f"[INFO] Predictions saved to {output_file}")

epoch_metrics = metrics_callback.get_epoch_metrics()

# Save test accuracy and AUC to a file
results_file = os.path.join(OUTPUT_DIR, "positive_samples.txt")
with open(results_file, "w") as f:
    for metrics in epoch_metrics:
        f.write(f"Epoch {metrics['epoch']}: Accuracy: {metrics['accuracy']:.4f}, AUC: {metrics['auc']:.4f}\n")
print(f"[INFO] Epoch metrics saved to {results_file}")

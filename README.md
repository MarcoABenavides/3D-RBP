# Predicting RNA-Binding Protein Sites and Protein Affinity using a Hybrid Convolutional Neural Network and Transformer Model with Distance-Based Attention for Spatial Dependencies

This project aims to predict RNA-protein interactions by leveraging a hybrid machine learning approach combining Convolutional Neural Networks (CNNs) and Transformer models. By integrating primary, secondary, and tertiary structural data of RNA and proteins, this method seeks to accurately identify RNA-binding protein (RBP) interactions and reveal the spatial dependencies that characterize RNA-protein interactions.

---

## Project Overview

### 1. Biological Question
The project explores how RNA nucleotides interact spatially with protein binding sites in 3D space. Modeling these relationships between RNA and protein structures enhances our understanding of RNA-protein interactions and helps predict which proteins bind to specific RNA regions.

### 2. Machine Learning Approach
This project implements a hybrid model that combines:
- **CNNs**: Capture short-distance relationships by analyzing primary and secondary structures.
- **Transformers**: Model long-distance spatial relationships with distance-based attention, focusing on 3D interactions.

**Inputs**:
- One-hot encoded RNA primary sequences (A, U, C, G).
- Binary matrices for RNA secondary structures (stems, loops, hairpins).
- One-hot encoded amino acid sequences.
- Binary encodings for protein secondary structures (alpha-helix, beta-sheet, coil).
- Distance matrices for protein tertiary structures (atomic coordinates: X, Y, Z).
- Predicted RNA tertiary structures generated using Rosetta or RNAComposer.

**Outputs**:
- Binary classification indicating RNA binding sites.
- Multi-class predictions for protein-specific binding probabilities.

---

## Data Sources

- **Primary Datasets**: RNA sequences sourced from the iONMF dataset, containing 24,000 training samples, 6,000 validation samples, and 10,000 independent test samples for 17 RBPs.
- **3D Structural Data**: Protein priamry, secondary and 3D information was obtained from the Protein Data Bank (PDB), and RNA secondary structures predicted using the ViennaRNA package. 

---

## Code Overview

### Python Scripts (once the data has been downloaded from the following link: https://drive.google.com/drive/folders/1ZjYIe1ekvKt9Xe3HMamf5_BPcrilMc5J and the libraries have been installed run the codes in the following order):
1. **RNA_features.py**: Encodes RNA sequences, secondary structures, and binding labels.
2. **Protein_features.py**: Extracts protein sequences, secondary structures, and tertiary structure coordinates from PDB files.
3. **Concatenate.py**: Concatenates the Protein encoded data to its respective RNA data pair.
4. **Data-distribution_analysis.ipynb**: Analyzes RNA and protein feature distributions, including nucleotide, structural, and binding class counts.
These can be run independently:

4. **CNN.py**: Implements the initial CNN for binary classification.
5. **Balanced-CNN.py**: Balances the dataset for improved predictions.
6. **Balanced-CNN + 3D Data.py**: Adds 3D protein data to the CNN.
7. **Balanced-CNN + 3D Data + Softmax.py**: Uses softmax for multi-class predictions.
8. **Balanced-Transformer-CNN + 3D Data + Softmax.py**: Integrates Transformers and positional embeddings.

---

## Feature Analysis

### Data Distribution Analysis
The `Data-distribution_analysis.ipynb` script performs the following:
- Assesses nucleotide distributions (A, C, G, U) in RNA sequences.
- Examines RNA secondary structure types (stems, loops, hairpins, no structure).
- Analyzes protein secondary structure distributions (helix, strand, coil).
- Visualizes binding class distributions (binding vs non-binding).

### Visual Outputs:
- Bar plots for nucleotide and RNA secondary structure distributions.
- Bar plots for protein secondary structure distributions.
- Confusion matrices for class predictions.
- PCA and t-SNE visualizations of encoded features for binding vs non-binding clustering.

---

## Model Architectures and Performance

| **Model Architecture**                      | **Purpose**                             | **Description**                                                                                               |
|---------------------------------------------|-----------------------------------------|---------------------------------------------------------------------------------------------------------------|
| **CNN**                                     | Binary classification of RNA-RBP binding | Convolutional layers with pooling operations and a sigmoid activation layer.                                  |
| **Balanced CNN**                            | Address data imbalance                  | CNN with balanced class representation for improved prediction.                                               |
| **Balanced CNN + 3D Protein Data**          | Enhance binding prediction              | Adds 3D protein structural features to the CNN architecture.                                                 |
| **Balanced CNN + 3D Data + Softmax**        | Multi-class prediction                  | Incorporates a softmax layer for predicting protein-specific binding probabilities.                           |
| **Balanced Transformer + CNN + 3D Data + Softmax** | Integrate sequence-specific and structural features | Adds Transformer layers after the CNN for feature integration.                                                |
| **Balanced Transformer + CNN + 3D Data + Positional Embedding** | Enhance feature representation           | Adds trainable positional embeddings with CNN and Transformer layers, concluding with softmax.                |

### Key Performance Metrics:
- **Binary Models**:
  - **CNN**: Accuracy: 81.16%, Validation Accuracy: 81.18%
  - **Balanced CNN**: Improved performance with balanced datasets.
  - **Balanced CNN + 3D Data**: Incorporates 3D structural data for enhanced accuracy.
- **Multi-Class Models**:
  - **Balanced CNN + 3D Data + Softmax**: Handles 34 classes, reaching a maximum accuracy of 50%.
  - **Balanced Transformer + CNN + 3D Data + Positional Embedding**: Improved test accuracy to 87% with reduced learning rate.

---

## Visualization Tools

### Combined Accuracy and AUC Plots:
The script analyzes `.txt` result files to extract epoch-wise accuracy and AUC metrics for all models, generating combined plots for comparison.

### Prediction Analysis:
1. Overall accuracy.
2. Classification report and confusion matrix.
3. Accuracy per class and train vs. test assignment analysis.
4. Misclassification trends using heatmaps.

---

## Dependencies

### Required Libraries:
```plaintext
numpy
pandas
tensorflow
scikit-learn
matplotlib
argparse
```

### Additional Imports:
```python
import os
import glob
import seaborn as sns
from sklearn.metrics import roc_auc_score, confusion_matrix, classification_report, accuracy_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
```

---

## Future Directions

1. **Optimize Transformers**: Fine-tune attention heads, layers, and learning rates.
2. **Refine Positional Embeddings**: Experiment with alternative strategies for better spatial representation.
3. **Data Quality**: Clean inconsistencies in datasets and standardize annotations across experiments.
4. **Expand Features**: Add protein surface properties and evolutionary conservation features.
5. **Scale Up**: Train on larger, more diverse datasets for improved generalizability.

--- 


# Predicting RNA-Binding Protein Sites and Protein Affinity using a Hybrid Convolutional Neural Network and Transformer Model with Distance-Based Attention for Spatial Dependencies

This project aims to predict RNA-protein interactions by leveraging a hybrid machine learning approach combining Convolutional Neural Networks (CNNs) and Transformer models. By integrating primary, secondary, and tertiary structural data of RNA and proteins, this method seeks to accurately identify RNA-binding protein (RBP) binding sites and reveal the spatial dependencies that characterize RNA-protein interactions.

## Project Proposal Overview

### 1. Biological Question
The project explores how RNA nucleotides interact spatially with protein binding sites in 3D space. Modeling these spatial relationships between RNA and protein structures will enhance our understanding of RNA-protein interactions and help predict which proteins bind to specific RNA regions.

### 2. Machine Learning Approach
This project implements a hybrid model that combines:
- **CNNs** to capture short-distance relationships by analyzing primary and secondary structures.
- **Transformers** to model long-distance spatial relationships with distance-based attention, focusing on 3D interactions.
  
**Inputs**:
- One-hot encoded primary structure of nucleotides (A, U, C, G)
- Binary matrix for RNA secondary structures (Stems, Loops, Hairpins)
- One-hot encoded primary structure of amino acids (e.g., Ala, Cys, Glu)
- Binary encoding for protein secondary structures (alpha-helix, beta sheets, or coils)
- Distance matrices representing protein tertiary structure (X, Y, Z alpha-carbon atomic coordinates)
- Predicted RNA tertiary structure generated using tools like Rosetta or RNAComposer

**Outputs**:
- Binary classification indicating binding sites for each RNA sequence 
- Likelihood of Proteins binding to non tested RNA sequences

### 3. Data Sources
- **Primary Datasets**: RNA sequences are sourced from the iONMF dataset (Blin, K., 2015), containing 24,000 training samples, 6,000 validation samples, and 10,000 independent test samples for 19 RBPs.
- **3D Structural Data**: Protein 3D structures are obtained from the Protein Data Bank (PDB), and RNA secondary structures are predicted using the Vienna RNA package. Tools like Rosetta or RNAComposer are used for RNA tertiary structures.

### 4. Interim Report Plans
To evaluate feasibility:
- **Data Preparation**: Download RNA sequences, Download protein primary sequences and secondary structures, generate RNA secondary structures.
- **Encoding**: Encode RNA and protein sequences and secondary structures, and construct Protein distance matrices for tertiary structures 
- **Model Testing**: Run a simplified CNN model to evaluate the architecture and adjust complexity as needed.

### 5. Performance Assessment
- **Supervised**: Validate binding sequences through against reference data.


### 6. Key References
- Pan, X., Rijnbeek, P., Yan, J., et al. (2018). *Prediction of RNA-protein sequence and structure binding preferences using deep convolutional and recurrent neural networks*. BMC Genomics, 19, 511. [doi:10.1186/s12864-018-4889-1](https://doi.org/10.1186/s12864-018-4889-1)
- Blin, K., et al. (2015). *DoRiNA 2.0—upgrading the doRiNA database of RNA interactions in post-transcriptional regulation*. Nucleic Acids Research, 43(D1), D160–D167. [doi:10.1093/nar/gku1153](https://doi.org/10.1093/nar/gku1153)
- Kloczkowski, A., et al. (2009). *Distance matrix-based approach to protein structure prediction*. Journal of Structural and Functional Genomics, 10(1), 67-81. [doi:10.1007/s10969-009-9062-2](https://doi.org/10.1007/s10969-009-9062-2)

---

## Code Overview

### Protein Feature Extraction (Protein_features.py)
- Extracts protein sequence and structural features from PDB files.
- Encodes primary and secondary structures and aligns coordinates to build a feature matrix.

### RNA Feature Extraction (RNA_features.py)
- Processes RNA sequences from .fa.gz files.
- Generates secondary structures using RNAfold, applies one-hot encoding, and assigns binding labels.

### Matrix Concatenation (Concatenate.py)
- Merges protein and RNA matrices, aligning features and zero-padding for compatibility.

### Convolutional Neural Network (CNN.py)
- A CNN model processes the concatenated matrices, iterating through each file and concatenating protein features with RNA segments to predict binding vs. non-binding sequences. The CNN drops 3-D data as this will be used later with the transformer.
### Dataset Summary

- **Total Samples**: 1,079,034
- **Binding Samples**: 295,375
- **Non-Binding Samples**: 783,659

Folders not processed due to missing data:
1. 1_PARCLIP_AGO1234_hg19
2. 2_PARCLIP_AGO2MNASE_hg19
3. 9_PARCLIP_ELAVL1MNASE_hg19
4. 14_PARCLIP_FUS_mut_hg19

---

### Important Notes

- **Processing Time**: RNA sequence processing takes approximately 2 days.

### Links to Protein Data

Below are links to the protein data sources:

- [1_PARCLIP_AGO1234_hg19](https://www.rcsb.org)
- [3_HITSCLIP_Ago2](https://www.rcsb.org/structure/4OLA)
- [4_HITSCLIP_Ago2](https://www.rcsb.org/structure/4OLA)
- [5_CLIPSEQ_AGO2_h19](https://www.rcsb.org/structure/4OLA)
- [6_CLIP-seq-eIF4AIII_1](https://www.rcsb.org/structure/2HXY)
- [7_CLIP-seq-eIF4AIII_2](https://www.rcsb.org/structure/2HXY)
- [8_PARCLIP_ELAVL1_h19](https://www.rcsb.org/structure/4FXV)
- [11_CLIPSEQ_ELAVL1_h19](https://www.rcsb.org/structure/4FXV)
- [12_PARCLIP_EWSR1_h19](https://www.rcsb.org/structure/2CPE)
- [13_PARCLIP_FUS_h19](https://www.rcsb.org/structure/6GBM)
- [15_PARCLIP_IGF2BP123_h19](https://www.rcsb.org/structure/6ROL)
- [16_ICLIP_hnRNPC_Hela_iCLIP_all_clusters](https://www.rcsb.org/structure/1TXP)
- [17_ICLIP_hnRNPC_hg19](https://www.rcsb.org/structure/1TXP)
- [18_ICLIP_hnRNPL](https://www.rcsb.org/sequence/3r27)
- [19_ICLIP_hnRNPL_U266](https://www.rcsb.org/sequence/3r27)
- [20_ICLIP_hnRNPlike_U266](https://www.rcsb.org/sequence/3r27)
- [21_PARCLIP_MOV10_Sievers_hg19](https://alphafold.ebi.ac.uk/entry/Q9HCE1)
- [22_ICLIP_NSUN2](https://alphafold.ebi.ac.uk/entry/Q08J23)
- [23_PARCLIP_PUM2_hg19](https://www.rcsb.org/structure/3q0p)
- [24_PARCLIP_QKI_hg19](https://www.rcsb.org/structure/4jvh)
- [25_CLIPSEQ_SFRS1_hg19](https://www.rcsb.org/structure/1X4A)
- [26_PARCLIP_TAF15_h19](https://www.rcsb.org/structure/8ONS)
- [27_ICLIP_TDP43_h19](https://www.rcsb.org/structure/8CGG)
- [28_ICLIP_TIA1_h19](https://www.rcsb.org/structure/2MJN)
- [29_ICLIP_TIAL1_h19](https://www.rcsb.org/structure/2MJN)
- [29_ICLIP_U2AF65_Hela_iCLIP_ctrl_all_clusters](https://www.rcsb.org/structure/5EV4)
- [30_ICLIP_U2AF65_Hela_iCLIP_ctrl+kd_all_clusters](https://www.rcsb.org/structure/5EV4)
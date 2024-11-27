import pandas as pd
import os

def find_rna_and_protein_files(base_folder):
    """Finds all RNA and protein file pairs in the specified base directory."""
    file_pairs = []
    
    # Walk through each folder in the "clip" directory
    for root, dirs, files in os.walk(base_folder):
        for dir_name in dirs:
            rna_file = os.path.join(root, dir_name, "RNA_sequences_combined_stacked.csv")
            protein_folder = os.path.join(root, dir_name, "Protein data")
            
            # Check if the RNA file and Protein folder exist
            if os.path.isfile(rna_file) and os.path.isdir(protein_folder):
                # Find the protein file in the protein folder
                for protein_file in os.listdir(protein_folder):
                    if protein_file.endswith(".csv"):
                        protein_file_path = os.path.join(protein_folder, protein_file)
                        file_pairs.append((rna_file, protein_file_path, os.path.join(root, dir_name)))
                        break  # Assuming only one protein file per "Protein data" folder
    return file_pairs

def concatenate_rna_protein_matrices(rna_file_path, protein_file_path, output_folder):
    """Concatenate RNA output below the protein output, adding zero-filled columns and moving RNA columns to the right."""
    # Define the output file path within the output folder
    output_file_path = os.path.join(output_folder, "Combined_RNA_Protein_Matrix.csv")
    
    # Check if the combined file already exists
    if os.path.exists(output_file_path):
        print(f"Skipping {output_file_path} as it already exists.")
        return

    # Load RNA and protein data matrices
    rna_df = pd.read_csv(rna_file_path)
    protein_df = pd.read_csv(protein_file_path)
    
    # Identify unique columns in each DataFrame
    protein_columns = protein_df.columns
    rna_columns = [col for col in rna_df.columns if col not in protein_columns]
    
    # Reorder RNA DataFrame to have protein columns first, followed by its unique columns
    for col in protein_columns:
        if col not in rna_df.columns:
            rna_df[col] = 0  # Add missing protein columns to RNA and fill with zeros
    rna_df = rna_df[list(protein_columns) + rna_columns]  # Order RNA columns as protein columns first, then RNA-specific
    
    # Add missing RNA columns to protein DataFrame, filled with zeros
    for col in rna_columns:
        protein_df[col] = 0
    
    # Concatenate the DataFrames vertically (protein on top, RNA below)
    combined_df = pd.concat([protein_df, rna_df], ignore_index=True)
    
    # Ensure the output folder exists
    os.makedirs(output_folder, exist_ok=True)
    
    # Save the combined DataFrame to a single CSV file
    combined_df.to_csv(output_file_path, index=False)
    
    # Print debug information to confirm save path
    print(f"Combined matrix saved to: {output_file_path}")
    print(f"Expected output folder: {output_folder}\n")

def process_all_pairs(base_folder):
    """Process all RNA and protein pairs found in the specified base directory."""
    file_pairs = find_rna_and_protein_files(base_folder)
    
    for rna_file, protein_file, output_folder in file_pairs:
        print(f"Processing RNA file: {rna_file}")
        print(f"Processing Protein file: {protein_file}")
        print(f"Output will be saved in: {output_folder}")
        concatenate_rna_protein_matrices(rna_file, protein_file, output_folder)

if __name__ == "__main__":
    # Dynamically calculate the base folder path
    script_dir = os.path.dirname(os.path.abspath(__file__))
    base_folder_path = os.path.join(script_dir, "Data", "datasets", "clip")

    # Ensure the folder exists
    if not os.path.exists(base_folder_path):
        print(f"Error: Base folder {base_folder_path} does not exist.")
        exit(1)

    # Process all pairs
    process_all_pairs(base_folder_path)

import os
import subprocess
import gzip
import pandas as pd
import numpy as np
import time

# File to keep track of processed sequences
PROCESSED_SEQUENCES_FILE = 'processed_sequences.txt'

# Open files to read RNA sequences, splitting the title of the protein from the RNA sequence
def read_sequences_from_gz(file_path):
    """Read sequences from a gzipped FASTA file."""
    print(f"Reading sequences from {file_path}...")
    with gzip.open(file_path, 'rt') as f:
        sequences = f.read().strip().split('>')
        print(f"Found {len(sequences) - 1} sequences.")
        return {seq[0]: seq[1].replace('\n', '') for seq in (s.split('\n', 1) for s in sequences[1:])}

# Predict secondary structure and folding energy
def run_rna_fold(sequence):
    """Run RNAfold and return the output."""
    print("Running RNAfold...")
    process = subprocess.Popen(['echo', sequence], stdout=subprocess.PIPE)
    output = subprocess.check_output(['RNAfold'], stdin=process.stdout)
    process.wait()
    print("RNAfold completed.")
    
    # Parse output to get the folding energy and structure
    output_lines = output.decode('utf-8').strip().splitlines()
    if len(output_lines) > 1:
        structure = output_lines[1].strip().split()[0]  # Get the structure from the second line
        folding_energy = output_lines[1].strip().split()[-1]  # Get the energy from the second line
        return structure, folding_energy
    else:
        return None, None

def map_nucleotides_to_structure(sequence, structure):
    """Map each nucleotide to its corresponding structure category (stem, loop, or no structure)."""
    mapping = ['NS'] * len(sequence)  # Default to 'No Structure' for all nucleotides
    stack = []  # To keep track of the indices of paired nucleotides

    # Iterate over the structure
    for i, char in enumerate(structure):
        if char == '(':
            if i < len(sequence):
                mapping[i] = 'S'  # Start of a stem
                stack.append(i)  # Push the index of the opening brace onto the stack
        elif char == ')':
            if stack:  # Ensure there is a corresponding opening brace
                opening_index = stack.pop()  # Get the index of the matching '('
                if opening_index + 1 < i:  # If there are bases between the stems
                    mapping[opening_index + 1:i] = ['H'] * (i - opening_index - 1)  # Mark hairpin region
                mapping[i] = 'S'  # Mark the current index as part of a stem
        elif char == '.':
            if i < len(sequence):
                mapping[i] = 'L'  # Loop (unpaired)

    return mapping  # Return the mapping directly for encoding

def one_hot_encode_sequence(sequence):
    """One-hot encode the nucleotide sequence, ignoring sequences with 'N'."""
    encoding_seq = {
        'A': [1, 0, 0, 0],
        'C': [0, 1, 0, 0],
        'G': [0, 0, 1, 0],
        'T': [0, 0, 0, 1]
    }
    # Skip any sequence with 'N'
    if 'N' in sequence:
        print("Skipping sequence due to 'N' character.")
        return None  # Return None for sequences with 'N'
    
    encoded_sequence = [encoding_seq[nuc] for nuc in sequence]
    print("One-hot encoding completed for sequence.")
    return encoded_sequence

def one_hot_encode_structure(structure_mapping):
    """One-hot encode the RNA secondary structure based on mapping."""
    encoding_structure = {
        'S': [1, 0, 0, 0],  # Stems
        'L': [0, 1, 0, 0],  # Loops
        'H': [0, 0, 1, 0],  # Hairpins
        'NS':[0, 0, 0, 1],  # No Structure
        
    }

    encoded_structure = [encoding_structure[mapping] for mapping in structure_mapping]
    print("One-hot encoding completed for structure.")
    return encoded_structure

def concatenate_encodings(encoded_sequence, encoded_structure):
    """Concatenate encoded sequence and encoded structure matrices."""
    return [seq + struct for seq, struct in zip(encoded_sequence, encoded_structure)]

def transform_encoding_to_matrix(sequence, concatenated_encoding, binding_class):
    """Transform encoding into an Nx9 matrix where N is the sequence length, with nucleotide, structure encoding, and binding class."""
    num_nucleotides = len(sequence)
    encoding_matrix = np.zeros((num_nucleotides, 9), dtype=int)  # Nx9 matrix to include binding class as well

    for i, encoding in enumerate(concatenated_encoding):
        nucleotide_encoding = encoding[:4]  # First 4 elements for nucleotide one-hot encoding
        structure_encoding = encoding[4:]   # Next 4 elements for structure one-hot encoding

        # Fill the matrix for each nucleotide position
        encoding_matrix[i, :4] = nucleotide_encoding  # Columns 0-3 for nucleotide encoding
        encoding_matrix[i, 4:8] = structure_encoding  # Columns 4-7 for structure encoding
        encoding_matrix[i, 8] = 1 if binding_class == "Binding" else 0  # Column 8 for binding class (1 for Binding, 0 for Not Binding)

    return encoding_matrix


def extract_binding_class(name):
    """Extracts binding class from sequence name."""
    if "class:1" in name:
        return "Binding"
    elif "class:0" in name:
        return "Not Binding"
    return "Unknown"

def load_processed_sequences():
    """Load already processed sequence names from file."""
    if os.path.exists(PROCESSED_SEQUENCES_FILE):
        with open(PROCESSED_SEQUENCES_FILE, 'r') as f:
            return set(line.strip() for line in f)
    return set()

def save_processed_sequence(name):
    """Save the name of a processed sequence to file."""
    with open(PROCESSED_SEQUENCES_FILE, 'a') as f:
        f.write(name + '\n')

def process_sequences_in_directory(directory):
    """Process sequences in the given directory and save to a single vertically stacked CSV file."""
    output_filename = os.path.join(directory, 'RNA_sequences_combined_stacked.csv')
    if os.path.exists(output_filename):
        print(f"Skipping {directory} as {output_filename} already exists.")
        return

    print(f"Processing sequences in directory: {directory}...")
    processed_sequences = load_processed_sequences()  # Load processed sequence names
    combined_df = pd.DataFrame()  # Initialize an empty DataFrame to hold all sequences

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.fa.gz'):  # Only process .fa.gz files
                file_path = os.path.join(root, filename)
                print(f"Processing file: {file_path}")
                
                sequences = read_sequences_from_gz(file_path)
                
                for name, seq in sequences.items():
                    if name in processed_sequences:
                        print(f"Skipping already processed sequence: {name}")
                        continue  # Skip sequences that have already been processed
                    
                    print(f'Processing sequence: {name}...')
                    structure_output, folding_energy = run_rna_fold(seq)
                    structure_mapping = map_nucleotides_to_structure(seq, structure_output)

                    # One-hot encode sequence and structure
                    encoded_sequence = one_hot_encode_sequence(seq)
                    encoded_structure = one_hot_encode_structure(structure_mapping)

                    # Skip sequences if encoding failed (None returned)
                    if encoded_sequence is None or encoded_structure is None:
                        print(f"Skipping sequence {name} due to encoding issues.")
                        continue

                    # Concatenate the encoded sequences and structures
                    concatenated_encoding = concatenate_encodings(encoded_sequence, encoded_structure)

                    # Extract binding class
                    binding_class = extract_binding_class(name)

                    # Convert concatenated encoding into a matrix format with binding class as an additional column
                    encoding_matrix = transform_encoding_to_matrix(seq, concatenated_encoding, binding_class)

                    # Create DataFrame and add to combined DataFrame
                    index_labels = ['Nucleotide_A', 'Nucleotide_C', 'Nucleotide_G', 'Nucleotide_T',
                                    'Structure_Stem', 'Structure_Loop', 'Structure_Hairpin',
                                    'Structure_No_Structure', 'Binding_Class']
                    df_sequence = pd.DataFrame(encoding_matrix, columns=index_labels)
                    df_sequence.insert(0, "Position", [f"Position_{i}" for i in range(1, len(seq) + 1)])
                    df_sequence.insert(0, "Sequence_Name", [name] * df_sequence.shape[0])

                    combined_df = pd.concat([combined_df, df_sequence], ignore_index=True)

                    save_processed_sequence(name)  # Save the sequence name as processed
                    print(f"Finished processing sequence: {name} from file {filename} in folder {directory}")

    # Save the combined DataFrame to a single CSV file
    combined_df.to_csv(output_filename, index=False)
    print(f"Saved all matrices to {output_filename}")
    print("Processing complete.")

def process_all_subdirectories(main_directory):
    """Process all subdirectories in the main directory if the output file does not already exist."""
    for subdir in os.listdir(main_directory):
        full_path = os.path.join(main_directory, subdir)
        if os.path.isdir(full_path):
            print(f"Starting processing for folder: {subdir}")
            process_sequences_in_directory(full_path)

if __name__ == "__main__":
    start_time = time.time()  # Start the timer
    main_directory = '/Users/marcobenavides/repos/ML-4-FG/3D-RBP/datasets/clip'
    process_all_subdirectories(main_directory)
    
    end_time = time.time()  # End the timer
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

import os
import subprocess
import gzip
import pandas as pd
import numpy as np
import time



# Helper Functions
def read_sequences_from_gz(file_path):
    """Read sequences from a gzipped FASTA file."""
    print(f"Reading sequences from {file_path}...")
    with gzip.open(file_path, 'rt') as f:
        sequences = f.read().strip().split('>')
        print(f"Found {len(sequences) - 1} sequences.")
        return {seq[0]: seq[1].replace('\n', '') for seq in (s.split('\n', 1) for s in sequences[1:])}


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
    mapping = ['NS'] * len(sequence)
    stack = []
    for i, char in enumerate(structure):
        if char == '(':
            mapping[i] = 'S'
            stack.append(i)
        elif char == ')':
            if stack:
                stack.pop()
                mapping[i] = 'S'
        elif char == '.':
            mapping[i] = 'L'
    return mapping


def one_hot_encode_sequence(sequence):
    """One-hot encode the nucleotide sequence."""
    encoding_seq = {'A': [1, 0, 0, 0], 'C': [0, 1, 0, 0], 'G': [0, 0, 1, 0], 'T': [0, 0, 0, 1]}
    if 'N' in sequence:
        print("Skipping sequence due to 'N' character.")
        return None
    return [encoding_seq[nuc] for nuc in sequence]


def one_hot_encode_structure(structure_mapping):
    """One-hot encode the RNA secondary structure based on mapping."""
    encoding_structure = {'S': [1, 0, 0, 0], 'L': [0, 1, 0, 0], 'H': [0, 0, 1, 0], 'NS': [0, 0, 0, 1]}
    return [encoding_structure[mapping] for mapping in structure_mapping]


def concatenate_encodings(encoded_sequence, encoded_structure):
    """Concatenate encoded sequence and encoded structure matrices."""
    return [seq + struct for seq, struct in zip(encoded_sequence, encoded_structure)]


def transform_encoding_to_matrix(sequence, concatenated_encoding, binding_class):
    """Transform encoding into an Nx9 matrix."""
    num_nucleotides = len(sequence)
    encoding_matrix = np.zeros((num_nucleotides, 9), dtype=int)
    for i, encoding in enumerate(concatenated_encoding):
        encoding_matrix[i, :4] = encoding[:4]
        encoding_matrix[i, 4:8] = encoding[4:]
        encoding_matrix[i, 8] = 1 if binding_class == "Binding" else 0
    return encoding_matrix


def extract_binding_class(name):
    """Extract binding class from sequence name."""
    if "class:1" in name:
        return "Binding"
    elif "class:0" in name:
        return "Not Binding"
    return "Unknown"


# Main Functions
def process_sequences_in_directory(directory):
    """Process sequences in a directory and save to a CSV file."""
    output_filename = os.path.join(directory, 'RNA_sequences_combined_stacked.csv')
    if os.path.exists(output_filename):
        print(f"Skipping {directory} as {output_filename} already exists.")
        return

    print(f"Processing sequences in directory: {directory}...")
    combined_df = pd.DataFrame()

    for root, _, files in os.walk(directory):
        for filename in files:
            if filename.endswith('.fa.gz'):
                file_path = os.path.join(root, filename)
                print(f"Processing file: {file_path}")
                
                sequences = read_sequences_from_gz(file_path)
                for name, seq in sequences.items():
                    print(f'Processing sequence: {name}...')
                    structure_output, _ = run_rna_fold(seq)
                    structure_mapping = map_nucleotides_to_structure(seq, structure_output)

                    encoded_sequence = one_hot_encode_sequence(seq)
                    encoded_structure = one_hot_encode_structure(structure_mapping)

                    if encoded_sequence is None or encoded_structure is None:
                        continue

                    concatenated_encoding = concatenate_encodings(encoded_sequence, encoded_structure)
                    binding_class = extract_binding_class(name)

                    encoding_matrix = transform_encoding_to_matrix(seq, concatenated_encoding, binding_class)
                    index_labels = ['Nucleotide_A', 'Nucleotide_C', 'Nucleotide_G', 'Nucleotide_T',
                                    'Structure_Stem', 'Structure_Loop', 'Structure_Hairpin',
                                    'Structure_No_Structure', 'Binding_Class']
                    df_sequence = pd.DataFrame(encoding_matrix, columns=index_labels)
                    df_sequence.insert(0, "Position", [f"Position_{i}" for i in range(1, len(seq) + 1)])
                    df_sequence.insert(0, "Sequence_Name", [name] * df_sequence.shape[0])

                    combined_df = pd.concat([combined_df, df_sequence], ignore_index=True)

    combined_df.to_csv(output_filename, index=False)
    print(f"Saved all matrices to {output_filename}")
    print("Processing complete.")


def process_all_subdirectories(main_directory):
    """Process all subdirectories in the main directory."""
    for subdir in os.listdir(main_directory):
        full_path = os.path.join(main_directory, subdir)
        if os.path.isdir(full_path):
            print(f"Starting processing for folder: {subdir}")
            process_sequences_in_directory(full_path)


if __name__ == "__main__":
    # Dynamically determine the main directory
    script_dir = os.path.dirname(os.path.abspath(__file__))
    main_directory = os.path.join(script_dir, "Data", "datasets", "clip")

    start_time = time.time()
    print(f"Processing sequences from directory: {main_directory}")

    if not os.path.exists(main_directory):
        print(f"Error: Directory {main_directory} does not exist.")
        exit(1)

    process_all_subdirectories(main_directory)
    end_time = time.time()
    print(f"Total time taken: {end_time - start_time:.2f} seconds")

import os
import pandas as pd

def extract_sequence_info_from_pdb(file_path):
    """Extract the amino acid count and sequence from SEQRES lines in a PDB file."""
    seqres_sequence = []
    expected_length = None

    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("SEQRES"):
                if expected_length is None:
                    expected_length = int(line[13:17].strip())
                seqres_sequence.extend(line[19:].strip().split())
    amino_acid_count = len(seqres_sequence)
    return expected_length, seqres_sequence[:expected_length]


def extract_ca_coordinates_with_fallback(file_path, seqres_sequence):
    """Extract CA coordinates with strict consecutive fallback matching."""
    exact_coordinates = {}
    fallback_coordinates = {}
    seq_index = 0  # Track the current index in SEQRES sequence

    # Step 1: Try exact matching by residue number and name
    residue_positions = {(res, i + 1): i for i, res in enumerate(seqres_sequence)}
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("ATOM") and line[13:15].strip() == "CA":
                residue_name = line[17:20].strip()
                residue_number = int(line[22:26].strip())
                x, y, z = map(float, (line[30:38].strip(), line[38:46].strip(), line[46:54].strip()))

                if (residue_name, residue_number) in residue_positions:
                    index = residue_positions[(residue_name, residue_number)]
                    exact_coordinates[index] = (x, y, z)

    # Step 2: Check if exact matching met criteria, otherwise use strict fallback
    if len(exact_coordinates) >= 10:
        print("Saved using exact coordinates")
        return exact_coordinates
    else:
        # Reset and start strict fallback matching
        with open(file_path, 'r') as f:
            prev_residue_number = None
            consecutive_matches = 0
            temp_fallback_coordinates = {}
            
            for line in f:
                if line.startswith("ATOM") and line[13:15].strip() == "CA":
                    residue_name = line[17:20].strip()
                    residue_number = int(line[22:26].strip())
                    
                    if seq_index < len(seqres_sequence) and residue_name == seqres_sequence[seq_index]:
                        # Check for strict consecutive in both SEQRES and ATOM
                        if prev_residue_number is None or residue_number == prev_residue_number + 1:
                            temp_fallback_coordinates[seq_index] = (
                                float(line[30:38].strip()),
                                float(line[38:46].strip()),
                                float(line[46:54].strip())
                            )
                            print(f"Fallback match aligned: SEQRES {residue_name} (Index {seq_index + 1}) with ATOM {residue_name} {residue_number}")
                            consecutive_matches += 1
                            prev_residue_number = residue_number
                            seq_index += 1
                        else:
                            # If not consecutive in ATOM, reset the temporary storage and start from this position
                            temp_fallback_coordinates = {}
                            consecutive_matches = 0
                            prev_residue_number = None
                            print("Non-consecutive ATOM found; resetting fallback alignment search.")

                        # If we achieve 3 strict consecutive matches, keep going to build alignment
                        if consecutive_matches >= 3:
                            fallback_coordinates.update(temp_fallback_coordinates)
                            temp_fallback_coordinates = {}
                    else:
                        # Reset if the residue names do not match, indicating an alignment issue
                        temp_fallback_coordinates = {}
                        consecutive_matches = 0
                        prev_residue_number = None

            # Confirm if enough fallback matches were achieved
            if len(fallback_coordinates) >= 10:
                return fallback_coordinates
            else:
                print("Fallback alignment insufficient; returning exact matches with 0 coordinates.")
                return exact_coordinates





def read_secondary_structure_from_pdb(file_path, sequence_length):
    """Read the secondary structure from a PDB file."""
    secondary_structure = ['C'] * sequence_length  # Default to 'C' for coil
    with open(file_path, 'r') as f:
        for line in f:
            if line.startswith("HELIX"):
                start_residue = int(line[21:25].strip())
                end_residue = int(line[33:37].strip())
                for i in range(start_residue - 1, end_residue):
                    if 0 <= i < sequence_length:
                        secondary_structure[i] = 'H'
            elif line.startswith("SHEET"):
                start_residue = int(line[22:26].strip())
                end_residue = int(line[33:37].strip())
                for i in range(start_residue - 1, end_residue):
                    if 0 <= i < sequence_length:
                        secondary_structure[i] = 'E'
    return secondary_structure



def one_hot_encode_sequence(sequence):
    """One-hot encode the amino acid sequence."""
    encoding_seq = {
        'ALA': [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'CYS': [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ASP': [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'GLU': [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'PHE': [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'GLY': [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'HIS': [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ILE': [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'LYS': [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'LEU': [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'MET': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
        'ASN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0],
        'PRO': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0],
        'GLN': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0],
        'ARG': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
        'SER': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],
        'THR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],
        'VAL': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],
        'TRP': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],
        'TYR': [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]
    }
    return [encoding_seq.get(amino, [0]*20) for amino in sequence]


def one_hot_encode_structure(structure):
    """One-hot encode the secondary structure and flatten the encoding."""
    encoding_structure = {
        'H': [1, 0, 0],
        'E': [0, 1, 0],
        'C': [0, 0, 1]
    }
    return [encoding_structure[sec] for sec in structure]


def concatenate_encodings(encoded_sequence, encoded_structure, coordinates, max_length):
    """Concatenate encoded sequences, structures, and coordinates into a matrix format and pad to max_length."""
    concatenated_data = []
    for i in range(max_length):
        # Sequence encoding: use actual encoding or zero-pad if beyond sequence length
        seq_encoding = encoded_sequence[i] if i < len(encoded_sequence) else [0] * 20
        
        # Structure encoding: use actual encoding or zero-pad if beyond sequence length
        struct_encoding = encoded_structure[i] if i < len(encoded_structure) else [0, 0, 0]
        
        # Coordinates: use actual coordinates or (0, 0, 0) if beyond sequence length
        coord = coordinates.get(i, (0.0, 0.0, 0.0)) if i < len(coordinates) else (0.0, 0.0, 0.0)
        
        # Append the row with sequence index, sequence encoding, structure encoding, and coordinates
        concatenated_data.append([i + 1] + seq_encoding + struct_encoding + list(coord))
    return concatenated_data



def process_protein(pdb_file_path, max_length):
    """Process the protein to extract amino acids, secondary structures, and coordinates."""
    expected_length, seqres_sequence = extract_sequence_info_from_pdb(pdb_file_path)
    coordinates = extract_ca_coordinates_with_fallback(pdb_file_path, seqres_sequence)
    secondary_structure = read_secondary_structure_from_pdb(pdb_file_path, expected_length)
    encoded_sequence = one_hot_encode_sequence(seqres_sequence)
    encoded_structure = one_hot_encode_structure(secondary_structure)
    concatenated_data = concatenate_encodings(encoded_sequence, encoded_structure, coordinates, max_length)
    
    amino_acids = ["ALA", "CYS", "ASP", "GLU", "PHE", "GLY", "HIS", "ILE", "LYS", "LEU", 
                   "MET", "ASN", "PRO", "GLN", "ARG", "SER", "THR", "VAL", "TRP", "TYR"]
    struct_labels = ["Helix", "Strand", "Coil"]
    columns = ["Index"] + amino_acids + struct_labels + ["X", "Y", "Z"]
    big_matrix_df = pd.DataFrame(concatenated_data, columns=columns)
    return big_matrix_df


def save_to_csv(big_matrix_df, output_file_prefix):
    """Save the concatenated matrix to a CSV file."""
    output_file_path = f"{output_file_prefix}_Concatenated_Encoding_Matrix.csv"
    big_matrix_df.to_csv(output_file_path, index=False)
    print(f"Results saved successfully to {output_file_path}")


def find_max_length(base_folder_path):
    """Find the maximum sequence length across all PDB files."""
    max_length = 0
    for root, dirs, files in os.walk(base_folder_path):
        if os.path.basename(root) == 'Protein data':
            for pdb_file in [f for f in files if f.endswith('.pdb')]:
                pdb_file_path = os.path.join(root, pdb_file)
                _, seqres_sequence = extract_sequence_info_from_pdb(pdb_file_path)
                max_length = max(max_length, len(seqres_sequence))
    return max_length


def process_all_pdb_files_in_clip_folder(base_folder_path):
    """Process all PDB files in each 'Protein data' folder within the specified base folder."""
    max_length = find_max_length(base_folder_path)  # Determine the max length across all files
    for root, dirs, files in os.walk(base_folder_path):
        if os.path.basename(root) == 'Protein data':
            for pdb_file in [f for f in files if f.endswith('.pdb')]:
                pdb_file_path = os.path.join(root, pdb_file)
                print(f"Processing file: {pdb_file_path}")
                big_matrix_df = process_protein(pdb_file_path, max_length)
                output_file_prefix = os.path.join(root, os.path.splitext(pdb_file)[0])
                save_to_csv(big_matrix_df, output_file_prefix)


if __name__ == "__main__":
    base_folder_path = '/Users/marcobenavides/repos/ML-4-FG/3D-RBP/datasets/clip'
    process_all_pdb_files_in_clip_folder(base_folder_path)

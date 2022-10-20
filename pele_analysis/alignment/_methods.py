from Bio import SeqIO, AlignIO
import json

def readFastaFile(fasta_file):
    """
    Read a fasta file and get the sequences as a dictionary

    Parameters
    ----------
    fasta_file : str
        Path to the input fasta file

    Returns
    -------
    sequences : dict
        Dictionary containing the IDs and squences in the fasta file.
    """

    sequences = {}
    for record in SeqIO.parse(fasta_file, "fasta"):
        sequences[record.id] = str(record.seq)

    return sequences

def writeFastaFile(sequences, output_file):
    """
    Write sequences to a fasta file.

    Parameters
    ----------
    sequences : dict
        Dictionary containing as values the strings representing the sequences
        of the proteins to align and their identifiers as keys.

    output_file : str
        Path to the output fasta file
    """

    # Write fasta file containing the sequences
    with open(output_file, 'w') as of:
        for name in sequences:
            of.write('>'+name+'\n')
            of.write(sequences[name]+'\n')

def writeMsaToFastaFile(msa, output_file):
    """
    Write sequences inside an MSA to a fasta file.

    Parameters
    ----------
    msa : Bio.AlignIO
        Multiple sequence aligment in Biopython format.

    output_file : str
        Path to the output fasta file
    """

    # Write fasta file containing the sequences
    with open(output_file, 'w') as of:
        for s in msa:
            of.write('>'+s.id+'\n')
            of.write(str(s.seq)+'\n')


def readMsaFromFastaFile(alignment_file):
    """
    Read an MSA from a fasta file.

    Parameters
    ----------
    alignment_file : str
        Path to the alignment fasta file

    msa : Bio.AlignIO
        Multiple sequence aligment in Biopython format.
    """

    msa = AlignIO.read(alignment_file, 'fasta')

    return msa

def savePSIBlastAsJson(psi_blast_result, output_file):
    """
    Save the results of a psiblast calculation (e.g., _blast_functions.PSIBlastDatabase())
    into a json file.

    Parameters
    ==========
    psi_blast_result : dict
        Output dictionary from PSIBlastDatabase() function.
    output_file : str
        Path to the output file.

    Returns
    =======
    output_file : str
        Path to the output file.
    """
    with open(output_file, 'w') as of:
        json.dump(psi_blast_result, of)
    return output_file

def readPSIBlastFromJson(json_file):
    """
    Save the results of a psiblast calculation (e.g., _blast_functions.PSIBlastDatabase())
    into a json file.

    Parameters
    ==========
    json_file : str
        Path to the json file contaning the blas result

    Returns
    =======
    psi_blast_result : dict
        Result from the PSI blast calculation
    """
    with open(json_file) as jf:
        psi_blast_result = json.load(jf)
    return psi_blast_result


def msaIndexesFromSequencePositions(msa, sequence_id, sequence_positions):
    """
    Get the multiple sequence alignment position indexes matching those positions (zero-based) of a specific target sequence.

    Parameters
    ==========
    msa : Bio.AlignIO
        Multiple sequence aligment in Biopython format.
    sequence_id : str
        ID of the target sequence
    sequence_positions : list
        Target sequence positions to match (one-based indexes)

    Returns
    =======
    msa_indexes : list
        MSA indexes matching the target sequence positions (zero-based indexes)
    """

    msa_indexes = []
    p = 1
    for i in range(msa.get_alignment_length()):
        for a in msa:
            if a.id == sequence_id:
                if a.seq[i] != '-':
                    p += 1
        if p in sequence_positions:
            msa_indexes.append(i)

    return msa_indexes

def getSequencePositionFromMSAindex(msa, index, return_identity=False):
    """
    For a MSA index position return the sequence position of each entry in the given MSA.
    If the entry has a '-' at that position, then return None. Optionally, the aminoacid
    identity can be returned with the option return_identity=True.

    Parameters
    ==========
    msa : Bio.AlignIO
        Multiple sequence aligment in Biopython format.
    index : int
        The MSA index
    return_identity : bool
        Return identities instead of sequence positions.
    """

    # Initialize a sequence counter per each sequence
    position = {}
    for entry in msa:
        position[entry.id] = 0

    # Count the sequence positions
    for i in range(msa.get_alignment_length()):
        for entry in msa:
            if entry.seq[i] != '-':
                position[entry.id] += 1

            if i == index:
                if return_identity:
                    # Return the amino acid identity of the corresponding position
                    position[entry.id] = entry.seq[i]
                else:
                    # Return None if gap is found at position
                    if entry.seq[i] == '-':
                        position[entry.id] = None

        # Beak if index matches the requested index.
        if i == index:
            break

    return position

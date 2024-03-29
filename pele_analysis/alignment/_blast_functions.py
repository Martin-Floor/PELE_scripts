import os
import numpy as np
from collections import OrderedDict
import subprocess
import uuid
import json

class blast:
    """
    Class to hold methods to work with blast executable.

    Methods
    -------
    calculatePIDs()
        Fast method to calculate the PID of a sequence against many.
    _getPIDsFromBlastpOutput()
        Method to parse the ouput of blast and retrieve the PIDs.
    """

    def calculatePIDs(target_sequence, comparison_sequences):
        """
        Calculate the percentage of identity (PID) of a target sequence against a group
        of other sequences.

        Parameters
        ----------
        target_sequence : str
            Target sequence.
        comparison_sequences : list of str
            List of sequences to compare the target sequence against.

        Returns
        -------
        pids : numpy.array
            Array containg the PIDs of the target_sequence to all the comparison_sequences.
        """

        if isinstance(comparison_sequences, str):
            comparison_sequences = [comparison_sequences]
        elif not isinstance(comparison_sequences, list):
            raise ValueError('Comparison sequences must be given a single string or \
            as a list of strings.')

        # Write sequences into a temporary file
        with open('seq1.fasta.tmp', 'w') as sf1:
            sf1.write('>seq1\n')
            sf1.write(str(target_sequence))
        with open('seq2.fasta.tmp', 'w') as sf2:
            for i,s in enumerate(comparison_sequences):
                sf2.write('>seq'+str(i)+'\n')
                sf2.write(str(s)+'\n')
        # Execute blastp calculation
        try:
            os.system('blastp -query seq1.fasta.tmp -subject seq2.fasta.tmp -out ssblast.out.tmp -max_target_seqs '+str(len(comparison_sequences)))
        except:
            raise ValueError('blastp executable failed!')

        # Parse blastp output to extract PIDs
        pids = blast._getPIDsFromBlastpOutput('ssblast.out.tmp', len(comparison_sequences))
        pids = np.array(list(pids.values()))

        # Remove temporary files
        os.remove('seq1.fasta.tmp')
        os.remove('ssblast.out.tmp')
        os.remove('seq2.fasta.tmp')

        return pids

    def blastDatabase(target_sequence, path_to_database_fasta, max_target_seqs=500):
        """
        Blast a specific sequence against a fasta database. The database fasta file
        must be supplied.
        """
        max_target_seqs = str(max_target_seqs)

        # Write target sequence into a temporary file
        target_file = '.'+str(uuid.uuid4())+'.fasta.tmp'
        output_file = '.'+str(uuid.uuid4())+'.out.tmp'
        with open(target_file, 'w') as sf1:
            sf1.write('>seq1\n')
            sf1.write(str(target_sequence))

        # Execute blastp calculation
        # Run blastp
        command = 'blastp -query '+target_file+' -subject '+path_to_database_fasta+' -out '+output_file+' -max_target_seqs '+max_target_seqs
        try:
            output = subprocess.check_output(command,stderr=subprocess.STDOUT,
                                             shell=True, universal_newlines=True)
        except subprocess.CalledProcessError as exc:
            print(exc.output, exc.returncode)
            raise Exception("Problem with blastp execution.")

        blast_results = blast._parseBlastpOutput(output_file)

        # Remove temporary files
        os.remove(target_file)
        os.remove(output_file)

        return blast_results

    def PSIBlastDatabase(target_sequence, path_to_database_fasta, output_file=None, num_iterations=5,
                         max_target_seqs=500, overwrite=False):
        """
        Run a PSI BLAST search for a specific sequence against a fasta database.
        The database fasta file must be supplied.
        """

        if output_file != None and os.path.exists(output_file) and not overwrite:
            print('PSIBlast output file found. Reading results from %s' % output_file)
            with open(output_file) as of:
                blast_results = json.load(of)
        else:
            max_target_seqs = str(max_target_seqs)

            # Write target sequence into a temporary file
            target_file = '.'+str(uuid.uuid4())+'.fasta.tmp'
            tmp_file = '.'+str(uuid.uuid4())+'.out.tmp'
            with open(target_file, 'w') as sf1:
                sf1.write('>seq1\n')
                sf1.write(str(target_sequence))

            # Execute blastp calculation
            # Run psiblast
            command = 'psiblast -query '+target_file+' '
            command += '-subject '+path_to_database_fasta+' '
            command += '-out '+tmp_file+' '
            command += '-max_target_seqs '+max_target_seqs+' '
            command += '-num_iterations '+str(num_iterations)+' '
            try:
                output = subprocess.check_output(command,stderr=subprocess.STDOUT,
                                                 shell=True, universal_newlines=True)

            except subprocess.CalledProcessError as exc:
                print(exc.output, exc.returncode)
                # Remove temporary files
                os.remove(target_file)
                os.remove(tmp_file)
                print("Problem with psiblast execution.")
                return None

            blast_results = blast._parsePSIBlastOutput(tmp_file)

            # Write results to output file if given
            if output_file != None:
                with open(output_file, 'w') as of:
                    json.dump(blast_results, of)

            # Remove temporary files
            os.remove(target_file)
            os.remove(tmp_file)

        return blast_results

    def _parsePSIBlastOutput(psiblast_outfile):
        """
        Reads information from a psiblast output file.

        Parameters
        ----------
        psiblast_outfile : str
            Path to the psiblast outputfile.
        """
        sequences = {}

        # Analyse sequences by round
        # Read blast file and extract sequences full ids
        r = None
        with open(psiblast_outfile) as bo:
            cond = False
            for l in bo:
                if l.startswith('>'):
                    cond = True
                    full_name = l[2:][:-1]
                elif cond:
                    if l.startswith('Length='):
                        sequences[r].append(full_name)
                        cond = False
                    else:
                        full_name += l[:-1]
                if l.startswith('Results from round'):
                    r = int(l.split()[-1])
                    sequences[r] = []

        # Read blast file again to extract e-values matched to partial sequence names
        r = None
        blast_results = {}
        with open(psiblast_outfile) as bo:
            cond = False
            for l in bo:
                if 'Sequences producing significant alignments:' in l and r == 1:
                    cond = True
                elif 'Sequences used in model and found again:' in l:
                    continue
                elif 'Sequences not found previously or not previously below threshold:' in l:
                    if r > 1:
                        cond = True
                    continue
                elif l.startswith('>'):
                    cond = False
                elif cond and l.split() != []:
                    e_value = float(l.split()[-1])
                    name = l[:-25]
                    blast_results[r][name] = {}
                    blast_results[r][name]['e-value'] = e_value
                if l.startswith('Results from round'):
                    r = int(l.split()[-1])
                    blast_results[r] = {}

        # Match partial sequence names with full sequence names
        full_names = {}
        for round in blast_results:
            full_names[round] = {}
            for k1 in blast_results[round]:
                for k2 in sequences[round]:
                    if k1 in k2:
                        full_names[round][k1] = k2

        # Replace dict entries with full names
        for round in full_names:
            for k in full_names[round]:
                blast_results[round][full_names[round][k]] = blast_results[round].pop(k)

        return blast_results

    def _parseBlastpOutput(blastp_outfile):
        """
        Reads information from blast output file.

        Parameters
        ----------
        blastp_outfile : str
            Path to the blastp outputfile.
        """
        sequences = []
        # Read blast file and extract sequences full ids
        with open(blastp_outfile) as bo:
            cond = False
            for l in bo:
                if l.startswith('>'):
                    cond = True
                    full_name = l[2:][:-1]
                elif cond:
                    if l.startswith('Length='):
                        sequences.append(full_name)
                        cond = False
                    else:
                        full_name += l[:-1]

        # Read blast file again to extract e-values matched to partial sequence names
        blast_results = {}
        with open(blastp_outfile) as bo:
            cond = False
            for l in bo:
                if 'Sequences producing significant alignments:' in l:
                    cond = True
                elif l.startswith('>'):
                    cond = False
                elif cond and l.split() != []:
                    e_value = float(l.split()[-1])
                    name = l[:-25]
                    blast_results[name] = {}
                    blast_results[name]['e-value'] = e_value

        # Match partial sequence names with full sequence names
        full_names = {}
        for k1 in blast_results:
            for k2 in sequences:
                if k1 in k2:
                    full_names[k1] = k2

        # Replace dict entries with full names
        for k in full_names:
            blast_results[full_names[k]] = blast_results.pop(k)

        return blast_results

    def _getPIDsFromBlastpOutput(blastp_outfile, n_sequences):
        """
        Parse output file from a blastp comparison to extract pids

        Parameters
        ----------
        blastp_outfile : str
            Path to the blastp outputfile.
        n_sequences : str
            number of sequences in the comparison file.

        Returns
        -------
        values : OrderedDict
            Dictionary containing the PID values.
        """

        # Create dictionary integer entries
        values = OrderedDict()
        for i in range(n_sequences):
            values[i] = 0

        # Read PID from blastp output file
        with open(blastp_outfile) as bf:
            for l in bf:
                if l.startswith('>'):
                    seq = int(l.split()[1].replace('seq',''))
                elif 'Identities' in l:
                    pid = eval(l.split()[2])
                    values[seq] = pid

        return values

from . import pele_read
from . import pele_trajectory
from . import pele_distances

import os
import shutil
import copy
import re

import pandas as pd
pd.options.mode.chained_assignment = None
import matplotlib as mpl
import matplotlib.pyplot as plt
import mdtraj as md
import numpy as np
from Bio import PDB
import gc
from Bio.PDB.PDBExceptions import PDBConstructionWarning
import warnings
warnings.simplefilter('ignore', PDBConstructionWarning)
import json
import io
from pkg_resources import resource_stream, Requirement, resource_listdir

from ipywidgets import interact, fixed, FloatSlider, IntSlider, FloatRangeSlider, VBox, HBox, interactive_output, Dropdown, Checkbox
import time

class peleAnalysis:
    """
    Analyse multiple PELE calculations in batch. This class assumes that calculations
    were run using the PELE Platform to generate the output and that trajectories
    are in XTC format.

    Attributes
    ==========
    """

    def __init__(self, pele_folder, pele_output_folder='output', separator='-', force_reading=False,
                 verbose=False, energy_by_residue=False, ebr_threshold=0.1, energy_by_residue_type='all',
                 read_equilibration=True, data_folder_name=None, global_pele=False, trajectories=False,
                 remove_original_trajectory=False, change_water_names=False):
        """
        When initiliasing the class it read the paths to the output report, trajectory,
        and topology files.

        Parameters
        ==========
        pele_folder : str
            Path to the pele folder containing one or several PELE calculations folders
        """

        # Set pele folder variables
        self.pele_folder = pele_folder
        self.pele_output_folder = pele_output_folder
        self.separator = separator

        # Set attributes for all PELE folders' paths
        if data_folder_name == None:
            self.data_folder  = '.pele_analysis' # Hidden if not given
        else:
            self.data_folder  = data_folder_name
        self.pele_directories = {}
        self.report_files = {}
        self.csv_files = {}
        self.csv_equilibration_files = {}
        self.trajectory_files = {}
        self.topology_files = {}
        self.spawning_files = {}
        self.conect_files = {}
        self.fixed_files = {}
        self.ligand_files = {}
        self.equilibration = {}
        self.pele_combinations = []
        self.ligand_names = {}
        self.chain_ids = {}
        self.atom_indexes = {}
        self.equilibration['report'] = {}
        self.equilibration['trajectory'] = {}
        self.verbose = verbose
        self.force_reading = force_reading
        self.energy_by_residue = energy_by_residue
        self.data = None
        self.distances = {}
        self.angles = {}
        self.nonbonded_energy = {}
        self.steps_matrix = None

        # System name attributes
        self.proteins = []
        self.ligands = []

        # Check given energy by residue type
        ebr_types = ['all', 'sgb', 'lennard_jones', 'electrostatic']
        if energy_by_residue_type not in ebr_types:
            raise ValueError('Energy by residue type not valid. valid options are: '+' '.join(energy_by_residue_type))

        # Check data folder for paths to csv files
        if self.verbose:
            print('Checking PELE analysis folder: %s' % self.data_folder)
        self._checkDataFolder(trajectories=trajectories)

        # Check PELE folder for paths to pele data
        if self.verbose:
            print('Checking PELE simulation folders: %s' % self.pele_folder)
        self._checkPELEFolder()

        # Copy PELE inputs to analysis folder
        if self.verbose:
            print('Copying PELE input files')
        self._copyPELEInputs(overwrite=self.force_reading)

        # Copy PELE configuration files to analysis folder
        if self.verbose:
            print('Copying PELE and Adaptive configuration files')
        self._copyPELEConfiguration(overwrite=self.force_reading)

        # Copy PELE topology files to analysis folder
        if self.verbose:
            print('Copying PELE topology files')
        self._copyPELETopology(overwrite=self.force_reading)

        # Copy PELE trajectories to analysis folder
        if trajectories:
            if self.verbose:
                print('Copying PELE trajectory files')
            self._copyPELETrajectories(overwrite=self.force_reading,
                                       remove_original_trajectory=remove_original_trajectory)

        # Set dictionary with Chain IDs to match mdtraj indexing
        print('Setting Chain IDs and Atom Indexes')
        self._setChainIDs(change_water_names=change_water_names)

        # Get protein and ligand cominations wither from pele or analysis folders
        self.pele_combinations = self._getProteinLigandCombinations()

        if self.verbose:
            print('Reading PELE information for:')

        # Read PELE simulation report data
        self._readReportData(energy_by_residue_type=energy_by_residue_type)

        if global_pele:
            self._readGlobalReportData()
            self._readReportData()

        if read_equilibration:
            if self.verbose:
                print('Reading equilibration information from report files from:')
            # Read PELE equilibration report data
            self._readReportData(equilibration=True)
        else:
            print('Skipping equilibration information from report files.')

        ### Read spawning infortmation
        self._checkSpawningInformation()

        # Sort protein and ligand names alphabetically for orderly iterations.
        self.proteins = sorted(self.proteins)
        self.ligands = sorted(self.ligands)

    def calculateDistances(self, atom_pairs, equilibration=False, overwrite=False, verbose=False,
                           skip_missing=False):
        """
        Calculate distances between pairs of atoms for each pele (protein+ligand)
        simulation. The atom pairs are given as a dictionary with the following format:

        The atom pairs must be given in a dicionary with each key representing the name
        of a model and each value a sub-dicionary with the ligands as keys and a list of the atom pairs
        to calculate in the format:
            {model_name: { ligand_name : [((chain1_id, residue1_id, atom1_name), (chain2_id, residue2_id, atom2_name)), ...],...} another_model_name:...}

        Parameters
        ==========
        atom_pairs : dict
            Atom pairs for each protein + ligand entry
        equilibration : bool
            Calculate distances for the equilibration steps also
        verbose : bool
            Display function messages
        overwrite : bool
            Force recalculation of distances.
        skip_missing : bool
            Skip models not found in the atom_pairs dictionary
        """

        if not os.path.exists(self.data_folder+'/distances'):
            os.mkdir(self.data_folder+'/distances')

        if not self.trajectory_files:
            raise ValueError('No trajectories were found!')

        # Iterate all PELE protein + ligand entries
        for protein in sorted(self.trajectory_files):
            self.distances[protein] = {}
            for ligand in sorted(self.trajectory_files[protein]):

                # Define a different distance output file for each pele run
                distance_file = self.data_folder+'/distances/'+protein+self.separator+ligand+'.csv'

                # Check if distance have been previously calculated
                if os.path.exists(distance_file) and not overwrite:
                    if verbose:
                        print('Distance file for %s + %s was found. Reading distances from there...' % (protein, ligand))
                    self.distances[protein][ligand] = pd.read_csv(distance_file, index_col=False)
                    self.distances[protein][ligand] = self.distances[protein][ligand].loc[:, ~self.distances[protein][ligand].columns.str.contains('^Unnamed')]
                    index_columns = ['Protein', 'Ligand', 'Epoch', 'Trajectory','Accepted Pele Steps']
                    if 'Step' in self.distances[protein][ligand].keys():
                        index_columns.append('Step')
                    self.distances[protein][ligand].set_index(index_columns, inplace=True)
                else:
                    self.distances[protein][ligand] = {}
                    self.distances[protein][ligand]['Protein'] = []
                    self.distances[protein][ligand]['Ligand'] = []
                    self.distances[protein][ligand]['Epoch'] = []
                    self.distances[protein][ligand]['Trajectory'] = []
                    self.distances[protein][ligand]['Accepted Pele Steps'] = []

                if verbose:
                    print('Calculating distances for %s + %s ' % (protein, ligand))

                # Load one trajectory at the time to save memory
                trajectory_files = self.trajectory_files[protein][ligand]
                topology_file = self.topology_files[protein][ligand]

                # Get atom pairs indexes
                topology = md.load(topology_file).topology

                # Get atom pair indexes to compute distances
                pairs = []
                dist_label = {}
                pair_lengths = []

                if skip_missing and protein not in atom_pairs:
                    continue
                elif skip_missing and ligand not in atom_pairs[protein]:
                    continue

                for pair in atom_pairs[protein][ligand]:

                    if len(pair) == 1:
                        if pair not in ['X', 'Y', 'Z']:
                            raise ValueError('You must ask for a X, Y, or Z coordinate!')
                        pairs.append(pair)
                        dist_label[pair] = 'coordinate_'

                    if len(pair) >= 2:
                        # Check if atoms are in the protein+ligand PELE topology
                        if pair[0] not in self.atom_indexes[protein][ligand]:
                            raise ValueError('Atom %s not found for protein %s and ligand %s' % (pair[0], protein, ligand))
                        if pair[1] not in self.atom_indexes[protein][ligand]:
                            raise ValueError('Atom %s not found for protein %s and ligand %s' % (pair[1], protein, ligand))

                        # Get the atom indexes
                        i1 = self.atom_indexes[protein][ligand][pair[0]]
                        i2 = self.atom_indexes[protein][ligand][pair[1]]

                        if len(pair) == 2:
                            pairs.append((i1, i2))
                            dist_label[(pair[0], pair[1])] = 'distance_'

                    if len(pair) >= 3:
                        # Check if atoms are in the protein+ligand PELE topology
                        if pair[0] not in self.atom_indexes[protein][ligand]:
                            raise ValueError('Atom %s not found for protein %s and ligand %s' % (pair[0], protein, ligand))
                        if pair[1] not in self.atom_indexes[protein][ligand]:
                            raise ValueError('Atom %s not found for protein %s and ligand %s' % (pair[1], protein, ligand))
                        if pair[2] not in self.atom_indexes[protein][ligand]:
                            raise ValueError('Atom %s not found for protein %s and ligand %s' % (pair[2], protein, ligand))

                        i3 = self.atom_indexes[protein][ligand][pair[2]]
                        if len(pair) == 3:
                            pairs.append((i1, i2, i3))
                            dist_label[(pair[0], pair[1], pair[2])] = 'angle_'

                    if len(pair) == 4:
                        # Check if atoms are in the protein+ligand PELE topology
                        if pair[0] not in self.atom_indexes[protein][ligand]:
                            raise ValueError('Atom %s not found for protein %s and ligand %s' % (pair[0], protein, ligand))
                        if pair[1] not in self.atom_indexes[protein][ligand]:
                            raise ValueError('Atom %s not found for protein %s and ligand %s' % (pair[1], protein, ligand))
                        if pair[2] not in self.atom_indexes[protein][ligand]:
                            raise ValueError('Atom %s not found for protein %s and ligand %s' % (pair[2], protein, ligand))
                        if pair[3] not in self.atom_indexes[protein][ligand]:
                            raise ValueError('Atom %s not found for protein %s and ligand %s' % (pair[3], protein, ligand))

                        i4 = self.atom_indexes[protein][ligand][pair[3]]
                        pairs.append((i1, i2, i3, i4))
                        dist_label[(pair[0], pair[1], pair[2], pair[3])] = 'torsion_'

                    pair_lengths.append(len(pair))

                # Check pairs
                pair_lengths = set(pair_lengths)
                if len(pair_lengths) > 1:
                    raise ValueError('Mixed number of atoms given!')
                pair_lengths = list(pair_lengths)[0]

                # Define labels
                labels = []
                for pair in atom_pairs[protein][ligand]:
                    label = dist_label[pair]
                    if len(pair) > 1:
                        for p in pair:
                            label += ''.join([str(x) for x in p])+'_'
                    else:
                        label += pair+'_'
                    labels.append(label[:-1])

                # Check if labels are already in distance_data
                missing_labels = []
                skip_index_append = False
                if isinstance(self.distances[protein][ligand], pd.DataFrame):
                    distances_keys = list(self.distances[protein][ligand].keys())
                    for l in labels:
                        if l not in distances_keys:
                            missing_labels.append(l)
                    if missing_labels != []:
                        # Convert DF into a dictionary for distance appending
                        self.distances[protein][ligand].reset_index(inplace=True)
                        self.distances[protein][ligand] = self.distances[protein][ligand].to_dict()
                        for k in self.distances[protein][ligand]:
                            nv = [self.distances[protein][ligand][k][x] for x in self.distances[protein][ligand][k]]
                            self.distances[protein][ligand][k] = nv
                        skip_index_append = True
                else:
                    missing_labels = labels

                # Update pairs based on missing labels
                if missing_labels != []:

                    # Create an entry for each distance
                    for label in missing_labels:
                        self.distances[protein][ligand][label] = []

                    # Update pairs based on missing labels
                    updated_pairs = []
                    for p,l in zip(pairs, labels):
                        if l in missing_labels:
                            updated_pairs.append(p)
                    pairs = updated_pairs

                    # Compute distances and them to the dicionary
                    for epoch in sorted(trajectory_files):
                        for t in sorted(trajectory_files[epoch]):

                            # Load trajectory
                            try:
                                traj = md.load(trajectory_files[epoch][t], top=topology_file)
                            except:
                                message = 'Problems with trajectory %s of epoch %s ' % (epoch, t)
                                message += 'of protein %s and ligand %s' % (protein, ligand)
                                raise ValueError(message)

                            # Calculate centroid coordinates
                            if pair_lengths == 1:

                                # Get all ligand atom coordinates
                                for r in traj.topology.residues:
                                    if r.name == self.ligand_names[protein][ligand]:
                                        ligand_atoms = [a.index for a in r.atoms]

                                # Get the requested coordinates
                                ligand_coordinates = traj.atom_slice(ligand_atoms).xyz
                                indexes = []
                                coord_to_index = {'X':0, 'Y':1, 'Z':2}
                                for p in pairs:
                                    indexes.append(coord_to_index[p])
                                d = np.average(ligand_coordinates[:,:,indexes], axis=1)*10

                            # Calculate distances
                            if pair_lengths == 2:
                                d = md.compute_distances(traj, pairs)*10
                            elif pair_lengths == 3:
                                d = md.compute_angles(traj, pairs)
                                d = np.rad2deg(md.compute_angles(traj, pairs))
                            elif pair_lengths == 4:
                                d = np.rad2deg(md.compute_dihedrals(traj, pairs))
                                # d = md.compute_dihedrals(traj, pairs)

                            # Store data
                            if not skip_index_append:
                                self.distances[protein][ligand]['Protein'] += [protein]*d.shape[0]
                                self.distances[protein][ligand]['Ligand'] += [ligand]*d.shape[0]
                                self.distances[protein][ligand]['Epoch'] += [epoch]*d.shape[0]
                                self.distances[protein][ligand]['Trajectory'] += [t]*d.shape[0]
                                self.distances[protein][ligand]['Accepted Pele Steps'] += list(range(d.shape[0]))
                            for i,l in enumerate(missing_labels):
                                self.distances[protein][ligand][l] += list(d[:,i])

                    # Convert distances into dataframe
                    self.distances[protein][ligand] = pd.DataFrame(self.distances[protein][ligand])

                    # Save distances to CSV file
                    self.distances[protein][ligand].to_csv(distance_file)

                    # Set indexes for DataFrame
                    index_columns = ['Protein', 'Ligand', 'Epoch', 'Trajectory','Accepted Pele Steps']
                    if 'Step' in self.distances[protein][ligand].keys():
                        index_columns.append('Step')
                    self.distances[protein][ligand].set_index(index_columns, inplace=True)

    def calculateDistancesParallel(self, atom_pairs, overwrite=False, verbose=False, cpus=None, skip_missing=False):
        """
        Calculate distances between pairs of atoms for each pele (protein+ligand)
        simulation. The atom pairs are given as a dictionary with the following format:

        The atom pairs must be given in a dicionary with each key representing the name
        of a model and each value a sub-dicionary with the ligands as keys and a list of the atom pairs
        to calculate in the format:
            {model_name: { ligand_name : [((chain1_id, residue1_id, atom1_name), (chain2_id, residue2_id, atom2_name)), ...],...} another_model_name:...}

        Parameters
        ==========
        atom_pairs : dict
            Atom pairs for each protein + ligand entry
        verbose : bool
            Display function messages
        overwrite : bool
            Force recalculation of distances.
        cpus : int
            Number of cpus to use in the distances calculation.
        """
        distance_calculation = pele_distances.distances(self)
        distance_calculation.calculateDistances(atom_pairs, overwrite=overwrite,
                                                verbose=verbose, cpus=cpus, skip_missing=skip_missing)

    def getTrajectory(self, protein, ligand, step, trajectory, equilibration=False):
        """
        Load trajectory file for the selected protein, ligand, step, and trajectory number.
        """

        if not os.path.isdir(self.pele_folder):
            raise ValueError('Pele folder not found. Cannot get trajectory without pele folder')

        if equilibration:
            traj = md.load(self.equilibration['trajectory'][protein][ligand][step][trajectory],
                            top=self.topology_files[protein][ligand])
        else:
            traj = md.load(self.trajectory_files[protein][ligand][step][trajectory],
                           top=self.topology_files[protein][ligand])
        return traj

    def calculateLigandRMSD(self, recalculate=False, production=True, equilibration=False,
                            reference_pdb=None, only_proteins=None, only_ligands=None,
                            verbose=False):
        """
        Calculate ligand RMSD using as reference the lowest binding energy pose or
        a reference PDB if given. It is added as "Ligand RMSD" column in the pele data frame'
        """

        if not production and not equilibration:
            raise ValueError('You must at least set production or equilibration to True!')

        if isinstance(only_proteins, str):
            only_proteins = [only_proteins]

        if isinstance(only_ligands, str):
            only_ligands = [only_ligands]

        calc_eq_rmsd = False
        if equilibration:
            if 'Ligand RMSD' in self.equilibration_data.keys() and not recalculate:
                print('Ligand RMSD equilibration data already computed. Give recalculate=True to recompute.')
            else:
                calc_eq_rmsd = True
                equilibration_RMSD = None

        calc_prod_rmsd = False
        if 'Ligand RMSD' in self.data.keys() and not recalculate:
            print('Ligand RMSD production data already computed. Give recalculate=True to recompute.')
        else:
            calc_prod_rmsd = True
            RMSD = None

        if reference_pdb == None:
            reference_pdb = {}

        RMSD = None
        for protein in self.proteins:

            for ligand in self.ligands:

                # Skip protein,ligand combinations without topology files
                if ligand not in self.topology_files[protein]:
                    continue

                if (only_proteins and protein not in only_proteins) or (only_ligands and ligand not in only_ligands):
                    ligand_data = self.getProteinAndLigandData(protein, ligand)

                    if equilibration:
                        ligand_data = self.getProteinAndLigandData(protein, ligand, equilibration=True)
                        if 'Ligand RMSD' in ligand_data:
                            rmsd = ligand_data['Ligand RMSD'].to_numpy()
                            if any(np.isnan(rmsd)):
                                rmsd = np.array([np.nan]*ligand_data.shape[0])
                        else:
                            rmsd = np.array([np.nan]*ligand_data.shape[0])
                        if isinstance(equilibration_RMSD, type(None)):
                            equilibration_RMSD = rmsd
                        else:
                            equilibration_RMSD = np.concatenate((equilibration_RMSD, rmsd))

                    if production:
                        ligand_data = self.getProteinAndLigandData(protein, ligand)
                        if 'Ligand RMSD' in ligand_data:
                            rmsd = ligand_data['Ligand RMSD'].to_numpy()
                            if any(np.isnan(rmsd)):
                                rmsd = np.array([np.nan]*ligand_data.shape[0])
                        else:
                            rmsd = np.array([np.nan]*ligand_data.shape[0])

                        if isinstance(RMSD, type(None)):
                            RMSD = rmsd
                        else:
                            RMSD = np.concatenate((RMSD, rmsd))

                    continue

                # Get topology PDB as reference for alignment
                topology_file = self.topology_files[protein][ligand]
                top_traj = md.load(topology_file)
                if protein in reference_pdb and ligand in reference_pdb[protein]:
                    print('Comparing RMSD values to %s' % reference_pdb[protein][ligand])
                    reference_file = reference_pdb[protein][ligand]
                    ref_traj = md.load(reference_file)

                else:
                    # Get best binding energy pose as reference
                    ligand_data = self.getProteinAndLigandData(protein, ligand)

                    # Skip empty data frames
                    if ligand_data.empty:
                        continue

                    ref_model = ligand_data.nsmallest(1, 'Binding Energy')
                    ref_trajectory = ref_model.index.get_level_values('Trajectory')[0]
                    ref_epoch = ref_model.index.get_level_values('Epoch')[0]
                    ref_step = ref_model.index.get_level_values('Accepted Pele Steps')[0]
                    ref_traj = self.getTrajectory(protein, ligand, ref_epoch, ref_trajectory)[ref_step]

                if equilibration and calc_eq_rmsd and recalculate:
                    # Calculate ligand RMSD
                    for epoch in sorted(self.equilibration['trajectory'][protein][ligand]):
                        for trajectory in sorted(self.equilibration['trajectory'][protein][ligand][epoch]):
                            traj = self.getTrajectory(protein, ligand, epoch, trajectory, equilibration=True)
                            traj.superpose(top_traj)
                            ligand_atoms = traj.topology.select('resname '+self.ligand_names[protein][ligand])

                            # Calculate RMSD
                            rmsd = md.rmsd(traj, ref_traj, atom_indices=ligand_atoms)*10
                            if isinstance(equilibration_RMSD, type(None)):
                                equilibration_RMSD = rmsd
                            else:
                                equilibration_RMSD = np.concatenate((equilibration_RMSD, rmsd))

                if (production and calc_prod_rmsd) or recalculate:
                    print(f'Computing RMSD for {protein} and {ligand}:')
                    # Calculate ligand RMSD
                    for epoch in sorted(self.trajectory_files[protein][ligand]):
                        for trajectory in sorted(self.trajectory_files[protein][ligand][epoch]):
                            if verbose:
                                print(f'\tfor epoch {epoch} and trajectory {trajectory}', end='\r')
                            traj = self.getTrajectory(protein, ligand, epoch, trajectory)
                            traj.superpose(top_traj)
                            ligand_atoms = traj.topology.select('resname '+self.ligand_names[protein][ligand])

                            # Calculate RMSD
                            rmsd = md.rmsd(traj, ref_traj, atom_indices=ligand_atoms)*10
                            if isinstance(RMSD, type(None)):
                                RMSD = rmsd
                            else:
                                RMSD = np.concatenate((RMSD, rmsd))
                    print()

        if equilibration and calc_eq_rmsd:
            self.equilibration_data['Ligand RMSD'] = equilibration_RMSD
            self._saveDataState(individually=True, equilibration=True)
        if production and calc_prod_rmsd:
            self.data['Ligand RMSD'] = RMSD
            self._saveDataState(individually=True)

    def calculateProteinRMSD(self, full_atom=True, equilibration=True, productive=True, recalculate=False):
        """
        Calculate the RMSD of all steps regarding the input (topology) structure'
        """

        if not os.path.isdir(self.pele_folder):
            raise ValueError('Pele folder not found. RMSD cannot be calculated without pele folder')

        if equilibration:
            if 'Protein RMSD' in self.equilibration_data.keys() and not recalculate:
                print('Equilibration data RMSD already computed. Give recalculate=True to recompute.')
            else:
                RMSD = None

                for protein in self.proteins:
                    for ligand in self.ligands:
                        if ligand in self.topology_files[protein]:
                            topology_file = self.topology_files[protein][ligand]
                            reference = md.load(topology_file)
                            for step in sorted(self.equilibration['trajectory'][protein][ligand]):
                                for trajectory in sorted(self.equilibration['trajectory'][protein][ligand][step]):
                                    traj = self.getTrajectory(protein, ligand, step, trajectory, equilibration=True)
                                    traj.superpose(reference)
                                    if full_atom:
                                        protein_atoms = traj.topology.select('protein')
                                    else:
                                        protein_atoms = traj.topology.select('protein and name CA')

                                    rmsd = md.rmsd(traj, reference, atom_indices=protein_atoms)*10
                                    if isinstance(RMSD, type(None)):
                                        RMSD = rmsd
                                    else:
                                        RMSD = np.concatenate((RMSD, rmsd))

                self.equilibration_data['Protein RMSD'] = RMSD
                self._saveDataState(equilibration=True)

        if productive:
            if 'Protein RMSD' in self.data.keys() and not recalculate:
                print('Data RMSD already computed. Give recalculate=True to recompute.')
            else:
                RMSD = None

                for protein in self.proteins:
                    for ligand in self.ligands:
                        if ligand in self.topology_files[protein]:
                            topology_file = self.topology_files[protein][ligand]
                            reference = md.load(topology_file)
                            for epoch in sorted(self.trajectory_files[protein][ligand]):
                                for trajectory in sorted(self.trajectory_files[protein][ligand][epoch]):
                                    traj = self.getTrajectory(protein, ligand, epoch, trajectory)
                                    traj.superpose(reference)
                                    protein_atoms = traj.topology.select('protein')

                                    # Define ligand chain as the last chain =S
                                    rmsd = md.rmsd(traj, reference, atom_indices=protein_atoms)*10
                                    if isinstance(RMSD, type(None)):
                                        RMSD = rmsd
                                    else:
                                        RMSD = np.concatenate((RMSD, rmsd))

                self.data['Protein RMSD'] = RMSD
                self._saveDataState()

    def plotSimulationMetric(self, metric_column, equilibration=True, productive=True):
        """
        Plot the progression of a specic metric in the simulation data.

        Parameters
        ==========
        metric_column : str
            The column name of the metric
        equilibration : bool
            Equilibration data is present or not
        productive : bool
            Productive data is present or not
        """

        plt.figure()

        if equilibration:

            last_step = {}
            for protein in self.proteins:
                last_step[protein] = {}
                protein_series = self.equilibration_data[self.equilibration_data.index.get_level_values('Protein') == protein]
                for ligand in self.ligands:
                    last_step[protein][ligand] = {}
                    ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                    if not ligand_series.empty:
                        trajectories = set(ligand_series.reset_index()['Trajectory'].tolist())
                        for trajectory in trajectories:
                            trajectory_series = ligand_series[ligand_series.index.get_level_values('Trajectory') == trajectory]
                            last_step[protein][ligand][trajectory] = trajectory_series.shape[0]
                            plt.plot(range(1,trajectory_series[metric_column].shape[0]+1),
                                     trajectory_series[metric_column],
                                     c='r',
                                     lw=0.1)

        if productive:
            for protein in self.proteins:
                protein_series = self.data[self.data.index.get_level_values('Protein') == protein]
                for ligand in self.ligands:
                    ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                    if not ligand_series.empty:
                        trajectories = set(ligand_series.reset_index()['Trajectory'].tolist())
                        for trajectory in trajectories:
                            trajectory_series = ligand_series[ligand_series.index.get_level_values('Trajectory') == trajectory]
                            if equilibration:
                                start = last_step[protein][ligand][trajectory]
                            else:
                                start = 1
                            steps = range(start, trajectory_series[metric_column].shape[0]+start)
                            plt.plot(steps,
                                     trajectory_series[metric_column],
                                     c='k',
                                     lw=0.1)

        plt.xlabel('Simulation step')
        plt.ylabel(metric_column)
        plt.show()

    def plotSimulationEnergy(self, equilibration=True, productive=True):
        """
        Plot the progression of a total energy in the simulation data.

        Parameters
        ==========
        equilibration : bool
            Equilibration data is present or not
        productive : bool
            Productive data is present or not
        """
        self.plotSimulationMetric('Total Energy',
                                  equilibration=equilibration,
                                  productive=productive)

    def plotAcceptanceProbability(self):
        """
        Plot the accepted steps by trajectory together with the overall acceptance
        probability.
        """

        def getLigands(Protein, equilibration=False):
            protein_series = self.data[self.data.index.get_level_values('Protein') == Protein]
            ligands = list(set(protein_series.index.get_level_values('Ligand').tolist()))
            interact(plotAcceptance, Protein=fixed(Protein), Ligand=ligands,
                     equilibration=fixed(equilibration))

        def plotAcceptance(Protein, Ligand, equilibration=False):

            if equilibration:
                data = self.equilibration_data
            else:
                data = self.data

            protein_series = data[data.index.get_level_values('Protein') == Protein]
            ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == Ligand]
            plt.figure(dpi=300, figsize=(2,8))
            plt.title(Protein+'-'+Ligand, size=5)
            acc_prob = {}
            n_traj = 0
            for t in ligand_series.index.levels[3]:
                n_traj += 1
                traj_series = ligand_series[ligand_series.index.get_level_values('Trajectory') == t]

                steps = max(traj_series.index.get_level_values('Step'))
                x = list(range(0, steps))
                y = [t if v in traj_series.index.get_level_values('Step').to_list() else t-0.5 for v in x]

                for s,a in zip(x,y):
                    acc_prob.setdefault(s, 0)
                    if not str(a).endswith('.5'):
                        acc_prob[s] += 1
                plt.plot(x,y, lw=0.5, c='k')

            acc_steps = [s for s in acc_prob]
            acc_prob = [(acc_prob[s]/n_traj)+n_traj+1 for s in acc_prob]
            plt.plot(acc_steps,acc_prob, lw=0.5, c='r')
            plt.axhline(n_traj+1, lw=0.1, c='k', ls='--')
            plt.axhline(n_traj+2, lw=0.1, c='k', ls='--')
            plt.xticks([0, max(acc_steps)], [1,max(acc_steps)+1], size=4)
            plt.yticks([*range(1,n_traj+1)]+[n_traj+1.5], ['T'+str(t) for t in range(1,n_traj+1)]+['Acc. Prob.'], size=4)
            plt.ylim(0, n_traj+3)
            plt.xlabel('MC step', size=5)
            plt.ylabel('Trajectory', size=5)

        interact(getLigands, Protein=sorted(self.proteins), equilibration=False)

    def plotSimulationProteinRMSD(self, equilibration=True, productive=True):
        """
        Plot the progression of the protein atoms RMSD in the simulation data.

        Parameters
        ==========
        equilibration : bool
            Equilibration data is present or not
        productive : bool
            Productive data is present or not
        """
        if 'Protein RMSD' not in self.data and productive:
            raise ValueError('You must call calculateRMSD() before calling this function.')
        elif 'Protein RMSD' not in self.equilibration_data and equilibration:
            raise ValueError('You must call calculateRMSD() before calling this function.')

        self.plotSimulationMetric('Protein RMSD',
                                  equilibration=equilibration,
                                  productive=productive)

    def scatterPlotIndividualSimulation(self, protein, ligand, x, y, vertical_line=None, color_column=None, size=1.0, labels_size=10.0, plot_label=None,
                                        xlim=None, ylim=None, metrics=None, labels=None, title=None, title_size=14.0, return_axis=False, dpi=300, show_legend=False,
                                        axis=None, xlabel=None, ylabel=None, vertical_line_color='k', vertical_line_width=0.5, marker_size=0.8, clim=None, show=False,
                                        clabel=None, legend_font_size=6, no_xticks=False, no_yticks=False, no_cbar=False, no_xlabel=False, no_ylabel=False,
                                        relative_color_values=False, dataframe=None, **kwargs):
        """
        Creates a scatter plot for the selected protein and ligand using the x and y
        columns. Data series can be filtered by specific metrics.

        Parameters
        ==========
        protein : str
            The target protein.
        ligand : str
            The target ligand.
        x : str
            The column name of the data to plot in the x-axis.
        y : str
            The column name of the data to plot in the y-axis.
        vertical_line : float
            Position to plot a vertical line.
        color_column : str
            The column name to use for coloring the plot. Also a color cna be given
            to use uniformly for the points.
        xlim : tuple
            The limits for the x-range.
        ylim : tuple
            The limits for the y-range.
        clim : tuple
            The limits for the color range.
        metrics : dict
            A set of metrics for filtering the data points.
        labels : dict
            Analog to metrics, use the label column values to filter the data.
        title : str
            The plot's title.
        return_axis : bool
            Whether to return the axis of this plot.
        axis : matplotlib.pyplot.axis
            The axis to use for plotting the data.
        """

        if not isinstance(dataframe, type(None)):
            protein_series = dataframe[dataframe.index.get_level_values('Protein') == protein]
        else:
            protein_series = self.data[self.data.index.get_level_values('Protein') == protein]

        if protein_series.empty:
            raise ValueError('Protein name %s not found in data!' % protein)
        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
        if ligand_series.empty:
            raise ValueError("Ligand name %s not found in protein's %s data!" % (ligand, protein))


        if len(ligand_series) != 0:
            # Add distance data to ligand_series
            if protein in self.distances:
                if ligand in self.distances[protein]:
                    if not isinstance(self.distances[protein][ligand], type(None)):
                        for distance in self.distances[protein][ligand]:
                            #if distance.startswith('distance_'):

                            if not isinstance(dataframe, type(None)):
                                indexes = dataframe.reset_index().set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps', 'Step']).index
                                ligand_series[distance] = self.distances[protein][ligand][self.distances[protein][ligand].index.isin(indexes)][distance].tolist()
                            else:
                                ligand_series[distance] = self.distances[protein][ligand][distance].tolist()

            # Add angle data to ligand_series
            if protein in self.angles:
                if ligand in self.angles[protein]:
                    if not isinstance(self.angles[protein][ligand], type(None)):
                        for angle in self.angles[protein][ligand]:
                            if not isinstance(dataframe, type(None)):
                                indexes = dataframe.reset_index().set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps', 'Step']).index
                                ligand_series[angle] = self.angles[protein][ligand][self.angles[protein][ligand].index.isin(indexes)][angle].tolist()
                            else:
                                ligand_series[angle] = self.angles[protein][ligand][angle].tolist()

        # Filter points by metric
        if not isinstance(metrics, type(None)):
            for metric in metrics:
                if isinstance(metrics[metric], float):
                    mask = ligand_series[metric] <= metrics[metric]
                elif isinstance(metrics[metric], tuple):
                    mask = (ligand_series[metric] >= metrics[metric][0]).to_numpy()
                    mask = mask & ((ligand_series[metric] <= metrics[metric][1]).to_numpy())
                ligand_series = ligand_series[mask]

        if not isinstance(labels, type(None)):
            for label in labels:
                if labels[label] != None:
                    mask = ligand_series[label] == labels[label]
                    ligand_series = ligand_series[mask]

        # Check if an axis has been given
        new_axis = False
        if axis == None:
            plt.figure(figsize=(4*size, 3.3*size), dpi=dpi)
            axis = plt.gca()
            new_axis = True

        # Check if label has been given
        if plot_label == None:
            plot_label = protein+self.separator+ligand

        # Define color columns
        color_columns = [k for k in ligand_series.keys()]
        color_columns = [k for k in color_columns if ':' not in k]
        color_columns = [k for k in color_columns if 'distance' not in k]
        color_columns = [k for k in color_columns if 'angle' not in k]
        color_columns = [k for k in color_columns if not k.startswith('metric_')]
        # color_columns.pop(color_columns.index('Step'))

        if color_column != None:

            if clim != None:
                vmin = clim[0]
                vmax = clim[1]
            else:
                vmin = None
                vmax = None

            ascending = False
            colormap='Blues_r'

            if color_column == 'Step':
                ascending = True
                colormap='Blues'

            elif color_column == 'Epoch' or color_column == 'Cluster':
                ascending = True
                color_values = ligand_series.reset_index()[color_column]
                cmap = plt.cm.jet
                cmaplist = [cmap(i) for i in range(cmap.N)]
                cmaplist[0] = (.5, .5, .5, 1.0)
                max_epoch = max(color_values.tolist())
                bounds = np.linspace(0, max_epoch+1, max_epoch+2)
                norm = mpl.colors.BoundaryNorm(bounds, cmap.N)
                colormap = mpl.colors.LinearSegmentedColormap.from_list(
                                                'Custom cmap', cmaplist, cmap.N)
                color_values = color_values+0.01
                sc = axis.scatter(ligand_series[x],
                    ligand_series[y],
                    c=color_values,
                    cmap=colormap,
                    norm=norm,
                    vmin=vmin,
                    vmax=vmax,
                    label=plot_label,
                    s=marker_size,
                    **kwargs)
                if new_axis:
                    if not no_cbar:
                        cbar = plt.colorbar(sc)
                        cbar.set_label(label=color_column, size=labels_size*size)
                        cbar.ax.tick_params(labelsize=labels_size*size)

            elif color_column in color_columns:
                ligand_series = ligand_series.sort_values(color_column, ascending=ascending)
                color_values = ligand_series[color_column]

                if relative_color_values:
                    color_values = color_values-np.min(color_values)

                sc = axis.scatter(ligand_series[x],
                    ligand_series[y],
                    c=color_values,
                    cmap=colormap,
                    vmin=vmin,
                    vmax=vmax,
                    label=plot_label,
                    s=marker_size*size,
                    **kwargs)
                if new_axis:
                    if not no_cbar:
                        cbar = plt.colorbar(sc)
                        if clabel == None:
                            clabel = color_column
                        cbar.set_label(label=clabel, size=labels_size*size)
                        cbar.ax.tick_params(labelsize=labels_size*size)
            else:
                sc = axis.scatter(ligand_series[x],
                    ligand_series[y],
                    c=color_column,
                    vmin=vmin,
                    vmax=vmax,
                    label=plot_label,
                    s=marker_size*size,
                    **kwargs)

        else:
            sc = axis.scatter(ligand_series[x],
                ligand_series[y],
                label=plot_label,
                s=marker_size*size,
                **kwargs)

        if not isinstance(vertical_line, type(None)):
            axis.axvline(vertical_line, c=vertical_line_color, lw=vertical_line_width, ls='--')

        if xlabel == None and not no_xlabel:
            xlabel = x
        if ylabel == None and not no_ylabel:
            ylabel = y

        axis.set_xlabel(xlabel, fontsize=labels_size*size)
        axis.set_ylabel(ylabel, fontsize=labels_size*size)
        axis.tick_params(axis='both', labelsize=labels_size*size)

        if no_xticks:
            for tick in axis.xaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        if no_yticks:
            for tick in axis.yaxis.get_major_ticks():
                tick.tick1line.set_visible(False)
                tick.tick2line.set_visible(False)
                tick.label1.set_visible(False)
                tick.label2.set_visible(False)

        # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.8)
    #     plt.tight_layout()

        if title != None:
            axis.set_title(title, fontsize=title_size*size)
        if xlim != None:
            axis.set_xlim(xlim)
        if ylim != None:
            axis.set_ylim(ylim)

        if show_legend:
            plt.legend(prop={'size': legend_font_size*size})

        if show:
            plt.show()

        if return_axis:
            return axis

    def boxPlotProteinSimulation(self, protein, column):
        """
        Creates a box plot for the selected protein using the specified column values
        for all ligands present.
        """

        protein_series = self.data[self.data.index.get_level_values('Protein') == protein]
        if protein_series.empty:
            raise ValueError('Protein name %s not found in data!' % protein)

        plt.figure()
        X = []
        labels = []

        for ligand in self.ligands:
            ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
            if not ligand_series.empty:
                X.append(ligand_series[column].tolist())
                labels.append(ligand)
        plt.boxplot(X, labels=labels)
        plt.xlabel('Ligands')
        plt.ylabel(column)
        plt.xticks(rotation=75)
        # plt.show()

    def boxPlotLigandSimulation(self, ligand, column):
        """
        Creates a box plot for the selected ligand using the specified column values
        for all proteins present.
        """

        ligand_series = self.data[self.data.index.get_level_values('Ligand') == ligand]
        if ligand_series.empty:
            raise ValueError('Ligand name %s not found in data!' % ligand)

        plt.figure()

        X = []
        labels = []
        for protein in self.proteins:
            protein_series = ligand_series[ligand_series.index.get_level_values('Protein') == protein]
            if not protein_series.empty:
                X.append(protein_series[column].tolist())
                labels.append(protein)

        plt.boxplot(X, labels=labels)
        plt.xlabel('Proteins')
        plt.ylabel(column)
        plt.xticks(rotation=90)
        plt.show()

    def bindingEnergyLandscape(self, initial_threshold=3.5, vertical_line=None, xlim=None, ylim=None, clim=None, color=None,
                               size=1.0, alpha=0.05, vertical_line_width=0.5, vertical_line_color='k', dataframe=None,
                               title=None, no_xticks=False, no_yticks=False, no_xlabel=False, no_ylabel=False,
                               no_cbar=False, xlabel=None, ylabel=None, clabel=None, relative_total_energy=False):
        """
        Plot binding energy as interactive plot.
        """

        if self.distances == {}:
            if os.path.isdir(self.pele_folder):
                raise ValueError('There are no distances in pele data. Use calculateDistances to show plot.')
            else:
                raise ValueError('There are no distances in pele data and there is no pele folder to calculate them')

        def getLigands(protein, dataframe=None):

            if not isinstance(dataframe, type(None)):
                protein_series = dataframe[dataframe.index.get_level_values('Protein') == protein]
            else:
                protein_series = self.data[self.data.index.get_level_values('Protein') == protein]

            ligands = list(set(protein_series.index.get_level_values('Ligand').tolist()))
            ligands_ddm = Dropdown(options=ligands, description='Ligand',
                                   style= {'description_width': 'initial'})

            interact(getDistance, protein_series=fixed(protein_series), protein=fixed(protein), ligand=ligands_ddm)

        def getDistance(protein_series, protein, ligand, by_metric=False):
            ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]

            distances = []
            distance_label = 'Distance'
            if by_metric:
                distances = []
                for d in ligand_series:
                    if d.startswith('metric_'):
                        if not ligand_series[d].dropna().empty:
                            distances.append(d)

            if distances == []:
                by_metric = False
            else:
                distance_label = 'Metric'

            if not by_metric:
                distances = []
                if ligand in self.distances[protein] and not isinstance(self.distances[protein][ligand], type(None)):
                    for d in self.distances[protein][ligand]:
                        if 'distance' in d:
                            distances.append(d)
                        elif '_coordinate' in d:
                            distances.append(d)

                # Add angle
                if ligand in self.angles[protein] and not isinstance(self.angles[protein][ligand], type(None)):
                    for d in self.angles[protein][ligand]:
                        if 'angle' in d:
                            distances.append(d)

                if 'Ligand RMSD' in self.data:
                    distances.append('Ligand RMSD')

                if distances == []:
                    raise ValueError('Not Ligand RMSD nor distances were found! Consider to calculate some distance.')

            distances_ddm = Dropdown(options=distances, description=distance_label,
                                     style= {'description_width': 'initial'})

            interact(getMetrics, distances=fixed(distances_ddm),
                     ligand_series=fixed(ligand_series),
                     protein=fixed(protein), ligand=fixed(ligand))

        def getMetrics(ligand_series, distances, protein, ligand, filter_by_metric=False, filter_by_label=False,
                       color_by_metric=False, color_by_labels=False):

            if color_by_metric or filter_by_metric:
                metrics = [k for k in ligand_series.keys() if 'metric_' in k]
                metrics_sliders = {}
                for m in metrics:

                    if self.metric_type[m] == 'distance':
                        m_slider = FloatSlider(
                                        value=initial_threshold,
                                        min=0,
                                        max=max(30, max(ligand_series[m])),
                                        step=0.1,
                                        description=m+':',
                                        disabled=False,
                                        continuous_update=False,
                                        orientation='horizontal',
                                        readout=True,
                                        readout_format='.2f',
                                    )

                    elif self.metric_type[m] == 'angle':
                        m_slider = FloatRangeSlider(
                                        value=[110, 130],
                                        min=-180,
                                        max=180,
                                        step=0.1,
                                        description=m+':',
                                        disabled=False,
                                        continuous_update=False,
                                        orientation='horizontal',
                                        readout=True,
                                        readout_format='.2f',
                                    )

                    elif self.metric_type[m] == 'torsion':
                        m_slider = FloatRangeSlider(
                                        value=[90, 120],
                                        min=-180,
                                        max=180,
                                        step=0.1,
                                        description=m+':',
                                        disabled=False,
                                        continuous_update=False,
                                        orientation='horizontal',
                                        readout=True,
                                        readout_format='.2f',
                                    )

                    metrics_sliders[m] = m_slider

            else:
                metrics_sliders = {}

            if filter_by_label:
                labels_ddms = {}
                labels = [l for l in ligand_series.keys() if 'label_' in l]
                for l in labels:
                    label_options = [None]+sorted(list(set(ligand_series[l])))
                    labels_ddms[l] = Dropdown(options=label_options, description=l,
                                              style= {'description_width': 'initial'})
            else:
                labels_ddms = {}

            interact(getColor, distance=distances, protein=fixed(protein), ligand=fixed(ligand),
                     metrics=fixed(metrics_sliders), ligand_series=fixed(ligand_series),
                     color_by_metric=fixed(color_by_metric), color_by_labels=fixed(color_by_labels), **labels_ddms)

        def getColor(distance, ligand_series, metrics, protein, ligand, color_by_metric=False,
                     color_by_labels=False, **labels):

            if color == None:
                color_columns = [k for k in ligand_series.keys()]
                color_columns = [k for k in color_columns if ':' not in k]
                color_columns = [k for k in color_columns if 'distance' not in k]
                color_columns = [k for k in color_columns if not k.startswith('metric_')]
                color_columns = [k for k in color_columns if not k.startswith('label_')]
                color_columns = [None, 'Epoch']+color_columns

                if 'Binding Energy' in ligand_series:
                    be_column = 'Binding Energy'
                elif 'bindingEnergy' in ligand_series:
                    be_column = 'bindingEnergy'

                del color_columns[color_columns.index(be_column)]

                color_ddm = Dropdown(options=color_columns, description='Color',
                                     style= {'description_width': 'initial'})
                if color_by_metric:
                    color_ddm.options = ['Color by metrics']
                    alpha = 0.10
                elif color_by_labels:
                    color_ddm.options = ['Color by labels']
                    alpha = 1.00
                else:
                    alpha = fixed(0.10)

                color_object = color_ddm

            else:
                color_object = fixed(color)

            interact(_bindingEnergyLandscape, color=color_object, ligand_series=fixed(ligand_series),
                     distance=fixed(distance), color_by_metric=fixed(color_by_metric), color_by_labels=fixed(color_by_labels),
                     Alpha=alpha, labels=fixed(labels), protein=fixed(protein), ligand=fixed(ligand), title=fixed(title),
                     no_xticks=fixed(no_xticks), no_yticks=fixed(no_yticks), no_cbar=fixed(no_cbar), clabel=fixed(clabel),
                     no_xlabel=fixed(no_xlabel), no_ylabel=fixed(no_ylabel), xlabel=fixed(xlabel), ylabel=fixed(ylabel),
                     relative_total_energy=fixed(relative_total_energy), clim=fixed(clim), **metrics)

        def _bindingEnergyLandscape(color, ligand_series, distance, protein, ligand,
                                    color_by_metric=False, color_by_labels=False,
                                    Alpha=0.10, labels=None, title=None, no_xticks=False,
                                    no_yticks=False, no_cbar=False, no_xlabel=True, no_ylabel=False,
                                    xlabel=None, ylabel=None, clabel=None, relative_total_energy=False,
                                    clim=None, **metrics):

            skip_fp = False
            show = True

            # Deactivate metrics for first plot and make it black
            return_axis = False
            if color_by_metric:
                color = 'k'
                color_metrics = metrics
                metrics = {}
                return_axis = True
                show = False

            elif color_by_labels:
                skip_fp = True
                return_axis = True
                show = False

            if color == 'Total Energy' and relative_total_energy:
                relative_color_values = True
                if clim == None:
                    clim = (0,27.631021116) # Calculated for a probability contribution of 1% at KT 6 (-(ln(0.01)*6)
            else:
                relative_color_values = None

            if 'Binding Energy' in ligand_series:
                be_column = 'Binding Energy'
            elif 'bindingEnergy' in ligand_series:
                be_column = 'bindingEnergy'

            if not skip_fp:
                axis = self.scatterPlotIndividualSimulation(protein, ligand, distance, be_column, xlim=xlim, ylim=ylim,
                                                            vertical_line=vertical_line, color_column=color, clim=clim, size=size,
                                                            vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                            metrics=metrics, labels=labels, return_axis=return_axis, show=show,
                                                            title=title, no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar,
                                                            no_xlabel=no_xlabel, no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel,
                                                            clabel=clabel, relative_color_values=relative_color_values, dataframe=ligand_series)

            # Make a second plot only coloring points passing the filters
            if color_by_metric:
                self.scatterPlotIndividualSimulation(protein, ligand, distance, be_column, xlim=xlim, ylim=ylim,
                                                     vertical_line=vertical_line, color_column='r', clim=clim, size=size,
                                                     vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                     metrics=color_metrics, labels=labels, axis=axis, show=True, alpha=Alpha,
                                                     no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar, no_xlabel=no_xlabel,
                                                     no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel, clabel=clabel, dataframe=ligand_series)
            elif color_by_labels:
                all_labels = {}
                for l in ligand_series.keys():
                    if 'label_' in l:
                        all_labels[l] = sorted(list(set(ligand_series[l].to_list())))

                for l in all_labels:
                    colors = iter([plt.cm.Set2(i) for i in range(len(all_labels[l]))])
                    for i,v in enumerate(all_labels[l]):
                        if i == 0:
                            axis = self.scatterPlotIndividualSimulation(protein, ligand, distance, be_column, xlim=xlim, ylim=ylim, plot_label=v,
                                                                   vertical_line=vertical_line, color_column=[next(colors)], clim=clim, size=size,
                                                                   vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                                   metrics=metrics, labels=labels, return_axis=return_axis, alpha=Alpha, show=show,
                                                                   no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar, no_xlabel=no_xlabel,
                                                                   no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel, clabel=clabel, dataframe=ligand_series)
                            continue
                        elif i == len(all_labels[l])-1:
                            show = True
                        axis = self.scatterPlotIndividualSimulation(protein, ligand, distance, be_column, xlim=xlim, ylim=ylim, plot_label=v,
                                                                    vertical_line=vertical_line, color_column=[next(colors)], clim=clim, size=size,
                                                                    vertical_line_color=vertical_line_color, vertical_line_width=vertical_line_width,
                                                                    metrics=metrics, labels={l:v}, return_axis=return_axis, axis=axis, alpha=Alpha, show=show,
                                                                    show_legend=True, title=title, no_xticks=no_xticks, no_yticks=no_yticks, no_cbar=no_cbar,
                                                                    no_xlabel=no_xlabel, no_ylabel=no_ylabel, xlabel=xlabel, ylabel=ylabel, clabel=clabel,
                                                                    dataframe=ligand_series)

        proteins = self.proteins
        proteins_ddm = Dropdown(options=proteins, description='Protein',
                                style= {'description_width': 'initial'})

        interact(getLigands, protein=proteins_ddm, dataframe=fixed(dataframe))

    def plotDistributions(self):
        """
        Plot distribution of different values in the simulation in a by-protein basis.
        """
        def _plotDistributionByProtein(Protein, Column):
            self.boxPlotProteinSimulation(Protein, Column)
        def _plotDistributionByLigand(Ligand, Column):
            self.boxPlotLigandSimulation(Ligand, Column)
        def selectLevel(By_protein, By_ligand):
            if By_protein:
                interact(_plotDistributionByProtein, Protein=self.proteins, Column=columns)
            if By_ligand:
                interact(_plotDistributionByLigand, Ligand=self.ligands, Column=columns)

        columns = [k for k in self.data.keys() if ':' not in k and 'distance' not in k]
        # del columns[columns.index('Task')]
        del columns[columns.index('Step')]

        interact(selectLevel, By_protein=True, By_ligand=False)

    def getDistances(self, protein, ligand, return_none=False):
        """
        Returns the distance associated to a specific protein and ligand simulation
        """

        if protein not in self.distances:
            #raise ValueError('There are no distances for protein %s. Use calculateDistances to obtain them.' % protein)
            print('WARNING: There are no distances for protein %s. Use calculateDistances to obtain them.' % protein)
        elif ligand not in self.distances[protein]:
            #raise ValueError('There are no distances for protein %s and ligand %s. Use calculateDistances to obtain them.' % (protein, ligand))
            print('WARNING: There are no distances for protein %s and ligand %s. Use calculateDistances to obtain them.' % (protein, ligand))

        #if not os.path.isdir(self.pele_folder):
        #    raise ValueError('There are no distances in pele data and there is no pele folder to calculate them')

        distances = []

        if protein not in self.distances:
            return distances
        elif ligand not in self.distances[protein]:
            return distances
        if isinstance(self.distances[protein][ligand], type(None)):
            return distances

        for d in self.distances[protein][ligand]:
            if 'distance_' in d:
                distances.append(d)
            elif 'angle_' in d:
                distances.append(d)
            elif 'torsion_' in d:
                distances.append(d)
            elif '_coordinate' in d:
                distances.append(d)

        return distances

    def plotCatalyticPosesFraction(self, initial_threshold=3.5):
        """
        Plot interactively the number of catalytic poses as a function of the threshold
        of the different catalytic metrics. The plot can be done by protein or by ligand.
        """
        metrics = [k for k in self.data.keys() if 'metric_' in k]

        if len(metrics)==0:
            raise ValueError('No calatytic metrics have been computed.')

        def _plotCatalyticPosesFractionByProtein(Protein, Separate_by_metric=True, **metrics):

            protein_series = self.data[self.data.index.get_level_values('Protein') == Protein]
            if protein_series.empty:
                raise ValueError('Protein name %s not found in data!' % Protein)

            plt.figure()

            if Separate_by_metric:
                catalytic_count = {}
                labels = []
                for metric in metrics:
                    catalytic_count[metric] = []
                    for ligand in self.ligands:
                        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                        if not ligand_series.empty:
                            catalytic_count[metric].append(sum(ligand_series[metric] <= metrics[metric])/ligand_series.shape[0])
                            if ligand not in labels:
                                labels.append(ligand)

                if len(labels) == 2:
                    bar_width = 0.2
                else:
                    bar_width = 0.5
                bar_width = bar_width/len(labels)
                pos = np.arange(len(labels))
                x_pos = pos-(bar_width/2)+(len(metrics)*bar_width)/2
                plt.xticks(x_pos, labels)
                for metric in catalytic_count:
                    counts = catalytic_count[metric]
                    plt.bar(pos, counts, bar_width)
                    pos = pos + bar_width

                plt.xlabel('Ligands')
                plt.ylabel('Catalytic Poses Fraction')
                plt.ylim(0,1)

            else:
                catalytic_count = []
                labels = []
                for ligand in self.ligands:
                    ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                    if not ligand_series.empty:
                        metric_series = ligand_series
                        for metric in metrics:
                            metric_series = metric_series[metric_series[metric] <= metrics[metric]]
                        catalytic_count.append(metric_series.shape[0]/ligand_series.shape[0])
                        labels.append(ligand)

                plt.bar(labels, catalytic_count)
                plt.xticks(rotation=45)
                plt.xlabel('Ligands')
                plt.ylabel('Catalytic Poses Fraction')
                plt.ylim(0,1)

        def _plotCatalyticPosesFractionByLigand(Ligand, Separate_by_metric=True, **metrics):

            ligand_series = self.data[self.data.index.get_level_values('Ligand') == Ligand]
            if ligand_series.empty:
                raise ValueError('Ligand name %s not found in data!' % Ligand)

            plt.figure()

            if Separate_by_metric:
                catalytic_count = {}
                labels = []
                for metric in metrics:
                    catalytic_count[metric] = []
                    for protein in self.proteins:
                        protein_series = ligand_series[ligand_series.index.get_level_values('Protein') == protein]
                        if not protein_series.empty:
                            catalytic_count[metric].append(sum(protein_series[metric] <= metrics[metric])/protein_series.shape[0])
                            if protein not in labels:
                                labels.append(protein)

                if len(labels) == 2:
                    bar_width = 0.2
                else:
                    bar_width = 0.5
                bar_width = bar_width/len(labels)
                pos = np.arange(len(labels))
                x_pos = pos-(bar_width/2)+(len(metrics)*bar_width)/2
                plt.xticks(x_pos, labels)
                for metric in catalytic_count:
                    counts = catalytic_count[metric]
                    plt.bar(pos, counts, bar_width)
                    pos = pos + bar_width

                plt.xticks(rotation=45)
                plt.xlabel('Protein')
                plt.ylabel('Catalytic Poses Fraction')
                plt.ylim(0,1)

            else:
                catalytic_count = []
                labels = []
                for protein in self.proteins:
                    protein_series = ligand_series[ligand_series.index.get_level_values('Protein') == protein]
                    if not protein_series.empty:
                        metric_series = protein_series
                        for metric in metrics:
                            metric_series = metric_series[metric_series[metric] <= metrics[metric]]
                        catalytic_count.append(metric_series.shape[0]/protein_series.shape[0])
                        labels.append(protein)

                plt.bar(labels, catalytic_count)
                plt.xticks(rotation=45)
                plt.xlabel('Proteins')
                plt.ylabel('Catalytic Poses Fraction')
                plt.ylim(0,1)

        def selectLevel(By_protein, By_ligand):
            if By_protein:
                if len(metrics) > 1:
                    interact(_plotCatalyticPosesFractionByProtein, Protein=self.proteins, Separate_by_metric=False, **metrics)
                else:
                    interact(_plotCatalyticPosesFractionByProtein, Protein=self.proteins, **metrics)
            if By_ligand:
                if len(metrics) > 1:
                    interact(_plotCatalyticPosesFractionByLigand, Ligand=self.ligands, Separate_by_metric=False, **metrics)
                else:
                    interact(_plotCatalyticPosesFractionByLigand, Ligand=self.ligands, **metrics)

        metrics = {m:initial_threshold for m in metrics}

        interact(selectLevel, By_protein=True, By_ligand=False)

    def plotCatalyticBindingEnergyDistributions(self, initial_threshold=3.5):
        """
        Plot interactively the binding energy distributions as a function of the threshold
        of the different catalytic metrics. The plot can be done by protein or by ligand.
        """
        metrics = [k for k in self.data.keys() if 'metric_' in k]

        if len(metrics)==0:
            raise ValueError('No calatytic metrics have been computed.')

        def _plotBindingEnergyByProtein(Protein, Separate_by_metric=True, **metrics):

            protein_series = self.data[self.data.index.get_level_values('Protein') == Protein]
            if protein_series.empty:
                raise ValueError('Protein name %s not found in data!' % Protein)

            plt.figure()

            if Separate_by_metric:
                catalytic_dist = {}
                labels = []
                for metric in metrics:
                    catalytic_dist[metric] = []
                    for ligand in self.ligands:
                        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                        if not ligand_series.empty:
                            catalytic_dist[metric].append(ligand_series[ligand_series[metric] <= metrics[metric]]['Binding Energy'])
                            if ligand not in labels:
                                labels.append(ligand)

                if len(labels) == 2:
                    bar_width = 0.2
                else:
                    bar_width = 0.5
                bar_width = bar_width/len(labels)
                pos = np.arange(len(labels))
                x_pos = pos-(bar_width/2)+(len(metrics)*bar_width)/2
                for metric in catalytic_dist:
                    plt.boxplot(catalytic_dist[metric], widths=bar_width*0.8, positions=pos)
                    pos = pos + bar_width
                plt.xticks(x_pos, labels, rotation=45)
                plt.xlabel('Ligands')
                plt.ylabel('Binding Energy')

            else:
                catalytic_dist = []
                labels = []
                for ligand in self.ligands:
                    ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                    if not ligand_series.empty:
                        metric_series = ligand_series
                        for metric in metrics:
                            metric_series = metric_series[metric_series[metric] <= metrics[metric]]
                        catalytic_dist.append(metric_series['Binding Energy'])
                        labels.append(ligand)

                plt.boxplot(catalytic_dist, labels=labels)
                plt.xticks(rotation=45)
                plt.xlabel('Ligands')
                plt.ylabel('Binding Energy')

        def _plotBindingEnergyByLigand(Ligand, Separate_by_metric=True, **metrics):

            ligand_series = self.data[self.data.index.get_level_values('Ligand') == Ligand]
            if ligand_series.empty:
                raise ValueError('Ligand name %s not found in data!' % Ligand)

            plt.figure()

            if Separate_by_metric:
                catalytic_dist = {}
                labels = []
                for metric in metrics:
                    catalytic_dist[metric] = []
                    for protein in self.proteins:
                        protein_series = ligand_series[ligand_series.index.get_level_values('Protein') == protein]
                        if not protein_series.empty:
                            catalytic_dist[metric].append(protein_series[protein_series[metric] <= metrics[metric]]['Binding Energy'])
                            if protein not in labels:
                                labels.append(protein)

                if len(labels) == 2:
                    bar_width = 0.2
                else:
                    bar_width = 0.5
                bar_width = bar_width/len(labels)
                pos = np.arange(len(labels))
                x_pos = pos-(bar_width/2)+(len(metrics)*bar_width)/2
                for metric in catalytic_dist:
                    plt.boxplot(catalytic_dist[metric], widths=bar_width*0.8, positions=pos)
                    pos = pos + bar_width

                plt.xticks(x_pos, labels, rotation=45)
                plt.xlabel('Proteins')
                plt.ylabel('Binding Energy')

            else:
                catalytic_dist = []
                labels = []
                for protein in self.proteins:
                    protein_series = ligand_series[ligand_series.index.get_level_values('Protein') == protein]
                    if not protein_series.empty:
                        metric_series = protein_series
                        for metric in metrics:
                            metric_series = metric_series[metric_series[metric] <= metrics[metric]]
                        catalytic_dist.append(metric_series['Binding Energy'])
                        labels.append(protein)

                plt.boxplot(catalytic_dist, labels=labels)
                plt.xticks(rotation=45)
                plt.xlabel('Proteins')
                plt.ylabel('Binding Energy')

        def selectLevel(By_protein, By_ligand):
            if By_protein:
                if len(metrics) > 1:
                    interact(_plotBindingEnergyByProtein, Protein=self.proteins, Separate_by_metric=False, **metrics)
                else:
                    interact(_plotBindingEnergyByProtein, Protein=self.proteins, **metrics)
            if By_ligand:
                if len(metrics) > 1:
                    interact(_plotBindingEnergyByLigand, Ligand=self.ligands, Separate_by_metric=False, **metrics)
                else:
                    interact(_plotBindingEnergyByLigand, Ligand=self.ligands, **metrics)

        metrics = {m:initial_threshold for m in metrics}

        interact(selectLevel, By_protein=True, By_ligand=False)

    def bindingFreeEnergyMatrix(self):

        def _bindingFreeEnergyMatrix(KT=0.593):
            # Create a matrix of length proteins times ligands
            M = np.zeros((len(self.proteins), len(self.ligands)))
            for i,protein in enumerate(self.proteins):
                protein_series = self.data[self.data.index.get_level_values('Protein') == protein]
                for j,ligand in enumerate(self.ligands):
                    ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                    if not ligand_series.empty:
                        total_energy = ligand_series['Total Energy']
                        relative_energy = total_energy-total_energy.min()
                        Z = np.sum(np.exp(-relative_energy/KT))
                        probability = np.exp(-relative_energy/KT)/Z
                        M[i][j] = np.sum(probability*ligand_series['Binding Energy'])
                    else:
                        M[i][j] = np.nan

            plt.matshow(M, cmap='autumn')
            plt.colorbar(label='Binding Free Energy')
            plt.xlabel('Ligands', fontsize=12)
            plt.xticks(range(len(self.ligands)), self.ligands, rotation=50)
            plt.ylabel('Proteins', fontsize=12)
            plt.yticks(range(len(self.proteins)), self.proteins)

        KT_slider = FloatSlider(
                        value=0.593,
                        min=0.593,
                        max=20.0,
                        step=0.1,
                        description='KT:',
                        disabled=False,
                        continuous_update=False,
                        orientation='horizontal',
                        readout=True,
                        readout_format='.1f',
                    )

        interact(_bindingFreeEnergyMatrix, KT=KT_slider)

    def bindingFreeEnergyCatalyticDifferenceMatrix(self, initial_threshold=3.5, initial_threshold_filter=3.5, measured_metrics=None,
                                                   store_values=False, lig_label_rot=90, observable='Binding Energy',
                                                   matrix_file='catalytic_matrix.npy', models_file='catalytic_models.json',
                                                   max_metric_threshold=30, pele_data=None, KT=5.93, to_csv=None,
                                                   only_proteins=None, only_ligands=None, average_binding_energy=False,
                                                   nan_to_zero=False):

        def _bindingFreeEnergyMatrix(KT=KT, sort_by_ligand=None, models_file='catalytic_models.json',
                                     lig_label_rot=90, pele_data=None, only_proteins=None, only_ligands=None,
                                     abc=False, avg_ebc=False, n_poses=10, **metrics):

            metrics_filter = {m:metrics[m] for m in metrics if m.startswith('metric_')}
            labels_filter = {l:metrics[l] for l in metrics if l.startswith('label_')}

            if isinstance(pele_data, type(None)):
                pele_data = self.data

            if only_proteins != None:
                proteins = [p for p in self.proteins if p in only_proteins]
            else:
                proteins = self.proteins

            if only_ligands != None:
                ligands = [l for l in self.ligands if l in only_ligands]
            else:
                ligands = self.ligands

            if len(proteins) == 0:
                raise ValueError('No proteins were found!')
            if len(ligands) == 0:
                raise ValueError('No ligands were found!')

            # Create a matrix of length proteins times ligands
            M = np.zeros((len(proteins), len(ligands)))

            # Calculate the probaility of each state
            for i, protein in enumerate(proteins):

                protein_series = pele_data[pele_data.index.get_level_values('Protein') == protein]

                for j, ligand in enumerate(ligands):

                    ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]

                    if not ligand_series.empty:

                        if abc:
                            # Calculate partition function
                            total_energy = ligand_series['Total Energy']
                            energy_minimum = total_energy.min()
                            relative_energy = total_energy-energy_minimum
                            Z = np.sum(np.exp(-relative_energy/KT))

                        # Calculate catalytic binding energy
                        catalytic_series = ligand_series

                        for metric in metrics_filter:
                            if isinstance(metrics_filter[metric], float):
                                mask = catalytic_series[metric] <= metrics_filter[metric]
                            elif isinstance(metrics[metric], tuple):
                                mask = (ligand_series[metric] >= metrics_filter[metric][0]).to_numpy()
                                mask = mask & ((ligand_series[metric] <= metrics_filter[metric][1]).to_numpy())
                            catalytic_series = catalytic_series[mask]

                        for l in labels_filter:
                            # Filter by labels
                            if labels_filter[l] != None:
                                catalytic_series = catalytic_series[catalytic_series[l] == labels_filter[l]]

                        if abc:
                            total_energy = catalytic_series['Total Energy']
                            relative_energy = total_energy-energy_minimum
                            probability = np.exp(-relative_energy/KT)/Z
                            M[i][j] = np.sum(probability*catalytic_series[observable])
                        elif avg_ebc:
                            M[i][j] = catalytic_series.nsmallest(n_poses, observable)[observable].mean()
                    else:
                        M[i][j] = np.nan

            if nan_to_zero:
                M[np.isnan(M)] = 0.0

            if abc:
                binding_metric_label = '$A_{B}^{C}$'
            elif avg_ebc:
                binding_metric_label = '$\overline{E}_{B}^{C}$'
            else:
                raise ValueError('You should mark at least one option: $A_{B}^{C}$ or $\overline{E}_{B}^{C}$!')

            if store_values:
                np.save(matrix_file, M)
                if not models_file.endswith('.json'):
                    models_file = models_file+'.json'
                with open(models_file, 'w') as of:
                    json.dump(proteins, of)

            if to_csv != None:
                catalytic_values = {}
                catalytic_values['Model'] = []
                catalytic_values['Ligand'] = []
                catalytic_values[binding_metric_label] = []

                for i,m in zip(M, proteins):
                    for v, l in zip(i,  ligands):
                        catalytic_values['Model'].append(m)
                        catalytic_values['Ligand'].append(l)
                        catalytic_values[binding_metric_label].append(v)
                catalytic_values = pd.DataFrame(catalytic_values)
                catalytic_values.set_index(['Model', 'Ligand'])
                catalytic_values.to_csv(to_csv)

            # Sort matrix by ligand or protein
            if sort_by_ligand == 'by_protein':
                protein_labels = proteins
            else:
                ligand_index = ligands.index(sort_by_ligand)
                sort_indexes = M[:, ligand_index].argsort()
                M = M[sort_indexes]
                protein_labels = [proteins[x] for x in sort_indexes]

            plt.figure(dpi=100, figsize=(0.28*len(ligands),0.2*len(proteins)))
            plt.imshow(M, cmap='autumn')
            plt.colorbar(label=binding_metric_label)

            plt.xlabel('Ligands', fontsize=12)
            ax = plt.gca()
            ax.set_xticklabels(ligands, rotation=lig_label_rot)
            plt.xticks(np.arange(0,len(ligands)), ligands, rotation=lig_label_rot)
            plt.ylabel('Proteins', fontsize=12)
            plt.yticks(range(len(proteins)), protein_labels)

            display(plt.show())

        # Check to_csv input
        if to_csv != None and not isinstance(to_csv, str):
            raise ValueError('to_csv must be a path to the output csv file.')
        if to_csv != None and not to_csv.endswith('.csv'):
            to_csv = to_csv+'.csv'

        # Define if PELE data is given
        if isinstance(pele_data, type(None)):
            pele_data = self.data

        # Add checks for the given pele data pandas df
        metrics = [k for k in pele_data.keys() if 'metric_' in k]
        labels = {}
        for m in metrics:
            for l in pele_data.keys():
                if 'label_' in l and l.replace('label_', '') == m.replace('metric_', ''):
                    labels[m] = sorted(list(set(pele_data[l])))

        metrics_sliders = {}
        labels_ddms = {}
        for m in metrics:
            if measured_metrics != None:
                if m in measured_metrics:
                    initial_threshold = initial_threshold
                else:
                    initial_threshold = initial_threshold_filter
            if self.metric_type[m] == 'distance':
                m_slider = FloatSlider(
                                value=initial_threshold,
                                min=0,
                                max=max_metric_threshold,
                                step=0.1,
                                description=m+':',
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='.2f',
                                style= {'description_width': 'initial'})

            elif self.metric_type[m] == 'angle':
                m_slider = FloatRangeSlider(
                                value=[110, 130],
                                min=-180,
                                max=180,
                                step=0.1,
                                description=m+':',
                                disabled=False,
                                continuous_update=False,
                                orientation='horizontal',
                                readout=True,
                                readout_format='.2f',
                            )

            metrics_sliders[m] = m_slider

            if m in labels and labels[m] != []:
                label_options = [None]+labels[m]
                label_ddm = Dropdown(options=label_options, description=m.replace('metric_', 'label_'), style= {'description_width': 'initial'})
                metrics_sliders[m.replace('metric_', 'label_')] = label_ddm

        if only_proteins != None:
            if isinstance(only_proteins, str):
                only_proteins = [only_proteins]

        # Get only ligands if given
        if only_ligands != None:
            if isinstance(only_ligands, str):
                only_ligands = [only_ligands]

            ligands = [l for l in self.ligands if l in only_ligands]
        else:
            ligands = self.ligands

        VB = []
        ligand_ddm = Dropdown(options=ligands+['by_protein'], description='Sort by ligand',
                              style= {'description_width': 'initial'})
        VB.append(ligand_ddm)

        abc = Checkbox(value=True,
                       description='$A_{B}^{C}$')
        VB.append(abc)

        if average_binding_energy:
            avg_ebc = Checkbox(value=False,
                               description='$\overline{E}_{B}^{C}$')

            VB.append(avg_ebc)

            Ebc_slider = IntSlider(
                     value=10,
                     min=1,
                     max=1000,
                     step=1,
                     description='N poses (only $\overline{E}_{B}^{C}$):',
                     disabled=False,
                     continuous_update=False,
                     orientation='horizontal',
                     readout=True)
            VB.append(Ebc_slider)

        KT_slider = FloatSlider(
                    value=KT,
                    min=0.593,
                    max=1000.0,
                    step=0.1,
                    description='KT:',
                    disabled=False,
                    continuous_update=False,
                    orientation='horizontal',
                    readout=True,
                    readout_format='.1f')

        for m in metrics_sliders:
            VB.append(metrics_sliders[m])
        for m in labels_ddms:
            VB.append(labels_ddms[m])
        VB.append(KT_slider)

        if average_binding_energy:
            plot = interactive_output(_bindingFreeEnergyMatrix, {'KT': KT_slider, 'sort_by_ligand' :ligand_ddm,
                                      'pele_data' : fixed(pele_data), 'models_file': fixed(models_file),
                                      'lig_label_rot' : fixed(lig_label_rot), 'only_proteins': fixed(only_proteins),
                                      'only_ligands': fixed(only_ligands), 'abc' : abc, 'avg_ebc' : avg_ebc,
                                      'n_poses' : Ebc_slider, **metrics_sliders})
        else:
            plot = interactive_output(_bindingFreeEnergyMatrix, {'KT': KT_slider, 'sort_by_ligand' :ligand_ddm,
                                      'pele_data' : fixed(pele_data), 'models_file': fixed(models_file),
                                      'lig_label_rot' : fixed(lig_label_rot), 'only_proteins': fixed(only_proteins),
                                      'only_ligands': fixed(only_ligands), 'abc' : abc, **metrics_sliders})

        VB.append(plot)
        VB = VBox(VB)

        display(VB)

    def visualiseBestPoses(self, pele_data=None, initial_threshold=3.5):

        if not os.path.isdir(self.pele_folder):
            raise ValueError('Pele folder not found. There are no trajectories.')


        def _visualiseBestPoses(Protein, Ligand, n_smallest=10, **metrics):
            protein_series = pele_data[pele_data.index.get_level_values('Protein') == Protein]
            ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == Ligand]

            # Filter by metric
            # Calculate catalytic binding energy
            catalytic_series = ligand_series
            for metric in metrics:
                catalytic_series = catalytic_series[catalytic_series[metric] <= metrics[metric]]

            catalytic_series = catalytic_series.nsmallest(n_smallest, observable)

            if catalytic_series.empty:
                raise ValueError('No frames were selected for the selected thresholds.')

            traj = pele_trajectory.loadTrajectoryFrames(catalytic_series,
                                                        self.trajectory_files[Protein][Ligand],
                                                        self.topology_files[Protein][Ligand])

            ligand_atoms = traj.topology.select('resname '+self.ligand_names[protein][ligand])
            neighbors = md.compute_neighbors(traj, 0.5, ligand_atoms)
            chain_ids = self.chain_ids[Protein][Ligand]

            # Get list of residues to depict
            residues = []
            for frame in neighbors:
                for x in frame:
                    residue = traj.topology.atom(x).residue
                    if residue.name != self.ligand_names[protein][ligand]:
                        chain = chain_ids[residue.chain.index]
                        resid = residue.resSeq
                        # The chain_ids dictiona must be done per residue
                        residue = str(resid)#+':'+chain
                        if residue not in residues:
                            residues.append(residue)
            # residues += [531]
            return pele_trajectory.showTrajectory(traj, residues=residues)

        def getLigands(Protein):
            protein_series = pele_data[pele_data.index.get_level_values('Protein') == Protein]
            ligands = list(set(protein_series.index.get_level_values('Ligand').tolist()))

            n_poses_slider = IntSlider(
                            value=10,
                            min=1,
                            max=50,
                            description='Poses:',
                            readout=True,
                            readout_format='.0f')

            interact(_visualiseBestPoses, Protein=fixed(Protein),
                     Ligand=ligands, n_smallest=n_poses_slider,
                     **metrics)

        # Define pele data as self.data if non given
        if isinstance(pele_data, type(None)):
            pele_data = self.data

        metrics = [k for k in pele_data.keys() if 'metric_' in k]

        widget_metrics = {}
        for metric in metrics:
            widget_metrics[metric] = FloatSlider(
                                    value=initial_threshold,
                                    min=0,
                                    max=30,
                                    step=0.05,
                                    description=metric+':',
                                    readout_format='.2f')
        metrics = widget_metrics

        # interactive_output(getLigands, {'Protein': protein})

        interact(getLigands, Protein=sorted(self.proteins))

    def getStepsMatrix(self, step='Accepted Pele Steps'):
        """
        Get the data for the last step of the last epoch of every protein and ligand trajectory.
        The matrix are stored in self.steps_matrix as (protein, ligand) dictionary. To plot these
        matrices use the plotTrajectoryLastSteps() matrix.
        """

        allowed_steps = ['Accepted Pele Steps', 'Step']
        if step not in ['Accepted Pele Steps', 'Step']:
            raise ValueError('The indicated step is not allowed. Try: %s' % allowed_steps)

        self.steps_matrix = {}
        for protein, ligand in self.pele_combinations:

            protein_data = self.data[self.data.index.get_level_values('Protein') == protein]
            ligand_data = protein_data[protein_data.index.get_level_values('Ligand') == ligand]

            epochs = [e for e in ligand_data.index.levels[2]]
            trajectories = [t for t in ligand_data.index.levels[3]]

            M = np.zeros((len(trajectories), len(epochs)))

            for i, epoch in enumerate(epochs):
                epoch_data = ligand_data[ligand_data.index.get_level_values('Epoch') == epoch]

                for j, traj in enumerate(trajectories):
                    traj_data = epoch_data[epoch_data.index.get_level_values('Trajectory') == traj]

                    if traj_data.empty:
                        M[j][i] = np.nan
                        continue

                    M[j][i] = traj_data.index.get_level_values(step).to_numpy().max()

            self.steps_matrix[(protein, ligand)] = M

        return self.steps_matrix

    def plotTrajectoryLastSteps(self):
        """
        Plot the last accepted step by each trajectory at each epoch
        """

        def getLigands(protein):
            protein_series = self.data[self.data.index.get_level_values('Protein') == protein]
            ligands = list(set(protein_series.index.get_level_values('Ligand').tolist()))
            ligands_ddm = Dropdown(options=ligands, description='Ligand',
                                   style= {'description_width': 'initial'})

            interact(plotLastSteps, protein=fixed(protein), ligand=ligands_ddm)

        def plotLastSteps(protein, ligand):

            M = self.steps_matrix[(protein, ligand)]
            plt.matshow(M)
            plt.xticks(range(M.shape[1]), labels=range(1,M.shape[1]+1))
            plt.yticks(range(M.shape[0]), labels=range(1,M.shape[0]+1))
            plt.xlabel('Epoch', size=15)
            plt.ylabel('Trajectory', size=15)
            cbar = plt.colorbar()
            cbar.set_label(label='Step', size=15)
            display(plt.show())

        if self.steps_matrix == None:
            getStepsMatrix(self)

        proteins = Dropdown(options=self.proteins, description='Protein',
                            style= {'description_width': 'initial'})

        interact(getLigands, protein=proteins)

    def visualiseInVMD(self, protein, ligand, resnames=None, resids=None, peptide=False,
                      num_trajectories='all', epochs=None, trajectories=None, equilibration=False):

        if not os.path.isdir(self.pele_folder):
            raise ValueError('Pele folder not found. There are no trajectories.')

        if isinstance(resnames, str):
            resnames = [resnames]

        if isinstance(resids, int):
            resids = [resids]

        if equilibration:
            traj_files = self.equilibration['trajectory'][protein][ligand]
            start_index = 1
        else:
            traj_files = self.trajectory_files[protein][ligand]
            start_index = 0

        trajs = [t for t in sorted(traj_files[start_index])]

        if num_trajectories == 'all' and isinstance(trajectories, type(None)):
            num_trajectories = len(trajs)

        if not isinstance(trajectories, type(None)):
            if isinstance(trajectories, list):
                trajectories = [trajs[i-1] for i in trajectories]
            else:
                raise ValueError('trajectories must be given as a list')

        elif isinstance(num_trajectories, int):
            trajectories = np.random.choice(trajs, size=num_trajectories, replace=False)

        if not isinstance(epochs, type(None)):
            if not isinstance(epochs, list):
                raise ValueError('epochs must be given as a list')

        with open('.load_vmd.tcl', 'w') as vmdf:

            topology = self.topology_files[protein][ligand]
            vmdf.write('color Display background white\n')
            vmdf.write('proc colorsel {selection color} {\n')
            vmdf.write('set colorlist {blue red gray orange yellow tan silver green white pink cyan purple lime \\\n')
            vmdf.write('               mauve ochre iceblue black yellow2 yellow3 green2 green3 cyan2 cyan3 blue2 \\\n')
            vmdf.write('               blue3 violet violet2 magenta magenta2 red2 red3 orange2 orange3}\n')
            vmdf.write('set num [lsearch $colorlist $color]\n')
            vmdf.write('set charlist {A D G J M O Q R T U V W X Y 0 1 2 3 4 5 6 7 8 9 ! @ # $ % ^ & + -}\n')
            vmdf.write('set char [lindex $charlist $num]\n')
            vmdf.write('$selection set type $char\n')
            vmdf.write('color Type $char $color}\n')

            for i,traj in enumerate(sorted(trajectories)):
                vindex = 0
                vmdf.write('mol new '+topology+'\n')
                for epoch in sorted(traj_files):
                    if isinstance(epochs, list):
                        if epoch in epochs:
                            vmdf.write('mol addfile '+traj_files[epoch][traj]+'\n')
                    else:
                        vmdf.write('mol addfile '+traj_files[epoch][traj]+'\n')
                vmdf.write('mol modselect '+str(vindex)+' '+str(i)+' "not chain L"\n')
                vmdf.write('mol modstyle '+str(vindex)+' '+str(i)+' newcartoon\n')
                vmdf.write('mol modcolor '+str(vindex)+' '+str(i)+' ColorID 3\n')
                vmdf.write('mol addrep '+str(i)+'\n')
                vindex += 1
                if peptide:
                    vmdf.write('mol modselect '+str(vindex)+' '+str(i)+' "chain L and not sidechain"\n')
                    vmdf.write('mol modstyle '+str(vindex)+' '+str(i)+' newcartoon\n')
                    vmdf.write('mol modcolor '+str(vindex)+' '+str(i)+' ColorID 11\n')
                    vmdf.write('mol addrep '+str(i)+'\n')
                    vindex += 1
                    vmdf.write('mol modselect '+str(vindex)+' '+str(i)+' "chain L"\n')
                    vmdf.write('mol modstyle '+str(vindex)+' '+str(i)+' Lines 0.1\n')
                    vmdf.write('mol modcolor '+str(vindex)+' '+str(i)+' Type\n')
                    vmdf.write('colorsel [atomselect top "chain L and carbon"] purple\n')
                else:
                    vmdf.write('mol modselect '+str(vindex)+' '+str(i)+' "chain L"\n')
                    vmdf.write('mol modstyle '+str(vindex)+' '+str(i)+' Licorice 0.2\n')
                    vmdf.write('mol modcolor '+str(vindex)+' '+str(i)+' Type\n')
                    vmdf.write('colorsel [atomselect top "chain L and carbon"] purple\n')

                vmdf.write('mol addrep '+str(i)+'\n')
                vindex += 1
                vmdf.write('mol modselect '+str(vindex)+' '+str(i)+' "not protein and not chain L and not ion"\n')
                vmdf.write('mol modstyle '+str(vindex)+' '+str(i)+' Lines\n')
                vmdf.write('mol modcolor '+str(vindex)+' '+str(i)+' Type\n')
                vmdf.write('colorsel [atomselect top "(not protein and not chain L and not ion) and carbon"] purple\n')
                vmdf.write('mol addrep '+str(i)+'\n')
                vindex += 1
                vmdf.write('mol modselect '+str(vindex)+' '+str(i)+' "ion"\n')
                vmdf.write('mol modstyle '+str(vindex)+' '+str(i)+' CPK\n')
                vmdf.write('mol modcolor '+str(vindex)+' '+str(i)+' Element\n')
                vmdf.write('mol addrep '+str(i)+'\n')
                vindex += 1
                vmdf.write('mol modselect '+str(vindex)+' '+str(i)+' "(all and same residue as within 5 of chain L) and not chain L"\n')
                vmdf.write('mol modstyle '+str(vindex)+' '+str(i)+' Lines 1.0\n')
                vmdf.write('mol modcolor '+str(vindex)+' '+str(i)+' Type\n')
                vmdf.write('colorsel [atomselect top "((all and same residue as within 5 of chain L) and not chain L) and carbon"] orange\n')
                if resnames != None:
                    vmdf.write('mol addrep '+str(i)+'\n')
                    vindex += 1
                    vmdf.write('mol modselect '+str(vindex)+' '+str(i)+' "resname '+' '.join(resnames)+'"\n')
                    vmdf.write('mol modstyle '+str(vindex)+' '+str(i)+' Licorice 0.2\n')
                    vmdf.write('mol modcolor '+str(vindex)+' '+str(i)+' Type\n')
                    vmdf.write('colorsel [atomselect top "(not chain L and not ion) and carbon"] orange\n')
                if resids != None:
                    vmdf.write('mol addrep '+str(i)+'\n')
                    vindex += 1
                    vmdf.write('mol modselect '+str(vindex)+' '+str(i)+' "resid '+' '.join([str(x) for x in resnames])+'"\n')
                    vmdf.write('mol modstyle '+str(vindex)+' '+str(i)+' Licorice 0.2\n')
                    vmdf.write('mol modcolor '+str(vindex)+' '+str(i)+' Type\n')
                    vmdf.write('colorsel [atomselect top "(not chain L and not ion) and carbon"] orange\n')

        return 'vmd -e .load_vmd.tcl'

    def combineDistancesIntoMetrics(self, catalytic_labels, labels=None, nonbonded_energy=False,
                                    overwrite=False, verbose=False):

        """
        Combine different equivalent distances into specific named metrics. The function
        takes as input a dictionary (catalytic_labels) composed of inner dictionaries as follows:

            catalytic_labels = {
                metric_name = {
                    protein = {
                        ligand = distances_list}}}

        The innermost distances_list object contains all equivalent distance names for
        a specific protein and ligand pair to be combined under the same metric_name column.

        The combination is done by taking the minimum value of all equivalent distances.

        Parameters
        ==========
        catalytic_labels : dict
            Dictionary defining which distances will be combined under a common name.
            (for details see above).
        """

        self.metric_type = {}
        metric_type_file = self.data_folder+'/metric_type.json'

        for name in catalytic_labels:
            if 'metric_'+name in self.data.keys() and not overwrite:
                print('Combined metric %s already added. Give overwrite=True to combine again the distances.' % name)
                if os.path.exists(metric_type_file):
                    self.metric_type = self._loadDictionaryFromJson(metric_type_file)
            else:
                distance_types = []
                values = []
                if labels != None:
                    label_values = []

                protein_ligand = []
                for i in [i[:2] for i in self.data.index]:
                    if i not in protein_ligand:
                        protein_ligand.append(i)

                for protein, ligand in protein_ligand:

                    if verbose:
                        print(f'Reading protein {protein} and ligand {ligand} distances')

                    ligand_data = self.getProteinAndLigandData(protein, ligand)

                    # Get best distance values
                    distances = catalytic_labels[name][protein][ligand]
                    distance_type = [x.split('_')[0] for x  in catalytic_labels[name][protein][ligand]][0]
                    distance_types += [distance_type]

                    if distance_type == 'distance':
                        distance_values = self.distances[protein][ligand][distances].min(axis=1)
                    elif distance_type == 'angle':
                        distance_values = self.angles[protein][ligand][distances].min(axis=1)

                    # Check that distances and ligand data matches
                    assert ligand_data.shape[0] == distance_values.to_numpy().shape[0]

                    values += distance_values.to_list()

                    if labels != None:
                        if name not in labels or protein not in labels[name] or ligand not in labels[name][protein]:
                            continue
                        if labels[name][protein][ligand] == {}:
                            continue

                        best_distances = self.distances[protein][ligand][distances].idxmin(axis=1).to_list()
                        label_values += [labels[name][protein][ligand][x] for x in best_distances]

                self.data['metric_'+name] = values
                if labels != None and label_values != []:
                    self.data['label_'+name] = label_values

                self._saveDataState()

                if len(set(distance_types)) > 1:
                    raise ValueError(f'Different distance types were combined under the metric {name}! Check your input!')
                elif len(set(distance_types)) == 0:
                    raise ValueError(f'No distance was found to be combined under the metric {name}! Check your input!')

                self.metric_type['metric_'+name] = list(set(distance_types))[0]

        self._saveDictionaryAsJson(self.metric_type, metric_type_file)

    def combineMetricsWithExclusions(self, combinations, exclusions, drop=True):
        """
        Function to combine metrics that are mutually exclusive. The function takes two inputs,
        combinations and exclusions.
        Combinations are the metrics that should be merged and have the following structure:

            combinations = {
                new_metric_name = (comb_metric_1,comb_metric_2),
                ...}

        Exclusions are the pairs of metrics that are mutually exclusive as a list of tuples.

        Parameters
        ==========
        combinations : dict
            Dictionary defining which distances will be combined under a common name.
        exclusions : list
            List of tuples of the incompatible metrics

        """

        # Get all metrics as index dictionary
        metrics_indexes = {}
        all_metrics = []
        i = 0
        for c in combinations:
            for m in combinations[c]:
                metrics_indexes[m] = i
                all_metrics.append('metric_'+m)
                i += 1

        # Get data as numpy array with only metric columns
        data = self.data[all_metrics]

        # Get labels of the shortest distance
        min_values = data.idxmin(axis=1)

        # Define columns to be excluded
        excluded_values = [] # Exclude for the same metric
        for i,m in enumerate(min_values):
            m = m.replace('metric_', '')
            for e in exclusions:
                if m in e:
                    x = list(set(e)-set([m]))[0]
                    excluded_values.append([i,metrics_indexes[x]])

            for c in combinations:
                if m in combinations[c]:
                    y = list(set(combinations[c])-set([m]))[0]
                    excluded_values.append([i,metrics_indexes[y]])

        # Set excluded values as np.inf
        data = data.to_numpy()
        for i,j in excluded_values:
            data[i,j] = np.inf

        # Add new metrics to data frame
        for c in combinations:
            c_indexes = [metrics_indexes[c] for c in combinations[c]]
            self.data['metric_'+c] = np.min(data[:,c_indexes], axis=1)

        if drop:
            self.data.drop(all_metrics, axis=1, inplace=True)

    def plotEnergyByResidue(self, initial_threshold=3.5):
        """
        Plot an energy by residue comparison between PELE runs. Two sets of selection
        are displayed to be compared in the same plot. The catalytic metrics are displayed
        to select the intervals of the two catalytic regions to be compared.

        Parameters
        ==========
        initial_threshold : float
            Starting upper value for the definition of the catalytic region range.
        """
        def getLigands(Protein1, Protein2, ebr_all=True, ebr_sgb=False, ebr_lj=False, ebr_ele=False):

            ps1 = self.data[self.data.index.get_level_values('Protein') == Protein1]
            ligands1 = list(set(ps1.index.get_level_values('Ligand').tolist()))

            ps2 = self.data[self.data.index.get_level_values('Protein') == Protein2]
            ligands2 = list(set(ps2.index.get_level_values('Ligand').tolist()))

            ligand1 = Dropdown(options=ligands1)
            ligand2 = Dropdown(options=ligands2)

            KT_slider1 = FloatSlider(
                            value=5.93,
                            min=0.593,
                            max=20.0,
                            step=0.1,
                            description='KT:',
                            readout=True,
                            readout_format='.1f')

            KT_slider2 = FloatSlider(
                            value=5.93,
                            min=0.593,
                            max=20.0,
                            step=0.1,
                            description='KT:',
                            readout=True,
                            readout_format='.1f')

            n_residue_slider = IntSlider(
                                value=10,
                                min=1,
                                max=50,
                                description='Number residues:',
                                readout=True,
                                readout_format='.1f')

            ebr_type = None
            if ebr_all:
                ebr_type = 'all'
            elif ebr_sgb:
                ebr_type = 'sgb'
            elif ebr_lj:
                ebr_type = 'lj'
            elif ebr_ele:
                ebr_type = 'ele'

            parameters = {'Protein1' : fixed(Protein1),
                          'Protein2' : fixed(Protein2),
                          'Ligand1' : ligand1,
                          'Ligand2' : ligand2,
                          'KT_slider1' : KT_slider1,
                          'KT_slider2' : KT_slider2,
                          'n_residue_slider' : n_residue_slider,
                          'ebr_type' : fixed(ebr_type)}

            widget_metrics1 = []
            for metric in metrics:
                widget_metric = FloatRangeSlider(
                                value=[0, initial_threshold],
                                min=0,
                                max=20,
                                step=0.05,
                                description=metric+':',
                                readout_format='.2f')
                widget_metrics1.append(widget_metric)
                parameters[metric+'_1'] = widget_metric

            widget_metrics2 = []
            for metric in metrics:
                widget_metric = FloatRangeSlider(
                                value=[0, initial_threshold],
                                min=0,
                                max=20,
                                step=0.05,
                                description=metric+':',
                                readout_format='.2f')
                widget_metrics2.append(widget_metric)
                parameters[metric+'_2'] = widget_metric

            plot = interactive_output(_plot, parameters)

            mVB1 = VBox(widget_metrics1)
            mVB2 = VBox(widget_metrics2)

            VB1 = VBox([ligand1, KT_slider1, mVB1])
            VB2 = VBox([ligand2, KT_slider2, mVB2])
            HB = HBox([VB1, VB2], width='100%')
            VB = VBox([HB, n_residue_slider, plot])
            display(VB)

        def _getPlotData(Protein, Ligand, KT=0.593, ebr_type='all', **metrics):

            if ebr_type == None:
                ebr_type = 'all'

            ebk = [x for x in self.data.keys() if x.startswith('L:1') and x.endswith(ebr_type)]
            series = self.getProteinAndLigandData(Protein, Ligand)
            total_energy = series['Total Energy']
            energy_minimum = total_energy.min()
            relative_energy = total_energy-energy_minimum
            Z = np.sum(np.exp(-relative_energy/KT))

            filtered_series = series
            for metric in metrics:
                filtered_series = filtered_series[metrics[metric][0] <= filtered_series[metric]]
                filtered_series = filtered_series[metrics[metric][1] >= filtered_series[metric]]

            relative_energy = filtered_series['Total Energy'].to_numpy()-energy_minimum
            probability = np.exp(-relative_energy/KT)/Z
            probability = np.reshape(probability, (-1, 1))
            ebr = np.sum(np.multiply(filtered_series[ebk], probability), axis=0)

            labels = []
            for x in ebk:
                x = x.split(':')[-1]
                try:
                    resname = PDB.Polypeptide.three_to_one(x.split('_')[1])
                except:
                    resname = x.split('_')[1]
                resid = x.split('_')[0]
                labels.append(resname+resid)

            argsort = np.abs(ebr).argsort()[::-1]
            ebr = dict(zip(np.array(labels)[argsort], ebr[argsort].to_numpy()))
            return ebr

        def _plot(Protein1, Protein2, Ligand1, Ligand2, KT_slider1, KT_slider2, n_residue_slider,
                  ebr_type='all', **metrics):

            metrics1 = {}
            metrics2 = {}
            for metric in metrics:
                if metric.endswith('_1'):
                    metrics1[metric[:-2]] = metrics[metric]
                elif metric.endswith('_2'):
                    metrics2[metric[:-2]] = metrics[metric]

            # Get all energy-by-residue metrics
            ebr1 = _getPlotData(Protein1, Ligand1, KT=KT_slider1, ebr_type=ebr_type, **metrics1)
            ebr2 = _getPlotData(Protein2, Ligand2, KT=KT_slider2, ebr_type=ebr_type, **metrics2)

            # Get all residues to plot and sort them by ebr
            residues = []
            for r in ebr1:
                residues.append((r+'_1', ebr1[r]))
                if len(residues) == n_residue_slider:
                    break
            for r in ebr2:
                residues.append((r+'_2', ebr2[r]))
                if len(residues) == n_residue_slider*2:
                    break

            done = []
            values1 = []
            values2 = []
            count = 0
            pos1 = []
            pos2 = []
            labels = []
            used = []
            for r in sorted(residues, key=lambda x:abs(x[1]), reverse=True):
                if r[0][:-2] in used:
                    continue
                g = int(r[0][-1:])
                if g == 1:
                    values1.append(r[1])
                    values2.append(ebr2[r[0][:-2]])
                if g == 2:
                    values1.append(ebr1[r[0][:-2]])
                    values2.append(r[1])
                pos1.append(count-0.2)
                pos2.append(count+0.2)
                labels.append(r[0][:-2])
                count += 1
                used.append(r[0][:-2])
                if count == n_residue_slider:
                    break

            if n_residue_slider >= 30:
                fontsize = 8
            else:
                fontsize = 10

            if ebr_type == 'all':
                energy_label = 'all'
            elif ebr_type == 'sgb':
                energy_label = 'SGB'
            elif ebr_type == 'lj':
                energy_label = 'lennard_jones'
            elif ebr_type == 'ele':
                energy_label = 'electrostatic'

            plt.figure(dpi=120)
            hist = plt.bar(pos1, values1, 0.4)
            hist = plt.bar(pos2, values2, 0.4)
            xt = plt.xticks(range(len(labels)), labels, rotation=90, fontsize=fontsize)
            plt.xlabel('Residue')
            plt.ylabel(energy_label+' Energy contribution [kcal/mol]')
            display(plt.show())

        metrics = [k for k in self.data.keys() if 'metric_' in k]
        metrics = {m:initial_threshold for m in metrics}

        protein1 = Dropdown(options=self.proteins)
        protein2 = Dropdown(options=self.proteins)

        ebr_all = Checkbox(value=True,
                           description='All')
        ebr_sgb = Checkbox(value=False,
                           description='SGB')
        ebr_lj = Checkbox(value=False,
                           description='Lennard-Jones')
        ebr_ele = Checkbox(value=False,
                           description='Electrostatics')

        plot_data = interactive_output(getLigands, {'Protein1': protein1, 'Protein2' :protein2,
                                                    'ebr_all' : ebr_all, 'ebr_sgb': ebr_sgb,
                                                    'ebr_lj' : ebr_lj, 'ebr_ele': ebr_ele})

        H0 = HBox([ebr_all, ebr_sgb, ebr_lj, ebr_ele])
        VB1 = VBox([protein1])
        VB2 = VBox([protein2])
        HB = HBox([VB1, VB2], width='100%')
        VB0 = VBox([H0, HB])
        VB = VBox([VB0, plot_data], width='100%')

        d = display(VB)

    def getProteinData(self, protein, equilibration=False):
        """
        Get PELE data for a protein and ligand combination.

        Paramters
        =========
        protein : str
            Protein name
        equilibration : bool
            Get equilibration data instead?

        Returns
        =======
        protein_series : pandas.DataFrame
            The pandas for the protein only data.
        """
        if equilibration:
            data = self.equilibration_data
        else:
            data = self.data
        protein_series = data[data.index.get_level_values('Protein') == protein]
        return protein_series

    def getLigandData(self, ligand, equilibration=False):
        """
        Get PELE data for a protein and ligand combination.

        Paramters
        =========
        ligand : str
            Ligand Name
        equilibration : bool
            Get equilibration data instead?

        Returns
        =======
        ligand_series : pandas.DataFrame
            The pandas for the ligand data.
        """
        if equilibration:
            data = self.equilibration_data
        else:
            data = self.data
        ligand_series = data[data.index.get_level_values('Ligand') == ligand]
        return ligand_series

    def getProteinAndLigandData(self, protein, ligand, equilibration=False):
        """
        Get PELE data for a protein and ligand combination.

        Paramters
        =========
        protein : str
            Protein name
        ligand : str
            Ligand Name
        equilibration : bool
            Get equilibration data instead?

        Returns
        =======
        ligand_series : pandas.DataFrame
            The pandas for the protein and ligand data.
        """
        if equilibration:
            data = self.equilibration_data
        else:
            data = self.data
        protein_series = data[data.index.get_level_values('Protein') == protein]
        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
        return ligand_series

    def readClusterDataFromGlobal(self):

        """
        Read cluster data from the data file generated by site-finder PELE.

        Parameters
        ==========
        """

        cluster_data = {}
        for protein,ligand in self.pele_combinations:
            df = pd.read_csv(self.pele_directories[protein][ligand]+'/output/data.csv')
            for line in df.iterrows():
                traj = line[1]['trajectory'].split('/')[-1].split('.')[0].split('_')[1]
                if line[1]['Cluster'] == '-':
                    cl = '-1'
                else:
                    cl =  line[1]['Cluster']
                cluster_data[(protein,ligand,line[1]['epoch'],traj,line[1]['numberOfAcceptedPeleSteps'])] = int(cl)

        self.data['Cluster'] = cluster_data.values()

    ### Extract poses methods

    def getBestPELEPoses(self, filter_values=None, proteins=None, ligands=None, column='Binding Energy',
                         n_models=1, return_failed=False, cluster_aware=True, label_aware=True, lower_limit=None):
        """
        Get best models based on the best column score and a set of metrics with specified thresholds.
        The filter thresholds must be provided with a dictionary using the metric names as keys
        and the thresholds as the values.

        Parameters
        ==========
        n_models : int
            The number of poses to select for each protein + ligand pele simulation.
        filter_values : dict
            Thresholds for the filter.
        return_failed : bool
            Whether to return a list of the pele without any poses fulfilling
            the selection criteria. It is returned as a tuple (index 0) alongside
            the filtered data frame (index 1).
        cluster_aware : bool
            Check if cluster column is inside the dataframe and extract best models also by cluster.
        """

        bp = []
        failed = []
        for protein in self.proteins:

            # If a list of proteins is given skip proteins not in the list
            if proteins != None:
                if protein not in proteins:
                    continue

            protein_series = self.data[self.data.index.get_level_values('Protein') == protein]
            for ligand in self.ligands:

                # If a list of ligands is given skip ligands not in the list
                if ligands != None:
                    if ligand not in ligands:
                        continue

                ligand_data = protein_series[protein_series.index.get_level_values('Ligand') == ligand]

                if filter_values != None:
                    for metric in filter_values:
                        if metric in ['RMSD', 'Ligand SASA', 'Total Energy', 'Binding Energy', 'Ligand RMSD']:
                            metric_name = metric
                        else:
                            metric_name = 'metric_'+metric
                        if lower_limit:
                            ligand_data = ligand_data[ligand_data[metric_name] > lower_limit]
                        ligand_data = ligand_data[ligand_data[metric_name] < filter_values[metric]]

                if ligand_data.empty:
                    failed.append((protein, ligand))
                    continue

                if 'Cluster' in self.data.keys() and cluster_aware:

                    clusters = [x for x in ligand_data['Cluster'] if x != '-']
                    clusters = list(set(clusters))

                    for c in clusters:

                        cluster_data = ligand_data[ligand_data['Cluster'] == c]
                        if cluster_data.shape[0] < n_models:
                            print('WARNING: less than %s models available for pele %s + %s simulation for cluster %s' % (n_models, protein, ligand, c))
                        for i in cluster_data[column].nsmallest(n_models).index:
                            bp.append(i)

                elif len([x for x in self.data.keys() if 'label_' in x]) > 0 and label_aware:
                    labels = [x for x in self.data.keys() if 'label_' in x]

                    for label in labels:
                        label_values = sorted(list(set(ligand_data[label])))
                        for value in label_values:
                            label_value_data = ligand_data[ligand_data[label] == value]

                            if label_value_data.shape[0] < n_models:
                                print('WARNING: less than %s models available for pele %s + %s simulation for label %s and value %s' % (n_models, protein, ligand, c))
                            for i in label_value_data[column].nsmallest(n_models).index:
                                bp.append(i)

                else:
                    if ligand_data.shape[0] < n_models:
                        print('WARNING: less than %s models available for pele %s + %s simulation' % (n_models, protein, ligand))
                    for i in ligand_data[column].nsmallest(n_models).index:
                        bp.append(i)

        if return_failed:

            return failed, self.data[self.docking_data.index.isin(bp)]

        return self.data[self.data.index.isin(bp)]

    def getBestPELEPosesIteratively(self, metrics, column='Binding Energy', ligands=None, proteins=None,
                                    min_threshold=3.5, max_threshold=5.0, step_size=0.1, label_aware=True,
                                    lower_limit=None, verbose=False):
        """
        Extract best poses iteratively using all given metrics simoultaneously.
        """
        extracted = []
        selected_indexes = []

        if len([x for x in self.data.keys() if 'label_' in x]) > 0 and label_aware:
            labels = [x for x in self.data.keys() if 'label_' in x]
        else:
            labels = []

        for t in np.arange(min_threshold, max_threshold+(step_size/10), step_size):
            if verbose:
                print(f'Getting best poses at threshold {t}', end='\r')
            filter_values = {m:t for m in metrics}
            best_poses = self.getBestPELEPoses(filter_values, column=column, n_models=1,
                                               proteins=proteins, ligands=ligands,
                                               label_aware=label_aware, lower_limit=lower_limit)
            mask = []
            if not isinstance(ligands, type(None)):
                for level in best_poses.index.get_level_values('Ligand'):
                    if level in ligands:
                        mask.append(True)
                    else:
                        mask.append(False)
                pele_data = best_poses[mask]
            else:
                pele_data = best_poses

            # Check that models were not written in a previous iteration (one by protein, ligand, and label)
            if label_aware and labels != []:
                for i,v in pele_data.iterrows():
                    for l in labels:
                        i_label = (*i[:2], v[l])
                        if i_label not in extracted:
                            selected_indexes.append(i)
                            extracted.append(i_label)

            # Check that models were not written in a previous iteration (one by protein and ligand)
            else:
                for row in pele_data.index:
                    if row[:2] not in extracted:
                        selected_indexes.append(row)
                        extracted.append(row[:2])

        final_mask = []
        for row in self.data.index:
            if row in selected_indexes:
                final_mask.append(True)
            else:
                final_mask.append(False)
        pele_data = self.data[final_mask]

        return pele_data

    def extractPELEPoses(self, pele_data, output_folder, separator=None, keep_chain_names=True,
                         label_aware=True, remote_pele_path=None, skip_missing=False, skip_connects=False):
        """
        Extract pele poses present in a pele dataframe. The PELE DataFrame
        contains the same structure as the self.data dataframe, attribute of
        this class.

        Parameters
        ==========
        pele_data : pandas.DataFrame
            Datframe containing the entries for the poses to be extracted
        output_folder : str
            Path to the folder where the pele poses structures will be saved.
        separator : str
            Symbol used to separate protein, ligand, epoch, trajectory and pele step for each pose filename.
        remote_pele_path : str
            Path if the trajectories are stored in a remote server. It should have the following structure:
                server_name:path/to/my/pele/folder
            Note that this will only work if you have a ssh key setup. If you do not check the group guidelines.
        skip_missing : bool
            Skip if trajectory files are missing.
        """

        if separator == None:
            separator = self.separator

        if len([x for x in self.data.keys() if 'label_' in x]) > 0 and label_aware:
            labels = [x for x in self.data.keys() if 'label_' in x]
        else:
            labels = []

        # Create output folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Crea Bio PDB parser and io
        parser = PDB.PDBParser()
        io = PDB.PDBIO()

        # Extract pele poses with mdtraj
        for protein in self.proteins:

            # Check the separator is not in protein names
            if separator in protein:
                raise ValueError('The separator %s was found in protein name %s. Please use a different separator symbol.' % (separator, protein))

            protein_data = pele_data[pele_data.index.get_level_values('Protein') == protein]

            for ligand in self.ligands:

                # Check the separator is not in ligand name
                if separator in ligand:
                    raise ValueError('The separator %s was found in ligand name %s. Please use a different separator symbol.' % (separator, ligand))

                ligand_data = protein_data[protein_data.index.get_level_values('Ligand') == ligand]

                if not ligand_data.empty:

                    if not os.path.exists(output_folder+'/'+protein):
                        os.mkdir(output_folder+'/'+protein)

                    # Check whether protein and ligand trajectories are available and if remote path was given.
                    if (protein not in self.trajectory_files or ligand not in self.trajectory_files[protein]) and remote_pele_path == None:
                        if skip_missing:
                            print('Trajectory files not found for protein %s and ligand %s.' % (protein, ligand))
                            continue
                        else:
                            raise ValueError('Trajectory files not found for protein %s and ligand %s.' % (protein, ligand))

                    elif (protein not in self.trajectory_files or ligand not in self.trajectory_files[protein]) and remote_pele_path != None:
                        traj = pele_trajectory.loadTrajectoryFrames(ligand_data,
                                                                    self.pele_folder+'/'+protein+separator+ligand+'/output/output/',
                                                                    self.topology_files[protein][ligand],
                                                                    remote_pele_path = remote_pele_path+'/'+protein+separator+ligand+'/output/output/')
                    else:
                        traj = pele_trajectory.loadTrajectoryFrames(ligand_data,
                                                                    self.trajectory_files[protein][ligand],
                                                                    self.topology_files[protein][ligand])

                    # Create atom names to traj indexes dictionary
                    atom_traj_index = {}
                    for chain in traj.topology.chains:
                        chain_index = chain.index
                        chain_name = self.chain_ids[protein][ligand][chain_index]
                        if chain_name not in atom_traj_index:
                            atom_traj_index[chain_name] = {}
                        for residue in chain.residues:
                            residue_label = residue.resSeq
                            atom_traj_index[chain_name][residue_label] = {}
                            for atom in residue.atoms:
                                if 'HOH' in residue.name and atom.name == 'O':
                                    atom_name = 'OW'
                                else:
                                    atom_name = atom.name

                                atom_traj_index[chain_name][residue_label][atom_name] = atom.index

                    # Create a topology file with Bio.PDB
                    pdb_topology = parser.get_structure(protein, self.topology_files[protein][ligand])
                    atoms = [a for a in pdb_topology.get_atoms()]

                    # Pass mdtraj coordinates to Bio.PDB structure to preserve correct chain names
                    for i, (entry, values) in enumerate(ligand_data.iterrows()):

                        if label_aware and labels != []:
                            label_string = '-'.join([values[l] for l in labels])
                            entry = (*entry[:2], label_string, * entry[2:])

                        filename = separator.join([str(x) for x in entry])+'.pdb'
                        xyz = traj[i].xyz[0]
                        for j in range(traj.n_atoms):

                            # Get chain and residue labels
                            chain = atoms[j].get_parent().get_parent()
                            residue = atoms[j].get_parent()
                            #chain_index = mdt_index[chain.id]

                            if residue.resname in ['HID', 'HIE', 'HIP']:
                                resname = 'HIS'
                            else:
                                resname = residue.resname
                            residue_label = residue.id[1]

                            # Give atom coordinates to Bio.PDB object
                            traj_index = atom_traj_index[chain.id][residue_label][atoms[j].name]
                            atoms[j].coord = xyz[traj_index]*10

                        # Save structure
                        io.set_structure(pdb_topology)
                        io.save(output_folder+'/'+protein+'/'+filename)

                        if protein not in self.conects:
                            continue
                        elif ligand not in self.conects[protein]:
                            continue

                        if self.conects[protein][ligand] != [] and not skip_connects:
                            conectLines._writeConectLines(output_folder+'/'+protein+'/'+filename,
                                                          self.conects[protein][ligand])

    ### Clustering methods

    def getLigandTrajectoryPerTrajectory(self, protein, ligand,  overwrite=False,
                                         return_dictionary=False, return_paths=False):
        """
        Generate a full-ligand (only ligand coordinates) trajectories per every PELE trajectory aligned to
        the protein framework for all epochs. Useful for clustering purposes. A dictionary mapping the trajectory
        indexes to the pele data DataFrame will also be stored.

        Parameters
        ==========
        protein : str
            Name of the protein system
        ligand : str
            Name of the ligand
        overwrite : bool
            Recalculate the ligand trajectory from the PELE output?
        return_dictionary : bool
            Return the trajectory indexes to PELE data mapping dicionary? Will return
            a tuple: (ligand_traj, traj_dict).
        return_paths : bool
            Return a list containing the path to the ligand trajectory files.

        Returns
        =======
        Default: ligand_traj : dict
            Dictionary by trajectory index with the mdtraj.Trajectory objects as values.
        """

        if not os.path.isdir(self.pele_folder):
            raise ValueError('Pele folder not found. Cannot acces trajectories.')


        ligand_traj_dir = self.data_folder+'/ligand_traj'
        if not os.path.exists(ligand_traj_dir):
            os.mkdir(ligand_traj_dir)

        ligand_path = ligand_traj_dir+'/'+protein+self.separator+ligand
        ligand_top_path = ligand_path+'.pdb'
        ligand_traj_dict_path = ligand_path+'.json'

        # Get trajectory and topology files
        trajectory_files = self.trajectory_files[protein][ligand]
        topology_file = self.topology_files[protein][ligand]

        # Get trajectory and topology files
        ligand_traj = None
        traj_dict = {}
        ligand_traj_paths = []

        # Define reference frame using the topology
        reference = md.load(topology_file)

        # Append to ligand-only trajectory
        ligand_atoms = reference.topology.select('resname '+self.ligand_names[protein][ligand])

        if not os.path.exists(ligand_top_path):
            lig_ref = reference.atom_slice(ligand_atoms)
            # Store topology as PDB
            for residue in lig_ref.topology.residues:
                residue.resSeq = 1
            lig_ref.save(ligand_top_path)

        # Read or save ligand trajectories by PELE trajectory
        ligand_traj = {}

        for t in sorted(trajectory_files[0]):
            i = 0
            ligand_traj[t] = None
            traj_dict[t] = {}
            ligand_traj_path = ligand_path+'_'+str(t)+'.dcd'

            if not os.path.exists(ligand_traj_path) or overwrite:
                for epoch in sorted(trajectory_files):
                    # Load trajectory
                    traj = md.load(trajectory_files[epoch][t], top=topology_file)

                    # Align trajectory to protein atoms only
                    protein_atoms = traj.topology.select('protein')
                    traj.superpose(reference, atom_indices=protein_atoms)

                    if isinstance(ligand_traj[t], type(None)):
                        ligand_traj[t] = traj.atom_slice(ligand_atoms)
                    else:
                        ligand_traj[t] = md.join([ligand_traj[t], traj.atom_slice(ligand_atoms)])

                    # Store PELE data into mapping dictionary
                    for x in range(i, ligand_traj[t].n_frames):
                        traj_dict[t][x] = (epoch, t, x-i)
                    i = ligand_traj[t].n_frames

                ligand_traj[t].save(ligand_traj_path)
                ligand_traj_paths.append(ligand_traj_path)
            else:
                ligand_traj[t] = md.load(ligand_traj_path, top=ligand_top_path)
                ligand_traj_paths.append(ligand_traj_path)

        if os.path.exists(ligand_traj_dict_path) and not overwrite:
            traj_dict = self._loadDictionaryFromJson(ligand_traj_dict_path)
        else:
            self._saveDictionaryAsJson(traj_dict, ligand_traj_dict_path)

        if return_dictionary:
            if return_paths:
                return ligand_traj_paths, ligand_top_path, traj_dict
            else:
                return ligand_traj, traj_dict
        elif return_paths:
                return ligand_traj_paths, ligand_top_path
        else:
            return ligand_traj

    def getLigandTrajectoryAsOneBundle(self, protein, ligand, overwrite=False, return_dictionary=False):
        """
        Generate a single trajectory containing only the ligand coordinates aligned to
        the protein framework for all epochs and trajectories. Useful for clustering purposes.
        A dictionary mapping the trajectory indexes to the pele data DataFrame will also be stored.

        Parameters
        ==========
        protein : str
            Name of the protein system
        ligand : str
            Name of the ligand
        overwrite : bool
            Recalculate the ligand trajectory from the PELE output?
        return_dictionary : bool
            Return the trajectory indexes to PELE data mapping dicionary? Will return
            a tuple: (ligand_traj, traj_dict).
        """

        if not os.path.isdir(self.pele_folder):
            raise ValueError('Pele folder not found. Cannot acces trajectories.')

        ligand_traj_dir = self.data_folder+'/ligand_traj'
        if not os.path.exists(ligand_traj_dir):
            os.mkdir(ligand_traj_dir)

        ligand_path = ligand_traj_dir+'/'+protein+self.separator+ligand
        ligand_top_path = ligand_path+'.pdb'
        ligand_traj_path = ligand_path+'.dcd'
        ligand_traj_dict_path = ligand_path+'.json'

        # Create ligand trajectory if it does not exists
        if not os.path.exists(ligand_traj_path) or overwrite:

            # Get trajectory and topology files
            trajectory_files = self.trajectory_files[protein][ligand]
            topology_file = self.topology_files[protein][ligand]

            # Get trajectory and topology files
            ligand_traj = None
            traj_dict = {}

            # Define reference frame using the topology
            reference = md.load(topology_file)

            # Append to ligand-only trajectory
            ligand_atoms = reference.topology.select('resname '+self.ligand_names[protein][ligand])
            i = 0
            sum = 0
            for epoch in sorted(trajectory_files):
                for t in sorted(trajectory_files[epoch]):
                    # Load trajectory
                    traj = md.load(trajectory_files[epoch][t], top=topology_file)
                    sum += traj.n_frames
                    # Align trajectory to protein atoms only
                    protein_atoms = traj.topology.select('protein')
                    traj.superpose(reference, atom_indices=protein_atoms)

                    if isinstance(ligand_traj, type(None)):
                        ligand_traj = traj.atom_slice(ligand_atoms)
                    else:
                        ligand_traj = md.join([ligand_traj, traj.atom_slice(ligand_atoms)])

                    # Store PELE data into mapping dictionary
                    for x in range(i, ligand_traj.n_frames):
                        traj_dict[x] = (epoch, t, x-i)
                    i = ligand_traj.n_frames

            reference.atom_slice(ligand_atoms).save(ligand_top_path)
            ligand_traj.save(ligand_traj_path)
            self._saveDictionaryAsJson(traj_dict, ligand_traj_dict_path)

        # Read ligand trajectory if found
        else:
            if not os.path.exists(ligand_traj_dict_path):
                print('Warining: Dictionary mapping ligand trajectory indexes to pele data is missing!')

            print('Ligand trajectory for %s and %s found. Reading it from file.' % (protein, ligand))
            ligand_traj = md.load(ligand_traj_path, top=ligand_top_path)
            traj_dict = self._loadDictionaryFromJson(ligand_traj_dict_path)
            traj_dict = {int(k):tuple(v) for k,v in traj_dict.items()} # Convert dict keys to int

        if return_dictionary:
            return ligand_traj, traj_dict
        else:
            return ligand_traj

    def computeLigandRMSDClusters(self, rmsd_threshold=5, overwrite=False):
        """
        Cluster ligand conformations by RMSD from a ligand-only trajectory aligned
        to the protein framework. Clustering belonging is added to the pele data
        DataFrame (self.data) as a column 'rmsd_cluster_i', where i is the
        rmsd_threshold employed.

        Parameters
        ==========
        rmsd_threshold : float
            RMSD threshold empoyed to separate clusters.
        overwrite : bool
            Whether to recalculate the ligand trajectory (see getLigandTrajectory()).
        """

        # Iterate by protein
        rmsd_column = []
        cluster_data = []
        for protein in sorted(self.trajectory_files):

            protein_series = self.getDataSeries(self.data, protein, 'Protein')

            # Iterate by ligand
            for ligand in sorted(self.trajectory_files[protein]):

                ligand_series = self.getDataSeries(protein_series, ligand, 'Ligand')

                # Create or read ligand trajectory
                ligand_traj, traj2pose = self.getLigandTrajectoryAsOneBundle(protein, ligand,
                                                 overwrite=overwrite, return_dictionary=True)
                # pose2traj = {tuple(v):k for k,v in traj2pose.items()}

                be = ligand_series['Binding Energy'].to_list()

                ite_traj = copy.copy(ligand_traj)

                cluster_num = 0
                clusters = {}
                while not len(ite_traj) == 0:
                    cluster_num += 1

                    # Get best energy pose
                    best_index = np.argmin(be)
                    be_traj = ite_traj[best_index]

                    # Calculate RMSD
                    rmsd = np.array((np.sqrt(3*np.mean((ite_traj.xyz - be_traj.xyz)**2, axis=(1,2)))))
                    if ligand_traj.n_frames == rmsd.shape[0]:
                        rmsd_column += list(rmsd*10.0)

                    assert rmsd[best_index] == 0.0

                    # Assign frames to cluster
                    cluster = np.where(rmsd<rmsd_threshold/10)[0]

                    ## Map cluster data to pose indexes
                    for index in cluster:
                        if traj2pose[index] in clusters:
                            raise ValueError('!')
                        clusters[traj2pose[index]] = cluster_num

                    # Update traj2pose dict to map new traj indexes
                    traj2posetmp = {}
                    count = 0
                    for i in range(ite_traj.n_frames):
                        if i not in cluster:
                            traj2posetmp[count] = traj2pose[i]
                            count += 1
                    traj2pose = traj2posetmp

                    # Slice be list and trajectory
                    be = [be[i] for i in range(ite_traj.n_frames) if i not in cluster]
                    mask_cluster = np.array([(i in cluster) for i in range(len(ite_traj))])
                    ite_traj = ite_traj[~mask_cluster]

                for index in ligand_series.index:
                    cluster_data.append(clusters[index[-3:]])

        self.data['Ligand RMSD'] = rmsd_column
        self.data['Ligand Clusters'] = cluster_data

    def setLigandMSMClusters(self):
        """
        Create a class for analysing PELE simulations under the MSM framework.
        """
        from ._msm_clustering import ligand_msm
        self.ligand_msm = ligand_msm(self)
        return self.ligand_msm

    def setUpPELERerun(self, pele_folder, protein_ligands, restart=False,
                       continuation=False):
        """
        Generate PELE simulations from original input yamls and PDBs.

        Parameters
        ==========
        pele_folder : str
            Path to the folder where PELE calcualtions will be located.
        protein_ligands : list
            List of (protein, ligand) tuples for which to set up PELE rerun.
        restart : bool
            Should the simulation run in two steps (two input yamls must be present).
        continuation : bool
            Do you which to continue a simulation?
        """

        # Check input
        if not isinstance(protein_ligands, list):
            raise ValueError('protein_ligands must be a list of 2-elements tuples: (protein, ligand)')
        else:
            for tup in protein_ligands:
                if not isinstance(tup, tuple) and len(tup) != 2:
                    raise ValueError('protein_ligands must be a list of 2-elements tuples')

        # Create PELE job folder
        if not os.path.exists(pele_folder):
            os.mkdir(pele_folder)

        jobs = []
        for protein, ligand in protein_ligands:
            pl_name = protein+self.separator+ligand

            # Create input folder
            if not os.path.exists(pele_folder+'/'+pl_name):
                os.mkdir(pele_folder+'/'+pl_name)

            # Copy files in input folder
            input_yaml = None
            restart_yaml = None
            for f in os.listdir(self.data_folder+'/pele_inputs/'+pl_name):
                if f.endswith('.yaml') and 'restart' not in f:
                    input_yaml = f
                elif f.endswith('.yaml') and 'restart' in f:
                    restart_yaml = f
                shutil.copyfile(self.data_folder+'/pele_inputs/'+pl_name+'/'+f,
                                pele_folder+'/'+pl_name+'/'+f)

            # Check debug flag at the input files
            if restart:
                input_yaml_path = self.data_folder+'/pele_inputs/'+pl_name+'/'+input_yaml
                debug = False
                with open(input_yaml_path) as yf:
                    lines = yf.readlines()
                    for l in lines:
                        if l.startswith('debug:') and l.endswith('true\n'):
                            debug = True

                # If debug not found add it
                if not debug:
                    with open(input_yaml_path, 'w') as yf:
                        for l in lines:
                            yf.write(l)
                        yf.write('debug: true\n')

                # Check for restart flags at restart yaml
                if restart_yaml != None:
                    input_yaml_path = self.data_folder+'/pele_inputs/'+pl_name+'/'+restart_yaml
                    restart_flag = False
                    with open(input_yaml_path) as yf:
                        lines = yf.readlines()
                        for l in lines:
                            if l.startswith('restart:') and l.endswith('true\n'):
                                restart_flag = True

                    # If restart not found add it
                    if not restart_flag:
                        with open(input_yaml_path, 'w') as yf:
                            for l in lines:
                                yf.write(l)
                            yf.write('restart: true\n')

            if continuation:
                if restart_yaml == None:
                    input_yaml_path = self.data_folder+'/pele_inputs/'+pl_name+'/'+input_yaml
                    debug = False
                    restart_flag = False
                    continuation_flag = False
                    with open(input_yaml_path) as yf:
                        lines = yf.readlines()
                        for l in lines:
                            if l.startswith('debug:') and l.endswith('true\n'):
                                debug = True
                            elif l.startswith('restart:') and l.endswith('true\n'):
                                restart_flag = True
                            elif l.startswith('adaptive_restart:') and l.endswith('true\n'):
                                continuation_flag = True

                    # If restart, debug or continuation not found add it
                    if not restart_flag or not debug or not continuation_flag:
                        with open('input_restart.yaml', 'w') as yf:
                            for l in lines:
                                yf.write(l)
                            if not debug:
                                yf.write('debug: true\n')
                            elif not restart_flag:
                                yf.write('restart: true\n')
                            elif not restart_flag:
                                yf.write('adaptive_restart: true\n')

            # Create command for PELE execution
            command = 'cd '+pele_folder+'/'+pl_name+'\n'
            if restart and restart_yaml != None:
                command += 'python -m pele_platform.main '+input_yaml+'\n'
                command += 'python -m pele_platform.main '+restart_yaml+'\n'
            elif continuation and restart_yaml == None:
                command += 'python -m pele_platform.main input_restart.yaml\n'
            else:
                command += 'python -m pele_platform.main '+input_yaml+'\n'
            command += 'cd ../..'
            jobs.append(command)

        return jobs

    def getTopologyStructures(self):
        """
        Iterate over the topology files loaded as structures.
        """
        for protein, ligand in self.pele_combinations:
            structure = self._readPDB(protein+'-'+ligand, self.topology_files[protein][ligand])
            yield (protein, ligand), structure

    def getFolderStructures(self, poses_folder, return_paths=False, only_proteins=None,
                            only_ligands=None):
        """
        Iterate over the PDB files in a folder as Biopython structures. The folder
        must be written in the format of the extractPELEPoses() function.

        Parameters
        ==========
        poses_folder : str
            Path to PELE poses extracted with extractPELEPoses() function.
        """

        if only_proteins == None:
            only_proteins = []
        elif isinstance(only_proteins, str):
            only_proteins = [only_proteins]

        if only_ligands == None:
            only_ligands = []
        elif isinstance(only_ligands, str):
            only_ligands = [only_ligands]

        for protein in os.listdir(poses_folder):

            if only_proteins != [] and protein not in only_proteins:
                continue

            for f in os.listdir(poses_folder+'/'+protein):
                fs = f.replace('.pdb','').split(self.separator)
                if fs[0] == protein:
                    ligand, epoch, trajectory, pele_step = fs[1:5]

                    if only_ligands != [] and ligand not in only_ligands:
                        continue

                    if return_paths:
                        yield (protein, ligand, epoch, trajectory, pele_step), poses_folder+'/'+protein+'/'+f
                    else:
                        structure = self._readPDB(protein+self.separator+ligand, poses_folder+'/'+protein+'/'+f)
                        yield (protein, ligand, epoch, trajectory, pele_step), structure

    def alignCommonPELEPoses(self, pele_poses_folder):
        """
        Align poses belonging to the same protein and ligand combination.
        The poses are located in a folder coming from the extractPELEPoses()
        function.

        Parameters
        ==========
        pele_poses_folder : str
            Path to a folder were poses were extracted with the function extractPELEPoses()
        """

        def alignStructures(reference, target):
            """
            Align to structures based on their C-Alpha atoms.
            """

            reference_model = reference[0]
            target_model = target[0]

            reference_residues = [r for r in reference_model.get_residues() if r.id[0] == ' ']
            target_residues = [r for r in target_model.get_residues() if r.id[0] == ' ']

            assert len(reference_residues) == len(target_residues)

            reference_atoms = []
            target_atoms = []
            for r1, r2 in zip(reference_residues, target_residues):
                reference_atoms.append(r1['CA'])
                target_atoms.append(r2['CA'])

            # Super impose
            super_imposer = PDB.Superimposer()
            super_imposer.set_atoms(reference_atoms, target_atoms)
            super_imposer.apply(target_model.get_atoms())

        # Defin Biopython objects
        io = PDB.PDBIO()

        # Get PDB files paths
        pdb_paths = {}
        for index, pdb_path in self.getFolderStructures(pele_poses_folder, return_paths=True):
            protein = index[0]
            ligand = index[1]
            pdb_paths.setdefault((protein, ligand), [])
            pdb_paths[(protein, ligand)].append(pdb_path)

        # Align PDB models
        for (protein, ligand) in pdb_paths:

            structures = []
            pdb_names = []
            for pdb_path in pdb_paths[(protein, ligand)]:
                pdb_name = pdb_path.split('/')[-1].replace('.pdb', '')
                structure = self._readPDB(pdb_name, pdb_path)
                structures.append(structure)
                pdb_names.append(pdb_name)

            if len(structures) > 1:
                reference = structures[0]
                for pdb, structure in zip(pdb_names[1:], structures[1:]):
                    alignStructures(reference, structure)
                    io.set_structure(structure)
                    io.save(pdb_path)


    def getNewBoxCenters(self, pele_poses_folder, center_atoms, verbose=False,
                         only_proteins=None, only_ligands=None):
        """
        Gets the new box centers for a group of extracted PELE poses. The new box centers
        are taken from a dictionary containing an atom-tuple in the format:

            (chain_id, residue_id, atom_name)

        and using as keys the corresponding (protein, ligand) tuples.

        Alternatively, a single 3-element tuple can be given with format:

            (chain_id, residue_name, atom_name)

        to be used with all poses

        When more than one pose is given for the same protein and ligand combination an
        average center will be calculated, therefore it is recommended that they will
        be aligned before calculating the new average box center.

        Parameters
        ==========
        pele_poses_folder : str
            Path to a folder were poses were extracted with the function extractPELEPoses()
        center_atoms : dict or tuple
            Atoms to be used as coordinate centers for the new box coordinates.
        only_proteins : (list, str)
            Only process the given proteins.
        only_ligands : (list, str)
            Only process the given ligands.

        Returns
        =======
        box_centers : dict
            Dictionary by (protein, ligand) that contains the new box center coordinates
        """

        if only_proteins == None:
            only_proteins = []
        elif isinstance(only_proteins, str):
            only_proteins = [only_proteins]

        if only_ligands == None:
            only_ligands = []
        elif isinstance(only_ligands, str):
            only_ligands = [only_ligands]

        # Get new box centers
        box_centers = {}
        for index, structure in self.getFolderStructures(pele_poses_folder, only_proteins=only_proteins,
                                                         only_ligands=only_ligands):

            protein = index[0]
            ligand = index[1]

            # Create entry for the protein and ligand box centers
            box_centers.setdefault((protein, ligand), [])

            # Check the format of the given center atoms
            if isinstance(center_atoms, dict):
                if (protein, ligand) not in center_atoms:
                    message = 'The protein and ligand combination %s-%s was not found' % (protein, ligand)
                    message += ' in the given center_atoms dictionary.'
                    raise ValueError(message)
                else:
                    chain_id, residue_id, atom_name = center_atoms[protein,ligand]
                    residue_name = None

            elif isinstance(center_atoms, tuple):
                chain_id, residue_name, atom_name = center_atoms
                residue_id = None

            else:
                raise ValueError('center_atoms must be a dict or tuple!')

            # Get center coordinates
            bc = None
            for residue in structure.get_residues():

                res_match = False

                # Check that chain matches
                chain = residue.get_parent()
                if chain.id != chain_id:
                    continue

                # Check that residue matches
                if residue_name == None:
                    if residue.id[1] == residue_id:
                        res_match = True
                elif residue_id == None:
                    if residue.resname == residue_name:
                        res_match = True

                # Check that the atom matches
                if res_match:
                    for atom in residue:
                        if atom.name == atom_name:
                            bc = [float(x) for x in atom.coord]

            # Store bc if found
            if bc != None:
                box_centers[(protein, ligand)].append(np.array(bc))
            else:
                raise ValueError('Atom could not be match for model %s-%s' % (protein, ligand))

        # Check if there are more than one box center
        for (protein, ligand) in box_centers:

            bc = np.array(box_centers[(protein, ligand)])

            if len(bc) > 1:
                if verbose:
                    message = 'Multiple protein centers from different poses found for '
                    message += '%s-%s. Calculating an average box center.' % (protein, ligand)
                    print(message)

                # Getting box center distance matrix
                M = np.zeros((len(bc), len(bc)))
                for i in range(len(bc)):
                    for j in range(len(bc)):
                        if i > j:
                            M[i][j] = np.linalg.norm(bc[i]-bc[j])
                            M[j][i] = M[i][j]

                    # Warn if the the difference between box centers is large
                    if np.amax(M) > 2.0 and verbose:
                        print('Warning: box centers differ more than 2.0 angstrom between them!')
                        print('Is recommended that common poses are aligned before calculating their new box centers.')
                        print()
                box_centers[(protein, ligand)] = np.average(bc, axis=0)

            else:
                box_centers[(protein, ligand)] = bc.reshape((3,))

        return box_centers

    def setUpPELECalculation(self, pele_folder, models_folder, input_yaml, box_centers=None, distances=None, ligand_index=1,
                             box_radius=10, steps=100, debug=False, iterations=3, cpus=96, equilibration_steps=100, ligand_energy_groups=None,
                             separator='-', use_peleffy=True, usesrun=True, energy_by_residue=False, ebr_new_flag=False, ninety_degrees_version=False,
                             analysis=False, energy_by_residue_type='all', peptide=False, equilibration_mode='equilibrationLastSnapshot',
                             spawning='independent', continuation=False, equilibration=True, skip_models=None, skip_ligands=None,
                             extend_iterations=False, only_models=None, only_ligands=None, ligand_templates=None, seed=12345, log_file=False,
                             simulation_type=None, nonbonded_energy=None, nonbonded_energy_type='all', nonbonded_new_flag=False,
                             covalent_ligands=None, old_pele_folder=None, skip_ligands_prep=None, ligand_equilibration_cst=True):
        """
        Generates a PELE calculation for extracted poses. The function reads all the
        protein ligand poses and creates input for a PELE platform set up run.

        Parameters
        ==========
        pele_folder : str
            Path to the folder where PELE calcualtions will be located
        models_folder : str
            Path to input docking poses folder.
        input_yaml : str
            Path to the input YAML file to be used as template for all the runs.
        ligand_energy_groups : dict
            Additional groups to consider when doing energy by residue reports.
        Missing!
        """

        if continuation:
            continue_all = True
        else:
            continue_all = False

        spawnings = ['independent', 'inverselyProportional', 'epsilon', 'variableEpsilon',
                     'independentMetric', 'UCB', 'FAST', 'ProbabilityMSM', 'MetastabilityMSM',
                     'IndependentMSM']

        simulation_types = ['induced_fit_fast', 'induced_fit_long', 'rescoring', None]

        if spawning not in spawnings:
            message = 'Spawning method %s not found.' % spawning
            message = 'Allowed options are: '+str(spawnings)
            raise ValueError(message)

        if simulation_type not in simulation_types:
            message = 'Simulation type method %s not found.' % simulation_type
            message = 'Allowed options are: '+str(simulation_types)
            raise ValueError(message)

        if isinstance(skip_ligands_prep, type(None)):
            skip_ligands_prep = []

        if isinstance(covalent_ligands,  type(None)):
            covalent_ligands = []

        if isinstance(covalent_ligands, str):
            covalent_ligands = [covalent_ligands]

        if covalent_ligands:
            covalent_indexes = {}

        # Create PELE job folder
        if not os.path.exists(pele_folder):
            os.mkdir(pele_folder)

        # Read docking poses information from models_folder and create pele input
        # folders.
        jobs = []
        structures = {}
        for d in os.listdir(models_folder):
            if os.path.isdir(models_folder+'/'+d):
                models = {}
                ligand_pdb_name = {}
                for f in os.listdir(models_folder+'/'+d):
                    fs = f.split(separator)
                    protein = fs[0]
                    ligand = fs[1]
                    pose = fs[2].replace('.pdb','')

                    # Skip given protein models
                    if skip_models != None:
                        if protein in skip_models:
                            continue

                    # Skip given ligand models
                    if skip_ligands != None:
                        if ligand in skip_ligands:
                            continue

                    # Skip proteins not in only_proteins list
                    if only_models != None:
                        if protein not in only_models:
                            continue

                    # Skip proteins not in only_ligands list
                    if only_ligands != None:
                        if ligand not in only_ligands:
                            continue

                    # Create PELE job folder for each docking
                    if not os.path.exists(pele_folder+'/'+protein+separator+ligand):
                        os.mkdir(pele_folder+'/'+protein+separator+ligand)

                    structure = self._readPDB(protein+separator+ligand, models_folder+'/'+d+'/'+f)
                    structures[protein+separator+ligand] = structure

                    # Change water names if any
                    for residue in structure.get_residues():
                        if residue.id[0] == 'W':
                            residue.resname = 'HOH'

                        if residue.get_parent().id == 'L':
                            ligand_pdb_name[ligand] = residue.resname

                    ## Add dummy atom if peptide docking ### Strange fix =)
                    if peptide:
                        for chain in structure.get_chains():
                            if chain.id == 'L':
                                # Create new residue
                                new_resid = max([r.id[1] for r in chain.get_residues()])+1
                                residue = PDB.Residue.Residue(('H', new_resid, ' '), 'XXX', ' ')
                                serial_number = max([a.serial_number for a in chain.get_atoms()])+1
                                atom = PDB.Atom.Atom('X', [0,0,0], 0, 1.0, ' ',
                                                     '%-4s' % 'X', serial_number+1, 'H')
                                residue.add(atom)
                                chain.add(residue)

                    self._saveStructureToPDB(structure, pele_folder+'/'+protein+separator+ligand+'/'+f)

                    if (protein, ligand) not in models:
                        models[(protein,ligand)] = []
                    models[(protein,ligand)].append(f)

                    # Change set up if covalent ligands were used in the previous PELE run
                    if covalent_ligands:
                        debug = True
                        continuation = True
                        _copyScriptFile(pele_folder+'/'+protein+separator+ligand, 'modifyProcessedForCovalentPELE.py')

                        if isinstance(old_pele_folder, type(None)):
                            raise ValueError('You must give the old_pele_folder to copy covalent paramters!')

                        # Copy DataLocal folders for covalently modified residues
                        for p in os.listdir(old_pele_folder):
                            if protein in p and ligand in p:
                                datalocal_dir = old_pele_folder+protein+separator+ligand+'/output/DataLocal'
                                output_dir = pele_folder+'/'+protein+separator+ligand+'/output'
                                if not os.path.exists(output_dir):
                                    os.mkdir(output_dir)
                                if not os.path.exists(output_dir+'/DataLocal'):
                                    shutil.copytree(datalocal_dir, output_dir+'/DataLocal')

                        # Search covalent indexes in PDBs
                        covalent_indexes[(protein, ligand)] = []
                        for r in structure.get_residues():
                            if r.resname in covalent_ligands:
                                covalent_indexes[(protein, ligand)].append(r.id[1])

                # If templates are given for ligands
                templates = {}
                if ligand_templates != None:

                    # Create templates folder
                    if not os.path.exists(pele_folder+'/templates'):
                        os.mkdir(pele_folder+'/templates')

                    for ligand in os.listdir(ligand_templates):

                        if not os.path.isdir(ligand_templates+'/'+ligand):
                            continue

                        # Create ligand template folder
                        if not os.path.exists(pele_folder+'/templates/'+ligand):
                            os.mkdir(pele_folder+'/templates/'+ligand)

                        templates[ligand] = []
                        for f in os.listdir(ligand_templates+'/'+ligand):
                            if f.endswith('.rot.assign') or f.endswith('z'):

                                # Copy template files
                                shutil.copyfile(ligand_templates+'/'+ligand+'/'+f,
                                                pele_folder+'/templates/'+ligand+'/'+f)

                                templates[ligand].append(f)

                # Create YAML file
                for model in models:
                    protein, ligand = model
                    keywords = ['system', 'chain', 'resname', 'steps', 'iterations', 'atom_dist', 'analyse',
                                'cpus', 'equilibration', 'equilibration_steps', 'traj', 'working_folder',
                                'usesrun', 'use_peleffy', 'debug', 'box_radius', 'box_center', 'equilibration_mode',
                                'seed' ,'spawning']

                    # Get distances from PELE data
                    if distances == None:
                        distances = {}
                    if protein not in distances:
                        distances[protein] = {}
                    if ligand not in distances[protein]:
                        distances[protein][ligand] = []

                    pele_distances = [(x.split('_')[1:3][0], x.split('_')[1:3][1]) for x in self.getDistances(protein, ligand) if '_coordinate' not in x]
                    pele_distances = list(set(pele_distances))

                    for d in pele_distances:
                        at1 = self._atomStringToTuple(d[0])
                        at2 = self._atomStringToTuple(d[1])
                        distances[protein][ligand].append((at1, at2))

                    # Write input yaml
                    with open(pele_folder+'/'+protein+separator+ligand+'/'+'input.yaml', 'w') as iyf:
                        if energy_by_residue or nonbonded_energy != None:
                            # Use new PELE version with implemented local nonbonded energies
                            iyf.write('pele_exec: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/bin/PELE-1.7.2_mpi"\n')
                            iyf.write('pele_data: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/Data"\n')
                            iyf.write('pele_documents: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/Documents/"\n')
                        elif ninety_degrees_version:
                            # Use new PELE version with implemented 90 degrees fix
                            iyf.write('pele_exec: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/bin/PELE-1.8_mpi"\n')
                            iyf.write('pele_data: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/Data"\n')
                            iyf.write('pele_documents: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/Documents/"\n')

                        if simulation_type != None:
                            iyf.write(simulation_type+": true\n")

                        if len(models[model]) > 1:
                            equilibration_mode = 'equilibrationCluster'
                            iyf.write("system: '*.pdb'\n")
                        else:
                            iyf.write("system: '"+" ".join(models[model])+"'\n")
                        iyf.write("chain: 'L'\n")
                        if peptide:
                            iyf.write("resname: 'XXX'\n")
                            iyf.write("skip_ligand_prep:\n")
                            iyf.write(" - 'XXX'\n")
                        else:
                            iyf.write("resname: '"+ligand_pdb_name[ligand]+"'\n")
                        iyf.write("steps: "+str(steps)+"\n")
                        iyf.write("iterations: "+str(iterations)+"\n")
                        iyf.write("cpus: "+str(cpus)+"\n")
                        if equilibration:
                            iyf.write("equilibration: true\n")
                            iyf.write("equilibration_mode: '"+equilibration_mode+"'\n")
                            iyf.write("equilibration_steps: "+str(equilibration_steps)+"\n")
                        else:
                            iyf.write("equilibration: false\n")
                        if spawning != None:
                            iyf.write("spawning: '"+str(spawning)+"'\n")

                        iyf.write("traj: trajectory.xtc\n")
                        iyf.write("working_folder: 'output'\n")
                        if usesrun:
                            iyf.write("usesrun: true\n")
                        else:
                            iyf.write("usesrun: false\n")
                        if use_peleffy:
                            iyf.write("use_peleffy: true\n")
                        else:
                            iyf.write("use_peleffy: false\n")
                        if analysis:
                            iyf.write("analyse: true\n")
                        else:
                            iyf.write("analyse: false\n")

                        skip_lig_line = False
                        if ligand in templates or ligand in skip_ligands_prep:
                            iyf.write("templates:\n")
                            iyf.write(' - "LIGAND_TEMPLATE_PATH_ROT"\n')
                            iyf.write(' - "LIGAND_TEMPLATE_PATH_Z"\n')
                            iyf.write("skip_ligand_prep:\n")
                            skip_lig_line = True
                            iyf.write(' - "'+ligand_pdb_name[ligand]+'"\n')

                        for l in covalent_ligands:
                            if not skip_lig_line:
                                iyf.write("skip_ligand_prep:\n")
                            iyf.write(' - "'+l+'"\n')

                        iyf.write("box_radius: "+str(box_radius)+"\n")
                        if isinstance(box_centers, type(None)) and peptide:
                            raise ValueError('You must give per-protein box_centers when docking peptides!')

                        if not isinstance(box_centers, type(None)):

                            if not all(isinstance(x, (float,int)) for x in box_centers[(protein, ligand)]):
                                # get coordinates from tuple
                                structure = structures[protein+separator+ligand]
                                for chain in structure.get_chains():
                                    if chain.id == box_centers[(protein,ligand)][0]:
                                        for r in chain:
                                            if r.id[1] == box_centers[(protein,ligand)][1]:
                                                for atom in r:
                                                    if atom.name == box_centers[(protein,ligand)][2]:
                                                        coordinates = atom.coord
                                #raise ValueError('This is not yet implemented!')
                            else:
                                coordinates = box_centers[model]

                            box_center = ''
                            for coord in coordinates:
                                #if not isinstance(coord, float):
                                #    raise ValueError('Box centers must be given as a (x,y,z) tuple or list of floats.')
                                box_center += '  - '+str(float(coord))+'\n'
                            iyf.write("box_center: \n"+box_center)

                        # energy by residue is not implemented in PELE platform, therefore
                        # a scond script will modify the PELE.conf file to set up the energy
                        # by residue calculation.
                        if any([debug, energy_by_residue, peptide, nonbonded_energy != None, ligand_equilibration_cst]):
                            iyf.write("debug: true\n")

                        if distances != None:
                            iyf.write("atom_dist:\n")
                            for d in distances[protein][ligand]:
                                if isinstance(d[0], str):
                                    d1 = "- 'L:"+str(ligand_index)+":"+d[0]+"'\n"
                                else:
                                    d1 = "- '"+d[0][0]+":"+str(d[0][1])+":"+d[0][2]+"'\n"
                                if isinstance(d[1], str):
                                    d2 = "- 'L:"+str(ligand_index)+":"+d[1]+"'\n"
                                else:
                                    d2 = "- '"+d[1][0]+":"+str(d[1][1])+":"+d[1][2]+"'\n"
                                iyf.write(d1)
                                iyf.write(d2)

                        if seed:
                            iyf.write('seed: '+str(seed)+'\n')

                        if log_file:
                            iyf.write('log: true\n')

                        iyf.write('\n')

                        if input_yaml != None:
                            iyf.write("#Options gathered from "+input_yaml+'\n')

                            with open(input_yaml) as tyf:
                                for l in tyf:
                                    if l.startswith('#'):
                                        continue
                                    elif l.startswith('-'):
                                        continue
                                    elif l.strip() == '':
                                        continue
                                    if l.split()[0].replace(':', '') not in keywords:
                                        iyf.write(l)

                    if energy_by_residue:
                        _copyScriptFile(pele_folder, 'addEnergyByResidueToPELEconf.py')
                        ebr_script_name = '._addEnergyByResidueToPELEconf.py'
                        if not isinstance(ligand_energy_groups, type(None)):
                            if not isinstance(ligand_energy_groups, dict):
                                raise ValueError('ligand_energy_groups, must be given as a dictionary')
                            with open(pele_folder+'/'+protein+separator+ligand+'/ligand_energy_groups.json', 'w') as jf:
                                json.dump(ligand_energy_groups[ligand], jf)

                    if nonbonded_energy != None:
                        _copyScriptFile(pele_folder, 'addAtomNonBondedEnergyToPELEconf.py')
                        nbe_script_name = '._addAtomNonBondedEnergyToPELEconf.py'
                        if not isinstance(nonbonded_energy, dict):
                            raise ValueError('nonbonded_energy, must be given as a dictionary')
                        with open(pele_folder+'/'+protein+separator+ligand+'/nonbonded_energy_atoms.json', 'w') as jf:
                            json.dump(nonbonded_energy[protein][ligand], jf)

                    if peptide:
                        _copyScriptFile(pele_folder, 'modifyPelePlatformForPeptide.py')
                        peptide_script_name = '._modifyPelePlatformForPeptide.py'

                    if ligand_equilibration_cst:
                        _copyScriptFile(pele_folder, 'addLigandConstraintsToPELEconf.py')
                        equilibration_script_name = '._addLigandConstraintsToPELEconf.py'
                        _copyScriptFile(pele_folder, 'changeAdaptiveIterations.py')
                        adaptive_script_name = '._changeAdaptiveIterations.py'


                    # Create command
                    command = 'cd '+pele_folder+'/'+protein+separator+ligand+'\n'

                    # Add commands to write template folder absolute paths
                    if ligand in templates:
                        command += "export CWD=$(pwd)\n"
                        command += 'cd ../templates\n'
                        command += 'export TMPLT_DIR=$(pwd)/'+ligand+'\n'
                        command += 'cd $CWD\n'
                        for tf in templates[ligand]:
                            if continuation:
                                yaml_file = 'input_restart.yaml'
                            else:
                                yaml_file = 'input.yaml'
                            if tf.endswith('.assign'):
                                command += "sed -i s,LIGAND_TEMPLATE_PATH_ROT,$TMPLT_DIR/"+tf+",g "+yaml_file+"\n"
                            elif tf.endswith('z'):
                                command += "sed -i s,LIGAND_TEMPLATE_PATH_Z,$TMPLT_DIR/"+tf+",g "+yaml_file+"\n"

                    if not continuation or covalent_ligands:
                        command += 'python -m pele_platform.main input.yaml\n'
                        if covalent_ligands:
                            covalent_command = 'cd output\n'
                            for covlig in covalent_indexes[(protein, ligand)]:
                                covalent_command += 'python ../._modifyProcessedForCovalentPELE.py '+str(covlig)+' \n'
                            covalent_command += 'cd ..\n'
                            command += covalent_command

                        if ligand_equilibration_cst:

                            # Copy input_yaml for equilibration
                            oyml = open(pele_folder+'/'+protein+separator+ligand+'/input_equilibration.yaml', 'w')
                            debug_line = False
                            restart_line = False
                            with open(pele_folder+'/'+protein+separator+ligand+'/input.yaml') as iyml:
                                for l in iyml:
                                    if 'debug: true' in l:
                                        debug_line = True
                                        oyml.write('restart: true\n')
                                        oyml.write('adaptive_restart: true\n')
                                        continue
                                    elif 'restart: true' in l:
                                        restart_line = True
                                    elif l.startswith('iterations:'):
                                        l = 'iterations: 1\n'
                                    elif l.startswith('steps:'):
                                        l = 'steps: 1\n'
                                    oyml.write(l)
                                if not debug_line and not restart_line:
                                    oyml.write('restart: true\n')
                                    oyml.write('adaptive_restart: true\n')
                            oyml.close()

                            # Add commands for adding ligand constraints
                            command += 'cp output/pele.conf output/pele.conf.backup\n'
                            command += 'cp output/adaptive.conf output/adaptive.conf.backup\n'

                            # Modify pele.conf to add ligand constraints
                            command += 'python ../'+equilibration_script_name+' '
                            command += "output " # I think we should change this for a variable
                            if isinstance(ligand_equilibration_cst, (int, float)) and ligand_equilibration_cst != 1.0:
                                command += "--constraint_value "+str(float(ligand_equilibration_cst))
                            command += '\n'

                            # Modify adaptive.conf to remove simulation steps
                            command += 'python ../'+adaptive_script_name+' '
                            command += "output " # I think we should change this for a variable
                            command += '--iterations 1 '
                            command += '--steps 1\n'

                            # Launch equilibration
                            command += 'python -m pele_platform.main input_equilibration.yaml\n'

                            # Recover conf files
                            command += 'cp output/pele.conf.backup output/pele.conf\n'
                            command += 'cp output/adaptive.conf.backup output/adaptive.conf\n'

                            continuation = True

                    if continuation:
                        debug_line = False
                        restart_line = False
                        # Copy input_yaml for equilibration
                        oyml = open(pele_folder+'/'+protein+separator+ligand+'/input_restart.yaml', 'w')
                        debug_line = False
                        restart_line = False
                        with open(pele_folder+'/'+protein+separator+ligand+'/input.yaml') as iyml:
                            for l in iyml:
                                if 'debug: true' in l:
                                    debug_line = True
                                    oyml.write('restart: true\n')
                                    oyml.write('adaptive_restart: true\n')
                                    continue
                                elif 'restart: true' in l:
                                    restart_line = True
                                oyml.write(l)
                            if not debug_line and not restart_line:
                                oyml.write('restart: true\n')
                                oyml.write('adaptive_restart: true\n')
                        oyml.close()

                        if extend_iterations:
                            _copyScriptFile(pele_folder, 'extendAdaptiveIteartions.py')
                            extend_script_name = '._extendAdaptiveIteartions.py'
                            command += 'python ../'+extend_script_name+' output '+str(iterations)+'\n'

                        if not any([energy_by_residue, peptide, nonbonded_energy]):
                            command += 'python -m pele_platform.main input_restart.yaml\n'

                        if any([ligand_equilibration_cst]) and not continue_all:
                            continuation = False
                            debug = False

                    if energy_by_residue:
                        command += 'python ../'+ebr_script_name+' output --energy_type '+energy_by_residue_type
                        if isinstance(ligand_energy_groups, dict):
                            command += ' --ligand_energy_groups ligand_energy_groups.json'
                            command += ' --ligand_index '+str(ligand_index)
                        if ebr_new_flag:
                            command += ' --new_version '
                        if peptide:
                            command += ' --peptide \n'
                            command += 'python ../'+peptide_script_name+' output '+" ".join(models[model])+'\n'
                        else:
                            command += '\n'

                        # Create second PELE iterations
                        if ligand_equilibration_cst:
                            command += 'python ../'+adaptive_script_name+' '
                            command += "output " # I think we should change this for a variable
                            command += '--iterations 2 '
                            command += '--steps '+str(steps)+'\n'
                            command += 'python -m pele_platform.main input_restart.yaml\n'

                            # Remove incomplete-header first iteration
                            command += 'rm -r output/output/0\n' # Change first output for a variable
                            # Move second iteration to first
                            command += 'mv output/output/1 output/output/0\n' # Change first output for a variable

                            # Add commands for recover original adaptive
                            command += 'cp output/adaptive.conf.backup output/adaptive.conf\n'

                        command += 'python -m pele_platform.main input_restart.yaml\n'
                    if peptide:
                        command += 'python ../'+peptide_script_name+' output '+" ".join(models[model])+'\n'
                        with open(pele_folder+'/'+protein+separator+ligand+'/'+'input_restart.yaml', 'w') as oyml:
                            with open(pele_folder+'/'+protein+separator+ligand+'/'+'input.yaml') as iyml:
                                for l in iyml:
                                    if 'debug: true' in l:
                                        l = 'restart: true\n'
                                    oyml.write(l)
                        if nonbonded_energy == None:
                            command += 'python -m pele_platform.main input_restart.yaml\n'

                    elif extend_iterations and not continuation:
                        raise ValueEror('extend_iterations must be used together with the continuation keyword')

                    if nonbonded_energy != None:
                        command += 'python ../'+nbe_script_name+' output --energy_type '+nonbonded_energy_type
                        command += ' --target_atoms nonbonded_energy_atoms.json'
                        protein_chain = [c for c in self.chain_ids[protein][ligand].values() if c != 'L'][0]
                        command += ' --protein_chain '+protein_chain
                        if ebr_new_flag or nonbonded_new_flag:
                            command += ' --new_version'
                        command += '\n'

                        if not os.path.exists(pele_folder+'/'+protein+separator+ligand+'/'+'input_restart.yaml'):
                            with open(pele_folder+'/'+protein+separator+ligand+'/'+'input_restart.yaml', 'w') as oyml:
                                with open(pele_folder+'/'+protein+separator+ligand+'/'+'input.yaml') as iyml:
                                    for l in iyml:
                                        if 'debug: true' in l:
                                            l = 'restart: true\n'
                                        oyml.write(l)
                        command += 'python -m pele_platform.main input_restart.yaml\n'

                    command += 'cd ../..\n'
                    jobs.append(command)

        return jobs


    def removeTrajectoryFiles(self):
        """
        Remove all trajectory files from PELE calculation.
        """

        # Remove PELE trajectories
        for protein in self.trajectory_files:
            for ligand in self.trajectory_files[protein]:
                for epoch in self.trajectory_files[protein][ligand]:
                    for trajectory in self.trajectory_files[protein][ligand][epoch]:
                        f = self.trajectory_files[protein][ligand][epoch][trajectory]
                        if f != {} and os.path.exists(f) and f.split('/')[0] != self.data_folder:
                            os.remove(self.trajectory_files[protein][ligand][epoch][trajectory])

        # Remove PELE equilibration trajectories
        for protein in self.equilibration['trajectory']:
            for ligand in self.equilibration['trajectory'][protein]:
                for epoch in self.equilibration['trajectory'][protein][ligand]:
                    for trajectory in self.equilibration['trajectory'][protein][ligand][epoch]:
                        f = self.equilibration['trajectory'][protein][ligand][epoch][trajectory]
                        if f != {} and os.path.exists(f) and f.split('/')[0] != self.data_folder:
                            os.remove(self.equilibration['trajectory'][protein][ligand][epoch][trajectory])

    def setUpSiteMapCalculation(self, job_folder, residue_selection, only_proteins=None, only_ligands=None,
                                site_box=10, resolution='fine', reportsize=100, sidechain=True, verbose=False,
                                keep_volpts=False, remove_ligand=False, overwrite=False, separator=None,
                                dataframe=None, recalculate=False, skip_models=None, skip_ligands=None,
                                check_separator=True):
        """
        Sets up sitemap calculations for the whole PELE simulation, so, yes it does consume a lot of space and computation.
        After the calculations are completed it is recommended to download only the log files, which contains the sitemap
        scores, e.g.:

        rsync -av --include='*/' --include='*.log' --exclude='*' cluster@user:sitemap_folder_path destination_path

        Then read the log files with readSiteMapCalculation() which will incorporate the data to the main dataframe.
        Any missing data point will be set to nan if it was not computed at all or if sitemap failed to compute a pocket for it.
        After executing this last function the sitemap data will be stored in the data_folder, so now it would be a good idea
        to remove all the calculation folder to save space for having fun with something else...

        Parameters
        ==========
        job_folder : str
            Path to the sitemap calculation folder.
        residue_selection : list or tuple
            List of residues as tuples: (chainid, resid) to be used as the pocket definition.
        only_proteins : list
            Limit the set up to only these models.
        only_proteins : list
            Limit the set up to only these ligand.
        site_box : int
            Sitemap site_box parameter.
        resolution : str
            Sitemap resolution parameter.
        sidechain : bool
            Use only the sidechain of the given residues to define the pocket.
        reportsize : int
            Sitemap resolution reportsize
        keep_volpts : bool
            Keep the volpts pdb files. Only use it to debug a few snapshots or if you are insane.
        remove_ligand : bool
            Remove the ligand from the structures to compute the pocket map without it interfering it.
            Recommended!
        overwrite : bool
            Rewrite any input pdb file already created.
        dataframe : pandas.DataFrame
            Restrict calculations to indexes in the given dataframe.

        Returns
        =======
        jobs : list
            List of jobs to execute to calculate sitemap for all structures.

        ## TODO: Implement a dataframe so only those points are computed.
        """

        def checkLogFiles(job_folder, separator):

            log_files = {}
            for p in os.listdir(job_folder):
                protein_folder = job_folder+'/'+p
                if not os.path.isdir(protein_folder):
                    continue
                log_files[p] = {}
                for l in os.listdir(protein_folder):
                    ligand_folder = protein_folder+'/'+l
                    if not os.path.isdir(ligand_folder):
                        continue
                    log_files[p][l] = []
                    sitemap_folder = ligand_folder+'/sitemap/output_models'
                    for log in os.listdir(sitemap_folder):
                        if not log.endswith('.log'):
                            continue
                        epoch, trajectory, step = log.split(separator)[-3:]
                        epoch = int(epoch)
                        trajectory = int(trajectory)
                        step = int(step.split('.')[0])
                        log_files[p][l].append((epoch, trajectory, step))
            return log_files

        def checkInputPDBfiles(job_folder, separator):

            pdb_input_files = {}
            for p in os.listdir(job_folder):
                protein_folder = job_folder+'/'+p
                if not os.path.isdir(protein_folder):
                    continue
                pdb_input_files[p] = {}
                for l in os.listdir(protein_folder):
                    ligand_folder = protein_folder+'/'+l
                    if not os.path.isdir(ligand_folder):
                        continue
                    pdb_input_files[p][l] = []
                    prepwizard_folder = ligand_folder+'/prepwizard/input_models'
                    for pdb in sorted(os.listdir(prepwizard_folder)):
                        if not pdb.endswith('.pdb'):
                            continue
                        epoch, trajectory, step = pdb.split(separator)[-3:]
                        epoch = int(epoch)
                        trajectory = int(trajectory)
                        step = int(step.split('.')[0])
                        pdb_input_files[p][l].append((epoch, trajectory, step))

            return pdb_input_files

        if not separator:
            separator = self.separator

        if not os.path.exists(job_folder):
            os.mkdir(job_folder)

        if only_proteins:
            if isinstance(only_proteins, str):
                only_proteins = [only_proteins]

        if only_ligands:
            if isinstance(only_ligands, str):
                only_ligands = [only_ligands]

        if skip_models:
            if isinstance(skip_models, str):
                skip_models = [skip_models]

        if skip_ligands:
            if isinstance(skip_ligands, str):
                skip_ligands = [skip_ligands]

        if not isinstance(dataframe, type(None)):
            selected_indexes = {}
            for index in dataframe.index:
                protein, ligand = index[:2]
                selected_indexes.setdefault(protein, {})
                selected_indexes[protein].setdefault(ligand, [])
                selected_indexes[protein][ligand].append(index[2:-1])

        _copyScriptFile(job_folder, 'prepareForSiteMap.py')
        script_name =  '._prepareForSiteMap.py'

        # Check input and output files
        log_files = checkLogFiles(job_folder, separator)
        pdb_input_files = checkInputPDBfiles(job_folder, separator)

        epoch_zf = 4
        trajectory_zf = 4
        step_zf = 5

        jobs = []
        for protein, ligand in self.pele_combinations:

            if only_proteins and protein not in only_proteins:
                continue

            if only_ligands and ligand not in only_ligands:
                continue

            if skip_models and protein in skip_models:
                continue

            if skip_ligands and protein in skip_ligands:
                continue

            if check_separator and separator in protein:
                raise ValueError(f'Separator "{separator}" was found in protein name. Please use a different one')
            if check_separator and separator in ligand:
                raise ValueError(f'Separator "{separator}" was found in ligand name. Please use a different one')

            # Skip if not in the given dataframe
            if not isinstance(dataframe, type(None)):
                if protein not in selected_indexes:
                    continue
                if ligand not in selected_indexes[protein]:
                    continue

            # Check if finished
            ligand_data = self.getProteinAndLigandData(protein, ligand)
            total_points = ligand_data.shape[0]

            if protein in log_files and ligand in log_files[protein]:
                if len(log_files[protein][ligand]) == total_points and not recalculate:
                    if verbose:
                        print(f'All sitemap calculations are done for {protein} and {ligand}')
                    continue

            print(f'Processing {protein} and {ligand}:')

            protein_folder = job_folder+'/'+protein
            if not os.path.exists(protein_folder):
                os.mkdir(protein_folder)

            ligand_folder = protein_folder+'/'+ligand
            if not os.path.exists(ligand_folder):
                os.mkdir(ligand_folder)

            # Create prepwizard folder
            prepwizard_folder = ligand_folder+'/prepwizard'
            if not os.path.exists(prepwizard_folder):
                os.mkdir(prepwizard_folder)

            # Create prepwizard input folder
            prepwizard_input_folder = prepwizard_folder+'/input_models'
            if not os.path.exists(prepwizard_input_folder):
                os.mkdir(prepwizard_input_folder)

            # Create prepwizard output folder
            prepwizard_output_folder = prepwizard_folder+'/output_models'
            if not os.path.exists(prepwizard_output_folder):
                os.mkdir(prepwizard_output_folder)

            # Create sitemap folder
            sitemap_folder = ligand_folder+'/sitemap'
            if not os.path.exists(sitemap_folder):
                os.mkdir(sitemap_folder)

            # Create sitemap input folder
            sitemap_input_folder = sitemap_folder+'/input_models'
            if not os.path.exists(sitemap_input_folder):
                os.mkdir(sitemap_input_folder)

            # Create sitemap output folder
            sitemap_output_folder = sitemap_folder+'/output_models'
            if not os.path.exists(sitemap_output_folder):
                os.mkdir(sitemap_output_folder)

            for epoch in sorted(self.trajectory_files[protein][ligand]):

                # Check if epoch pdbs were already extracted
                epoch_data = ligand_data[ligand_data.index.get_level_values('Epoch') == epoch]

                if protein in pdb_input_files  and ligand in pdb_input_files[protein]:
                    epoch_pdbs = [p for p in pdb_input_files[protein][ligand] if p[0] == epoch]
                    epoch_logs = [p for p in log_files[protein][ligand] if p[0] == epoch]
                else:
                    epoch_pdbs = []
                    epoch_logs = []

                if len(epoch_logs) == epoch_data.shape[0] and not recalculate:
                    if verbose:
                        print(f'\tAll sitemap calculations for epoch {epoch} are done')
                    continue

                for trajectory in sorted(self.trajectory_files[protein][ligand][epoch]):

                    # Check if trajectory pdbs were already extracted
                    trajectory_data = epoch_data[epoch_data.index.get_level_values('Trajectory') == trajectory]
                    trajectory_pdbs = [p for p in epoch_pdbs if p[1] == trajectory]
                    trajectory_logs = [p for p in epoch_logs if p[1] == trajectory]

                    if len(trajectory_logs) == trajectory_data.shape[0] and not recalculate:
                        if verbose:
                            print(f'\tAll sitemap calculations for epoch {epoch} and trajectory {trajectory} are done')
                        continue

                    if verbose:
                        print(f'\tExtracting models for epoch {epoch} and trajectory {trajectory}', end='\r')

                    pdb_files = []
                    extract_pdbs = True
                    if len(trajectory_pdbs) == trajectory_data.shape[0]:

                        extract_pdbs = False

                        for p in trajectory_pdbs:
                            if p in trajectory_logs and not recalculate:
                                continue

                            # Skip if not in dataframe
                            if not isinstance(dataframe, type(None)):
                                if p not in selected_indexes[protein][ligand]:
                                    continue

                            pdb_file = protein+separator
                            pdb_file += ligand+separator
                            pdb_file += str(epoch).zfill(epoch_zf)+separator
                            pdb_file += str(trajectory).zfill(trajectory_zf)+separator
                            pdb_file += str(p[2]).zfill(step_zf)+'.pdb'
                            pdb_files.append(pdb_file)

                    if extract_pdbs:
                        traj_file = self.trajectory_files[protein][ligand][epoch][trajectory]
                        traj = md.load(traj_file, top=self.topology_files[protein][ligand])

                        if remove_ligand:
                            atom_indexes = traj.topology.select('not (resname '+self.ligand_names[protein][ligand]+')')
                            traj = traj.atom_slice(atom_indexes)

                        for i,frame in enumerate(traj):

                            # Check this code
                            index = (epoch, trajectory, frame)
                            # Skip if not in dataframe
                            if not isinstance(dataframe, type(None)):
                                if index not in selected_indexes[protein][ligand]:
                                    continue

                            # Write input PDB
                            pdb_file = protein+separator
                            pdb_file += ligand+separator
                            pdb_file += str(epoch).zfill(epoch_zf)+separator
                            pdb_file += str(trajectory).zfill(trajectory_zf)+separator
                            pdb_file += str(i).zfill(step_zf)+'.pdb'
                            pdb_files.append(pdb_file)

                            input_file = prepwizard_input_folder+'/'
                            input_file += pdb_file

                            if not os.path.exists(input_file) or overwrite:
                                frame.save(input_file)

                    for pdb_file in pdb_files:

                        input_file = prepwizard_input_folder+'/'
                        input_file += pdb_file

                        # Write prepwizard command
                        command = 'cd '+prepwizard_output_folder+'\n'
                        command += '"${SCHRODINGER}/utilities/prepwizard" '
                        command += '../input_models/'+pdb_file+' '
                        command += pdb_file+' '
                        command += '-keepfarwat '
                        command += '-noepik '
                        command += '-noprotassign '
                        command += '-noimpref '
                        command += '-JOBNAME '+pdb_file.replace('.pdb','')+' '
                        command += '-HOST localhost:1 '
                        command += '-WAIT\n'
                        command += 'cd '+'../'*len(prepwizard_output_folder.split('/'))+'\n'

                        # Write pdb to mae for sitemap command
                        command += 'cd '+sitemap_input_folder+'\n'
                        command += '"${SCHRODINGER}/run" '
                        command += '../../../../'+script_name+' '
                        command += '../../prepwizard/output_models/'+pdb_file+' '
                        command += './ '
                        command += '--protein_only \n'
                        command += 'cd '+'../'*len(sitemap_input_folder.split('/'))+'\n'

                        # Write sitemap command
                        command += 'cd '+sitemap_output_folder+'\n'
                        command += '"${SCHRODINGER}/sitemap" '
                        command += '-j '+pdb_file+' '
                        command += '-prot ../input_models/'+pdb_file.replace('.pdb','_protein.mae')+' '
                        command += '-sitebox '+str(site_box)+' '
                        command += '-resolution '+str(resolution)+' '
                        if keep_volpts:
                            command += '-keepvolpts yes '
                        command += '-keeplogs yes '
                        command += '-reportsize '+str(reportsize)+' '

                        # For chain and residue index
                        for r in residue_selection:
                            if isinstance(r, tuple) and len(r) == 2:
                                command += '-siteasl \"chain.name '+str(r[0])+' and res.num {'+str(r[1])+'} '
                            else:
                                raise ValueError('Incorrect residue definition!')

                            if sidechain:
                                command += 'and not (atom.pt ca,c,n,h,o)'
                            command += '\" '
                        command += '-HOST localhost:1 '
                        command += '-TMPLAUNCHDIR '
                        command += '-WAIT\n'

                        command += 'cd '+'../'*len(sitemap_output_folder.split('/'))+'\n'
                        jobs.append(command)

        return jobs

    def readSiteMapCalculation(self, sitemap_folder, separator=None, verbose=False,
                               fill_with_zeros=False, force_reading=False):
        """
        Reads sitemap scores for the PELE trajectories. It also
        stores the sitemap data at pele_data_folder/sitemap_data.

        Parameters
        ==========
        sitemap_folder : str
            Sitemap calculation folder
        separator : str
            String employed to separate the data in file names.
        verbose : bool
            Verbode mode
        fill_with_zeros : bool
            Fill with zeros data points where sites were not found.
        force_reading : bool
            Force reading data from log files even if CSV files are found for the
            protein and ligand combination.

        Returns
        =======
        sitemap_data : pandas.DataFrame
            The pele dataframe points with the sitemap data added.
        """
        def readSitemapLog(log_file):
            cond = False
            scores = {}
            with open(log_file) as lf:
                for l in lf:
                    if l.startswith('SiteScore'):
                        sm_scores = l.split()
                        cond = True
                        continue
                    if cond:
                        for s,v in zip(sm_scores, l.split()):
                            try:
                                scores[s] = int(v)
                            except:
                                scores[s] = float(v)
                        cond = False
                    if l.startswith('Output files will not be written since no sites were found.'):
                        return

            return scores

        if not separator:
            separator = self.separator

        sitemap_scores = ['SiteScore', 'size', 'Dscore', 'volume', 'exposure', 'enclosure',
                          'contact', 'phobic', 'philic', 'balance', 'don/acc']

        # Create sitemap data folder at pele data
        sitemap_data_folder = self.data_folder+'/sitemap_data'
        if not os.path.exists(sitemap_data_folder):
            os.mkdir(sitemap_data_folder)

        sitemap_data = {}
        if verbose:
            counts = {}
            read_from_csv = {}

        for protein in self.proteins:

            protein_folder = sitemap_folder+'/'+protein

            if verbose:
                counts[protein] = {}
                read_from_csv[protein] = {}

            for ligand in self.ligands:

                csv_file = sitemap_data_folder+'/'+protein+self.separator+ligand+'.csv'

                ligand_folder = protein_folder+'/'+ligand
                if not os.path.isdir(ligand_folder) and not os.path.exists(csv_file):
                    continue

                sitemap_data.setdefault(protein, {})
                sitemap_data[protein][ligand] = {}
                sitemap_data[protein][ligand]['Epoch'] = []
                sitemap_data[protein][ligand]['Trajectory'] = []
                sitemap_data[protein][ligand]['Accepted Pele Steps'] = []

                if verbose:
                    counts[protein][ligand] = 0
                    read_from_csv[protein][ligand] = False

                # Check if sitemap data is available as CSV file
                sitemap_output_folder = ligand_folder+'/sitemap/output_models'
                if (not os.path.exists(csv_file) or force_reading) and os.path.exists(sitemap_output_folder):

                    for f in sorted(os.listdir(sitemap_output_folder)):

                        if f.endswith('.log'):

                            # Get pele trajectory values
                            split = f.split(separator)

                            # Check problems if incorrect separtor was used
                            if len(f.split(separator)) < 5:
                                raise ValueError(f'Are sure the used separator "{separator}" is the correct one?')
                            elif len(f.split(separator)) > 5:
                                protein_splits = []
                                ligand_splits = []
                                for s in split[:-3]:
                                    if s in protein:
                                        protein_splits.append(s)
                                    elif s in ligand:
                                        ligand_splits.append(s)
                                assert separator.join(protein_splits) == protein
                                assert separator.join(ligand_splits) == ligand
                            else:
                                assert split[0] == protein and split[1] == ligand

                            epoch = int(split[-3:][0])
                            trajectory = int(split[-3:][1])
                            step = int(split[-3:][2].split('.')[0])

                            scores = readSitemapLog(sitemap_output_folder+'/'+f)

                            if verbose:
                                counts[protein][ligand] += 1

                            # Add data
                            sitemap_data[protein][ligand]['Epoch'].append(epoch)
                            sitemap_data[protein][ligand]['Trajectory'].append(trajectory)
                            sitemap_data[protein][ligand]['Accepted Pele Steps'].append(step)

                            if not scores:
                                for s in sitemap_scores:
                                    sitemap_data[protein][ligand].setdefault(s, [])
                                    if fill_with_zeros:
                                        sitemap_data[protein][ligand][s].append(0.0)
                                    else:
                                        sitemap_data[protein][ligand][s].append(np.nan)
                            else:
                                for s in scores:
                                    sitemap_data[protein][ligand].setdefault(s, [])
                                    sitemap_data[protein][ligand][s].append(scores[s])

                    # Convert to dataframe and store in the pele data folder.
                    sitemap_data[protein][ligand] = pd.DataFrame(sitemap_data[protein][ligand])
                    indexes = ['Epoch', 'Trajectory', 'Accepted Pele Steps']
                    sitemap_data[protein][ligand] = sitemap_data[protein][ligand].set_index(indexes)
                    sitemap_data[protein][ligand].to_csv(csv_file)
                else:
                    sitemap_data[protein][ligand] = pd.read_csv(csv_file)
                    indexes = ['Epoch', 'Trajectory', 'Accepted Pele Steps']
                    sitemap_data[protein][ligand] = sitemap_data[protein][ligand].set_index(indexes)
                    if verbose:
                        read_from_csv[protein][ligand] = True

        # Get mapping of all sitemap data
        if verbose:
            print('Points read:')
        all_sitemap_values = {}
        for protein in sitemap_data:
            all_sitemap_values[protein] = {}
            for ligand in sitemap_data[protein]:

                if verbose:
                    message = f'\t{protein} {ligand}: '
                    if read_from_csv[protein][ligand]:
                        csv_file = sitemap_data_folder+'/'+protein+self.separator+ligand+'.csv'
                        message += f'data read from CSV: {csv_file}'
                    else:
                        ligand_data = self.getProteinAndLigandData(protein, ligand)
                        message += f'{counts[protein][ligand]} of {ligand_data.shape[0]} points '
                        message += 'have sitemap data. '
                        message += '(%.2f%%)' % (100*(counts[protein][ligand]/ligand_data.shape[0]))
                    print(message)

                all_sitemap_values[protein][ligand] = {}
                smd = sitemap_data[protein][ligand].to_dict()
                for s in smd:
                    for index in smd[s]:
                        all_sitemap_values[protein][ligand].setdefault(index, [])
                        all_sitemap_values[protein][ligand][index].append(smd[s][index])

        # Compile data to merge with self.data
        vectors = []
        for index in self.data.index:
            protein, ligand = index[:2]
            if protein not in all_sitemap_values or ligand not in all_sitemap_values[protein]:
                vectors.append([np.nan]*11)
            elif index[2:-1] not in all_sitemap_values[protein][ligand]:
                vectors.append([np.nan]*11)
            elif index[2:-1] in all_sitemap_values[protein][ligand]:
                vectors.append(all_sitemap_values[protein][ligand][index[2:-1]])
            else:
                raise ValueError('Check this loop.')

        # Merge with self.data
        vectors = np.array(vectors)
        for i, s in zip(range(vectors.shape[1]), sitemap_scores):
            self.data[s] = vectors[:,i]

        sitemap_data = self.data[~self.data['volume'].isna()]

        return sitemap_data

    def _readSiteMapData(self):
        """
        Read sitemap data in pele_data_folder.
        """

        sitemap_data_folder = self.data_folder+'/sitemap_data'
        sitemap_data = {}
        for f in os.listdir(sitemap_data_folder):
            if f.endswith('.csv'):
                protein, ligand = f.split(self.separator)
                ligand = ligand.replace('.csv', '')
                sitemap_data.setdefault(protein, {})
                sitemap_data[protein][ligand] = pd.read_csv(sitemap_data_folder+'/'+f)
                indexes = ['Epoch', 'Trajectory', 'Accepted Pele Steps']
                sitemap_data[protein][ligand] = sitemap_data[protein][ligand].set_index(indexes)
                scores = list(sitemap_data[protein][ligand].keys())

        # Get mapping of all sitemap data
        all_sitemap_values = {}
        for protein in sitemap_data:
            all_sitemap_values[protein] = {}
            for ligand in sitemap_data[protein]:
                all_sitemap_values[protein][ligand] = {}
                smd = sitemap_data[protein][ligand].to_dict()
                for s in smd:
                    for index in smd[s]:
                        all_sitemap_values[protein][ligand].setdefault(index, [])
                        all_sitemap_values[protein][ligand][index].append(smd[s][index])

        # Compile data to merge with self.data
        vectors = []
        for index in self.data.index:
            protein, ligand = index[:2]
            if protein not in all_sitemap_values or ligand not in all_sitemap_values[protein]:
                vectors.append([np.nan]*11)
            elif index[2:-1] in all_sitemap_values[protein][ligand]:
                vectors.append(all_sitemap_values[protein][ligand][index[2:-1]])
            else:
                vectors.append([np.nan]*11)

        # Merge with self.data
        vectors = np.array(vectors)
        for i, s in zip(range(vectors.shape[1]), scores):
            self.data[s] = vectors[:,i]

        return self.data

    def _saveDataState(self, equilibration=False, individually=False, only_proteins=None, only_ligands=None):
        """
        Save the data state of all protein and ligand into the CSV files.

        Parameters
        ==========
        equilibration : bool
            Save the equilibration data instead?
        individually : bool
            Save the DataFrames individually by protein and ligand combination?
        only_proteins : (str, list)
            Only save individual data for the given proteins.
        only_ligands : (str, list)
            Only save individual data for the given ligands.
        """

        if only_proteins != None:
            if not individually:
                raise ValueError('only_proteins only works for saving data individually.')
            if isinstance(only_proteins, str):
                only_proteins = [only_proteins]
            if not isinstance(only_proteins, list):
                raise ValueError('only_proteins should a be a string or a list.')

        if only_ligands != None:
            if not individually:
                raise ValueError('only_ligands only works for saving data points individually.')
            if isinstance(only_ligands, str):
                only_ligands = [only_ligands]
            if not isinstance(only_ligands, list):
                raise ValueError('only_ligands should a be a string or a list.')

        if individually:
            for protein, ligand in self.pele_combinations:

                # Skip proteins not given in only_proteins
                if only_proteins != None:
                    if protein not in only_proteins:
                        continue

                # Skip ligands not given in only_ligands
                if only_ligands != None:
                    if ligand not in only_ligands:
                        continue

                ligand_data = self.getProteinAndLigandData(protein, ligand, equilibration=equilibration)
                if equilibration:
                    data_file = self.data_folder+'/equilibration_data_'+protein+self.separator+ligand+'.csv'
                else:
                    data_file = self.data_folder+'/data_'+protein+self.separator+ligand+'.csv'
                ligand_data.to_csv(data_file)

        else:
            if equilibration:
                self.equilibration_data.to_csv(self.data_folder+'/equilibration_data.csv')
            else:
                self.data.to_csv(self.data_folder+'/data.csv')

    def _recoverDataState(self, remove=False, equilibration=False):
        if equilibration:
            csv_file = self.data_folder+'/equilibration_data.csv'
            data = self.equilibration_data
        else:
            csv_file = self.data_folder+'/data.csv'
            data = self.data

        if os.path.exists(csv_file):
            # Read CSV file
            data = pd.read_csv(csv_file)

            # Convert protein and ligand columns to strings
            data = data.astype({'Protein':'string'})
            data = data.astype({'Ligand':'string'})

            # Set indexes
            data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps', 'Step'], inplace=True)
            data = data.loc[:, ~data.columns.str.contains('^Unnamed')]

            if remove:
                os.remove(csv_file)

        if equilibration:
            self.equilibration_data = data
        else:
            self.data = data

    def getDataSeries(self, data, value, column):
        return data[data.index.get_level_values(column) == value]

    def _saveDictionaryAsJson(self, dictionary, output_file):
        with open(output_file, 'w') as of:
            json.dump(dictionary, of)

    def _loadDictionaryFromJson(self, json_file):
        with open(json_file) as jf:
            dictionary = json.load(jf)
        return dictionary

    def _readPDB(self, name, pdb_file):
        """
        Read PDB file to a structure object
        """
        parser = PDB.PDBParser()
        structure = parser.get_structure(name, pdb_file)
        return structure

    def _saveStructureToPDB(self, structure, output_file, remove_hydrogens=False,
                            remove_water=False, only_protein=False, keep_residues=[]):
        """
        Saves a structure into a PDB file

        Parameters
        ----------
        structure : list or Bio.PDB.Structure
            Structure to save
        remove_hydrogens : bool
            Remove hydrogen atoms from model?
        remove_water : bool
            Remove water residues from model?
        only_protein : bool
            Remove everything but the protein atoms?
        keep_residues : list
            List of residue indexes to keep when using the only_protein selector.
        """

        io = PDB.PDBIO()
        io.set_structure(structure)

        selector = None
        if remove_hydrogens:
            selector = _atom_selectors.notHydrogen()
        elif remove_water:
            selector = _atom_selectors.notWater()
        elif only_protein:
            selector = _atom_selectors.onlyProtein(keep_residues=keep_residues)
        if selector != None:
            io.save(output_file, selector)
        else:
            io.save(output_file)

    def _getInputPDB(self, protein, ligand):
        """
        Returns the input PDB for the PELE simulation.
        """

        # If PELE input folder is not found return None
        if protein not in self.pele_directories or ligand not in self.pele_directories[protein]:
            if protein not in self.topology_files or ligand not in self.topology_files[protein]:
                return
            else:
                return self.topology_files[protein][ligand]

        # Load input PDB with Bio.PDB and mdtraj
        pele_dir = self.pele_directories[protein][ligand]
        folder = pele_dir+'/'+self.pele_output_folder+'/input/'
        for d in os.listdir(folder):
            if d.endswith('processed.pdb'):
                return folder+'/'+d

    def _getInputLigandPDB(self, protein, ligand):
        """
        Returns the input PDB for the PELE simulation.
        """

        # If PELE input folder is not found return None
        if protein not in self.pele_directories or ligand not in self.pele_directories[protein]:
            if protein not in self.ligand_files or ligand not in self.ligand_files[protein]:
                return
            else:
                return self.ligand_files[protein][ligand]

        # Load input PDB with Bio.PDB and mdtraj
        pele_dir = self.pele_directories[protein][ligand]
        folder = pele_dir+'/'+self.pele_output_folder+'/input/'
        for d in os.listdir(folder):
            if d.endswith('ligand.pdb'):
                return folder+'/'+d

    def _atomStringToTuple(self, atom_string):
        """
        Reads a PELE platform atom string and outputs a 3-element tuple version.
        """
        index = ''
        name = ''
        index_done = False
        for i,s in enumerate(atom_string):
            if i == 0:
                chain = s
            elif s.isdigit() and not index_done:
                index += s
            elif not s.isdigit():
                index_done = True
                name += s
            elif index_done:
                name += s

        return (chain, index, name)

    def _checkSeparator(self, folder):
        # Check for separator
        if self.separator not in folder:
            raise ValueError('Separator %s not found in PELE folder names' % self.separator)
        if folder.count(self.separator) > 1:
            raise ValueError('Separator %s appears more than once in the PELE folder %s' % (self.separator, folder))

    def _checkDataFolder(self, trajectories=True):
        """
        Check protein and ligand information contained in the CSV files of the
        PELE data folder.
        """
        if os.path.exists(self.data_folder):

            # Iterate over CSV files and get protein and ligand names
            for d in os.listdir(self.data_folder):
                if d.endswith('.csv') and d.startswith('data_'):
                    protein = d.split(self.separator)[-2].replace('data_','')
                    ligand = d.split(self.separator)[-1].replace('.csv','')
                    if protein not in self.csv_files:
                        self.csv_files[protein] = {}
                    if ligand not in self.csv_files[protein]:
                        self.csv_files[protein][ligand] = self.data_folder+'/'+d

                    # Add protein and ligand to attribute lists
                    if protein not in self.proteins:
                        self.proteins.append(protein)
                    if ligand not in self.ligands:
                        self.ligands.append(ligand)# Create PELE input folders

                if d.endswith('.csv') and d.startswith('equilibration_data_'):
                    protein = d.split(self.separator)[-2].replace('equilibration_data_','')
                    ligand = d.split(self.separator)[-1].replace('.csv','')
                    if protein not in self.csv_equilibration_files:
                        self.csv_equilibration_files[protein] = {}
                    if ligand not in self.csv_equilibration_files[protein]:
                        self.csv_equilibration_files[protein][ligand] = self.data_folder+'/'+d

                    # Add protein and ligand to attribute lists
                    if protein not in self.proteins:
                        self.proteins.append(protein)
                    if ligand not in self.ligands:
                        self.ligands.append(ligand)# Create PELE input folders

            # Read trajectories if found
            traj_dir = self.data_folder+'/pele_trajectories'
            if os.path.exists(traj_dir):
                for d in os.listdir(traj_dir):
                    protein = d.split(self.separator)[-2]
                    ligand = d.split(self.separator)[-1]
                    if protein not in self.trajectory_files:
                        self.trajectory_files[protein] = {}
                    if protein not in self.equilibration['trajectory']:
                        self.equilibration['trajectory'][protein] = {}
                    self.trajectory_files[protein][ligand] = pele_read.getTrajectoryFiles(traj_dir+'/'+d)
                    self.equilibration['trajectory'][protein][ligand] = pele_read.getEquilibrationTrajectoryFiles(traj_dir+'/'+d)

            # Read PELE topology if found
            topology_dir = self.data_folder+'/pele_topologies'
            for d in os.listdir(topology_dir):
                protein = d.split(self.separator)[-2].replace('.pdb', '')
                ligand = d.split(self.separator)[-1]
                if protein not in self.topology_files:
                    self.topology_files[protein] = {}
                for f in os.listdir(topology_dir+'/'+d):
                    if f.endswith('.pdb'):
                        self.topology_files[protein][ligand] = topology_dir+'/'+d+'/'+f

            # Read conects if found
            conects_dir = self.data_folder+'/pele_conects'
            if not os.path.exists(conects_dir):
                os.mkdir(conects_dir)
            for d in os.listdir(conects_dir):
                protein = d.split(self.separator)[-2].replace('.json', '')
                ligand = d.split(self.separator)[-1]
                if protein not in self.conect_files:
                    self.conect_files[protein] = {}
                for f in os.listdir(conects_dir+'/'+d):
                    if f.endswith('.json'):
                        self.conect_files[protein][ligand] = conects_dir+'/'+d+'/'+f

        # Create analysis folder
        if not os.path.exists(self.data_folder):
            os.mkdir(self.data_folder)
        # Create PELE input folders
        if not os.path.exists(self.data_folder+'/pele_inputs'):
            os.mkdir(self.data_folder+'/pele_inputs')
        # Create PELE configuration folders
        if not os.path.exists(self.data_folder+'/pele_configuration'):
            os.mkdir(self.data_folder+'/pele_configuration')
        # Create PELE input folders
        if not os.path.exists(self.data_folder+'/pele_trajectories'):
            if trajectories:
                os.mkdir(self.data_folder+'/pele_trajectories')

        # Create PELE toplogies folder
        if not os.path.exists(self.data_folder+'/pele_topologies'):
            os.mkdir(self.data_folder+'/pele_topologies')

        # Create PELE conects folder
        if not os.path.exists(self.data_folder+'/pele_conects'):
            os.mkdir(self.data_folder+'/pele_conects')

        # Create PELE conects folder
        if not os.path.exists(self.data_folder+'/pele_spawnings'):
            os.mkdir(self.data_folder+'/pele_spawnings')

        # Create PELE ligands folders
        if not os.path.exists(self.data_folder+'/pele_ligands'):
            os.mkdir(self.data_folder+'/pele_ligands')

        # Create PELE distances
        if not os.path.exists(self.data_folder+'/distances'):
            os.mkdir(self.data_folder+'/distances')

        # Create PELE angles
        if not os.path.exists(self.data_folder+'/angles'):
            os.mkdir(self.data_folder+'/angles')

        # Create PELE non bonded energy
        if not os.path.exists(self.data_folder+'/nonbonded_energy'):
            os.mkdir(self.data_folder+'/nonbonded_energy')

    def _getProteinLigandCombinations(self):
        """
        Get existing protein and ligand combinations from the analysis or PELE folders.
        """
        # Gather protein and ligand combinations
        pele_combinations = []
        for protein in sorted(self.report_files):
            for ligand in sorted(self.report_files[protein]):
                if (protein, ligand) not in pele_combinations:
                    pele_combinations.append((protein, ligand))

        for protein in sorted(self.csv_files):
            for ligand in sorted(self.csv_files[protein]):
                if (protein, ligand) not in pele_combinations:
                    pele_combinations.append((protein, ligand))

        return pele_combinations

    def _readReportData(self, equilibration=False, energy_by_residue_type=None):
        """
        Read report data from PELE simulation report files.
        """

        if equilibration:
            reports_dict = self.equilibration['report']
        else:
            reports_dict = self.report_files

        # Iterate PELE folders to read report files
        report_data = []
        remove = [] # Put protein and ligands to remove them
        for protein, ligand in self.pele_combinations:

            # Check whether protein and ligand report files are present
            if protein not in reports_dict or ligand not in reports_dict[protein]:
                report_files = None
            else:
                report_files = reports_dict[protein][ligand]

            if report_files == None and equilibration and self.csv_equilibration_files == {}:
                print('WARNING: No equilibration data found for simulation %s-%s' % (protein, ligand))
                continue

            if self.verbose:
                print('\t'+protein+self.separator+ligand, end=' ')
                start = time.time()

            # Read report files into panda dataframes
            data, distance_data, angle_data, nonbonded_energy_data  = pele_read.readReportFiles(report_files,
                                                                                                protein,
                                                                                                ligand,
                                                                                                ebr_threshold=0.1,
                                                                                                separator=self.separator,
                                                                                                equilibration=equilibration,
                                                                                                data_folder_name=self.data_folder)

            # Skip of dataframe is None
            if isinstance(data, type(None)):
                remove.append((protein, ligand)) # Add protein and ligand to remove them
                continue

            # Check which dataframe columns are to be kept
            if not equilibration:
                keep = [k for k in data.keys() if not k.startswith('L:1_')]
                if self.energy_by_residue:
                    keep += [k for k in data.keys() if k.startswith('L:1_') and k.endswith(energy_by_residue_type)]
                data = data[keep]

            report_data.append(data)

            if self.verbose:
                print('\t in %.2f seconds.' % (time.time()-start))

            # Save distance, angles and nonbonded energy data
            if not equilibration:
                self.distances.setdefault(protein,{})
                self.distances[protein][ligand] = distance_data

                self.angles.setdefault(protein,{})
                self.angles[protein][ligand] = angle_data

                if not isinstance(distance_data, type(None)) and not distance_data.empty:

                    # Define a different distance output file for each pele run
                    distance_file = self.data_folder+'/distances/'+protein+self.separator+ligand+'.csv'

                    # Save distances to CSV file
                    self.distances[protein][ligand].to_csv(distance_file)

                if not isinstance(angle_data, type(None)) and not angle_data.empty:

                    # Define a different angle output file for each pele run
                    angle_file = self.data_folder+'/angles/'+protein+self.separator+ligand+'.csv'

                    # Save angles to CSV file
                    self.angles[protein][ligand].to_csv(angle_file)

                self.nonbonded_energy.setdefault(protein,{})
                self.nonbonded_energy[protein][ligand] = nonbonded_energy_data

                if not isinstance(nonbonded_energy_data, type(None)):

                    # Define a different distance output file for each pele run
                    nonbonded_energy_file = self.data_folder+'/nonbonded_energy/'+protein+self.separator+ligand+'.csv'

                    # Save distances to CSV file
                    self.nonbonded_energy[protein][ligand].to_csv(nonbonded_energy_file)

        if report_data == [] and equilibration:
            self.equilibration_data = None
            return

        if equilibration:
            # Merge all dataframes into a single dataframe
            self.equilibration_data = pd.concat(report_data)
            # self.equilibration_data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps', 'Step'], inplace=True)
            self._saveDataState(equilibration=equilibration)
            self.equilibration_data = None
            gc.collect()
            self._recoverDataState(remove=True, equilibration=equilibration)
        else:
            # Merge all dataframes into a single dataframe
            self.data = pd.concat(report_data)
            # self.data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps', 'Step'], inplace=True)
            # Save and reload dataframe to avoid memory fragmentation
            self._saveDataState()
            self.data = None
            gc.collect()
            self._recoverDataState(remove=True)

            # Remove protein and ligands from pele combinations
            for protein, ligand in remove:
                self.pele_combinations.pop(self.pele_combinations.index((protein, ligand)))

    def _readGlobalReportData(self,overwrite=False):

        for protein, ligand in self.pele_combinations:
            df = pd.read_csv(self.data_folder+'/data_'+protein+self.separator+ligand+'.csv')
            if 'Cluster' not in df:
                df_glo = pd.read_csv(self.pele_directories[protein][ligand]+'/output/'+'data.csv')
                df = df.join(df_glo['Cluster'], how='right')
                df.to_csv(self.data_folder+'/data_'+protein+self.separator+ligand+'.csv')

        #    shutil.move(self.pele_directories[protein][ligand]+'/output/'+'data.csv',self.data_folder+'/data_'+protein+'_'+ligand+'.csv')

    def _copyPELEInputs(self, overwrite=False):
        """
        Copy PELE input files to analysis folder for easy setup PELE reruns.
        """
        for protein in self.pele_directories:
            for ligand in self.pele_directories[protein]:
                dir = self.data_folder+'/pele_inputs/'+protein+self.separator+ligand
                if not os.path.exists(dir):
                    os.mkdir(dir)
                for f in os.listdir(self.pele_directories[protein][ligand]):
                    dest = self.data_folder+'/pele_inputs/'+protein+self.separator+ligand+'/'+f
                    if os.path.exists(dest) and not overwrite:
                        continue
                    orig = self.pele_directories[protein][ligand]+'/'+f
                    if f.endswith('.pdb'):
                        shutil.copyfile(orig, dest)
                    elif f.endswith('.yaml'):
                        shutil.copyfile(orig, dest)

    def _copyPELEConfiguration(self, overwrite=False):

        for protein in self.pele_directories:
            for ligand in self.pele_directories[protein]:

                output_dir = self.pele_directories[protein][ligand]+'/'+self.pele_output_folder+'/output'

                # Check if output folder exists
                if not os.path.exists(output_dir):
                    print('Output folder not found for %s%s%s PELE calculation.' % (protein, self.separator, ligand))
                    continue

                dir = self.data_folder+'/pele_configuration/'+protein+self.separator+ligand
                if not os.path.exists(dir):
                    os.mkdir(dir)
                for f in os.listdir(self.pele_directories[protein][ligand]+'/'+self.pele_output_folder):
                    dest = self.data_folder+'/pele_configuration/'+protein+self.separator+ligand+'/'+f
                    if os.path.exists(dest) and not overwrite:
                        continue
                    orig = self.pele_directories[protein][ligand]+'/'+self.pele_output_folder+'/'+f
                    if f.endswith('.conf'):
                        shutil.copyfile(orig, dest)

    def _copyPELETopology(self, overwrite=False):
        """
        Copy topology files to analysis folder.
        """
        for protein in self.topology_files:
            for ligand in self.topology_files[protein]:

                # Copy topologies
                dir = self.data_folder+'/pele_topologies/'+protein+self.separator+ligand
                if not os.path.exists(dir):
                    os.mkdir(dir)
                dest = dir+'/'+protein+self.separator+ligand+'.pdb'

                # Skip found in pele data folder
                if os.path.exists(dest) and not overwrite:
                    # self.topology_files.setdefault(protein, {})
                    # self.topology_files[protein][ligand] = dest
                    continue

                # Skip if not found in pele folder
                if protein not in self.topology_files:
                    if os.path.exists(dest):
                        self.topology_files.setdefault(protein, {})
                        self.topology_files[protein][ligand] = dest
                    continue
                elif ligand not in self.topology_files[protein]:
                    if os.path.exists(dest):
                        self.topology_files[protein][ligand] = dest
                    continue

                orig = self.topology_files[protein][ligand]
                if orig != dest:
                    shutil.copyfile(orig, dest)
                    self.topology_files[protein][ligand] = dest

            for ligand in self.topology_files[protein]:
                # Copy ligands (only one file per ligand)
                dir = self.data_folder+'/pele_ligands/'+ligand
                if not os.path.exists(dir):
                    os.mkdir(dir)
                dest = dir+'/'+ligand+'.pdb'

                if os.path.exists(dest) and not overwrite:
                    self.ligand_files.setdefault(protein, {})
                    self.ligand_files[protein][ligand] = dest
                    continue

                # Skip if not found in pele folder
                if protein not in self.ligand_files:
                    if os.path.exists(dest):
                        self.ligand_files.setdefault(protein, {})
                        self.ligand_files[protein][ligand] = dest
                    continue

                elif ligand not in self.ligand_files[protein]:
                    if os.path.exists(dest):
                        self.ligand_files[protein][ligand] = dest
                    continue

                orig = self.ligand_files[protein][ligand]
                if orig != dest:
                    shutil.copyfile(orig, dest)
                    self.ligand_files[protein][ligand] = dest

    def _copyPELETrajectories(self, overwrite=False, remove_original_trajectory=False):
        """
        Copy PELE output trajectories to analysis folder.
        """

        if not os.path.exists(self.data_folder+'/pele_trajectories'):
            os.mkdir(self.data_folder+'/pele_trajectories')

        # Copy PELE trajectories
        for protein in self.trajectory_files:
            for ligand in self.trajectory_files[protein]:
                dir = self.data_folder+'/pele_trajectories/'+protein+self.separator+ligand
                if not os.path.exists(dir):
                    os.mkdir(dir)
                for epoch in self.trajectory_files[protein][ligand]:
                    epoch_folder = dir+'/'+str(epoch)
                    if not os.path.exists(epoch_folder):
                        os.mkdir(epoch_folder)
                    for traj in self.trajectory_files[protein][ligand][epoch]:
                        orig = self.trajectory_files[protein][ligand][epoch][traj]
                        dest = epoch_folder+'/'+orig.split('/')[-1]
                        if os.path.exists(dest) and not overwrite:
                            if remove_original_trajectory:
                                if self.pele_folder+'/' in orig: #Only remove pele folder (original)
                                    os.remove(orig)
                            continue
                        elif orig != dest: # Copy only they are not found in the analysis folder
                            shutil.copyfile(orig, dest)
                            self.trajectory_files[protein][ligand][epoch][traj] = dest
                            if remove_original_trajectory:
                                if self.pele_folder+'/' in orig: #Only remove pele folder (original)
                                    os.remove(orig)

        # Copy equilibration PELE trajectories
        for protein in self.equilibration['trajectory']:
            for ligand in self.equilibration['trajectory'][protein]:
                dir = self.data_folder+'/pele_trajectories/'+protein+self.separator+ligand
                if not os.path.exists(dir):
                    os.mkdir(dir)
                for epoch in self.equilibration['trajectory'][protein][ligand]:
                    epoch_folder = dir+'/equilibration_'+str(epoch)
                    if not os.path.exists(epoch_folder):
                        os.mkdir(epoch_folder)
                    for traj in self.equilibration['trajectory'][protein][ligand][epoch]:
                        orig = self.equilibration['trajectory'][protein][ligand][epoch][traj]
                        dest = epoch_folder+'/'+orig.split('/')[-1]
                        if os.path.exists(dest) and not overwrite:
                            continue
                        if orig != dest: # Copy only they are not found in the analysis folder
                            shutil.copyfile(orig, dest)
                            self.equilibration['trajectory'][protein][ligand][epoch][traj] = dest

    def _checkPELEFolder(self):
        """
        Check for the paths to files in the PELE folder.
        """

        # Check if PELE folder exists
        if os.path.isdir(self.pele_folder):
            if self.verbose:
                print('Getting paths to PELE files')
        else:
            print('Pele directory %s not found. Checking %s folder...' % (self.pele_folder, self.data_folder))
            return

        # Check pele_folder for PELE runs.
        remove = [] # Put protein and ligand here for their removal
        for d in os.listdir(self.pele_folder):
            if os.path.isdir(self.pele_folder+'/'+d):

                self._checkSeparator(d)

                # set pele folder dir
                pele_dir = self.pele_folder+'/'+d

                # Get protein and ligand name based on separator
                protein = d.split(self.separator)[0]
                ligand = d.split(self.separator)[1]

                # Add protein name ligand names to dictionaries
                if protein not in self.pele_directories:
                    self.pele_directories[protein] = {}
                if protein not in self.report_files:
                    self.report_files[protein] = {}
                if protein not in self.report_files:
                    self.report_files[protein] = {}
                if protein not in self.trajectory_files:
                    self.trajectory_files[protein] = {}
                if protein not in self.topology_files:
                    self.topology_files[protein] = {}
                if protein not in self.spawning_files:
                    self.spawning_files[protein] = {}
                if protein not in self.fixed_files:
                    self.fixed_files[protein] = {}
                if protein not in self.ligand_files:
                    self.ligand_files[protein] = {}
                if protein not in self.equilibration['report']:
                    self.equilibration['report'][protein] = {}
                if protein not in self.equilibration['trajectory']:
                    self.equilibration['trajectory'][protein] = {}
                self.pele_directories[protein][ligand] = pele_dir

                # Set names for output and input folders
                output_dir = pele_dir+'/'+self.pele_output_folder+'/output'
                input_dir = pele_dir+'/'+self.pele_output_folder+'/input'

                # Check for spawning dictionaries
                self.spawning_files[protein][ligand] = pele_read.getSpawningDictionaries(pele_dir)

                # Check if output folder exists
                if not os.path.exists(output_dir):
                    print('Output folder not found for %s-%s PELE calculation.' % (protein, ligand))
                    remove.append((protein, ligand)) # Append protein and ligand for removal
                    continue

                # Get paths to PELE folders
                else:

                    self.report_files[protein][ligand] = pele_read.getReportFiles(output_dir)
                    self.equilibration['report'][protein][ligand] = pele_read.getEquilibrationReportFiles(output_dir)

                    # Read trajectory files if not in analysis folder
                    trajectories = pele_read.getTrajectoryFiles(output_dir)
                    if trajectories != {}:
                        self.trajectory_files[protein][ligand] = trajectories
                    trajectories = pele_read.getEquilibrationTrajectoryFiles(output_dir)
                    if trajectories != {}:
                        self.equilibration['trajectory'][protein][ligand] = trajectories

                # Check if input folder exists
                if not os.path.exists(input_dir):
                    print('PELE input folder not found for %s-%s PELE calculation.' % (protein, ligand))
                    continue
                # Get path to topology file
                else:
                    self.topology_files[protein][ligand] = pele_read.getTopologyFile(input_dir)
                    self.fixed_files[protein][ligand] = pele_read.getFixedFile(input_dir)
                    self.ligand_files[protein][ligand] = pele_read.getLigandFile(input_dir)

                # Add protein and ligand to attribute lists
                if protein not in self.proteins:
                    self.proteins.append(protein)
                if ligand not in self.ligands:
                    self.ligands.append(ligand)

        # Sort protein and ligands for easy iteration
        self.proteins = sorted(self.proteins)
        self.ligands = sorted(self.ligands)

        # Remove protein and ligand from paths
        for protein, ligand in remove:
            self.pele_directories[protein].pop(ligand)
            if self.pele_directories[protein] == {}:
                self.pele_directories.pop(protein)
            if protein in self.report_files  and self.report_files[protein] == {}:
                self.report_files.pop(protein)

    def _setChainIDs(self, change_water_names=False):

        # Crea Bio PDB parser and io
        parser = PDB.PDBParser()
        io = PDB.PDBIO()

        # Read complex chain ids
        self.structure = {}
        self.conects = {}
        self.ligand_structure = {}
        self.md_topology = {}

        # Check that chain ids match the number of proteins
        if os.path.exists(self.data_folder+'/chains_ids.json'):
            chain_ids = self._loadDictionaryFromJson(self.data_folder+'/chains_ids.json')
            if len(chain_ids) != len(self.proteins):
                # print(len(chain_ids), len(self.proteins))
                os.remove(self.data_folder+'/chains_ids.json')

        # Check that atom indexes match the number of proteins
        if os.path.exists(self.data_folder+'/atom_indexes.json'):
            atom_indexes = self._loadDictionaryFromJson(self.data_folder+'/atom_indexes.json')
            if len(atom_indexes) != len(self.proteins):
                # print(len(atom_indexes), len(self.proteins))
                os.remove(self.data_folder+'/atom_indexes.json')

        if not os.path.exists(self.data_folder+'/chains_ids.json') or not os.path.exists(self.data_folder+'/atom_indexes.json') or self.force_reading:
            for protein in self.proteins:
                self.structure.setdefault(protein, {})
                self.conects.setdefault(protein, {})
                self.ligand_structure.setdefault(protein, {})
                self.md_topology.setdefault(protein, {})
                self.chain_ids.setdefault(protein, {})
                self.atom_indexes.setdefault(protein, {})

                for ligand in self.ligands:
                    self.chain_ids[protein].setdefault(ligand, {})
                    self.atom_indexes[protein].setdefault(ligand, {})

                    # Load input PDB with Bio.PDB and mdtraj
                    input_pdb = self._getInputPDB(protein, ligand)

                    # Skip combinations with missing input PDB
                    if not input_pdb:
                        continue

                    self.structure[protein][ligand] = parser.get_structure(protein, input_pdb)

                    if protein in self.fixed_files and ligand in self.fixed_files[protein]:
                        fixed_pdb = self.fixed_files[protein][ligand]
                        self.conects[protein][ligand] = conectLines._readPDBConectLines(fixed_pdb,
                                                                                        change_water=change_water_names)
                        conect_folder =  conects_dir = self.data_folder+'/pele_conects/'
                        conect_folder += protein+self.separator+ligand+'/'
                        if not os.path.exists(conect_folder):
                            os.mkdir(conect_folder)
                        conect_file = conect_folder+protein+self.separator+ligand+'.json'
                        self._saveDictionaryAsJson(self.conects[protein][ligand], conect_file)

                    input_ligand_pdb = self._getInputLigandPDB(protein, ligand)
                    self.ligand_structure[protein][ligand] = parser.get_structure(protein, input_ligand_pdb)

                    # Add ligand three letter code ligand_names
                    if protein not in self.ligand_names or ligand not in self.ligand_names[protein]:
                        self.ligand_names.setdefault(protein, {})
                        for residue in self.ligand_structure[protein][ligand].get_residues():
                            self.ligand_names[protein][ligand] = residue.resname

                    # Read topology mdtraj trajectory object
                    self.md_topology[protein][ligand] = md.load(input_pdb)

                    # Match the MDTraj chain with the PDB chain id
                    biopdb_residues = [r for r in self.structure[protein][ligand].get_residues()]
                    mdtrj_residues = [r for r in self.md_topology[protein][ligand].topology.residues]
                    for r_pdb, r_md in zip(biopdb_residues, mdtrj_residues):
                        chain_pdb = r_pdb.get_parent().id
                        chain_md = r_md.chain.index
                        self.chain_ids[protein][ligand][chain_md] = chain_pdb

                        # Get a dictionary mapping (chain, residue, atom) to traj atom_index
                        biopdb_atoms = [a for a in r_pdb]
                        mdtrj_atoms = [a for a in r_md.atoms]
                        for a_pdb, a_md in zip(biopdb_atoms, mdtrj_atoms):
                            # Check water residues and remove final W
                            if r_pdb.id[0] == 'W':
                                atom_name = a_pdb.name[:-1]
                            else:
                                atom_name = a_pdb.name
                            # Join tuple as a string to save as json.
                            atom_map = '-'.join([r_pdb.get_parent().id, str(r_pdb.id[1]), atom_name])
                            self.atom_indexes[protein][ligand][atom_map] = a_md.index

            self._saveDictionaryAsJson(self.chain_ids, self.data_folder+'/chains_ids.json')
            self._saveDictionaryAsJson(self.atom_indexes, self.data_folder+'/atom_indexes.json')

            # Recover atom_indexes tuple status
            atom_indexes = {}
            for protein in self.atom_indexes:
                atom_indexes[protein] = {}
                for ligand in self.atom_indexes[protein]:
                    atom_indexes[protein][ligand] = {}
                    for am in self.atom_indexes[protein][ligand]:
                        ams = am.split('-')
                        atom_indexes[protein][ligand][(ams[0], int(ams[1]), ams[2])] = self.atom_indexes[protein][ligand][am]
            self.atom_indexes = atom_indexes
        else:
            self.chain_ids = self._loadDictionaryFromJson(self.data_folder+'/chains_ids.json')
            # Recover chain as integer in the dictionary
            chain_ids = {}
            for protein in self.chain_ids:
                self.structure[protein] = {}
                self.ligand_structure[protein] = {}
                self.md_topology[protein] = {}
                chain_ids[protein] = {}
                for ligand in self.chain_ids[protein]:
                    chain_ids[protein][ligand] = {}
                    input_pdb = self._getInputPDB(protein, ligand)

                    # If input file is not found continue
                    if input_pdb == None:
                        continue

                    self.structure[protein][ligand] = parser.get_structure(protein, input_pdb)
                    input_ligand_pdb = self._getInputLigandPDB(protein, ligand)

                    if input_ligand_pdb:
                        self.ligand_structure[protein][ligand] = parser.get_structure(protein, input_ligand_pdb)

                    # Add ligand three letter code ligand_names
                    if (protein not in self.ligand_names or ligand not in self.ligand_names[protein]) and ligand in self.ligand_structure[protein]:
                        self.ligand_names.setdefault(protein, {})
                        for residue in self.ligand_structure[protein][ligand].get_residues():
                            self.ligand_names[protein][ligand] = residue.resname

                    # Read topology mdtraj trajectory object
                    self.md_topology[protein][ligand] = md.load(input_pdb)
                    for chain in self.chain_ids[protein][ligand]:
                        chain_ids[protein][ligand][int(chain)] = self.chain_ids[protein][ligand][chain]

                    # Read conect lines
                    if protein in self.conect_files and ligand in self.conect_files[protein]:
                        self.conects.setdefault(protein, {})
                        self.conects[protein][ligand] = self._loadDictionaryFromJson(self.conect_files[protein][ligand])

                        # Convert json lists to tuples for hashing
                        conects = []
                        for cl in self.conects[protein][ligand]:
                            nc = []
                            for atom in cl:
                                nc.append(tuple(atom))
                            conects.append(nc)
                        self.conects[protein][ligand] = conects

            self.chain_ids = chain_ids

            self.atom_indexes = self._loadDictionaryFromJson(self.data_folder+'/atom_indexes.json')
            # Recover atom_indexes tuple status
            atom_indexes = {}
            for protein in self.atom_indexes:
                atom_indexes[protein] = {}
                for ligand in self.atom_indexes[protein]:
                    atom_indexes[protein][ligand] = {}
                    for am in self.atom_indexes[protein][ligand]:
                        ams = am.split('-')
                        atom_indexes[protein][ligand][(ams[0], int(ams[1]), ams[2])] = self.atom_indexes[protein][ligand][am]
            self.atom_indexes = atom_indexes

    def _checkSpawningInformation(self):

        # Compute spawning membership
        spawning_values = []
        regional_values = []
        for protein, ligand in self.pele_combinations:

            ligand_data = self.getProteinAndLigandData(protein, ligand)

            if protein in self.spawning_files and ligand in self.spawning_files[protein]:

                with open(self.spawning_files[protein][ligand]['metrics']) as jf:
                    metrics = json.load(jf)

                with open(self.spawning_files[protein][ligand]['thresholds']) as jf:
                    metrics_thresholds = json.load(jf)

                with open(self.spawning_files[protein][ligand]['mappings']) as jf:
                    spawning_mapping = json.load(jf)

                # Add spawning information
                spawning_mapping = {int(e):spawning_mapping[e] for e in spawning_mapping}
                spawning_values += [spawning_mapping[e] for e in ligand_data.index.get_level_values('Epoch')]

                # Add regional membership
                metric_data = ligand_data.copy()
                acceptance = np.ones(metric_data.shape[0], dtype=bool)
                for metric in metrics:

                    # Skip undefined metrics
                    if metric not in metrics_thresholds:
                        continue

                    # Check how metrics will be combined
                    distances = False
                    angles = False
                    for x in metrics[metric]:
                        if 'distance_' in x:
                            distances = True
                        elif 'angle' in x:
                            angles = True

                    if distances and angles:
                        raise ValueError(f'Metric {metric} combines distances and angles which is not supported.')

                    # Combine metrics
                    if distances:
                        distances = self.distances[protein][ligand][metrics[metric]]
                        metric_data[metric] = distances.min(axis=1).tolist()

                    elif angles:
                        angles = self.angles[protein][ligand][metrics[metric]]
                        if len(metrics[metric]) > 1:
                            raise ValueError('Combining more than one angle into a metric is not currently supported.')
                        metric_data[metric] = angles.min(axis=1).tolist()

                    if isinstance(metrics_thresholds[metric], float):
                        acceptance = acceptance & ((metric_data[metric] <= metrics_thresholds[metric]).to_numpy())

                    elif isinstance(metrics_thresholds[metric], (list, tuple)):
                        acceptance = acceptance & ((metric_data[metric] >= metrics_thresholds[metric][0]).to_numpy())
                        acceptance = acceptance & ((metric_data[metric] <= metrics_thresholds[metric][1]).to_numpy())

                regional_values += list(acceptance)
            else:
                spawning_values += ligand_data.shape[0]*[None]
                regional_values += ligand_data.shape[0]*[None]

        if any(spawning_values):
            self.data['Regional spawning'] = spawning_values
            self.data['Regional membership'] = regional_values


class conectLines:

    def _readPDBConectLines(pdb_file, only_hetatoms=False, change_water=False):
        """
        Read PDB file and get conect lines only
        """

        # Get atom indexes by tuple and objects
        atoms = conectLines._getAtomIndexes(pdb_file, change_water=change_water)
        if only_hetatoms:
            atoms_objects = conectLines._getAtomIndexes(pdb_file, return_objects=True)
        conects = []

        # Read conect lines as dictionaries linking atoms
        with open(pdb_file) as pdbf:
            for l in pdbf:
                if l.startswith('CONECT'):
                    l = l.replace("CONECT", "")
                    l = l.strip("\n").rstrip()
                    num = len(l) / 5
                    new_l = [int(l[i * 5:(i * 5) + 5]) for i in range(int(num))]
                    if only_hetatoms:
                        het_atoms = [True if atoms_objects[int(x)].get_parent().id[0] != ' ' else False for x in new_l]
                        if True not in het_atoms:
                            continue
                    conects.append([atoms[int(x)] for x in new_l])

        return conects

    def _getAtomIndexes(pdb_file, invert=False, return_objects=False, change_water=False):

        def _get_atom_tuple(atom):

            return (atom.get_parent().get_parent().id,
                    atom.get_parent().id[1],
                    atom.name)

        def _readPDB(name, pdb_file):
            """
            Read PDB file to a structure object
            """
            parser = PDB.PDBParser()
            structure = parser.get_structure(name, pdb_file)
            return structure

        # Define atom name changes for water residues
        if change_water:
            water_name = {'O': 'OW', 'H1': '1HW', 'H2': '2HW'}

        # Read PDB file
        atom_indexes = {}
        with open(pdb_file, 'r') as f:
            for l in f:
                if l.startswith('ATOM') or l.startswith('HETATM'):
                    index, name, resname, chain, resid = (int(l[6:11]), l[12:16].strip(), l[17:20], l[21], int(l[22:26]))
                    if change_water and resname == 'HOH':
                        if name not in water_name.values():
                            name =  water_name[name]
                    atom_indexes[(chain, resid, name)] = index

        # Read structure
        structure = _readPDB('fixed', pdb_file)

        # Assign PDB indexes to each Bio.PDB atom
        atoms = {}
        for chain in structure[0]:
            for residue in chain:

                for atom in residue:

                    if change_water and residue.id[0] == 'W':
                        if name not in water_name.values():
                            atom.name = water_name[atom.name]

                    # Get atom PDB index
                    index = atom_indexes[(chain.id, residue.id[1], atom.name)]

                    # Return atom objects instead of tuples
                    if return_objects:
                        _atom = atom
                    else:
                        _atom = _get_atom_tuple(atom)

                    # Invert the returned dictionary
                    if invert:
                        atoms[_atom] = index
                    else:
                        atoms[index] = _atom
        return atoms

    def _writeConectLines(pdb_file, conects, atom_mapping=None, hydrogens=True):
        """
        Write stored conect lines for a particular model into the given PDB file.

        Parameters
        ==========
        model : str
            Model name
        pdb_file : str
            Path to PDB file to modify
        """

        def check_atom_in_atoms(atom, atoms, atom_mapping):

            if atom not in atoms and atom_mapping != None and atom in atom_mapping:
                if isinstance(atom_mapping[atom], str):
                    atom = (atom[0], atom[1], atom_mapping[atom])
                elif isinstance(atom_mapping[atom], tuple) and len(atom_mapping[atom]) == 3:
                    atom = atom_mapping[atom]

            if atom not in atoms:
                residue_atoms = ' '.join([ac[-1] for ac in atoms if atom[1] == ac[1]])
                message = "Conect atom %s not found in %s's topology\n\n" % (atom, pdb_file)
                message += "Topology's residue %s atom names: %s" % (atom[1], residue_atoms)
                raise ValueError(message)

            return atom

        # Get atom indexes map
        atoms = conectLines._getAtomIndexes(pdb_file, invert=True)

        # Check atoms not found in conects
        with open(pdb_file+'.tmp', 'w') as tmp:
            with open(pdb_file) as pdb:

                # write all lines but skip END line
                for line in pdb:
                    if not line.startswith('END'):
                        tmp.write(line)

                # Write new conect line mapping
                for entry in conects:
                    line = 'CONECT'
                    for x in entry:
                        if not hydrogens:
                            type_index = (x[2].find(next(filter(str.isalpha, x[2]))))
                            if x[2][type_index] != 'H':
                                x = check_atom_in_atoms(x, atoms, atom_mapping=atom_mapping)
                                line += '%5s' % atoms[x]
                        else:
                            x = check_atom_in_atoms(x, atoms, atom_mapping=atom_mapping)
                            line += '%5s' % atoms[x]

                    line += '\n'
                    tmp.write(line)
            tmp.write('END\n')
        shutil.move(pdb_file+'.tmp', pdb_file)

def _copyScriptFile(output_folder, script_name, no_py=False, subfolder=None, hidden=True):
    """
    Copy a script file from the prepare_proteins package.

    Parameters
    ==========

    """
    # Get script
    path = "pele_analysis/scripts"
    if subfolder != None:
        path = path+'/'+subfolder

    script_file = resource_stream(Requirement.parse("pele_analysis"),
                                     path+'/'+script_name)
    script_file = io.TextIOWrapper(script_file)

    # Write control script to output folder
    if no_py == True:
        script_name = script_name[:-3]

    if hidden:
        output_path = output_folder+'/._'+script_name
    else:
        output_path = output_folder+'/'+script_name

    with open(output_path, 'w') as sof:
        for l in script_file:
            sof.write(l)

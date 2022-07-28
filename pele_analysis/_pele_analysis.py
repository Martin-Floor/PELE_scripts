import os
from . import pele_read
from . import pele_trajectory
from . import clustering

import pandas as pd
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

from ipywidgets import interact, fixed, FloatSlider, IntSlider, FloatRangeSlider, VBox, HBox, interactive_output, Dropdown, Checkbox
import time
import random

class peleAnalysis:
    """
    Analyse multiple PELE calculations in batch. This class assumes that calculations
    were run using the PELE Platform to generate the output and that trajectories
    are in XTC format.

    Attributes
    ==========
    """

    def __init__(self, pele_folder, pele_output_folder='output', force_reading=False, separator='_',
                 verbose=False, energy_by_residue=False, ebr_threshold=0.1, energy_by_residue_type='all'):
        """
        When initiliasing the class it read the paths to the output folder report,
        trajectory, and topology files.

        Parameters
        ==========
        pele_folder : str
            Path to the pele folder containing one or several PELE calculations folders
        """

        # start = time.time()

        # Check if dataframe exists
        self.pele_folder = pele_folder
        self.pele_output_folder = pele_output_folder
        self.separator = separator

        # Get all PELE folders' paths
        self.pele_directories = {}
        self.report_files = {}
        self.trajectory_files = {}
        self.topology_files = {}
        self.equilibration = {}
        self.ligand_names = {}
        self.chain_ids = {}
        self.atom_indexes = {}
        self.equilibration['report'] = {}
        self.equilibration['trajectory'] = {}

        # Clustering Attributes
        self.proteins = []
        self.ligands = []

        # Check energy by residue type
        ebr_types = ['all', 'sgb', 'lennard_jones', 'electrostatic']
        if energy_by_residue_type not in ebr_types:
            raise ValueError('Energy by residue type not valid. valid options are: '+' '.join(energy_by_residue_type))

        # Create analysis folder
        if not os.path.exists('.pele_analysis'):
            os.mkdir('.pele_analysis')

        parser = PDB.PDBParser()

        if verbose:
            print('Getting paths to PELE files')

        for d in os.listdir(self.pele_folder):
            if os.path.isdir(self.pele_folder+'/'+d):

                # Store paths to the pele folders
                pele_dir = self.pele_folder+'/'+d
                protein = d.split(self.separator)[0]
                ligand = d.split(self.separator)[1]
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
                if protein not in self.equilibration['report']:
                    self.equilibration['report'][protein] = {}
                if protein not in self.equilibration['trajectory']:
                    self.equilibration['trajectory'][protein] = {}

                self.pele_directories[protein][ligand] = pele_dir

                if not os.path.exists(pele_dir+'/'+self.pele_output_folder+'/output'):
                    print('Output folder not found for %s-%s PELE calculation.' % (protein, ligand))
                    continue
                else:
                    self.report_files[protein][ligand] = pele_read.getReportFiles(pele_dir+'/'+self.pele_output_folder+'/output')
                    self.trajectory_files[protein][ligand] = pele_read.getTrajectoryFiles(pele_dir+'/'+self.pele_output_folder+'/output')
                    self.equilibration['report'][protein][ligand] = pele_read.getEquilibrationReportFiles(pele_dir+'/'+self.pele_output_folder+'/output')
                    self.equilibration['trajectory'][protein][ligand] = pele_read.getEquilibrationTrajectoryFiles(pele_dir+'/'+self.pele_output_folder+'/output')

                if not os.path.exists(pele_dir+'/'+self.pele_output_folder+'/input'):
                    print('PELE input folder not found for %s-%s PELE calculation.' % (protein, ligand))
                    continue
                else:
                    self.topology_files[protein][ligand] = pele_read.getTopologyFile(pele_dir+'/'+self.pele_output_folder+'/input')

                if protein not in self.proteins:
                    self.proteins.append(protein)
                if ligand not in self.ligands:
                    self.ligands.append(ligand)

        self.proteins = sorted(self.proteins)
        self.ligands = sorted(self.ligands)

        # Read complex chain ids
        self.structure = {}
        self.ligand_structure = {}
        self.md_topology = {}

        if not os.path.exists('.pele_analysis/chains_ids.json') or not os.path.exists('.pele_analysis/atom_indexes.json') or force_reading:
            for protein in self.report_files:
                if protein not in self.chain_ids:
                    self.structure[protein] = {}
                    self.ligand_structure[protein] = {}
                    self.md_topology[protein] = {}
                    self.chain_ids[protein] = {}
                    self.atom_indexes[protein] = {}
                for ligand in self.report_files[protein]:
                    if ligand not in self.chain_ids[protein]:
                        self.chain_ids[protein][ligand] = {}

                    if ligand not in self.atom_indexes[protein]:
                        self.atom_indexes[protein][ligand] = {}

                    # Load input PDB with Bio.PDB and mdtraj
                    input_pdb = self._getInputPDB(self.pele_directories[protein][ligand])
                    self.structure[protein][ligand] = parser.get_structure(protein, input_pdb)
                    input_ligand_pdb = self._getInputLigandPDB(self.pele_directories[protein][ligand])
                    self.ligand_structure[protein][ligand] = parser.get_structure(protein, input_ligand_pdb)

                    # Add ligand three letter code ligand_names
                    if ligand not in self.ligand_names:
                        for residue in self.ligand_structure[protein][ligand].get_residues():
                            self.ligand_names[ligand] = residue.resname

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

            self._saveDictionaryAsJson(self.chain_ids, '.pele_analysis/chains_ids.json')
            self._saveDictionaryAsJson(self.atom_indexes, '.pele_analysis/atom_indexes.json')

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
            self.chain_ids = self._loadDictionaryFromJson('.pele_analysis/chains_ids.json')
            # Recover chain as integer in the dictionary
            chain_ids = {}
            for protein in self.chain_ids:
                self.structure[protein] = {}
                self.ligand_structure[protein] = {}
                self.md_topology[protein] = {}
                chain_ids[protein] = {}
                for ligand in self.chain_ids[protein]:
                    chain_ids[protein][ligand] = {}
                    input_pdb = self._getInputPDB(self.pele_directories[protein][ligand])
                    self.structure[protein][ligand] = parser.get_structure(protein, input_pdb)
                    input_ligand_pdb = self._getInputLigandPDB(self.pele_directories[protein][ligand])
                    self.ligand_structure[protein][ligand] = parser.get_structure(protein, input_ligand_pdb)

                    # Add ligand three letter code ligand_names
                    if ligand not in self.ligand_names:
                        for residue in self.ligand_structure[protein][ligand].get_residues():
                            self.ligand_names[ligand] = residue.resname

                    # Read topology mdtraj trajectory object
                    self.md_topology[protein][ligand] = md.load(input_pdb)
                    for chain in self.chain_ids[protein][ligand]:
                        chain_ids[protein][ligand][int(chain)] = self.chain_ids[protein][ligand][chain]

            self.chain_ids = chain_ids

            self.atom_indexes = self._loadDictionaryFromJson('.pele_analysis/atom_indexes.json')
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

        if verbose:
            print('Reading information from report files from:')

        # Read report files into pandas
        report_data = []
        for protein in sorted(self.report_files):
            for ligand in sorted(self.report_files[protein]):
                if verbose:
                    print('\t'+protein+self.separator+ligand, end=' ')
                    start = time.time()
                data = pele_read.readReportFiles(self.report_files[protein][ligand],
                                                 protein,
                                                 ligand,
                                                 force_reading=force_reading,
                                                 ebr_threshold=0.1)
                if isinstance(data, type(None)):
                    continue

                keep = [k for k in data.keys() if not k.startswith('L:1_')]
                if energy_by_residue:
                    keep += [k for k in data.keys() if k.startswith('L:1_') and k.endswith(energy_by_residue_type)]
                data = data[keep]

                data = data.reset_index()
                data['Protein'] = protein
                data['Ligand'] = ligand
                data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps'], inplace=True)
                report_data.append(data)
                if verbose:
                    print('\t in %.2f seconds.' % (time.time()-start))

        self.data = pd.concat(report_data)
        # Remove Task column
        self.data.drop(['Task'], axis=1)

        # Save and reaload dataframe to avoid memory fragmentation
        self._saveDataState()
        self.data = None
        gc.collect()
        self._recoverDataState(remove=True)

        if verbose:
            print('Reading equilibration information from report files from:')

        # Read equlibration files
        equilibration_data = []
        for protein in sorted(self.equilibration['report']):
            for ligand in sorted(self.equilibration['report'][protein]):

                if self.equilibration['report'][protein][ligand] == {}:
                    print('WARNING: No equilibration data found for simulation %s-%s' % (protein, ligand))
                    continue

                if verbose:
                    print('\t'+protein+self.separator+ligand, end=' ')
                    start = time.time()

                data = pele_read.readReportFiles(self.equilibration['report'][protein][ligand],
                                                 protein,
                                                 ligand,
                                                 force_reading=force_reading,
                                                 equilibration=True)

                if isinstance(data, type(None)):
                    continue

                data = data.reset_index()
                data['Protein'] = protein
                data['Ligand'] = ligand
                data.set_index(['Protein', 'Ligand', 'Step', 'Trajectory', 'Accepted Pele Steps'], inplace=True)
                equilibration_data.append(data)
                if verbose:
                    print('\t in %.2f seconds.' % (time.time()-start))

        if equilibration_data != []:
            self.equilibration_data = pd.concat(equilibration_data)
            self._saveEquilibrationDataState()
            self.equilibration_data = None
            gc.collect()
            self._recoverEquilibrationDataState(remove=True)

    def calculateDistances(self, atom_pairs, equilibration=False, verbose=False, overwrite=False):
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
            Atom pairs for each protein + ligand entry.
        equilibration : bool
            Calculate distances for the equilibration steps also
        verbose : bool
            Print the analysis progression.
        overwrite : bool
            Force recalculation of distances.
        """

        if not os.path.exists('.pele_analysis/distances'):
            os.mkdir('.pele_analysis/distances')

        # Iterate all PELE protein + ligand entries
        distances = {}
        for protein in sorted(self.trajectory_files):
            distances[protein] = {}
            for ligand in sorted(self.trajectory_files[protein]):

                # Define a different distance output file for each pele run
                distance_file = '.pele_analysis/distances/'+protein+self.separator+ligand+'.csv'

                # Check if distance have been previously calculated
                if os.path.exists(distance_file) and not overwrite:
                    if verbose:
                        print('Distance file for %s + %s was found. Reading distances from there...' % (protein, ligand))
                    distances[protein][ligand] = pd.read_csv(distance_file, index_col=False)
                    distances[protein][ligand] = distances[protein][ligand].loc[:, ~distances[protein][ligand].columns.str.contains('^Unnamed')]
                    distances[protein][ligand].set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory','Accepted Pele Steps'], inplace=True)
                else:
                    distances[protein][ligand] = {}
                    distances[protein][ligand]['Protein'] = []
                    distances[protein][ligand]['Ligand'] = []
                    distances[protein][ligand]['Epoch'] = []
                    distances[protein][ligand]['Trajectory'] = []
                    distances[protein][ligand]['Accepted Pele Steps'] = []
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
                    for pair in atom_pairs[protein][ligand]:
                        if len(pair) >= 2:
                            i1 = self.atom_indexes[protein][ligand][pair[0]]
                            i2 = self.atom_indexes[protein][ligand][pair[1]]
                            if len(pair) == 2:
                                pairs.append((i1, i2))
                                dist_label[(pair[0], pair[1])] = 'distance_'
                        if len(pair) >= 3:
                            i3 = self.atom_indexes[protein][ligand][pair[2]]
                            if len(pair) == 3:
                                pairs.append((pair[0], pair[1], pair[2]))
                                dist_label[(i1, i2, i3)] = 'angle_'
                        if len(pair) == 4:
                            i4 = self.atom_indexes[protein][ligand][pair[3]]
                            pairs.append((i1, i2, i3, i4))
                            dist_label[(pair[0], pair[1], pair[2], pair[3])] = 'torsion_'
                        pair_lengths.append(len(pair))

                    pair_lengths = set(pair_lengths)
                    if len(pair_lengths) > 1:
                        raise ValueError('Mixed number of atoms given!')
                    pair_lengths = list(pair_lengths)[0]

                    # Define labels
                    labels = [dist_label[p]+''.join([str(x) for x in p[0]])+'_'+\
                                            ''.join([str(x) for x in p[1]]) for p in atom_pairs[protein][ligand]]

                    # Create an entry for each distance
                    for label in labels:
                        distances[protein][ligand][label] = []

                    # Compute distances and them to the dicionary
                    for epoch in sorted(trajectory_files):
                        for t in sorted(trajectory_files[epoch]):
                            # Load trajectory
                            traj = md.load(trajectory_files[epoch][t], top=topology_file)
                            # Calculate distances
                            if pair_lengths == 2:
                                d = md.compute_distances(traj, pairs)*10
                            elif pair_lengths == 3:
                                d = md.compute_angles(traj, pairs)*10
                            elif pair_lengths == 4:
                                d = md.compute_dihedrals(traj, pairs)*10

                            # Store data
                            distances[protein][ligand]['Protein'] += [protein]*d.shape[0]
                            distances[protein][ligand]['Ligand'] += [ligand]*d.shape[0]
                            distances[protein][ligand]['Epoch'] += [epoch]*d.shape[0]
                            distances[protein][ligand]['Trajectory'] += [t]*d.shape[0]
                            distances[protein][ligand]['Accepted Pele Steps'] += list(range(d.shape[0]))
                            for i,l in enumerate(labels):
                                distances[protein][ligand][l] += list(d[:,i])

                    # Convert distances into dataframe
                    distances[protein][ligand] = pd.DataFrame(distances[protein][ligand])

                    # Save distances to CSV file
                    distances[protein][ligand].to_csv(distance_file)

                    # Set indexes for DataFrame
                    distances[protein][ligand].set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory','Accepted Pele Steps'], inplace=True)

        # Concatenate individual distances into a single data frame
        all_distances = []
        for protein in distances:
            for ligand in distances[protein]:
                all_distances.append(distances[protein][ligand])
        all_distances = pd.concat(all_distances)

    # Add distances to main dataframe
        self.data = self.data.merge(all_distances, left_index=True, right_index=True)

    def getTrajectory(self, protein, ligand, step, trajectory, equilibration=False):
        """
        Load trajectory file for the selected protein, ligand, step, and trajectory number.
        """
        if equilibration:
            traj = md.load(self.equilibration['trajectory'][protein][ligand][step][trajectory],
                            top=self.topology_files[protein][ligand])
        else:
            traj = md.load(self.trajectory_files[protein][ligand][step][trajectory],
                            top=self.topology_files[protein][ligand])
        return traj

    def calculateRMSD(self, equilibration=True, productive=True, recalculate=False):
        """
        Calculate the RMSD of all steps regarding the input (topology) structure.
        """

        if equilibration:
            if 'RMSD' in self.equilibration_data.keys() and not recalculate:
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
                                    protein_atoms = traj.topology.select('protein')
                                    rmsd = md.rmsd(traj, reference, atom_indices=protein_atoms)*10
                                    if isinstance(RMSD, type(None)):
                                        RMSD = rmsd
                                    else:
                                        RMSD = np.concatenate((RMSD, rmsd))

                self.equilibration_data['Protein RMSD'] = RMSD
                self._saveEquilibrationDataState()

        if productive:
            if 'RMSD' in self.data.keys() and not recalculate:
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
        if 'Protein RMSD' not in self.data:
            raise ValueError('You must call calculateRMSD() before calling this function.')
        elif 'Protein RMSD' not in self.equilibration_data:
            raise ValueError('You must call calculateRMSD() before calling this function.')

        self.plotSimulationMetric('Protein RMSD',
                                  equilibration=equilibration,
                                  productive=productive)

    def scatterPlotIndividualSimulation(self, protein, ligand, x, y, vertical_line=None, color_column=None):
        """
        Creates a scatter plot for the selected protein and ligand using the x and y
        columns.
        """

        protein_series = self.data[self.data.index.get_level_values('Protein') == protein]
        if protein_series.empty:
            raise ValueError('Protein name %s not found in data!' % protein)
        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
        if ligand_series.empty:
            raise ValueError('Ligand name %s not found in protein %s data!' % (ligand, protein))

        plt.figure(figsize=(10, 8))
        if color_column != None:

            ascending = False
            colormap='Blues_r'

            if color_column == 'Step':
                ascending = True
                colormap='Blues'

            if color_column == 'Epoch':
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
                sc = plt.scatter(ligand_series[x],
                    ligand_series[y],
                    c=color_values,
                    cmap=colormap,
                    norm=norm,
                    label=protein+self.separator+ligand)
            else:
                ligand_series = ligand_series.sort_values(color_column, ascending=ascending)
                color_values = ligand_series[color_column]
                sc = plt.scatter(ligand_series[x],
                    ligand_series[y],
                    c=color_values,
                    cmap=colormap,
                    label=protein+self.separator+ligand)
            cbar = plt.colorbar(sc, label=color_column)
        else:
            sc = plt.scatter(ligand_series[x],
                ligand_series[y],
                label=protein+self.separator+ligand)

        if not isinstance(vertical_line, type(None)):
            plt.axvline(vertical_line, ls='--', lw=0.5)

        plt.xlabel(x)
        plt.ylabel(y)
        plt.show()

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
        plt.xticks(rotation=90)
        plt.show()

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

    def bindingEnergyLandscape(self, vertical_line=None):
        """
        Plot binding energy as interactive plot.
        """
        def getLigands(Protein, by_metric=True, vertical_line=None):
            protein_series = self.data[self.data.index.get_level_values('Protein') == Protein]
            ligands = list(set(protein_series.index.get_level_values('Ligand').tolist()))
            interact(getDistance, Protein=fixed(Protein), Ligand=ligands, vertical_line=fixed(vertical_line), by_metric=fixed(by_metric))

        def getDistance(Protein, Ligand, vertical_line=None, by_metric=True):
            protein_series = self.data[self.data.index.get_level_values('Protein') == Protein]
            ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == Ligand]

            distances = []
            if by_metric:
                distances = []
                for d in ligand_series:
                    if d.startswith('metric_'):
                        if not ligand_series[d].dropna().empty:
                            distances.append(d)

            if distances == []:
                by_metric = False

            if not by_metric:
                distances = []
                for d in ligand_series:
                    if 'distance' in d:
                        if not ligand_series[d].dropna().empty:
                            distances.append(d)

            color_columns = [k for k in ligand_series.keys()]
            color_columns = [k for k in color_columns if ':' not in k]
            color_columns = [k for k in color_columns if 'distance' not in k]
            color_columns = [k for k in color_columns if not k.startswith('metric_')]
            color_columns = [None, 'Epoch']+color_columns

            del color_columns[color_columns.index('Task')]
            del color_columns[color_columns.index('Binding Energy')]

            interact(_bindingEnergyLandscape,
                     Protein=fixed(Protein),
                     Ligand=fixed(Ligand),
                     Distance=distances,
                     Color=color_columns,
                     vertical_line=fixed(vertical_line))

        def _bindingEnergyLandscape(Protein, Ligand, Distance, Color, vertical_line=None):
            self.scatterPlotIndividualSimulation(Protein, Ligand, Distance, 'Binding Energy',
                                                 vertical_line=vertical_line, color_column=Color)

        interact(getLigands, Protein=sorted(self.proteins), vertical_line=fixed(vertical_line), by_metric=False)

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
        del columns[columns.index('Task')]
        del columns[columns.index('Step')]

        interact(selectLevel, By_protein=True, By_ligand=False)

    def getDistances(self, protein, ligand):
        """
        Returns the distance associated to a specific protein and ligand simulation
        """
        protein_series = self.data[self.data.index.get_level_values('Protein') == protein]
        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
        if not ligand_series.empty:
            distances = []
            for d in ligand_series:
                if 'distance' in d:
                    if not ligand_series[d].dropna().empty:
                        distances.append(d)
            return distances
        else:
            return None

    def plotCatalyticPosesFraction(self, initial_threshold=4.5):
        """
        Plot interactively the number of catalytic poses as a function of the threshold
        of the different catalytic metrics. The plot can be done by protein or by ligand.
        """
        metrics = [k for k in self.data.keys() if 'metric_' in k]

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

    def plotCatalyticBindingEnergyDistributions(self, initial_threshold=4.5):
        """
        Plot interactively the binding energy distributions as a function of the threshold
        of the different catalytic metrics. The plot can be done by protein or by ligand.
        """
        metrics = [k for k in self.data.keys() if 'metric_' in k]

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

    def bindingFreeEnergyCatalyticDifferenceMatrix(self, initial_threshold=3.5, store_values=False, lig_label_rot=50,
                matrix_file='catalytic_matrix.npy', models_file='catalytic_models.json', max_metric_threshold=30, pele_data=None):

        def _bindingFreeEnergyMatrix(KT=0.593, sort_by_ligand=None, dA=True, Ec=False, Enc=False, models_file='catalytic_models.json',
                                     lig_label_rot=50, pele_data=None, **metrics):

            if isinstance(pele_data, type(None)):
                pele_data = self.data

            # Create a matrix of length proteins times ligands
            M = np.zeros((len(self.proteins), len(self.ligands)))

            # Calculate the probaility of each state
            for i,protein in enumerate(self.proteins):
                protein_series = pele_data[pele_data.index.get_level_values('Protein') == protein]

                for j, ligand in enumerate(self.ligands):
                    ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]

                    if not ligand_series.empty:

                        # Calculate partition function
                        total_energy = ligand_series['Total Energy']
                        energy_minimum = total_energy.min()
                        relative_energy = total_energy-energy_minimum
                        Z = np.sum(np.exp(-relative_energy/KT))

                        # Calculate catalytic binding energy
                        catalytic_series = ligand_series
                        for metric in metrics:
                            catalytic_series = catalytic_series[catalytic_series[metric] <= metrics[metric]]

                        total_energy = catalytic_series['Total Energy']
                        relative_energy = total_energy-energy_minimum
                        probability = np.exp(-relative_energy/KT)/Z
                        Ebc = np.sum(probability*catalytic_series['Binding Energy'])

                        # Calculate non-catalytic binding energy
                        catalytic_indexes = catalytic_series.index.values.tolist()
                        noncatalytic_series = ligand_series.loc[~ligand_series.index.isin(catalytic_indexes)]

                        total_energy = noncatalytic_series['Total Energy']
                        relative_energy = total_energy-energy_minimum
                        probability = np.exp(-relative_energy/KT)/Z
                        Ebnc = np.sum(probability*noncatalytic_series['Binding Energy'])

                        if dA:
                            M[i][j] = Ebc-Ebnc
                        elif Ec:
                            M[i][j] = Ebc
                        elif Enc:
                            M[i][j] = Ebnc
                    else:
                        M[i][j] = np.nan

            # Sort matrix by ligand or protein
            if sort_by_ligand == 'by_protein':
                protein_labels = self.proteins
            else:
                ligand_index = self.ligands.index(sort_by_ligand)
                sort_indexes = M[:, ligand_index].argsort()
                M = M[sort_indexes]
                protein_labels = [self.proteins[x] for x in sort_indexes]

            plt.matshow(M, cmap='autumn')
            if dA:
                plt.colorbar(label='${E_{B}^{C}}-{E_{B}^{NC}}$')
            elif Ec:
                plt.colorbar(label='$E_{B}^{C}$')
            elif Enc:
                plt.colorbar(label='$E_{B}^{NC}$')

            if store_values:
                np.save(matrix_file, M)
                if not models_file.endswith('.json'):
                    models_file = models_file+'.json'
                with open(models_file, 'w') as of:
                    json.dump(protein_labels, of)

            plt.xlabel('Ligands', fontsize=12)
            plt.xticks(range(len(self.ligands)), self.ligands, rotation=lig_label_rot)
            plt.ylabel('Proteins', fontsize=12)
            plt.yticks(range(len(self.proteins)), protein_labels)

            display(plt.show())

        if isinstance(pele_data, type(None)):
            pele_data = self.data

        # Add checks for the given pele data pandas df
        metrics = [k for k in pele_data.keys() if 'metric_' in k]

        metrics_sliders = {}
        for m in metrics:
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
                        )
            metrics_sliders[m] = m_slider

        metrics = {m:initial_threshold for m in metrics}

        KT_slider = FloatSlider(
                        value=0.593,
                        min=0.593,
                        max=60.0,
                        step=0.1,
                        description='KT:',
                        disabled=False,
                        continuous_update=False,
                        orientation='horizontal',
                        readout=True,
                        readout_format='.1f',
                    )

        dA = Checkbox(value=True,
                     description='$\delta A$')
        Ec = Checkbox(value=False,
                     description='$E_{B}^{C}$')
        Enc = Checkbox(value=False,
                     description='$E_{B}^{NC}$')

        ligand_ddm = Dropdown(options=self.ligands+['by_protein'])

        interact(_bindingFreeEnergyMatrix, KT=KT_slider, sort_by_ligand=ligand_ddm, pele_data=pele_data,
                 dA=dA, Ec=Ec, Enc=Enc, models_file=fixed(models_file), lig_label_rot=fixed(lig_label_rot), **metrics_sliders)

    def visualiseBestPoses(self, pele_data=None, initial_threshold=3.5):

        def _visualiseBestPoses(Protein, Ligand, n_smallest=10, **metrics):
            protein_series = pele_data[pele_data.index.get_level_values('Protein') == Protein]
            ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == Ligand]

            # Filter by metric
            # Calculate catalytic binding energy
            catalytic_series = ligand_series
            for metric in metrics:
                catalytic_series = catalytic_series[catalytic_series[metric] <= metrics[metric]]

            catalytic_series = catalytic_series.nsmallest(n_smallest, 'Binding Energy')

            if catalytic_series.empty:
                raise ValueError('No frames were selected for the selected thresholds.')

            traj = pele_trajectory.loadTrajectoryFrames(catalytic_series,
                                                        self.trajectory_files[Protein][Ligand],
                                                        self.topology_files[Protein][Ligand])

            ligand_atoms = traj.topology.select('resname '+self.ligand_names[Ligand])
            neighbors = md.compute_neighbors(traj, 0.5, ligand_atoms)
            chain_ids = self.chain_ids[Protein][Ligand]

            # Get list of residues to depict
            residues = []
            for frame in neighbors:
                for x in frame:
                    residue = traj.topology.atom(x).residue
                    if residue.name != self.ligand_names[Ligand]:
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

    def visualiseInVMD(self, protein, ligand, resnames=None, resids=None, peptide=False,
                      num_trajectories='all', epochs=None, trajectories=None):

        if isinstance(resnames, str):
            resnames = [resnames]

        if isinstance(resids, int):
            resids = [resids]

        traj_files = self.trajectory_files[protein][ligand]
        trajs = [t for t in sorted(traj_files[0])]

        if num_trajectories == 'all' and isinstance(trajectories, type(None)):
            num_trajectories = len(trajs)

        if not isinstance(trajectories, type(None)):
            if isinstance(trajectories, list):
                trajectories = [trajs[i-1] for i in trajectories]
            else:
                raise ValueError('trajectories must be given as a list')

        elif isinstance(num_trajectories, int):
            trajectories = random.choices(trajs, k=num_trajectories)


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

    def combineDistancesIntoMetrics(self, catalytic_labels, overwrite=False):
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

        changed = False
        for name in catalytic_labels:
            if 'metric_'+name in self.data.keys() and not overwrite:
                print('Combined metric %s already added. Give overwrite=True to recombine' % name)
            else:
                changed = True
                values = []
                for protein in sorted(self.report_files):
                    protein_series = self.data[self.data.index.get_level_values('Protein') == protein]
                    for ligand in sorted(self.report_files[protein]):
                        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                        distances = catalytic_labels[name][protein][ligand]
                        values += ligand_series[distances].min(axis=1).tolist()

                self.data['metric_'+name] = values

        if changed:
            self._saveDataState()

    def plotEnergyByResidue(self, initial_threshold=4.5):
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
                            value=0.593,
                            min=0.593,
                            max=20.0,
                            step=0.1,
                            description='KT:',
                            readout=True,
                            readout_format='.1f')

            KT_slider2 = FloatSlider(
                            value=0.593,
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

    def getProteinAndLigandData(self, protein, ligand):
        protein_series = self.data[self.data.index.get_level_values('Protein') == protein]
        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
        return ligand_series

    ### Extract poses methods

    def getBestPELEPoses(self, filter_values, column='Binding Energy', n_models=1, return_failed=False):
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
        """

        best_poses = pd.DataFrame()
        bp = []
        failed = []
        for model in self.proteins:
            protein_series = self.data[self.data.index.get_level_values('Protein') == model]
            for ligand in self.ligands:
                ligand_data = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
                for metric in filter_values:
                    ligand_data = ligand_data[ligand_data['metric_'+metric] < filter_values[metric]]
                if ligand_data.empty:
                    failed.append((model, ligand))
                    continue
                if ligand_data.shape[0] < n_models:
                    print('WARNING: less than %s models available for pele %s + %s simulation' % (n_models, model, ligand))
                for i in ligand_data[column].nsmallest(n_models).index:
                    bp.append(i)

        if return_failed:

            return failed, self.data[self.docking_data.index.isin(bp)]

        return self.data[self.data.index.isin(bp)]

    def getBestPELEPosesIteratively(self, metrics, column='Binding Energy', ligands=None,
                                    min_threshold=3.5, max_threshold=5.0, step_size=0.1):
        """
        """
        extracted = []
        selected_indexes = []

        for t in np.arange(min_threshold, max_threshold+(step_size/10), step_size):
            filter_values = {m:t for m in metrics}
            best_poses = self.getBestPELEPoses(filter_values, column=column, n_models=1)
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

            for row in pele_data.index:
                if row[:2] not in extracted:
                    selected_indexes.append(row)
                if row[:2] not in extracted:
                    extracted.append(row[:2])

        final_mask = []
        for row in self.data.index:
            if row in selected_indexes:
                final_mask.append(True)
            else:
                final_mask.append(False)
        pele_data = self.data[final_mask]

        return pele_data

    def extractPELEPoses(self, pele_data, output_folder, separator='-', keep_chain_names=True):
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
        """

        # Create output folder
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)

        # Crea Bio PDB parser and io
        parser = PDB.PDBParser()
        io = PDB.PDBIO()

        # Extract pele poses with mdtraj

        # Check the separator is not in protein or ligand names
        for protein in self.proteins:
            if separator in protein:
                raise ValueError('The separator %s was found in protein name %s. Please use a different separator symbol.' % (separator, model))

            protein_data = pele_data[pele_data.index.get_level_values('Protein') == protein]

            for ligand in self.ligands:

                if separator in ligand:
                    raise ValueError('The separator %s was found in ligand name %s. Please use a different separator symbol.' % (separator, ligand))

                ligand_data = protein_data[protein_data.index.get_level_values('Ligand') == ligand]

                if not ligand_data.empty:

                    if not os.path.exists(output_folder+'/'+protein):
                        os.mkdir(output_folder+'/'+protein)

                    traj = pele_trajectory.loadTrajectoryFrames(ligand_data,
                                                                self.trajectory_files[protein][ligand],
                                                                self.topology_files[protein][ligand])

                    # Create atom names to traj indexes dictionary
                    atom_traj_index = {}
                    for residue in traj.topology.residues:
                        residue_label = residue.name+str(residue.resSeq)
                        atom_traj_index[residue_label] = {}
                        for atom in residue.atoms:
                            atom_traj_index[residue_label][atom.name] = atom.index

                    # Create a topology file with Bio.PDB
                    pdb_topology = parser.get_structure(protein, self.topology_files[protein][ligand])
                    atoms = [a for a in pdb_topology.get_atoms()]

                    # Pass mdtraj coordinates to Bio.PDB structure to preserve correct chains
                    for i, entry in enumerate(ligand_data.index):
                        filename = separator.join([str(x) for x in entry])+'.pdb'
                        xyz = traj[i].xyz[0]
                        for j in range(traj.n_atoms):

                            # Get residue label
                            residue = atoms[j].get_parent()
                            if residue.resname in ['HID', 'HIE', 'HIP']:
                                resname = 'HIS'
                            else:
                                resname = residue.resname
                            residue_label = resname+str(residue.id[1])

                            # Give atom coordinates to Bio.PDB object
                            traj_index = atom_traj_index[residue_label][atoms[j].name]
                            atoms[j].coord = xyz[traj_index]*10

                    # Save structure
                    io.set_structure(pdb_topology)
                    io.save(output_folder+'/'+protein+'/'+filename)

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

        ligand_traj_dir = '.pele_analysis/ligand_traj'
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
        ligand_atoms = reference.topology.select('resname '+self.ligand_names[ligand])

        # Store topology as PDB
        reference.atom_slice(ligand_atoms).save(ligand_top_path)

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
                return ligand_traj_paths, traj_dict
            else:
                return ligand_traj, traj_dict
        elif return_paths:
                return ligand_traj_paths
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

        ligand_traj_dir = '.pele_analysis/ligand_traj'
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
            ligand_atoms = reference.topology.select('resname '+self.ligand_names[ligand])
            i = 0
            for epoch in sorted(trajectory_files):
                for t in sorted(trajectory_files[epoch]):
                    # Load trajectory
                    traj = md.load(trajectory_files[epoch][t], top=topology_file)

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

        if return_dictionary:
            return ligand_traj, traj_dict
        else:
            return ligand_traj

    def computeLigandRMSDClusters(self, rmsd_threshold=3, overwrite=False):
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

        clustering_dir = '.pele_analysis/clustering'
        if not os.path.exists(clustering_dir):
            os.mkdir(clustering_dir)

        # Iterate by protein
        for protein in sorted(self.trajectory_files):

            # Iterate by ligand
            for ligand in sorted(self.trajectory_files[protein]):

                # Create or read ligand trajectory
                ligand_traj, ligand_traj_dict = self.getLigandTrajectory(protein, ligand, overwrite=overwrite,
                                                                         return_dictionary=True)

                # Start RMSD clustering
                rmsd_matrix_file = ligand_traj_dir+'/'+protein+self.separator+ligand+'.npy'
                clusters = clustering.clusterByRMSD(ligand_traj, threshold=rmsd_threshold/10.0,
                                                    rmsd_matrix_file=rmsd_matrix_file)
                return ligand_traj, clusters

    def setUpPELECalculation(self, pele_folder, models_folder, input_yaml, box_centers=None, distances=None, ligand_index=1,
                             box_radius=10, steps=100, debug=False, iterations=3, cpus=96, equilibration_steps=100,
                             separator='-', use_peleffy=True, usesrun=True, energy_by_residue=False, ninety_degrees_version=False,
                             analysis=False, energy_by_residue_type='all', peptide=False, equilibration_mode='equilibrationLastSnapshot'):
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
        Missing!
        """

        # Create PELE job folder
        if not os.path.exists(pele_folder):
            os.mkdir(pele_folder)

        # Read docking poses information from models_folder and create pele input folders.
        jobs = []
        for d in os.listdir(models_folder):
            if os.path.isdir(models_folder+'/'+d):
                models = {}
                ligand_pdb_name = {}
                for f in os.listdir(models_folder+'/'+d):
                    fs = f.split(separator)
                    protein = fs[0]
                    ligand = fs[1]
                    pose = fs[2].replace('.pdb','')

                    # Create PELE job folder for each docking
                    if not os.path.exists(pele_folder+'/'+protein+'_'+ligand):
                        os.mkdir(pele_folder+'/'+protein+'_'+ligand)

                    structure = self._readPDB(protein+'_'+ligand, models_folder+'/'+d+'/'+f)

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

                    self._saveStructureToPDB(structure, pele_folder+'/'+protein+'_'+ligand+'/'+f)

                    if (protein, ligand) not in models:
                        models[(protein,ligand)] = []
                    models[(protein,ligand)].append(f)

                # Create YAML file
                for model in models:
                    protein, ligand = model
                    keywords = ['system', 'chain', 'resname', 'steps', 'iterations', 'atom_dist', 'analyse',
                                    'cpus', 'equilibration', 'equilibration_steps', 'traj', 'working_folder',
                                    'usesrun', 'use_peleffy', 'debug', 'box_radius', 'equilibration_mode']

                    # Get distances from PELE data
                    if distances == None:
                        distances = {}
                    if protein not in distances:
                        distances[protein] = {}
                    if ligand not in distances[protein]:
                        distances[protein][ligand] = []

                    pele_distances = [(x.split('_')[1:3][0], x.split('_')[1:3][1]) for x in self.getDistances(protein, ligand)]
                    pele_distances = list(set(pele_distances))

                    for d in pele_distances:
                        at1 = self._atomStringToTuple(d[0])
                        at2 = self._atomStringToTuple(d[1])
                        distances[protein][ligand].append((at1, at2))

                    with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input.yaml', 'w') as iyf:
                        if energy_by_residue:
                            # Use new PELE version with implemented energy_by_residue
                            iyf.write('pele_exec: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/bin/PELE-1.7.2_mpi"\n')
                            iyf.write('pele_data: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/Data"\n')
                            iyf.write('pele_documents: "/gpfs/projects/bsc72/PELE++/mniv/V1.7.2-b6/Documents/"\n')
                        elif ninety_degrees_version:
                            # Use new PELE version with implemented energy_by_residue
                            iyf.write('pele_exec: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/bin/PELE-1.8_mpi"\n')
                            iyf.write('pele_data: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/Data"\n')
                            iyf.write('pele_documents: "/gpfs/projects/bsc72/PELE++/mniv/V1.8_pre_degree_fix/Documents/"\n')
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
                        iyf.write("equilibration: true\n")
                        iyf.write("equilibration_mode: '"+equilibration_mode+"'\n")
                        iyf.write("equilibration_steps: "+str(equilibration_steps)+"\n")
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

                        iyf.write("box_radius: "+str(box_radius)+"\n")
                        if isinstance(box_centers, type(None)) and peptide:
                            raise ValuError('You must give per-protein box_centers when docking peptides!')
                        if not isinstance(box_centers, type(None)):
                            box_center = ':'.join([str(x) for x in box_centers[protein]])
                            iyf.write("box_center: '"+box_center+"'\n")

                        # energy by residue is not implemented in PELE platform, therefore
                        # a scond script will modify the PELE.conf file to set up the energy
                        # by residue calculation.
                        if debug or energy_by_residue or peptide:
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

                        iyf.write('\n')
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

                    if peptide:
                        _copyScriptFile(pele_folder, 'modifyPelePlatformForPeptide.py')
                        peptide_script_name = '._modifyPelePlatformForPeptide.py'

                    # Create command
                    command = 'cd '+pele_folder+'/'+protein+'_'+ligand+'\n'
                    command += 'python -m pele_platform.main input.yaml\n'
                    if energy_by_residue:
                        command += 'python ../'+ebr_script_name+' output --energy_type '+energy_by_residue_type
                        if peptide:
                            command += ' --peptide \n'
                            command += 'python ../'+peptide_script_name+' output '+" ".join(models[model])+'\n'
                        else:
                            command += '\n'
                        with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input_restart.yaml', 'w') as oyml:
                            with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input.yaml') as iyml:
                                for l in iyml:
                                    if 'debug: true' in l:
                                        l = 'restart: true\n'
                                    oyml.write(l)
                        command += 'python -m pele_platform.main input_restart.yaml\n'
                    elif peptide:
                        command += 'python ../'+peptide_script_name+' output '+" ".join(models[model])+'\n'
                        with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input_restart.yaml', 'w') as oyml:
                            with open(pele_folder+'/'+protein+'_'+ligand+'/'+'input.yaml') as iyml:
                                for l in iyml:
                                    if 'debug: true' in l:
                                        l = 'restart: true\n'
                                    oyml.write(l)
                        command += 'python -m pele_platform.main input_restart.yaml\n'
                    command += 'cd ../..'
                    jobs.append(command)

        return jobs

    def _saveDataState(self):
        self.data.to_csv('.pele_analysis/data.csv')

    def _saveEquilibrationDataState(self):
        self.equilibration_data.to_csv('.pele_analysis/equilibration_data.csv')

    def _recoverDataState(self, remove=False):
        csv_file = '.pele_analysis/data.csv'
        if os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file, index_col=False)
            self.data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps'], inplace=True)
            if remove:
                os.remove(csv_file)

    def _recoverEquilibrationDataState(self, remove=False):
        csv_file = '.pele_analysis/equilibration_data.csv'
        if os.path.exists(csv_file):
            self.equilibration_data = pd.read_csv(csv_file, index_col=False)
            self.equilibration_data.set_index(['Protein', 'Ligand', 'Step', 'Trajectory', 'Accepted Pele Steps'], inplace=True)
            if remove:
                os.remove(csv_file)

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

    def _getInputPDB(self, pele_dir):
        """
        Returns the input PDB for the PELE simulation.
        """
        # Load input PDB with Bio.PDB and mdtraj
        folder = pele_dir+'/'+self.pele_output_folder+'/input/'
        for d in os.listdir(folder):
            if d.endswith('processed.pdb'):
                return folder+'/'+d

    def _getInputLigandPDB(self, pele_dir):
        """
        Returns the input PDB for the PELE simulation.
        """
        # Load input PDB with Bio.PDB and mdtraj
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

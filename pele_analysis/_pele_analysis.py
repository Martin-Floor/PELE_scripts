import os
from . import pele_read
from . import pele_trajectory

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
            print('Getting paths to trajectory files')

        for d in os.listdir(self.pele_folder):
            if os.path.isdir(self.pele_folder+'/'+d):

                # Store paths to the pele folders
                pele_dir = self.pele_folder+'/'+d
                protein = d.split(separator)[0]
                ligand = d.split(separator)[1]
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

                # Read ligand resname
                if ligand not in self.ligand_names:
                    ligand_structure = parser.get_structure(ligand, pele_dir+'/'+self.pele_output_folder+'/input/ligand.pdb')
                    for residue in ligand_structure.get_residues():
                        self.ligand_names[ligand] = residue.resname

                self.pele_directories[protein][ligand] = pele_dir
                self.report_files[protein][ligand] = pele_read.getReportFiles(pele_dir+'/'+self.pele_output_folder+'/output')
                self.trajectory_files[protein][ligand] = pele_read.getTrajectoryFiles(pele_dir+'/'+self.pele_output_folder+'/output')
                self.topology_files[protein][ligand] = pele_read.getTopologyFile(pele_dir+'/'+self.pele_output_folder+'/input')
                self.equilibration['report'][protein][ligand] = pele_read.getEquilibrationReportFiles(pele_dir+'/'+self.pele_output_folder+'/output')
                self.equilibration['trajectory'][protein][ligand] = pele_read.getEquilibrationTrajectoryFiles(pele_dir+'/'+self.pele_output_folder+'/output')

                if protein not in self.proteins:
                    self.proteins.append(protein)
                if ligand not in self.ligands:
                    self.ligands.append(ligand)

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
                    print('\t'+protein+'_'+ligand, end=' ')
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
                data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Pele Step'], inplace=True)
                report_data.append(data)
                if verbose:
                    print('\t in %.2f seconds.' % (time.time()-start))

        self.data = pd.concat(report_data)
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
                if verbose:
                    print('\t'+protein+'_'+ligand, end=' ')
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
                data.set_index(['Protein', 'Ligand', 'Step', 'Trajectory', 'Pele Step'], inplace=True)
                equilibration_data.append(data)
                if verbose:
                    print('\t in %.2f seconds.' % (time.time()-start))

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

        # # Get all distance labels
        # all_labels = set()
        # for protein in atom_pairs:
        #     for ligand in atom_pairs[protein]:
        #         for pair in  atom_pairs[protein][ligand]:
        #             label = 'distance_'+''.join([str(x) for x in pair[0]])+'_'+\
        #                                 ''.join([str(x) for x in pair[1]])
        #             all_labels.add(label)

        if not os.path.exists('.pele_analysis/distances'):
            os.mkdir('.pele_analysis/distances')

        # # Check whether previous distance calculation exists.
        # if os.path.exists('.pele_analysis/distances.csv') and not overwrite:
        #     distances = pd.read_csv('.pele_analysis/distances.csv')
        #     distances.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory','Pele Step'], inplace=True)

        # Calculate distances for all trajectory files
        # else:
        #     # Define dicionary to store distance calculations
        #     distances = {}
        #     distances['Protein'] = []
        #     distances['Ligand'] = []
        #     distances['Epoch'] = []
        #     distances['Trajectory'] = []
        #     distances['Pele Step'] = []
        #     for label in all_labels:
        #         distances[label] = []

        # Iterate all PELE protein + ligand entries
        distances = {}
        for protein in sorted(self.trajectory_files):
            distances[protein] = {}
            for ligand in sorted(self.trajectory_files[protein]):

                # Define a different distance output file for each pele run
                distance_file = '.pele_analysis/distances/'+protein+'_'+ligand+'.csv'

                # Check if distance have been previously calculated
                if os.path.exists(distance_file) and not overwrite:
                    if verbose:
                        print('Distance file for %s + %s was found. Reading distances from there...' % (protein, ligand))
                    distances[protein][ligand] = pd.read_csv(distance_file, index_col=False)
                    distances[protein][ligand].set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory','Pele Step'], inplace=True)

                else:
                    distances[protein][ligand] = {}
                    distances[protein][ligand]['Protein'] = []
                    distances[protein][ligand]['Ligand'] = []
                    distances[protein][ligand]['Epoch'] = []
                    distances[protein][ligand]['Trajectory'] = []
                    distances[protein][ligand]['Pele Step'] = []
                    if verbose:
                        print('Calculating distances for %s + %s ' % (protein, ligand))

                    # Load one trajectory at the time to save memory
                    trajectory_files = self.trajectory_files[protein][ligand]
                    topology_file = self.topology_files[protein][ligand]

                    # Get atom pairs indexes
                    topology = md.load(topology_file).topology

                    # Get atom pair indexes to compute distances
                    pairs = []
                    for pair in atom_pairs[protein][ligand]:
                        i1 = self.atom_indexes[protein][ligand][pair[0]]
                        i2 = self.atom_indexes[protein][ligand][pair[1]]
                        pairs.append([i1, i2])

                    # Define labels
                    labels = ['distance_'+''.join([str(x) for x in p[0]])+'_'+\
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
                            d = md.compute_distances(traj, pairs)*10
                            # Store data
                            distances[protein][ligand]['Protein'] += [protein]*d.shape[0]
                            distances[protein][ligand]['Ligand'] += [ligand]*d.shape[0]
                            distances[protein][ligand]['Epoch'] += [epoch]*d.shape[0]
                            distances[protein][ligand]['Trajectory'] += [t]*d.shape[0]
                            distances[protein][ligand]['Pele Step'] += list(range(d.shape[0]))
                            for i,l in enumerate(labels):
                                distances[protein][ligand][l] += list(d[:,i])

                    # # Fill distances values with none to match the dictionary length
                    # for label in all_labels:
                    #     delta = len(distances[protein][ligand]['Protein'])-len(distances[protein][ligand][label])
                    #     for x in range(delta):
                    #         distances[label].append(None)

                    # Convert distances into dataframe
                    distances[protein][ligand] = pd.DataFrame(distances[protein][ligand])

                    # Save distances to CSV file
                    distances[protein][ligand].to_csv(distance_file)

                    # Set indexes for DataFrame
                    distances[protein][ligand].set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory','Pele Step'], inplace=True)

        # Merge individual distances into a single data frame
        all_distances = None
        for protein in distances:
            for ligand in distances[protein]:
                if isinstance(all_distances, type(None)):
                    all_distances = distances[protein][ligand]
                else:
                    all_distances.merge(distances[protein][ligand], how='outer', left_index=True, right_index=True)

        # Add distances to main dataframe
        self.data = self.data.merge(all_distances, how='outer', left_index=True, right_index=True)

        return distances

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

    def scatterPlotIndividualSimulation(self, protein, ligand, x, y, color_column=None):
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

        plt.figure()
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
                    label=protein+'_'+ligand)
            else:
                ligand_series = ligand_series.sort_values(color_column, ascending=ascending)
                color_values = ligand_series[color_column]
                sc = plt.scatter(ligand_series[x],
                    ligand_series[y],
                    c=color_values,
                    cmap=colormap,
                    label=protein+'_'+ligand)
            cbar = plt.colorbar(sc, label=color_column)
        else:
            sc = plt.scatter(ligand_series[x],
                ligand_series[y],
                label=protein+'_'+ligand)

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

    def bindingEnergyLandscape(self):
        """
        Plot binding energy as interactive plot.
        """
        def getLigands(Protein, by_metric=True):
            protein_series = self.data[self.data.index.get_level_values('Protein') == Protein]
            ligands = list(set(protein_series.index.get_level_values('Ligand').tolist()))
            interact(getDistance, Protein=fixed(Protein), Ligand=ligands, by_metric=fixed(by_metric))

        def getDistance(Protein, Ligand, by_metric=True):
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
                     Color=color_columns)

        def _bindingEnergyLandscape(Protein, Ligand, Distance, Color):
            self.scatterPlotIndividualSimulation(Protein, Ligand, Distance, 'Binding Energy', color_column=Color)

        interact(getLigands, Protein=self.proteins, by_metric=False)

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

    def bindingFreeEnergyCatalyticDifferenceMatrix(self, initial_threshold=4.5):

        def _bindingFreeEnergyMatrix(KT=0.593, sort_by_ligand=None, dA=True, Ec=False, Enc=False, **metrics):

            # Create a matrix of length proteins times ligands
            M = np.zeros((len(self.proteins), len(self.ligands)))

            # Calculate the probaility of each state
            for i,protein in enumerate(self.proteins):
                protein_series = self.data[self.data.index.get_level_values('Protein') == protein]

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


            plt.xlabel('Ligands', fontsize=12)
            plt.xticks(range(len(self.ligands)), self.ligands, rotation=50)
            plt.ylabel('Proteins', fontsize=12)
            plt.yticks(range(len(self.proteins)), protein_labels)

        metrics = [k for k in self.data.keys() if 'metric_' in k]
        metrics = {m:initial_threshold for m in metrics}

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

        dA = Checkbox(value=True,
                     description='$\delta A$')
        Ec = Checkbox(value=False,
                     description='$E_{B}^{C}$')
        Enc = Checkbox(value=False,
                     description='$E_{B}^{NC}$')

        ligand_ddm = Dropdown(options=self.ligands)

        interact(_bindingFreeEnergyMatrix, KT=KT_slider, sort_by_ligand=ligand_ddm, dA=dA, Ec=Ec, Enc=Enc, **metrics)

    def visualiseBestPoses(self, initial_threshold=4.5):

        def _visualiseBestPoses(Protein, Ligand, n_smallest=10, **metrics):
            protein_series = self.data[self.data.index.get_level_values('Protein') == Protein]
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
            protein_series = self.data[self.data.index.get_level_values('Protein') == Protein]
            ligands = list(set(protein_series.index.get_level_values('Ligand').tolist()))
            interact(_visualiseBestPoses, Protein=fixed(Protein),
                     Ligand=ligands,
                     **metrics)

        metrics = [k for k in self.data.keys() if 'metric_' in k]

        metrics = {m:initial_threshold for m in metrics}

        interact(getLigands, Protein=self.proteins)

    def visualiseInVMD(self, protein, ligand, resnames=None, peptide=False, num_trajectories='all'):

        if isinstance(resnames, str):
            resnames = [resnames]

        traj_files = self.trajectory_files[protein][ligand]
        trajectories = [t for t in sorted(traj_files[0])]
        if isinstance(num_trajectories, int):
            trajectories = random.choices(trajectories, k=num_trajectories)

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

    def _saveDataState(self):
        self.data.to_csv('.pele_analysis/data.csv')

    def _saveEquilibrationDataState(self):
        self.equilibration_data.to_csv('.pele_analysis/equilibration_data.csv')

    def _recoverDataState(self, remove=False):
        csv_file = '.pele_analysis/data.csv'
        if os.path.exists(csv_file):
            self.data = pd.read_csv(csv_file, index_col=False)
            self.data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Pele Step'], inplace=True)
            if remove:
                os.remove(csv_file)

    def _recoverEquilibrationDataState(self, remove=False):
        csv_file = '.pele_analysis/equilibration_data.csv'
        if os.path.exists(csv_file):
            self.equilibration_data = pd.read_csv(csv_file, index_col=False)
            self.equilibration_data.set_index(['Protein', 'Ligand', 'Step', 'Trajectory', 'Pele Step'], inplace=True)
            if remove:
                os.remove(csv_file)

    def _saveDictionaryAsJson(self, dictionary, output_file):
        with open(output_file, 'w') as of:
            json.dump(dictionary, of)

    def _loadDictionaryFromJson(self, json_file):
        with open(json_file) as jf:
            dictionary = json.load(jf)
        return dictionary

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

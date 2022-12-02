from . import pele_read
from . import pele_trajectory

import os
import shutil
import copy

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
import random

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
                 read_equilibration=True, data_folder_name=None, trajectories=False):
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
        self._copyPELEInputs()

        # Copy PELE configuration files to analysis folder
        if self.verbose:
            print('Copying PELE and Adaptive configuration files')
        self._copyPELEConfiguration()

        # Copy PELE topology files to analysis folder
        if self.verbose:
            print('Copying PELE topology files')
        self._copyPELETopology()

        # Copy PELE trajectories to analysis folder
        if trajectories:
            if self.verbose:
                print('Copying PELE trajectory files')
            self._copyPELETrajectories()

        # Set dictionary with Chain IDs to match mdtraj indexing
        self._setChainIDs()

        # Get protein and ligand cominations wither from pele or analysis folders
        self.pele_combinations = self._getProteinLigandCombinations()

        if self.verbose:
            print('Reading PELE information for:')

        # Read PELE simulation report data
        self._readReportData()

        if read_equilibration:
            if self.verbose:
                print('Reading equilibration information from report files from:')
            # Read PELE equilibration report data
            self._readReportData(equilibration=True)
        else:
            print('Skipping equilibration information from report files.')

        # Sort protein and ligand names alphabetically for orderly iterations.
        self.proteins = sorted(self.proteins)
        self.ligands = sorted(self.ligands)

    def calculateDistances(self, atom_pairs, equilibration=False, overwrite=False,
                           verbose=False):
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
            Display function messages
        overwrite : bool
            Force recalculation of distances.
        """

        if not os.path.exists(self.data_folder+'/distances'):
            os.mkdir(self.data_folder+'/distances')

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
                    self.distances[protein][ligand].set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory','Accepted Pele Steps'], inplace=True)
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
                    for pair in atom_pairs[protein][ligand]:
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
                                pairs.append((pair[0], pair[1], pair[2]))
                                dist_label[(i1, i2, i3)] = 'angle_'

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

                    pair_lengths = set(pair_lengths)
                    if len(pair_lengths) > 1:
                        raise ValueError('Mixed number of atoms given!')
                    pair_lengths = list(pair_lengths)[0]

                    # Define labels
                    labels = [dist_label[p]+''.join([str(x) for x in p[0]])+'_'+\
                                            ''.join([str(x) for x in p[1]]) for p in atom_pairs[protein][ligand]]

                    # Create an entry for each distance
                    for label in labels:
                        self.distances[protein][ligand][label] = []

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

                            # Calculate distances
                            if pair_lengths == 2:
                                d = md.compute_distances(traj, pairs)*10
                            elif pair_lengths == 3:
                                d = md.compute_angles(traj, pairs)*10
                            elif pair_lengths == 4:
                                d = md.compute_dihedrals(traj, pairs)*10

                            # Store data
                            self.distances[protein][ligand]['Protein'] += [protein]*d.shape[0]
                            self.distances[protein][ligand]['Ligand'] += [ligand]*d.shape[0]
                            self.distances[protein][ligand]['Epoch'] += [epoch]*d.shape[0]
                            self.distances[protein][ligand]['Trajectory'] += [t]*d.shape[0]
                            self.distances[protein][ligand]['Accepted Pele Steps'] += list(range(d.shape[0]))
                            for i,l in enumerate(labels):
                                self.distances[protein][ligand][l] += list(d[:,i])

                    # Convert distances into dataframe
                    self.distances[protein][ligand] = pd.DataFrame(self.distances[protein][ligand])

                    # Save distances to CSV file
                    self.distances[protein][ligand].to_csv(distance_file)

                    # Set indexes for DataFrame
                    self.distances[protein][ligand].set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory','Accepted Pele Steps'], inplace=True)

    #     # Concatenate individual distances into a single data frame
    #     all_distances = []
    #     for protein in distances:
    #         for ligand in distances[protein]:
    #             all_distances.append(distances[protein][ligand])
    #     all_distances = pd.concat(all_distances)
    #
    # # Add distances to main dataframe
    #     self.data = self.data.merge(all_distances, left_index=True, right_index=True)

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

    def calculateRMSD(self, equilibration=True, productive=True, recalculate=False):
        """
        Calculate the RMSD of all steps regarding the input (topology) structure.
        """

        if not os.path.isdir(self.pele_folder):
            raise ValueError('Pele folder not found. RMSD cannot be calculated without pele folder')

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
                self._saveDataState(equilibration=True)

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
                steps = max(traj_series['Step'])
                x = list(range(0,steps))
                y = [t if v in traj_series['Step'].to_list() else t-0.5 for v in x]
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
        if 'Protein RMSD' not in self.data:
            raise ValueError('You must call calculateRMSD() before calling this function.')
        elif 'Protein RMSD' not in self.equilibration_data:
            raise ValueError('You must call calculateRMSD() before calling this function.')

        self.plotSimulationMetric('Protein RMSD',
                                  equilibration=equilibration,
                                  productive=productive)

    def scatterPlotIndividualSimulation(self, protein, ligand, x, y, vertical_line=None, color_column=None, size=1.0, labels_size=10.0,
                                        xlim=None, ylim=None, metrics=None, title=None, title_size=14.0, return_axis=False, dpi=300,
                                        axis=None, xlabel=None, ylabel=None, vertical_line_color='k', **kwargs):
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
        metrics : dict
            A set of metrics for filtering the data points.
        title : str
            The plot's title.
        return_axis : bool
            Whether to return the axis of this plot.
        axis : matplotlib.pyplot.axis
            The axis to use for plotting the data.
        """

        protein_series = self.data[self.data.index.get_level_values('Protein') == protein]
        if protein_series.empty:
            raise ValueError('Protein name %s not found in data!' % protein)
        ligand_series = protein_series[protein_series.index.get_level_values('Ligand') == ligand]
        if ligand_series.empty:
            raise ValueError("Ligand name %s not found in protein's %s data!" % (ligand, protein))

        # Filter points by metric
        mask = {}
        if not isinstance(metrics, type(None)):
            for metric in metrics:
                mask[(protein, ligand)] = ligand_series[metric] <= metrics[metric]
                ligand_series = ligand_series[mask[(protein, ligand)]]

        # Add distance data to ligand_series
        if len(ligand_series) != 0:
            if protein in self.distances:
                if ligand in self.distances[protein]:
                    for distance in self.distances[protein][ligand]:
                        ligand_series[distance] = self.distances[protein][ligand][distance]

        # Check if an axis has been given
        new_axis = False
        if axis == None:
            plt.figure(figsize=(4*size, 3.3*size), dpi=dpi)
            axis = plt.gca()
            new_axis = True

        # Define color columns
        color_columns = [k for k in ligand_series.keys()]
        color_columns = [k for k in color_columns if ':' not in k]
        color_columns = [k for k in color_columns if 'distance' not in k]
        color_columns = [k for k in color_columns if not k.startswith('metric_')]
        color_columns.pop(color_columns.index('Step'))

        if color_column != None:

            ascending = False
            colormap='Blues_r'

            if color_column == 'Step':
                ascending = True
                colormap='Blues'

            elif color_column == 'Epoch':
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
                    label=protein+self.separator+ligand,
                    **kwargs)
                if new_axis:
                    cbar = plt.colorbar(sc, label=color_column)
            elif color_column in color_columns:
                ligand_series = ligand_series.sort_values(color_column, ascending=ascending)
                color_values = ligand_series[color_column]
                sc = axis.scatter(ligand_series[x],
                    ligand_series[y],
                    c=color_values,
                    cmap=colormap,
                    label=protein+self.separator+ligand,
                    **kwargs)
                if new_axis:
                    cbar = plt.colorbar(sc, label=color_column)
            else:
                sc = axis.scatter(ligand_series[x],
                    ligand_series[y],
                    c=color_column,
                    label=protein+self.separator+ligand,
                    **kwargs)

        else:
            sc = axis.scatter(ligand_series[x],
                ligand_series[y],
                label=protein+self.separator+ligand,
                **kwargs)

        if not isinstance(vertical_line, type(None)):
            axis.axvline(vertical_line, c=vertical_line_color, ls='--')

        if xlabel == None:
            xlabel = x
        if ylabel == None:
            ylabel = y

        axis.set_xlabel(xlabel, fontsize=labels_size*size)
        axis.set_ylabel(ylabel, fontsize=labels_size*size)
        axis.tick_params(axis='both', labelsize=labels_size*size)

        # plt.subplots_adjust(bottom=0.1, right=0.8, top=0.8)
        plt.tight_layout()

        if title != None:
            axis.set_title(title, fontsize=title_size*size)
        if xlim != None:
            axis.set_xlim(xlim)
        if ylim != None:
            axis.set_ylim(ylim)

        if return_axis:
            return plt.gca()
        else:
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
        plt.xticks(rotation=90, fontsize=size)
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

    def bindingEnergyLandscape(self, vertical_line=None, xlim=None, ylim=None, color=None):
        """
        Plot binding energy as interactive plot.
        """

        if self.distances == {}:
            if os.path.isdir(self.pele_folder):
                raise ValueError('There are no distances in pele data. Use calculateDistances to show plot.')
            else:
                raise ValueError('There are no distances in pele data and there is no pele folder to calculate them')

        def getLigands(Protein, by_metric=True, vertical_line=None, filter_by_metric=False,color_by_metric=False,color=None):
            protein_series = self.data[self.data.index.get_level_values('Protein') == Protein]
            ligands = list(set(protein_series.index.get_level_values('Ligand').tolist()))
            interact(getDistance, Protein=fixed(Protein), Ligand=ligands, vertical_line=fixed(vertical_line),
                     by_metric=fixed(by_metric), filter_by_metric=fixed(filter_by_metric),color_by_metric=fixed(color_by_metric),color=fixed(color))

        def getDistance(Protein, Ligand, vertical_line=None, by_metric=True, filter_by_metric=False,color_by_metric=False,color=None):
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
                for d in self.distances[Protein][Ligand]:
                    if 'distance' in d:
                        distances.append(d)
                if 'RMSD' in self.data:
                    distances.append('RMSD')

            if color == None:
                color_columns = [k for k in ligand_series.keys()]
                color_columns = [k for k in color_columns if ':' not in k]
                color_columns = [k for k in color_columns if 'distance' not in k]
                color_columns = [k for k in color_columns if not k.startswith('metric_')]
                color_columns = [None, 'Epoch']+color_columns
                del color_columns[color_columns.index('Binding Energy')]
                color_object = color_columns
            else:
                color_object = fixed(color)

            if filter_by_metric or color_by_metric:# Add checks for the given pele data pandas df
                metrics = [k for k in ligand_series.keys() if 'metric_' in k]
                metrics_sliders = {}

                for m in metrics:
                    m_slider = FloatSlider(
                                    value=4.0,
                                    min=0,
                                    max=30,
                                    step=0.1,
                                    description=m+':',
                                    disabled=False,
                                    continuous_update=False,
                                    orientation='horizontal',
                                    readout=True,
                                    readout_format='.2f',
                                )
                    metrics_sliders[m] = m_slider

                if color_by_metric:
                    color_object = 'Color by metric'

                interact(_bindingEnergyLandscape,
                         Protein=fixed(Protein),
                         Ligand=fixed(Ligand),
                         Distance=distances,
                         Color=color_object,
                         vertical_line=fixed(vertical_line),
                         **metrics_sliders)
            else:
                interact(_bindingEnergyLandscape,
                         Protein=fixed(Protein),
                         Ligand=fixed(Ligand),
                         Distance=distances,
                         Color=color_object,
                         vertical_line=fixed(vertical_line))

        def _bindingEnergyLandscape(Protein, Ligand, Distance, Color, vertical_line=None, **metrics_sliders):

            if isinstance(metrics_sliders, type(None)):
                    self.scatterPlotIndividualSimulation(Protein, Ligand, Distance, 'Binding Energy', xlim=xlim, ylim=ylim,
                                                     vertical_line=vertical_line, color_column=Color)
            elif Color == 'Color by metric':

                axis = self.scatterPlotIndividualSimulation(Protein, Ligand, Distance, 'Binding Energy', xlim=xlim, ylim=ylim,
                                                     vertical_line=vertical_line, color_column='k',return_axis=True,
                                                     metrics=None)

                self.scatterPlotIndividualSimulation(Protein, Ligand, Distance, 'Binding Energy', xlim=xlim, ylim=ylim,
                                     vertical_line=vertical_line, color_column='r',
                                     metrics=metrics_sliders,axis=axis, alpha=0.05)

            else:
                self.scatterPlotIndividualSimulation(Protein, Ligand, Distance, 'Binding Energy', xlim=xlim, ylim=ylim,
                                                     vertical_line=vertical_line, color_column=Color,
                                                     metrics=metrics_sliders)


        interact(getLigands, Protein=sorted(self.proteins), vertical_line=fixed(vertical_line),
                 by_metric=False, filter_by_metric=False,color_by_metric=False,color=fixed(color))

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

    def getDistances(self, protein, ligand):
        """
        Returns the distance associated to a specific protein and ligand simulation
        """

        if protein not in self.distances:
            raise ValueError('There are no distances for protein %s. Use calculateDistances to obtain them.' % protein)
            #print('WARNING: There are no distances for protein %s. Use calculateDistances to obtain them.' % protein)
        elif ligand not in self.distances[protein]:
            raise ValueError('There are no distances for protein %s and ligand %s. Use calculateDistances to obtain them.' % (protein, ligand))
            #print('WARNING: There are no distances for protein %s and ligand %s. Use calculateDistances to obtain them.' % (protein, ligand))

        if not os.path.isdir(self.pele_folder):
            raise ValueError('There are no distances in pele data and there is no pele folder to calculate them')

        distances = []
        for d in self.distances[protein][ligand]:
            if 'distance' in d:
                distances.append(d)
        return distances

    def plotCatalyticPosesFraction(self, initial_threshold=4.5):
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

    def plotCatalyticBindingEnergyDistributions(self, initial_threshold=4.5):
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

    def bindingFreeEnergyCatalyticDifferenceMatrix(self, initial_threshold=3.5, store_values=False, lig_label_rot=90,
                matrix_file='catalytic_matrix.npy', models_file='catalytic_models.json', max_metric_threshold=30, pele_data=None, KT=0.593):

        def _bindingFreeEnergyMatrix(KT=KT, sort_by_ligand=None, dA=True, Ec=False, Enc=False, models_file='catalytic_models.json',
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
            ax = plt.gca()
            ax.set_xticklabels(self.ligands, rotation=lig_label_rot)
            plt.xticks(np.arange(0,len(self.ligands)), self.ligands, rotation=lig_label_rot)
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
                        readout_format='.1f',
                    )

        dA = Checkbox(value=False,
                     description='$\delta A$')
        Ec = Checkbox(value=True,
                     description='$E_{B}^{C}$')
        Enc = Checkbox(value=False,
                     description='$E_{B}^{NC}$')

        ligand_ddm = Dropdown(options=self.ligands+['by_protein'])

        interact(_bindingFreeEnergyMatrix, KT=KT_slider, sort_by_ligand=ligand_ddm, pele_data=fixed(pele_data),
                 dA=dA, Ec=Ec, Enc=Enc, models_file=fixed(models_file), lig_label_rot=fixed(lig_label_rot), **metrics_sliders)

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

        if not os.path.isdir(self.pele_folder):
            raise ValueError('Pele folder not found. There are no trajectories.')

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

        for name in catalytic_labels:
            if 'metric_'+name in self.data.keys() and not overwrite:
                print('Combined metric %s already added. Give overwrite=True to combine again the distances.' % name)
            else:
                values = []
                for protein, ligand in sorted(self.pele_combinations):
                    distances = catalytic_labels[name][protein][ligand]
                    values += self.distances[protein][ligand][distances].min(axis=1).tolist()
                self.data['metric_'+name] = values
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

    def getBestPELEPoses(self, filter_values, proteins=None, ligands=None, column='Binding Energy', n_models=1, return_failed=False):
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

            # If a list of proteins is given skip proteins not in the list
            if proteins != None:
                if protein not in proteins:
                    continue

            protein_series = self.data[self.data.index.get_level_values('Protein') == model]
            for ligand in self.ligands:

                # If a list of ligands is given skip ligands not in the list
                if ligands != None:
                    if ligand not in ligands:
                        continue

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

    def getBestPELEPosesIteratively(self, metrics, column='Binding Energy', ligands=None, proteins=None,
                                    min_threshold=3.5, max_threshold=5.0, step_size=0.1):
        """
        Extract best poses iteratively using all given metrics simoultaneously.
        """
        extracted = []
        selected_indexes = []

        for t in np.arange(min_threshold, max_threshold+(step_size/10), step_size):
            filter_values = {m:t for m in metrics}
            best_poses = self.getBestPELEPoses(filter_values, column=column, n_models=1,
                                               proteins=proteins, ligands=ligands)
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

    def extractPELEPoses(self, pele_data, output_folder, separator=None, keep_chain_names=True):
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

        if separator == None:
            separator = self.separator

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

                    traj = pele_trajectory.loadTrajectoryFrames(ligand_data,
                                                                self.trajectory_files[protein][ligand],
                                                                self.topology_files[protein][ligand])

                    # Create atom names to traj indexes dictionary
                    atom_traj_index = {}
                    for residue in traj.topology.residues:
                        residue_label = residue.name+str(residue.resSeq)
                        atom_traj_index[residue_label] = {}
                        for atom in residue.atoms:
                            if 'HOH' in residue_label and atom.name == 'O':
                                atom_name = 'OW'
                            else:
                                atom_name = atom.name
                            atom_traj_index[residue_label][atom_name] = atom.index

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
        ligand_atoms = reference.topology.select('resname '+self.ligand_names[ligand])

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
            ligand_atoms = reference.topology.select('resname '+self.ligand_names[ligand])
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

    def getFolderStructures(self, poses_folder, return_paths=False):
        """
        Iterate over the PDB files in a folder as Biopython structures. Th folder must be written
        in the format of the extractPELEPoses() function.

        Parameters
        ==========
        poses_folder : str
            Path to PELE poses extracted with extractPELEPoses() function.
        """

        for protein in os.listdir(poses_folder):
            for f in os.listdir(poses_folder+'/'+protein):
                fs = f.replace('.pdb','').split(self.separator)
                if fs[0] == protein:
                    ligand, epoch, trajectory, pele_step = fs[1:5]
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

    def getNewBoxCenters(self, pele_poses_folder, center_atoms, verbose=False):
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

        Returns
        =======
        box_centers : dict
            Dictionary by (protein, ligand) that contains the new box center coordinates
        """

        # Get new box centers
        box_centers = {}
        for index, structure in self.getFolderStructures(pele_poses_folder):

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
                    chain_id, residue_id, atom_name = center_atoms
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
                    if residue.resname == residue_name:
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
                raise ValueError('Atom could not be match for model %s' % index)

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

        return box_centers

    def setUpPELECalculation(self, pele_folder, models_folder, input_yaml, box_centers=None, distances=None, ligand_index=1,
                             box_radius=10, steps=100, debug=False, iterations=3, cpus=96, equilibration_steps=100, ligand_energy_groups=None,
                             separator='-', use_peleffy=True, usesrun=True, energy_by_residue=False, ebr_new_flag=False, ninety_degrees_version=False,
                             analysis=False, energy_by_residue_type='all', peptide=False, equilibration_mode='equilibrationLastSnapshot',
                             spawning='independent', continuation=False, equilibration=True, skip_models=None, skip_ligands=None,
                             extend_iterations=False, only_models=None, only_ligands=None, ligand_templates=None, seed=12345, log_file=False,
                             simulation_type=None, nonbonded_energy=None, nonbonded_energy_type='all', nonbonded_new_flag=False):
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

                    pele_distances = [(x.split('_')[1:3][0], x.split('_')[1:3][1]) for x in self.getDistances(protein, ligand)]
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

                        if ligand in templates:
                            iyf.write("templates:\n")
                            iyf.write(' - "LIGAND_TEMPLATE_PATH_ROT"\n')
                            iyf.write(' - "LIGAND_TEMPLATE_PATH_Z"\n')
                            iyf.write("skip_ligand_prep:\n")
                            iyf.write(' - "'+ligand_pdb_name[ligand]+'"\n')

                        iyf.write("box_radius: "+str(box_radius)+"\n")
                        if isinstance(box_centers, type(None)) and peptide:
                            raise ValueError('You must give per-protein box_centers when docking peptides!')
                        if not isinstance(box_centers, type(None)):
                            if not all(isinstance(x, float) for x in box_centers[model]):
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
                                box_center += '  - '+str(coord)+'\n'
                            iyf.write("box_center: \n"+box_center)

                        # energy by residue is not implemented in PELE platform, therefore
                        # a scond script will modify the PELE.conf file to set up the energy
                        # by residue calculation.
                        if debug or energy_by_residue or peptide or nonbonded_energy != None:
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
                    if not continuation:
                        command += 'python -m pele_platform.main input.yaml\n'
                    if continuation:
                        debug_line = False
                        restart_line = False
                        with open(pele_folder+'/'+protein+separator+ligand+'/'+'input_restart.yaml', 'w') as oyml:
                            with open(pele_folder+'/'+protein+separator+ligand+'/'+'input.yaml') as iyml:
                                for l in iyml:
                                    if 'debug: true' in l:
                                        debug_line = True
                                        oyml.write('restart: true\n')
                                        oyml.write('adaptive_restart: true\n')
                                    elif 'restart: true' in l:
                                        continue
                                    oyml.write(l)
                                if not debug_line:
                                    oyml.write('restart: true\n')
                                    oyml.write('adaptive_restart: true\n')

                        if extend_iterations:
                            _copyScriptFile(pele_folder, 'extendAdaptiveIteartions.py')
                            extend_script_name = '._extendAdaptiveIteartions.py'
                            command += 'python ../'+extend_script_name+' output\n'

                        command += 'python -m pele_platform.main input_restart.yaml\n'

                    elif energy_by_residue:
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
                        with open(pele_folder+'/'+protein+separator+ligand+'/'+'input_restart.yaml', 'w') as oyml:
                            with open(pele_folder+'/'+protein+separator+ligand+'/'+'input.yaml') as iyml:
                                for l in iyml:
                                    if 'debug: true' in l:
                                        l = 'restart: true\n'
                                    oyml.write(l)
                        command += 'python -m pele_platform.main input_restart.yaml\n'
                    elif peptide:
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

    def _saveDataState(self, equilibration=False):
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
            data = pd.read_csv(csv_file)
            data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps'], inplace=True)
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
            return

        # Load input PDB with Bio.PDB and mdtraj
        pele_dir = self.pele_directories[protein][ligand]
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
        else:
            # Create analysis folder
            os.mkdir(self.data_folder)

            # Create PELE input folders
            os.mkdir(self.data_folder+'/pele_inputs')

            # Create PELE configuration folders
            os.mkdir(self.data_folder+'/pele_configuration')

            # Create PELE input folders
            if trajectories:
                os.mkdir(self.data_folder+'/pele_trajectories')

            # Create PELE input folders
            os.mkdir(self.data_folder+'/pele_topologies')

            # Create PELE distances
            os.mkdir(self.data_folder+'/distances')

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

        if pele_combinations == []:
            raise ValueError('No PELE data was found.')

        return pele_combinations

    def _readReportData(self, equilibration=False):
        """
        Read report data from PELE simulation report files.
        """

        if equilibration:
            reports_dict= self.equilibration['report']
        else:
            reports_dict = self.report_files

        # Iterate PELE folders to read report files
        report_data = []
        for protein, ligand in self.pele_combinations:
            if self.verbose:
                print('\t'+protein+self.separator+ligand, end=' ')
                start = time.time()

            # Check whether protein and ligand report files are present
            if protein not in reports_dict or ligand not in reports_dict[protein]:
                report_files = None
            else:
                report_files = reports_dict[protein][ligand]

            if report_files == {} and equilibration and self.csv_equilibration_files == {}:
                print('WARNING: No equilibration data found for simulation %s-%s' % (protein, ligand))
                continue

            # Read report files into panda dataframes
            data, distance_data = pele_read.readReportFiles(report_files,
                                                            protein,
                                                            ligand,
                                                            ebr_threshold=0.1,
                                                            separator=self.separator,
                                                            equilibration=equilibration,
                                                            data_folder_name=self.data_folder)

            # Skip of dataframe is None
            if isinstance(data, type(None)):
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

            # Save distance data
            if not equilibration:
                self.distances.setdefault(protein,{})
                self.distances[protein][ligand] = distance_data

                # Define a different distance output file for each pele run
                distance_file = self.data_folder+'/distances/'+protein+self.separator+ligand+'.csv'

                # Save distances to CSV file
                self.distances[protein][ligand].to_csv(distance_file)

        if report_data == [] and equilibration:
            self.equilibration_data = None
            return

        if equilibration:
            # Merge all dataframes into a single dataframe
            self.equilibration_data = pd.concat(report_data)
            self._saveDataState(equilibration=equilibration)
            self.equilibration_data = None
            gc.collect()
            self._recoverDataState(remove=True, equilibration=equilibration)
        else:
            # Merge all dataframes into a single dataframe
            self.data = pd.concat(report_data)

            # Save and reload dataframe to avoid memory fragmentation
            self._saveDataState()
            self.data = None
            gc.collect()
            self._recoverDataState(remove=True)

    def _copyPELEInputs(self):
        """
        Copy PELE input files to analysis folder for easy setup PELE reruns.
        """
        for protein in self.pele_directories:
            for ligand in self.pele_directories[protein]:
                dir = self.data_folder+'/pele_inputs/'+protein+self.separator+ligand
                if not os.path.exists(dir):
                    os.mkdir(dir)
                for f in os.listdir(self.pele_directories[protein][ligand]):
                    orig = self.pele_directories[protein][ligand]+'/'+f
                    dest = self.data_folder+'/pele_inputs/'+protein+self.separator+ligand+'/'+f
                    if f.endswith('.pdb'):
                        shutil.copyfile(orig, dest)
                    elif f.endswith('.yaml'):
                        shutil.copyfile(orig, dest)

    def _copyPELEConfiguration(self):
        for protein in self.pele_directories:
            for ligand in self.pele_directories[protein]:

                output_dir = self.pele_directories[protein][ligand]+'/'+self.pele_output_folder+'/output'

                # Check if output folder exists
                if not os.path.exists(output_dir):
                    print('Output folder not found for %s-%s PELE calculation.' % (protein, ligand))
                    continue

                dir = self.data_folder+'/pele_configuration/'+protein+self.separator+ligand
                if not os.path.exists(dir):
                    os.mkdir(dir)
                for f in os.listdir(self.pele_directories[protein][ligand]+'/'+self.pele_output_folder):
                    orig = self.pele_directories[protein][ligand]+'/'+self.pele_output_folder+'/'+f
                    dest = self.data_folder+'/pele_configuration/'+protein+self.separator+ligand+'/'+f
                    if f.endswith('.conf'):
                        shutil.copyfile(orig, dest)

    def _copyPELETopology(self):
        """
        Copy topology files to analysis folder.
        """
        for protein in self.topology_files:
            for ligand in self.topology_files[protein]:
                dir = self.data_folder+'/pele_topologies/'+protein+self.separator+ligand
                if not os.path.exists(dir):
                    os.mkdir(dir)
                orig = self.topology_files[protein][ligand]
                dest = dir+'/'+protein+self.separator+ligand+'.pdb'
                if orig != dest:
                    shutil.copyfile(orig, dest)
                    self.topology_files[protein][ligand] = dest

    def _copyPELETrajectories(self):
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
                        if orig != dest: # Copy only they are not found in the analysis folder
                            shutil.copyfile(orig, dest)
                            self.trajectory_files[protein][ligand][epoch][traj] = dest

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
                if protein not in self.equilibration['report']:
                    self.equilibration['report'][protein] = {}
                if protein not in self.equilibration['trajectory']:
                    self.equilibration['trajectory'][protein] = {}
                self.pele_directories[protein][ligand] = pele_dir

                # Set names for output and input folders
                output_dir = pele_dir+'/'+self.pele_output_folder+'/output'
                input_dir = pele_dir+'/'+self.pele_output_folder+'/input'

                # Check if output folder exists
                if not os.path.exists(output_dir):
                    print('Output folder not found for %s-%s PELE calculation.' % (protein, ligand))
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

                # Add protein and ligand to attribute lists
                if protein not in self.proteins:
                    self.proteins.append(protein)
                if ligand not in self.ligands:
                    self.ligands.append(ligand)

        # Sort protein and ligands for easy iteration
        self.proteins = sorted(self.proteins)
        self.ligands = sorted(self.ligands)

    def _setChainIDs(self):

        # Crea Bio PDB parser and io
        parser = PDB.PDBParser()
        io = PDB.PDBIO()

        # Read complex chain ids
        self.structure = {}
        self.ligand_structure = {}
        self.md_topology = {}

        if not os.path.exists(self.data_folder+'/chains_ids.json') or not os.path.exists(self.data_folder+'/atom_indexes.json') or self.force_reading:
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
                    input_pdb = self._getInputPDB(protein, ligand)
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

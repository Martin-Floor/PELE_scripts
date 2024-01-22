import pandas as pd
from multiprocessing import Pool, cpu_count
import os
import mdtraj as md
import numpy as np

class distances:

    def __init__(self, pele_analysis):

        self.pele_analysis = pele_analysis
        self.atom_indexes = pele_analysis.atom_indexes
        self.distances = pele_analysis.distances
        self.trajectory_files = pele_analysis.trajectory_files
        self.topology_files = pele_analysis.topology_files
        self.data_folder = pele_analysis.data_folder
        self.separator = pele_analysis.separator

        if not os.path.exists(self.data_folder+'/distances'):
            os.mkdir(self.data_folder+'/distances')

    def calculateDistances(self, atom_pairs, overwrite=False, verbose=False, cpus=None, skip_missing=False):

        # Iterate all PELE protein + ligand entries

        distance_jobs = []
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
                    self.distances[protein][ligand].set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps'], inplace=True)
                else:
                    self.distances[protein][ligand] = {}
                    self.distances[protein][ligand]['Protein'] = []
                    self.distances[protein][ligand]['Ligand'] = []
                    self.distances[protein][ligand]['Epoch'] = []
                    self.distances[protein][ligand]['Trajectory'] = []
                    self.distances[protein][ligand]['Accepted Pele Steps'] = []

                pairs = []
                dist_label = {}
                pair_lengths = []

                if skip_missing and protein not in atom_pairs:
                    continue
                elif skip_missing and ligand not in atom_pairs[protein]:
                    continue

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

                    # Update pairs based on missing labels
                    updated_pairs = []
                    for p,l in zip(pairs, labels):
                        if l in missing_labels:
                            updated_pairs.append(p)
                    pairs = updated_pairs

                    distance_jobs.append([protein, ligand, pairs, missing_labels,
                                          skip_index_append, pair_lengths])

        if len(distance_jobs) >= 1:
            # Launch distance calculations in parallel
            if cpus == None:
                cpus = cpu_count()

            pool = Pool(cpus)
            results = pool.map(self._calculateDistance, distance_jobs)
            protein_ligands = [(x[0],x[1]) for x in distance_jobs]

            for (protein, ligand), distance_data in zip(protein_ligands, results):

                # Convert distances into dataframe if dicionary
                if not isinstance(distance_data, pd.DataFrame):
                    distance_data = pd.DataFrame(distance_data)

                # Save distances to CSV file
                distance_data.reset_index()
                distance_file = self.data_folder+'/distances/'+protein+self.separator+ligand+'.csv'
                distance_data.to_csv(distance_file)

                # Set indexes for DataFrame
                distance_data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory','Accepted Pele Steps'], inplace=True)

                self.pele_analysis.distances[protein][ligand] = distance_data

    def _calculateDistance(self, arguments):
        """
        Calculate distances for PELE calculation
        """

        protein, ligand, pairs, labels, skip_index_append, pair_lengths = arguments

        # Load one trajectory at the time
        trajectory_files = self.trajectory_files[protein][ligand]
        topology_file = self.topology_files[protein][ligand]

        # Get topology
        topology = md.load(topology_file).topology

        # Create an entry for each distance
        for label in labels:
            self.distances[protein][ligand][label] = []

        # Compute distances and add them to the dicionary
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
                    d = np.rad2deg(md.compute_angles(traj, pairs))
                    # d = md.compute_angles(traj, pairs)

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
                for i,l in enumerate(labels):
                    self.distances[protein][ligand][l] += list(d[:,i])

        return self.distances[protein][ligand]

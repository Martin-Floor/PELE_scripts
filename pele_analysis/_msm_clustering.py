try:
    import pyemma
except ImportError as e:
    raise ValueError('pyemma python module not avaiable. Please install it to use this function.')

import mdtraj as md
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import itertools
from scipy.ndimage import gaussian_filter
from ipywidgets import interact, fixed #FloatSlider, IntSlider, FloatRangeSlider, VBox, HBox, interactive_output, Dropdown, Checkbox

class ligand_msm:

    def __init__(self, pele_analysis):
        """
        MSM class for analysing ligand trajectories.
        """

        # Define attributes
        self.pele_analysis = pele_analysis
        self.trajectories = {}
        self.all_trajectories = []
        self.topology = {}
        self.ligand_atoms = {}
        self.features = {}
        self.data = {}
        self.tica_output = {}
        self.tica_concatenated = {}
        self.all_data = {}
        self.all_tica = {}
        self.all_tica_output = {}
        self.all_tica_concatenated = {}
        self.metric_features = {}

        # Get individual ligand-only trajectories
        print('Getting individual ligand trajectories')
        for protein, ligand in self.pele_analysis.pele_combinations:

            self.trajectories[(protein, ligand)], self.topology[ligand] = self.pele_analysis.getLigandTrajectoryPerTrajectory(protein, ligand, return_paths=True)

            # Gather all trajectories
            for t in self.trajectories[(protein, ligand)]:
                self.all_trajectories.append(t)

            # Get ligand atom names
            if ligand not in self.ligand_atoms:
                top_traj = md.load(self.topology[ligand])
                self.ligand_atoms[ligand] = [a.name for a in top_traj.topology.atoms]

            # Create featurizer
            if ligand not in self.features:
                self.features[ligand] = pyemma.coordinates.featurizer(self.topology[ligand])

    def addFeature(self, feature, ligand):
        """
        """
        implemented_features = ['positions', 'metrics']

        if feature not in implemented_features:
            raise ValueError('Feature %s not implemented. try: %s' % (feature, implemented_features))

        if feature == 'positions':
            self.features[ligand].add_selection(self.features[ligand].select('all'))

        if feature == 'metrics':

            # Add metric features to ligand
            if ligand not in self.metric_features:
                self.metric_features[ligand] = {}

            # Get metrics
            metrics = []
            for m in self.pele_analysis.data.keys():
                if m.startswith('metric_'):
                    metrics.append(m)

            # Get metrics data
            ligand_data = self.pele_analysis.data[self.pele_analysis.data.index.get_level_values('Ligand') == ligand]
            for protein in self.pele_analysis.proteins:

                # Add metric features to ligand
                if protein not in self.metric_features[ligand]:
                    self.metric_features[ligand][protein] = []

                protein_data = ligand_data[ligand_data.index.get_level_values('Protein') == protein]
                for trajectory in ligand_data.index.levels[3]:
                    trajectory_data = protein_data[protein_data.index.get_level_values('Trajectory') == trajectory]
                    self.metric_features[ligand][protein].append(trajectory_data[metrics].to_numpy())

    def getFeaturesData(self, ligand):
        """
        """
        if ligand not in self.data:
            self.data[ligand] = {}

        if ligand not in self.all_data:
            self.all_data[ligand] = []

        for protein in self.pele_analysis.proteins:

            self.data[ligand][protein] = pyemma.coordinates.load(self.trajectories[(protein, ligand)], features=self.features[ligand])
            self.all_data[ligand] += self.data[ligand][protein]

            # Add metric features
            if ligand in self.metric_features:
                for t in range(len(self.metric_features[ligand][protein])):

                    assert self.metric_features[ligand][protein][t].shape[0] == self.data[ligand][protein][t].shape[0]

                    cct = np.concatenate([self.data[ligand][protein][t],
                                    self.metric_features[ligand][protein][t]],
                                    axis=0)
                    print(cct.shape)

    def calculateTICA(self, ligand, lag_time):
        """
        """

        # Create TICA based on all ligand simulations
        self.all_tica[ligand] = pyemma.coordinates.tica(self.all_data[ligand], lag=lag_time)
        self.all_tica_output[ligand] = self.all_tica[ligand].get_output()
        self.all_tica_concatenated[ligand] = np.concatenate(self.all_tica_output[ligand])
        self.ndims = self.all_tica_concatenated[ligand].shape[1]

        # Transorm individual protein+ligand trajectories into TICA mammping
        if ligand not in self.tica_output:
            self.tica_output[ligand] = {}
        if ligand not in self.tica_concatenated:
            self.tica_concatenated[ligand] = {}
        for protein in self.data[ligand]:
            self.tica_output[ligand][protein] = self.all_tica[ligand].transform(self.data[ligand][protein])
            self.tica_concatenated[ligand][protein] = np.concatenate(self.tica_output[ligand][protein])

    def plotLagTimeVsTICADim(self, ligand, max_lag_time):
        lag_times = []
        dims = []
        for lt in range(1, max_lag_time+1):
            self.calculateTICA(ligand,lt)
            ndim = self.all_tica_concatenated[ligand].shape[1]
            lag_times.append(lt)
            dims.append(ndim)

        plt.figure(figsize=(4,2))
        Xa = np.array(lag_times)
        plt.plot(Xa,dims)
        plt.xlabel('Lag time [ns]', fontsize=12)
        plt.ylabel('Nbr. of dimensions holding\n95% of the kinetic variance', fontsize=12)

    def plotTICADistribution(self, ligand, max_tica=10):
        """
        """
        fig, axes = plt.subplots(1, 1, figsize=(12, max_tica))
        pyemma.plots.plot_feature_histograms(
            self.all_tica_concatenated[ligand][:,:max_tica],
            ax=axes,
            feature_labels=['IC'+str(i) for i in range(1, max_tica+1)],
            ylog=True)
        fig.tight_layout()

    def plotTICADensity(self, ligand, ndims=4):
        """
        """
        IC = {}
        for i in range(ndims):
            IC[i] = self.all_tica_concatenated[ligand][:, i]

        combinations = list(itertools.combinations(range(ndims), r=2))
        fig, axes = plt.subplots(len(combinations), figsize=(7, 5*len(combinations)), sharey=True, sharex=True)
        for i,c in enumerate(combinations):
            if len(combinations) <= 1:
                pyemma.plots.plot_density(*np.array([IC[c[0]], IC[c[1]]]), ax=axes, logscale=True)
                axes.set_xlabel('IC '+str(c[0]+1))
                axes.set_ylabel('IC '+str(c[1]+1))
            else:
                pyemma.plots.plot_density(*np.array([IC[c[0]], IC[c[1]]]), ax=axes[i], logscale=True)
                axes[i].set_xlabel('IC '+str(c[0]+1))
                axes[i].set_ylabel('IC '+str(c[1]+1))

    def plotFreeEnergy(self):
        """
        """

        def getLigands(Protein):
            ligands = []
            for protein, ligand in self.pele_analysis.pele_combinations:
                if Protein == 'all':
                    ligands.append(ligand)
                else:
                    if protein == Protein:
                        ligands.append(ligand)

            if Protein == 'all':
                ligands = list(set(ligands))

            interact(_plotBindingEnergy, Protein=fixed(Protein), Ligand=ligands)

        def _plotBindingEnergy(Protein, Ligand):
            if Protein == 'all':
                _plot_Nice_PES(self.all_tica_concatenated[Ligand], bins=100, size=2, sigma=1.0)
            else:
                _plot_Nice_PES(self.tica_concatenated[Ligand][Protein], bins=100, size=2, sigma=1.0)

        interact(getLigands, Protein=sorted(self.pele_analysis.proteins)+['all'])


def _plot_Nice_PES(input_data, bins=90, sigma=0.99, title=False, size = 1):

    matplotlib.style.use("seaborn-paper")

    plt.figure(figsize=(4*size, 3.3*size))

    fig, ax = plt.subplots()

    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    alldata=np.vstack(input_data)

    min1=np.min(alldata[:,0])
    max1=np.max(alldata[:,0])
    min2=np.min(alldata[:,1])
    max2=np.max(alldata[:,1])

    tickspacing1=1.0
    tickspacing2=1.0

    z,x,y = np.histogram2d(alldata[:,0], alldata[:,1], bins=bins)
    z += 0.1

    # compute free energies
    F = -np.log(z)

    # contour plot
    extent = [x[0], x[-1], y[0], y[-1]]

    plt.xticks([])
    plt.yticks([])

    data = gaussian_filter((F.T)*0.592-np.min(F.T)*0.592, sigma)

    levels=np.linspace(0, np.max(data)-0.5, num=9)

    plt.contour(data, colors='black', linestyles='solid', alpha=0.7,
                cmap=None, levels=levels, extent=extent)

    plt.contourf(data, alpha=0.5, cmap='jet', levels=levels, extent=extent)
    plt.xlabel('IC 1', fontsize=10*size)
    plt.ylabel('IC 2', fontsize=10*size)

    if title:
        plt.title(title, fontsize = 20*size, y=1.02)

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.8)

    cax = plt.axes([0.81, 0.1, 0.02, 0.7])

    plt.colorbar(cax=cax, format='%.1f').set_label('Free energy [kcal/mol]',
                                                   fontsize=10*size,
                                                   labelpad=5,
                                                   y=0.5)
    cax.axes.tick_params(labelsize=10*size)

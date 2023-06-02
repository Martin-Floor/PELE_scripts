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
        self.kmeans_clusters = {}

        # Get individual ligand-only trajectories
        print('Getting individual ligand trajectories')
        for protein, ligand in self.pele_analysis.pele_combinations:

            self.topology.setdefault(protein, {})

            self.trajectories[(protein, ligand)], self.topology[protein][ligand] = self.pele_analysis.getLigandTrajectoryPerTrajectory(protein, ligand, return_paths=True)

            # Gather all trajectories
            for t in self.trajectories[(protein, ligand)]:
                self.all_trajectories.append(t)

            # Get ligand atom names
            if ligand not in self.ligand_atoms:
                top_traj = md.load(self.topology[protein][ligand])
                self.ligand_atoms[ligand] = [a.name for a in top_traj.topology.atoms]

            # Create featurizer
            if ligand not in self.features:
                self.features[ligand] = pyemma.coordinates.featurizer(self.topology[protein][ligand])

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

            # Add metric features
            if ligand in self.metric_features:
                for t in range(len(self.metric_features[ligand][protein])):

                    assert self.metric_features[ligand][protein][t].shape[0] == self.data[ligand][protein][t].shape[0]

                    self.data[ligand][protein][t] = np.concatenate([self.data[ligand][protein][t],
                                                    self.metric_features[ligand][protein][t]],
                                                    axis=1)
            self.all_data[ligand] += self.data[ligand][protein]

    def calculateTICA(self, ligand, lag_time, dim=-1):
        """
        """

        # Create TICA based on all ligand simulations
        self.all_tica[ligand] = pyemma.coordinates.tica(self.all_data[ligand], lag=lag_time, dim=dim)
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

    def plotTICADensity(self, ligand, ndims=None):
        """
        """

        if ndims == None:
            ndims = self.ndims

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

    def getMetricData(self, ligand, protein, metric):
        """
        """
        metric_data = []
        ligand_data = self.pele_analysis.data[self.pele_analysis.data.index.get_level_values('Ligand') == ligand]
        protein_data = ligand_data[ligand_data.index.get_level_values('Protein') == protein]
        for trajectory in ligand_data.index.levels[3]:
            trajectory_data = protein_data[protein_data.index.get_level_values('Trajectory') == trajectory]
            metric_data.append(trajectory_data[metric].to_numpy())

        return metric_data

    def calculateKMeans(self, ligand, n_clusters=5, max_iter=500, stride=1, include_metrics=None):
        """
        Calculate clusters for the sampled TICA space usign the kmeans algorithm.

        Parameters
        ==========
        ligand : str
            Name of the ligand to clusterise its trajectories.
        n_clusters : int
            Number of clusters to compute.
        max_iter : int
            Maximum number of iterations for the Kmean algorithm (pyemma.coordinates.cluster_kmeans)
        stride : int
            Stride paramter of the Kmean algorithm (pyemma.coordinates.cluster_kmeans)
        include_metrics : (str, list)
            Do you want to include metrics in the clustering? which ones?
        """

        if isinstance(include_metrics, str):
            include_metrics = [include_metrics]

        if isinstance(include_metrics, type(None)):
            include_metrics = []

        if not isinstance(include_metrics, list):
            raise ValueError('include_metrics should be a single metric string or a list of metric strings')

        for i,m in enumerate(include_metrics):
            if not m.startswith('metric_'):
                include_metrics[i] = 'metric_'+m

        self.kmeans = {}
        self.kmeans_centers = {}
        self.kmeans_metrics = {}
        self.kmeans_metrics_conversion = {}
        self.kmeans_clusters = {}
        self.kmeans_clusters_concatenated = {}


        self.kmeans[ligand] = {}
        self.kmeans_clusters[ligand] = {}
        self.kmeans_clusters_concatenated[ligand] = {}
        self.kmeans_centers[ligand] = {}
        self.kmeans_metrics[ligand] = []
        self.kmeans_metrics_conversion[ligand] = {}

        for protein in self.pele_analysis.proteins:

            self.kmeans_metrics_conversion[ligand][protein] = {}

            # Get clustering data
            data = self.tica_output[ligand][protein]

            # Add normalised metrics if given
            for m in include_metrics:
                avg_m = np.average(self.pele_analysis.data[m])
                std_m = np.std(self.pele_analysis.data[m])

                n_metric = (self.pele_analysis.getProteinAndLigandData(protein, ligand)[m]-avg_m)/std_m

                # Add average and std to recover clustering coordinates
                self.kmeans_metrics[ligand].append(m)
                self.kmeans_metrics_conversion[ligand][protein][m] = (avg_m, std_m)

                # Add metric columns to each trajectory separatedly
                for i,t in enumerate(n_metric.index.levels[3]):
                    traj_data = n_metric[n_metric.index.get_level_values('Trajectory') == t]
                    traj_data = traj_data.to_numpy() # convert to numpy
                    traj_data = traj_data.reshape((traj_data.shape[0], 1)) # reshape for concatenation
                    data[i] = np.concatenate([data[i], traj_data], axis=1) # Paste with the clustering data

            # Calculate kmeans clusters
            self.kmeans[ligand][protein] = pyemma.coordinates.cluster_kmeans(data,
                                                                             k=n_clusters,
                                                                             max_iter=max_iter,
                                                                             stride=stride)

            # Store kmeans data
            self.kmeans_clusters[ligand][protein] = self.kmeans[ligand][protein].dtrajs
            self.kmeans_clusters_concatenated[ligand][protein] = np.concatenate(self.kmeans_clusters[ligand][protein])
            self.kmeans_centers[ligand][protein] = self.kmeans[ligand][protein].clustercenters

            return self.kmeans


    def plotFreeEnergy(self, max_tica=10, metric_line=None, size=1.0, sigma=1.0, bins=100, xlim=None, ylim=None,
                       mark_cluster=None, plot_clusters=True):
        """
        Plot free energy maps employing the metrics or TICA dimmensions calculated

        Paramters
        =========
        max_tica : int
            Maximum number of TICA dimmensions to calculate
        metric_line : float
            Dashed line to plot for the selected metric
        size : float
            Size paramter to control overall size of the plot
        sigma : float
            Sigma parameter for the gaussian filter employed on the contour plot (scipy.ndimage.gaussian_filter)
        bins : int
            Number of bins to divide the contourplot
        xlim : float
            X-axis limit to show
        ylim : float
            Y-axis limit to show
        mark_cluster : int
            Index of the kmean cluster to show as a bigger red dot in the map.
        plot_clusters : bool
            Do you want to plot clusters?
        """

        def getLigands(Protein, max_tica=10, metric_line=None):
            ligands = []
            for protein, ligand in self.pele_analysis.pele_combinations:
                if Protein == 'all':
                    ligands.append(ligand)
                else:
                    if protein == Protein:
                        ligands.append(ligand)

            if Protein == 'all':
                ligands = list(set(ligands))

            interact(getCoordinates, Protein=fixed(Protein), Ligand=ligands, max_tica=fixed(max_tica),
                    metric_line=fixed(metric_line))

        def getCoordinates(Protein, Ligand, max_tica=10, metric_line=None):

            if max_tica > self.ndims:
                max_tica = self.ndims

            dimmensions = []
            # Add metrics
            for m in self.pele_analysis.data.keys():
                if m.startswith('metric_'):
                    dimmensions.append(m)

            # Add TICA dimmensions
            for i in range(max_tica):
                dimmensions.append('IC'+str(i+1))

            interact(_plotBindingEnergy, Protein=fixed(Protein), Ligand=fixed(Ligand), X=dimmensions, Y=dimmensions,
                     metric_line=fixed(metric_line))

        def _plotBindingEnergy(Protein, Ligand, X='IC1', Y='IC2', metric_line=None):

            if X.startswith('IC'):
                index = int(X.replace('IC', ''))-1
                input_data1 = self.tica_concatenated[Ligand][Protein][:, index]
                x_metric_line = None
            elif X.startswith('metric_'):
                input_data1 = np.concatenate(self.getMetricData(Ligand, Protein, X))
                x_metric_line = metric_line

            if Y.startswith('IC'):
                index = int(Y.replace('IC', ''))-1
                input_data2 = self.tica_concatenated[Ligand][Protein][:, index]
                y_metric_line = None
            elif Y.startswith('metric_'):
                input_data2 = np.concatenate(self.getMetricData(Ligand, Protein, Y))
                y_metric_line = metric_line

            ax = _plot_Nice_PES(input_data1, input_data2, xlabel=X, ylabel=Y, bins=bins, size=size, sigma=sigma,
                                x_metric_line=x_metric_line, y_metric_line=y_metric_line, xlim=xlim, ylim=ylim)

            # Plot clusters if found
            while True: # Used for stoping if statments

                if not plot_clusters:
                    break

                if self.kmeans_clusters != {}:

                    # Get cluster centers
                    centers = self.kmeans_centers[Ligand][Protein]

                    if X.startswith('metric_'):
                        if X not in self.kmeans_metrics[Ligand]:
                            break # Continue if metric has not been used in the clustering
                        mi = self.kmeans_metrics[Ligand].index(X)
                        x_avg, x_std = self.kmeans_metrics_conversion[Ligand][Protein][X]
                        cpx = (centers[:,self.ndims+mi]*x_std)+x_avg # Convert cluster z-score back to metric values

                    elif X.startswith('IC'):
                        x_index = int(X.replace('IC', ''))-1
                        cpx = centers[:,x_index]

                    if Y.startswith('metric_'):
                        if Y not in self.kmeans_metrics[Ligand]:
                            break # Continue if metric has not been used in the clustering
                        mi = self.kmeans_metrics[Ligand].index(Y)
                        y_avg, y_std = self.kmeans_metrics_conversion[Ligand][Protein][Y]
                        cpy = (centers[:,self.ndims+mi]*y_std)+y_avg # Convert cluster z-score back to metric values

                    elif Y.startswith('IC'):
                        y_index = int(Y.replace('IC', ''))-1
                        cpy = centers[:,y_index]

                    ax.scatter(cpx, cpy, c='k', s=5)
                    if not isinstance(mark_cluster, type(None)):
                        ax.scatter(cpx[mark_cluster], cpy[mark_cluster], c='r', s=10)
                break

        interact(getLigands, Protein=sorted(self.pele_analysis.proteins)+['all'], max_tica=fixed(max_tica),
                 metric_line=fixed(metric_line))

def _plot_Nice_PES(input_data1, input_data2, xlabel=None, ylabel=None, bins=90, sigma=0.99, title=False, size=1,
                   x_metric_line=None, y_metric_line=None, dpi=300, title_size=14, cmax=None, title_rotation=None,
                   title_location=None, title_x=0.5, title_y=1.02, show_xticks=False, show_yticks=False,
                   xlim=None, ylim=None):

    matplotlib.style.use("seaborn-paper")

    plt.figure(figsize=(4*size, 3.3*size), dpi=dpi)

    fig, ax = plt.subplots()

    # alldata1=np.vstack(input_data1)
    # alldata2=np.vstack(input_data2)

    min1=np.min(input_data1)
    max1=np.max(input_data1)
    min2=np.min(input_data2)
    max2=np.max(input_data2)

    z,x,y = np.histogram2d(input_data1, input_data2, bins=bins)
    z += 0.1

    # compute free energies
    F = -np.log(z)

    # contour plot
    extent = [x[0], x[-1], y[0], y[-1]]

    ax.spines['right'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)

    if not show_xticks:
        plt.xticks([])
    else:
        ax.spines['bottom'].set_visible(True)

    if not show_yticks:
        plt.yticks([])
    else:
        ax.spines['left'].set_visible(True)

    data = gaussian_filter((F.T)*0.592-np.min(F.T)*0.592, sigma)

    if cmax != None:
        levels=np.linspace(0, cmax, num=9)
    else:
        levels=np.linspace(0, np.max(data)-0.5, num=9)

    plt.contour(data, colors='black', linestyles='solid', alpha=0.7,
                cmap=None, levels=levels, extent=extent)

    cnt = plt.contourf(data, alpha=0.5, cmap='jet', levels=levels, extent=extent)

    plt.xlabel(xlabel, fontsize=10*size)
    plt.ylabel(ylabel, fontsize=10*size)

    if xlim != None:
        plt.xlim(xlim)
    if ylim != None:
        plt.ylim(ylim)

    if x_metric_line != None:
        plt.axvline(x_metric_line, ls='--', c='k')
    if y_metric_line != None:
        plt.axhline(y_metric_line, ls='--', c='k')

    if title:
        plt.title(title, fontsize = title_size*size, rotation=title_rotation,
                  loc=title_location, x=title_x, y=title_y)

    plt.subplots_adjust(bottom=0.1, right=0.8, top=0.8)

    cax = plt.axes([0.81, 0.1, 0.02, 0.7])

    cbar = plt.colorbar(cax=cax, format='%.1f')
    cbar.set_label('Free energy [kcal/mol]',
                   fontsize=10*size,
                   labelpad=5,
                   y=0.5)
    cax.axes.tick_params(labelsize=10*size)

    return ax

try:
    import pyemma
    import mdtraj as md
    import numpy as np
    import matplotlib.pyplot as plt
except ImportError as e:
    raise ValueError('pyemma python module not avaiable. Please install it to use this function.')


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
        self.all_data = {}
        self.tica = {}
        self.tica_output = {}
        self.tica_concatenated = {}

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
        implemented_features = ['positions']
        if feature not in implemented_features:
            raise ValueError('Feature %s not implemented. try: %s' % (feature, implemented_features))

        if feature == 'positions':
            self.features[ligand].add_selection(self.features[ligand].select('all'))

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

    def calculateTICA(self, ligand, lag_time):
        """
        """
        self.tica[ligand] = pyemma.coordinates.tica(self.all_data[ligand], lag=lag_time)
        self.tica_output[ligand] = self.tica[ligand].get_output()
        self.tica_concatenated[ligand] = np.concatenate(self.tica_output[ligand])
        ndim = self.tica_concatenated[ligand].shape[1]

    def plotLagTimeVsTICADim(self, ligand, max_lag_time):
        lag_times = []
        dims = []
        for lt in range(1, max_lag_time+1):
            self.calculateTICA(ligand,lt)
            ndim = self.tica_concatenated[ligand].shape[1]
            lag_times.append(lt)
            dims.append(ndim)

        plt.figure(figsize=(4,2))
        Xa = np.array(lag_times)
        plt.plot(Xa,dims)
        plt.xlabel('Lag time [ns]', fontsize=12)
        plt.ylabel('Nbr. of dimensions holding\n95% of the kinetic variance', fontsize=12)
        #plt.savefig('ticadim_vs_lagtime.svg')
        print(*zip(Xa,dims))

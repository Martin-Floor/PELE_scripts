import mdtraj as md
import nglview as nv

def loadTrajectoryFrames(report_data, trajectory_files, topology_file):
    """
    Load only trajectory frames contained in the report_data object (see readReportFiles()
    function at pele_read on how is generated).

    Parameters
    ==========
    report_data : pandas.DataFrame
        Data frame containing PELE output data. Is generated by pele_read.readReportFiles()
        function.
    trajectory_files : dict
        Contains path to PELE output trajectory files. Is generated by pele_read.getTrajectoryFiles()
        function.
    topology_file : str
        Path to the PDB topology file. Can be obtained with pele_read.getTopologyFile()
        function.

    Returns
    =======
    trajectory : mdtraj.core.trajectory.Trajectory
        Trajectory containing all frames contained in the report_data object.
    """
    e_values = report_data.index.get_level_values('Epoch')
    t_values = report_data.index.get_level_values('Trajectory')
    p_steps = report_data.index.get_level_values('Pele Step')

    values = list(zip(e_values, t_values, p_steps))
    trajs = []
    for v in values:
        trajs.append(md.load_frame(trajectory_files[v[0]][v[1]], v[2], top=topology_file))
    trajectory = md.join(trajs)

    return trajectory

def showTrajectory(trajectory, residues=[]):
    """
    Show PELE trajectory center on ligand. It takes an optional list of residues to depict.

    Parameters
    ==========
    trajectory : mdtraj.core.trajectory.Trajectory
        PELE trajectory to depict
    residues : list
        List of residue indexes to depict.
    """

    show = nv.show_mdtraj(trajectory)
    show.clear_representations()
    show.add_representation('licorice', selection='ligand')
    for r in residues:
        show.add_representation('licorice', selection=str(r))
    show.add_representation('cartoon', selection='protein', color='w')
    show.center(selection='ligand')

    return show

def addMetricFromFunction(metric_function, report_data, trajectory_files, topology_file, labels=None, inplace=False):

    metrics = []
    for e in trajectory_files:
        for t in trajectory_files[e]:
            traj = md.load(trajectory_files[e][t], top=topology_file)
            metrics.append(metric_function(traj))

    metrics = np.hstack(metrics)

    n_metrics = metrics.shape[0]
    new_labels = ['Metric '+str(x).zfill(len(str(n_metrics))) for x in range(n_metrics)]
    if isinstance(labels, type(None)):
        labels = new_labels
    elif len(labels) != n_metrics:
        print('Number of labels is different from number of metrics. Generic labels were given.')
        labels = new_labels

    if inplace:
        report_data = report_data
    else:
        report_data = report_data.copy()

    for i in range(n_metrics):
        report_data[labels[i]] = metrics[i]

    return report_data
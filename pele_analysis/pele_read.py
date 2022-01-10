import os
import pandas as pd

def getTrajectoryFiles(pele_output_folder):
    """
    Retrieves the paths to the trajectory files from an Adaptive-PELE Platform output folder.
    Be mindful of iterate using the sorted() function upon the returned dictionary!

    Parameters
    ==========
    pele_output_folder : str
        Adaptive-PELE Platform output folder

    Returns
    =======
    trajectory_file : dict
        Dictionary containing the path to the trajectory files separated by epochs and then trajectories.
    """

    trajectory_file = {}
    for e in sorted(os.listdir(pele_output_folder)):
        try:
            epoch = int(e)
            trajectory_file[epoch] = {}
            for f in sorted(os.listdir(pele_output_folder+'/'+e)):
                if f.endswith('.xtc'):
                    t = int(f.split('_')[-1].split('.')[0])
                    trajectory_file[epoch][t] = pele_output_folder+'/'+e+'/'+f
        except:
            continue

    return trajectory_file

def getReportFiles(pele_output_folder):
    """
    Retrieves the paths to the report files from an Adaptive-PELE Platform output folder.
    Be mindful of iterate using the sorted() function upon the returned dictionary!

    Parameters
    ==========
    pele_output_folder : str
        Adaptive-PELE Platform output folder

    Returns
    =======
    report_file : dict
        Dictionary containing the path to the report files separated by epochs and then trajectories.
    """

    report_file = {}

    for e in sorted(os.listdir(pele_output_folder)):
        try:
            epoch = int(e)
            report_file[epoch] = {}
            for f in sorted(os.listdir(pele_output_folder+'/'+e)):
                if 'report' in f:
                    t = int(f.split('_')[-1])
                    report_file[epoch][t] = pele_output_folder+'/'+e+'/'+f
        except:
            continue

    return report_file

def getTopologyFile(pele_output_folder):
    """
    Retrieves the path to the topology (pdb) file of the Adaptive-PELE Platform
    output folder.

    Parameters
    ==========
    pele_output_folder : str
        Adaptive-PELE Platform output folder

    Returns
    =======
    topology_file : str
        Path to the topology file
    """
    for f in os.listdir(pele_output_folder+'/topologies'):
        if f.endswith('.pdb'):
            topology_file = pele_output_folder+'/topologies'+'/'+f

    return topology_file

def readReportFiles(report_files):
    """
    Merge a list of report files data into a single data frame. It adds epoch and trajectory
    information based on the input dictionary structure: report_files[epoch][trajectory].

    Parameters
    ==========
    report_files : dict
        Dictionary containing report files paths sorted by epochs and trajectories.
        See get_report_files() function's output.

    Returns
    =======
    report_data : pandas.DataFrame
        Panda dataframe contatining all reporters file information. Epoch and Trajectory
        are defined as th DataFrame indexes.
    """
    report_data = []
    for epoch in sorted(report_files):
        for traj in sorted(report_files[epoch]):
            rd = _readReportFile(report_files[epoch][traj])
            rd['Epoch'] = epoch
            rd['Trajectory'] = traj
            report_data.append(rd)
    report_data = pd.concat(report_data)
    report_data.set_index(['Epoch', 'Trajectory', 'Pele Step'], inplace=True)
    return report_data

def _readReportFile(report_file):
    """
    Reads a single report file as a pandas DataFrame.

    Parameters
    ==========
    report_file : str
        Path to the report file.

    Returns
    =======
    report_values : pandas.DataFrame
        Panda dataframe containing the report file information.

    """
    report_values = {}
    int_terms = ['Step', 'Task', 'Pele Step']
    with open(report_file) as rf:
        for i,l in enumerate(rf):
            if i == 0:
                terms = [t for t in l[1:].strip().split('    ')]
                distance_index = 1
                for t in terms:
                    if t == 'numberOfAcceptedPeleSteps':
                        t = 'Pele Step'
                    if t == 'sasaLig':
                        t = 'Ligand SASA'
                    if t == 'currentEnergy':
                        t = 'Total Energy'
                    if t == 'Binding Energy':
                        t = 'Binding Energy'
                    if 'distance' in t:
                        t = 'Relevant Distance '+str(distance_index)
                        distance_index += 1
                    report_values[t] = []
            else:
                for t,x in zip(report_values, l.split()):
                    if t in int_terms:
                        report_values[t].append(int(x))
                    else:
                        report_values[t].append(float(x))

    report_values = pd.DataFrame(report_values)

    return report_values

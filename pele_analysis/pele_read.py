import os
import pandas as pd
import numpy as np

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

def getEquilibrationReportFiles(pele_output_folder):
    """
    Retrieves the paths to the equilibration files from an Adaptive-PELE Platform output folder.
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
    for s in sorted(os.listdir(pele_output_folder)):
        if s.startswith('equilibration'):
            step = s.split('_')[1]
            report_file[step] = {}
            for f in sorted(os.listdir(pele_output_folder+'/'+s)):
                if 'report' in f:
                    t = int(f.split('_')[-1])
                    report_file[step][t] = pele_output_folder+'/'+s+'/'+f

    return report_file

def getEquilibrationTrajectoryFiles(pele_output_folder):
    """
    Retrieves the paths to the equilibration files from an Adaptive-PELE Platform output folder.
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

    trajectory_file = {}
    for s in sorted(os.listdir(pele_output_folder)):
        if s.startswith('equilibration'):
            step = s.split('_')[1]
            trajectory_file[step] = {}
            for f in sorted(os.listdir(pele_output_folder+'/'+s)):
                if f.endswith('.xtc'):
                    t = int(f.split('_')[-1].split('.')[0])
                    trajectory_file[step][t] = pele_output_folder+'/'+s+'/'+f

    return trajectory_file

def getTopologyFile(pele_input_folder):
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
    for f in os.listdir(pele_input_folder):
        if f.endswith('processed.pdb'):
            topology_file = pele_input_folder+'/'+f

    return topology_file

def readReportFiles(report_files, protein, ligand, equilibration=False, force_reading=False,
                    ebr_threshold=0.1,data_folder_name='.pele_analysis'):
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

    if equilibration:
        csv_file_name = data_folder_name+'/equilibration_data_'+protein+'_'+ligand+'.csv'
    else:
        csv_file_name = data_folder_name+'/data_'+protein+'_'+ligand+'.csv'

    if os.path.exists(csv_file_name) and not force_reading:
        report_data = pd.read_csv(csv_file_name)
    else:
        report_data = []
        for epoch in sorted(report_files):
            for traj in sorted(report_files[epoch]):
                rd = _readReportFile(report_files[epoch][traj],
                                     equilibration=equilibration,
                                     ebr_threshold=ebr_threshold)
                if equilibration:
                    step = 'Step'
                else:
                    step = 'Epoch'
                rd[step] = epoch
                rd['Trajectory'] = traj
                report_data.append(rd)

        # Check pele data can be read by the library
        try:
            report_data = pd.concat(report_data)
        except:
            if equilibration:
                print('Failed to read PELE equilibration data for %s + %s' % (protein, ligand))
            else:
                print('Failed to read PELE data for %s + %s' % (protein, ligand))
            return

        if report_data.empty:
            print('Failed to read PELE data for %s + %s' % (protein, ligand))
            return

        report_data.set_index([step, 'Trajectory', 'Accepted Pele Steps'], inplace=True)

        _saveDataToCSV(report_data, protein, ligand, equilibration=equilibration,data_folder_name=data_folder_name)

    return report_data

def _readReportFile(report_file, equilibration=False, ebr_threshold=0.1):
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
    int_terms = ['Step', 'Task', 'Accepted Pele Steps']

    with open(report_file) as rf:
        for i,l in enumerate(rf):
            if i == 0:
                terms = [t for t in l[1:].strip().split('    ')]
                for t in terms:
                    if t == 'numberOfAcceptedPeleSteps':
                        t = 'Accepted Pele Steps'
                    if t == 'sasaLig':
                        t = 'Ligand SASA'
                    if t == 'currentEnergy':
                        t = 'Total Energy'
                    if t == 'BindingEnergy':
                        t = 'Binding Energy'
                    if equilibration:
                        if t.startswith('L:1_'):
                            continue
                    report_values[t] = []
            else:
                for t,x in zip(report_values, l.split()):
                    if equilibration:
                        if t.startswith('L:1_'):
                            continue
                    if t in int_terms:
                        report_values[t].append(int(x))
                    else:
                        report_values[t].append(float(x))

        # Check for energy by residue data which is below threshold
        to_remove = []
        for t in report_values:
            if t.startswith('L:1_'):
                if np.abs(np.average(report_values[t])) <= ebr_threshold:
                    to_remove.append(t)

        for t in to_remove:
            del report_values[t]

        report_values = pd.DataFrame(report_values)

    return report_values

def _saveDataToCSV(dataframe, protein, ligand, equilibration=False,data_folder_name='.pele_analysis'):
    if not os.path.exists(data_folder_name):
        os.mkdir(data_folder_name)
    if equilibration:
        csv_file_name = data_folder_name+'/equilibration_data_'+protein+'_'+ligand+'.csv'
    else:
        csv_file_name = data_folder_name+'/data_'+protein+'_'+ligand+'.csv'
    dataframe.to_csv(csv_file_name)

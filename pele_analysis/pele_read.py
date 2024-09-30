import os
import pandas as pd
import numpy as np
import mdtraj as md

def getTrajectoryFiles(pele_output_folder):
    """
    Retrieves the paths to the trajectory files from an Adaptive-PELE Platform output folder.

    Parameters
    ==========
    pele_output_folder : str
        Adaptive-PELE Platform output folder

    Returns
    =======
    trajectory_file : dict
        Dictionary containing the path to the trajectory files separated by epochs and then trajectories.
        Be mindful of iterate using the sorted() function upon the returned dictionary!
    """

    trajectory_file = {}

    pdb_warn = True
    for e in sorted(os.listdir(pele_output_folder)):
        # Check that folder is a simulation folder
        is_sim = False
        if not os.path.isdir(pele_output_folder+'/'+e):
            continue
        else:
            for f in os.listdir(pele_output_folder+'/'+e):
                if f.startswith('trajectory_'):
                    is_sim = True
                    break
        if not is_sim:
            continue
        # Skip equilibration files
        if e.startswith('equilibration_'):
            continue

        # Add epoch to trajectory file
        epoch_directory = pele_output_folder+'/'+e
        epoch = int(e)
        trajectory_file[epoch] = {}

        for f in sorted(os.listdir(epoch_directory)):
            if f.endswith('.xtc'):
                t = int(f.split('_')[-1].split('.')[0])
                trajectory_file[epoch][t] = epoch_directory+'/'+f

            # Check if trajectory has been given as PDB
            elif f.endswith('.pdb') and f.startswith('trajectory_'):
                if pdb_warn:
                    print('Trajectories found as PDB at %s' % pele_output_folder)
                    pdb_warn = False
                    print('Converting trajectories to xtc...')

                print('Converting file %s' % epoch_directory+'/'+f, end='\r')
                traj = md.load(epoch_directory+'/'+f)
                traj.save(epoch_directory+'/'+f.replace('.pdb', '.xtc'))
                os.remove(epoch_directory+'/'+f)
                t = int(f.split('_')[-1].split('.')[0])
                trajectory_file[epoch][t] = epoch_directory+'/'+f.replace('.pdb', '.xtc')

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

    pdb_warn = True
    trajectory_file = {}
    for e in sorted(os.listdir(pele_output_folder)):

        # Check that folder is a simulation folder
        is_sim = False
        if not os.path.isdir(pele_output_folder+'/'+e):
            continue
        else:
            for f in os.listdir(pele_output_folder+'/'+e):
                if f.startswith('trajectory_'):
                    is_sim = True
                    break

        if not is_sim:
            continue

        # Skip non equilibration folders
        if not e.startswith('equilibration_'):
            continue

        # Add epoch to trajectory file
        epoch_directory = pele_output_folder+'/'+e
        epoch = int(e.split('_')[1])
        trajectory_file[epoch] = {}
        for f in sorted(os.listdir(epoch_directory)):
            if f.endswith('.xtc'):
                t = int(f.split('_')[-1].split('.')[0])
                trajectory_file[epoch][t] = epoch_directory+'/'+f

            # Check if trajectory has been given as PDB
            elif f.endswith('.pdb') and f.startswith('trajectory_'):
                if pdb_warn:
                    print('Trajectories found as PDB at %s' % epoch_directory)
                    pdb_warn = False
                    print('Converting trajectories to xtc...')

                print('Converting file %s' % epoch_directory+'/'+f, end='\r')
                traj = md.load(epoch_directory+'/'+f)
                traj.save(epoch_directory+'/'+f.replace('.pdb', '.xtc'))
                os.remove(epoch_directory+'/'+f)
                t = int(f.split('_')[-1].split('.')[0])
                trajectory_file[epoch][t] = epoch_directory+'/'+f.replace('.pdb', '.xtc')

    return trajectory_file

def getTopologyFile(pele_input_folder):
    """
    Retrieves the path to the topology (pdb) file of the Adaptive-PELE Platform
    input folder.

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

def getSpawningDictionaries(pele_folder):
    """
    Retrieves the path to the regional spawning metrics dictionaries if found

    Parameters
    ==========
    pele_folder : str
        Base protein and ligand PELE folder

    Returns
    =======
    spawning_files : dict
        Path to the spawning 'metrics' and 'thresholds' dictionaries
    """

    spawning_files = {}
    for f in os.listdir(pele_folder):
        if f == 'metrics.json':
            spawning_files['metrics'] = pele_folder+'/'+f
        elif f == 'metrics_thresholds.json':
            spawning_files['thresholds'] = pele_folder+'/'+f
        elif f == '._spawning_mapping.json':
            spawning_files['mappings'] = pele_folder+'/'+f

    return spawning_files

def getFixedFile(pele_input_folder):
    """
    Retrieves the path to the fixed (pdb) file of the Adaptive-PELE Platform
    input folder.

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
        if f.endswith('fixed.pdb'):
            fixed_file = pele_input_folder+'/'+f

    return fixed_file

def getLigandFile(pele_input_folder):
    """
    Retrieves the path to the ligand (pdb) file of the Adaptive-PELE Platform
    input folder.

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
        if f == 'ligand.pdb':
            ligand_file = pele_input_folder+'/'+f

    return ligand_file

def readReportFiles(report_files, protein, ligand, equilibration=False, force_reading=False,
                    ebr_threshold=0.1, data_folder_name='.pele_analysis', separator='-'):
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
        csv_file_name = data_folder_name+'/equilibration_data_'+protein+separator+ligand+'.csv'
    else:
        csv_file_name = data_folder_name+'/data_'+protein+separator+ligand+'.csv'

    if os.path.exists(csv_file_name) and not force_reading:
        report_data = pd.read_csv(csv_file_name)

        # Convert protein and ligand columns to strings
        report_data = report_data.astype({'Protein':'string'})
        report_data = report_data.astype({'Ligand':'string'})

        if equilibration:
            distance_data = None
            angle_data = None
            dihedral_data = None
            nonbonded_energy_data = None
        else:
            csv_distances_file = data_folder_name+'/distances/'+protein+separator+ligand+'.csv'
            csv_angles_file = data_folder_name+'/angles/'+protein+separator+ligand+'.csv'
            csv_dihedrals_file = data_folder_name+'/dihedrals/'+protein+separator+ligand+'.csv'
            csv_nonbonded_energy_file = data_folder_name+'/nonbonded_energy/'+protein+separator+ligand+'.csv'

            if os.path.exists(csv_distances_file):
                distance_data = pd.read_csv(csv_distances_file)
                distance_data = distance_data.loc[:, ~distance_data.columns.str.contains('^Unnamed')]
                indexes = ['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps']
                if 'Step' in distance_data.keys():
                    indexes.append('Step')
                distance_data.set_index(indexes, inplace=True)
            else:
                distance_data = None

            if os.path.exists(csv_angles_file):
                angle_data = pd.read_csv(csv_angles_file)
                angle_data = angle_data.loc[:, ~angle_data.columns.str.contains('^Unnamed')]
                indexes = ['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps']
                if 'Step' in angle_data.keys():
                    indexes.append('Step')
                angle_data.set_index(indexes, inplace=True)
            else:
                angle_data = None

            if os.path.exists(csv_dihedrals_file):
                dihedral_data = pd.read_csv(csv_dihedrals_file)
                dihedral_data = dihedral_data.loc[:, ~dihedral_data.columns.str.contains('^Unnamed')]
                indexes = ['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps']
                if 'Step' in dihedral_data.keys():
                    indexes.append('Step')
                dihedral_data.set_index(indexes, inplace=True)
            else:
                dihedral_data = None

            if os.path.exists(csv_nonbonded_energy_file):
                nonbonded_energy_data = pd.read_csv(csv_nonbonded_energy_file)
                nonbonded_energy_data = nonbonded_energy_data.loc[:, ~nonbonded_energy_data.columns.str.contains('^Unnamed')]
                nonbonded_energy_data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps'], inplace=True)
            else:
                nonbonded_energy_data = None

    elif report_files != None:
        report_data = []
        distance_data = []
        angle_data = []
        dihedral_data = []
        nonbonded_energy_data = []

        for epoch in sorted(report_files):
            for traj in sorted(report_files[epoch]):
                report_return = _readReportFile(report_files[epoch][traj],
                                         equilibration=equilibration,
                                         ebr_threshold=ebr_threshold,
                                         protein=protein,
                                         ligand=ligand,
                                         epoch=epoch,
                                         trajectory=traj)

                if report_return == None:
                    m = 'For protein %s and ligand %s,' % (protein, ligand)
                    m += ' no PELE data was found for epoch %s and trajectory %s.'  % (epoch, traj)
                    print(m)
                    continue

                rd, dd, ad, td, nbed = report_return

                report_data.append(rd)
                distance_data.append(dd)
                angle_data.append(ad)
                dihedral_data.append(td)
                nonbonded_energy_data.append(nbed)

        # Check pele data can be read by the library
        try:
            report_data = pd.concat(report_data)
            distance_data = pd.concat(distance_data)
            angle_data = pd.concat(angle_data)
            dihedral_data = pd.concat(dihedral_data)
            nonbonded_energy_data = pd.concat(nonbonded_energy_data)

        except:
            if equilibration:
                print('Failed to read PELE equilibration data for %s + %s' % (protein, ligand))
            else:
                print('Failed to read PELE data for %s + %s' % (protein, ligand))
            return (None, None, None)

        if report_data.empty:
            print('Failed to read PELE data for %s + %s' % (protein, ligand))
            return (None, None, None)

        report_data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps', 'Step'], inplace=True)
        distance_data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps', 'Step'], inplace=True)
        angle_data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps', 'Step'], inplace=True)
        dihedral_data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps', 'Step'], inplace=True)
        nonbonded_energy_data.set_index(['Protein', 'Ligand', 'Epoch', 'Trajectory', 'Accepted Pele Steps', 'Step'], inplace=True)

        _saveDataToCSV(report_data, protein, ligand, equilibration=equilibration,
                       separator=separator, data_folder_name=data_folder_name)
    else:
        report_data = None
        distance_data = None
        angle_data = None
        dihedral_data = None
        nonbonded_energy_data = None

    return report_data, distance_data, angle_data, dihedral_data, nonbonded_energy_data

def _readReportFile(report_file, equilibration=False, ebr_threshold=0.1, protein=None,
                    ligand=None, epoch=None, trajectory=None):
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
    distance_values = {}
    angle_values = {}
    dihedral_values = {}
    nonbonded_energy_values = {}
    all_values = []
    int_terms = ['Step', 'Accepted Pele Steps']

    with open(report_file) as rf:
        for i,l in enumerate(rf):

            # Remove C-language null characters if found.
            if '\x00' in l:
                l = l.replace('\x00', '')

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

                    if t in int_terms:
                        distance_values[t] = []
                        angle_values[t] = []
                        dihedral_values[t] = []
                        report_values[t] = []
                        nonbonded_energy_values[t] = []
                    elif t.startswith('distance_'):
                        distance_values[t] = []
                    elif t.startswith('angle_'):
                        angle_values[t] = []
                    elif t.startswith('torsion_'):
                        dihedral_values[t] = []
                    elif t.endswith(':all') or t.endswith(':lennard_jones') or t.endswith(':sgb') or t.endswith(':electrostatic'):
                        nonbonded_energy_values[t] = []
                    else:
                        report_values[t] = []
                    all_values.append(t)
            else:
                for t,x in zip(all_values, l.split()):
                    if equilibration:
                        if t.startswith('L:1_'):
                            continue
                    if t in int_terms:
                        report_values[t].append(int(x))
                        distance_values[t].append(int(x))
                        angle_values[t].append(int(x))
                        dihedral_values[t].append(int(x))
                        nonbonded_energy_values[t].append(int(x))
                    else:
                        if t.startswith('distance_'):
                            distance_values[t].append(float(x))
                        elif t.startswith('angle_'):
                            angle_values[t].append(float(x))
                        elif t.startswith('torsion_'):
                            dihedral_values[t].append(float(x))
                        elif t.endswith(':all') or t.endswith(':lennard_jones') or t.endswith(':sgb') or t.endswith(':electrostatic'):
                            nonbonded_energy_values[t].append(float(x))
                        else:
                            report_values[t].append(float(x))

        # Check for energy by residue data which is below threshold
        to_remove = []
        for t in report_values:
            if t.startswith('L:1_'):
                if np.abs(np.average(report_values[t])) <= ebr_threshold:
                    to_remove.append(t)

        # Delete ebr data below threshold
        for t in to_remove:
            del report_values[t]

        report_values = pd.DataFrame(report_values)
        distance_values = pd.DataFrame(distance_values)
        angle_values = pd.DataFrame(angle_values)
        dihedral_values = pd.DataFrame(dihedral_values)
        nonbonded_energy_values = pd.DataFrame(nonbonded_energy_values)

        if report_values.empty:
            return None

        # Add epoch and trajectory to DF
        report_values['Protein'] = protein
        report_values['Ligand'] = ligand
        report_values['Epoch'] = epoch
        report_values['Trajectory'] = trajectory
        report_values.drop(['Task'], axis=1, inplace=True)

        # Add epoch and trajectory to distances DF
        distance_values['Protein'] = protein
        distance_values['Ligand'] = ligand
        distance_values['Epoch'] = epoch
        distance_values['Trajectory'] = trajectory

        # Add epoch and trajectory to angles DF
        angle_values['Protein'] = protein
        angle_values['Ligand'] = ligand
        angle_values['Epoch'] = epoch
        angle_values['Trajectory'] = trajectory

        # Add epoch and trajectory to dihedrals DF
        dihedral_values['Protein'] = protein
        dihedral_values['Ligand'] = ligand
        dihedral_values['Epoch'] = epoch
        dihedral_values['Trajectory'] = trajectory

        # Add epoch and trajectory to nonbonded energy DF
        nonbonded_energy_values['Protein'] = protein
        nonbonded_energy_values['Ligand'] = ligand
        nonbonded_energy_values['Epoch'] = epoch
        nonbonded_energy_values['Trajectory'] = trajectory

    return report_values, distance_values, angle_values, dihedral_values, nonbonded_energy_values

def _saveDataToCSV(dataframe, protein, ligand, equilibration=False, separator='-', data_folder_name='.pele_analysis'):
    if not os.path.exists(data_folder_name):
        os.mkdir(data_folder_name)
    if equilibration:
        csv_file_name = data_folder_name+'/equilibration_data_'+protein+separator+ligand+'.csv'
    else:
        csv_file_name = data_folder_name+'/data_'+protein+separator+ligand+'.csv'
    dataframe.to_csv(csv_file_name)

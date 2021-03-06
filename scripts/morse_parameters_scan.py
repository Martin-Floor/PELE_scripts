import argparse
import schrodinger
from schrodinger.application.jaguar import input
from schrodinger import structure
import numpy as np
import os
from scipy.optimize import curve_fit
import time
import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='Input optimized structure file.')
parser.add_argument('atom_1_index', help='Index of atom 1 in scanned bond.')
parser.add_argument('atom_2_index', help='Index of atom 2 in scanned bond.')
parser.add_argument('--cpus', default=8, help='Number of CPUs to run the job.')
parser.add_argument('--simultaneous_jobs', default=5, help='Simultaneous jobs.')
parser.add_argument('--skip_optimization', default=False, action='store_true', help='Debug option: skip optimization step')
parser.add_argument('--get_atom_indexes', default=False, action='store_true', help='Debug option: helps you check the atom index selection')
parser.add_argument('--functional', default='B3LYP-D3', help='DFT functional.')
parser.add_argument('--basis_set', default='CC-PVTZ', help='DFT basis set.')
parser.add_argument('--gas_phase', default=False, action='store_true', help='Use water solvent.')
parser.add_argument('--system_charge', default=0, help='Charge of the full QM system.')
parser.add_argument('--multiplicity', default=1, help='Multiplicity of the system (default singlet)')

args=parser.parse_args()

input_file = args.input_file
try:
    atom_1_index = int(args.atom_1_index)
    atom_2_index = int(args.atom_2_index)
except:
    raise ValueError('Atoms to scan must be identified by their indexes! Use the --get_atom_indexes option when unsure.')
cpus = args.cpus
simultaneous_jobs = args.simultaneous_jobs
get_atom_indexes = args.get_atom_indexes
functional = args.functional
basis_set = args.basis_set
gas_phase = args.gas_phase
system_charge = int(args.system_charge)
multiplicity = int(args.multiplicity)

match_attractive = True # Match better the attractive region of the potential.

# Debug options
skip_optimization = args.skip_optimization

# Run options
if skip_optimization:
    run_optimization = False
else:
    run_optimization = True

if get_atom_indexes:
    print('The input structure has the following Atoms:')

# Read file as a Schordinger structure object
for st in structure.StructureReader(input_file):
    input_struct = st
    if get_atom_indexes:
        print('Index Name PDB_Name')
        for a in st.atom:
            print('%-5s %-3s %-4s' % (a.index, a.name, a.pdbname))
        print('Did you select them correctly?')
        exit()
    break

# Check that atom names are present in the structure
atom1 = None
atom2 = None
for a in st.atom:
    # Copy pdb_atom_name to regular atom_name
    if atom_1_index == a.index:
        atom1 = a
    elif atom_2_index == a.index:
        atom2 = a

if atom1 == None:
    raise ValueError('Atom index %s not found in structure.' % atom_1_index)
if atom2 == None:
    raise ValueError('Atom index %s not found in structure.' % atom_2_index)

# Job name
job_name = '.'.join(input_file.split('.mae')[:-1])
job_name += '_'+atom1.name
job_name += '_'+atom2.name
# Log files
output_folder = job_name+'/'+job_name+'_scan'
log_file = job_name+'/'+job_name+'.log'

def check_finished_jobs(log_file):
    finished = False
    count = 0
    total = 0
    with open(log_file) as lf:
        for l in lf:
            if l.startswith('Number of jobs:'):
                total = int(l.split()[-1])
            elif l.startswith('Finished:'):
                finished = True
            elif 'finished' in l:
                count += 1
    if total != 0:
        print('Finished %s of %s scan jobs' % (count, total), end='\r')
    return finished

def addScanCoordinate(jaguar_input, atoms, initial_value, final_value, steps):
    lines = []
    zmat = False
    end_zmat = False
    with open(jaguar_input, 'r') as ji:
        for l in ji:
            lines.append(l)
            if l.startswith('&zmat'):
                zmat = True
                continue
            if l.startswith('&') and zmat:
                end_zmat = True
            if end_zmat:
                lines.append('&zvar\n')
                r_line = 'r = %12s to %12s in %s\n' % (initial_value, final_value, steps)
                lines.append(r_line)
                lines.append('&\n')
                lines.append('&coord\n')
                lines.append(' '+atoms[0]+' '+atoms[1]+' # r\n')
                lines.append('&')
                end_zmat = False

    with open(jaguar_input, 'w') as ji:
        for l in lines:
            ji.write(l)

if run_optimization:
    print('Running distance rigid scan for %s-%s Bond' % (atom1.name, atom2.name))
    # Get current coordinate distance
    coord1 = np.array([atom1.property['r_m_'+x+'_coord'] for x in ['x', 'y', 'z']])
    coord2 = np.array([atom2.property['r_m_'+x+'_coord'] for x in ['x', 'y', 'z']])
    current_distance = np.linalg.norm(coord1-coord2)

    # Define coordinate scan parameters
    initial_distance = current_distance-0.4
    final_distance = current_distance+4.0
    delta = 0.05
    steps = np.arange(initial_distance,final_distance, delta).shape[0]+1

    # Create working directory
    if not os.path.exists(job_name):
        os.mkdir(job_name)

    os.chdir(job_name)

    # Create Jaguar input file
    jaguar_input = input.read('../'+input_file)
    jaguar_input.resetStructure(input_struct, molchg=system_charge, multip=multiplicity)
    qm_values = {}
    qm_values['basis'] = basis_set
    qm_values['dftname'] = functional
    if not gas_phase:
        qm_values['isolv'] = '7'
    # qm_values['molchg'] = system_charge
    qm_values['nogas'] = '0'

    jaguar_input.setValues(qm_values)
    jaguar_input.saveAs(job_name+'.in')
    addScanCoordinate(job_name+'.in', [atom1.name, atom2.name], initial_distance,
                      final_distance, steps)

    command = '"${SCHRODINGER}/jaguar" '
    command += 'run distributed_scan.py '
    command += '-jobname='+job_name+' '
    command += job_name+'.in '
    command += '-HOST localhost:'+str(simultaneous_jobs)+' '
    command += '-PARALLEL '+str(cpus)
    os.system(command)

    # Wait until log file is created
    while not os.path.exists('../'+log_file):
        time.sleep(1)

    # Wait until scan calculation has finished
    finished = False
    while not finished:
        time.sleep(1)
        # Check log file for job finish line
        finished = check_finished_jobs('../'+log_file)
    finished = check_finished_jobs('../'+log_file)
    print('Finished distance rigid scan for %s-%s Bond' % (atom1.name, atom2.name))
    os.chdir('..')

if not os.path.exists(output_folder):
    print('Output folder: %s does not exist!' % output_folder)
    print('Did you forget to run the scan optimization step?')

print('Adjusting Morse potential')
# Check failed calculations
failed = []
with open(log_file) as lf:
    for l in lf:
        if '| died' in l:
            failed.append(l.split()[6])
failed = list(set(failed))
if len(failed) > 0:
    print('%s scan calculation failed.' % len(failed))
    print('Skipping failed files')
    print('Check carefully the correctness of the resulting parameters!')

# Get path to scan output files
scan_files = {}
for d in os.listdir(output_folder):
    if d.endswith('01.mae'):
        try:
            index = int(d.split('.')[-3].split('_')[-1])
        except:
            continue
        if not d.replace('.01.mae', '') in failed:
            scan_files[index] = output_folder+'/'+d

# Read structures
structures = {}
for index in sorted(scan_files):
    for st in structure.StructureReader(scan_files[index]):
        structures[index] = st
        break

# Create dictionary to store property values
qm_data = {}

qm_data['Optimization step'] = []
distance_label = atom1.name+'_'+atom2.name+' distance [$\AA$]'
qm_data[distance_label] = []
final_energy_label = 'Final Energy'
qm_data[final_energy_label] = []
gas_phase_energy_label = 'Gas Phase Energy'
qm_data[gas_phase_energy_label] = []
solution_phase_energy_label = 'Solution Phase Energy'
qm_data[solution_phase_energy_label] = []
solvent_energy_label = 'Solvent Energy'
qm_data[solvent_energy_label] = []

# Get gas phase energy
for index in structures:
    optimization_step = structures[index].property['s_j_Jaguar_input_file'].split('_')[-1].split('.')[0]
    if 'isomer' not in optimization_step:
        qm_data['Optimization step'].append(int(optimization_step))
        final_energy = structures[index].property['r_j_Final_Energy']
        qm_data[final_energy_label].append(final_energy)
        gas_phase_energy = structures[index].property['r_j_Gas_Phase_Energy']
        qm_data[gas_phase_energy_label].append(gas_phase_energy)
        solution_phase_energy = structures[index].property['r_j_Solution_Phase_Energy']
        qm_data[solution_phase_energy_label].append(solution_phase_energy)

# Get relative energies in kcal/mol
hartree_to_kcal_mol = 627.503
qm_data[final_energy_label] = np.array(qm_data[final_energy_label])
qm_data[final_energy_label] = np.array(qm_data[final_energy_label])*hartree_to_kcal_mol
qm_data[final_energy_label] = qm_data[final_energy_label] - np.min(qm_data[final_energy_label])

qm_data[gas_phase_energy_label] = np.array(qm_data[gas_phase_energy_label])
qm_data[gas_phase_energy_label] = np.array(qm_data[gas_phase_energy_label])*hartree_to_kcal_mol
qm_data[gas_phase_energy_label] = qm_data[gas_phase_energy_label] - np.min(qm_data[gas_phase_energy_label])

qm_data[solution_phase_energy_label] = np.array(qm_data[solution_phase_energy_label])
qm_data[solution_phase_energy_label] = np.array(qm_data[solution_phase_energy_label])*hartree_to_kcal_mol
qm_data[solution_phase_energy_label] = qm_data[solution_phase_energy_label] - np.min(qm_data[solution_phase_energy_label])

qm_data[solvent_energy_label] = qm_data[solution_phase_energy_label] - qm_data[gas_phase_energy_label]

energies = qm_data[final_energy_label]

# Calculate scanned bond distance
for index in sorted(structures):
    optimization_step = structures[index].property['s_j_Jaguar_input_file'].split('_')[-1].split('.')[0]
    if 'isomer' not in optimization_step:
        coord1 = None
        coord2 = None
        for a in structures[index].atom:
            if atom_1_index == a.index:
                coord1 = np.array([a.property['r_m_'+x+'_coord'] for x in ['x', 'y', 'z']])
            if atom_2_index == a.index:
                coord2 = np.array([a.property['r_m_'+x+'_coord'] for x in ['x', 'y', 'z']])
        if isinstance(coord1, type(None)):
            raise ValueError('Atom index %s not found in structure.' % atom_1_index)
        if isinstance(coord2, type(None)):
            raise ValueError('Atom index %s not found in structure.' % atom_2_index)
        qm_data[distance_label].append(np.linalg.norm(coord1-coord2))

qm_data[distance_label] = np.array(qm_data[distance_label])
distances = qm_data[distance_label]

# Get minimum energy distance
re = distances[np.argmin(energies)]

# Store distance and energy values
qm_data = pd.DataFrame(qm_data)
qm_data.to_csv('scan_qm_data.csv')

# Define Morse Potential function
def morse(x, De, a):
    return De*(1-np.exp(-a*(x-re)))**2

# Fit De and a parameters to morse function
params, covs = curve_fit(morse, distances, energies)
with open('morse_parameters.txt', 'w') as mpf:
    mpf.write('Scanned Bond: %s-%s\n' % (atom1.name, atom2.name))
    mpf.write('Morse parameters [De(1-exp(-a(r-re)))^2]:\n')
    mpf.write('\tDe: %.5f\n' % params[0])
    mpf.write('\ta: %.5f\n' % params[1])
    mpf.write('\tre: %.5f\n' % re)

if match_attractive:
    # Refit the plot to match attractive points only
    attr_distances = distances[np.argmin(energies):]
    attr_energies = energies[np.argmin(energies):]

    # Fit De and a parameters to morse function
    params, covs = curve_fit(morse, attr_distances, attr_energies, p0=params)
    with open('morse_parameters.txt', 'w') as mpf:
        mpf.write('Scanned Bond: %s-%s\n' % (atom1.name, atom2.name))
        mpf.write('Morse parameters [De(1-exp(-a(r-re)))^2]:\n')
        mpf.write('\tDe: %.5f\n' % params[0])
        mpf.write('\ta: %.5f\n' % params[1])
        mpf.write('\tre: %.5f\n' % re)

# plot soft repulsion potential parameters

def lj12(x, Aii, Ajj):
    # print(np.sqrt(Aii)*np.sqrt(Ajj))
    # print(x**12)
    return (np.sqrt(Aii)*np.sqrt(Ajj))/(x**12)

def soft(x, Ci, Cj):
    return Ci*Cj*np.exp(-1.581*1.581*x)

Aii = 1802.2385
Ajj = 601.1488
Ci = Aii*0.1
Cj = Ajj*0.1

# Create bond potential plot
plt.figure()
plt.scatter(distances, energies, marker='*', c='red', label='QM rigid-scan points')
plt.plot(distances, morse(distances, params[0],  params[1]), c='k', label='Morse potential')

plt.xlabel(atom1.name+'_'+atom2.name+' distance [$\AA$]')
plt.ylabel('Potential Energy [kcal/mol]')
plt.legend()
plt.savefig('morse_plot.svg')

print('Finished adjusting Morse potential')
print('Morse parameters were written at: morse_parameters.txt')
print('Fitted points were plotted at: morse_plot.svg')

from schrodinger import structure
from schrodinger.structutils.minimize import minimize_structure
import schrodinger
import numpy as  np
import pymmlibs
import os
import time
import subprocess
import shutil
import argparse

# ## Define input variables
parser = argparse.ArgumentParser()

parser.add_argument('input_file', help='Input PDB file contatining ligand.')
parser.add_argument('ligand_name', default="UNK", help='Three letter code of the PDB ligand name.')
parser.add_argument('--max_min_steps', default=1000, help='Maximum ligand minimization steps.')
parser.add_argument('--n_conformers', default=10, help='Number of conformers to generate.')
parser.add_argument('--gas_phase', default=False, action='store_true', help='Use water solvent.')
parser.add_argument('--functional', default='B3LYP-D3', help='DFT functional to use for conformer optimization (Jaguar).')
parser.add_argument('--basis_set', default='CC-PVTZ', help='DFT basis set to use for conformer optimization (Jaguar).')
parser.add_argument('--cpus', default=8, help='Number of CPUs to use in conformer optimization (Jaguar).')
parser.add_argument('--simultaneous_jobs', default=5, help='Simultaneous jobs')
parser.add_argument('--skip_minimization', default=False, action='store_true', help='Debug option: skip ligand minimization step')
parser.add_argument('--skip_conformers', default=False, action='store_true', help='Debug option: skip conformers generation step')
parser.add_argument('--skip_optimization', default=False, action='store_true', help='Debug option: skip optimization step')
parser.add_argument('--skip_resp', default=False, action='store_true', help='Debug option: skip resp fitting step')
parser.add_argument('--min_conv_threshold', default=0.0001, help='Threshold to stop ligand minimization before total number of steps are reached.')
parser.add_argument('--min_step_report', default=10, help='Number of steps to report the change in minimized energy.')
parser.add_argument('--system_charge', default=0, help='Charge to use in the optimization and RESP fitting.')

args=parser.parse_args()

input_file = args.input_file
ligand_name = args.ligand_name
gas_phase = args.gas_phase
max_min_steps = int(args.max_min_steps)
n_conformers = int(args.n_conformers)
functional = args.functional
basis_set = args.basis_set
cpus = int(args.cpus)
min_conv_threshold = float(args.min_conv_threshold)
min_step_report = int(args.min_step_report)
simultaneous_jobs = int(args.simultaneous_jobs)
system_charge = int(args.system_charge)

# Debug variables
skip_minimization = args.skip_minimization
skip_conformers = args.skip_conformers
skip_optimization = args.skip_optimization
skip_resp = args.skip_resp

if skip_minimization:
    run_minimization = False
else:
    run_minimization = True

if skip_conformers:
    run_conformers = False
else:
    run_conformers = True

if skip_optimization:
    run_optimization = False
else:
    run_optimization = True

if skip_resp:
    run_resp = False
else:
    run_resp = True

# Define folders
conformer_folder = '01_conformers'
optimization_folder = '02_optimization'
resp_folder = '03_resp_charges'

### Read input ligand ###
print('Reading ligand file: %s' % input_file)
for st in structure.StructureReader(input_file):
    st
    break

### Ligand minimization ###
if run_minimization:
    # Set the energy property name
    print('Minimizing ligand')
    energy_name = 'r_ff_Potential_Energy-OPLS_2005'
    # Constants
    OPLS_2005 = 14
    minimize_structure(st, ffld_version=OPLS_2005, max_steps=0)
    initial_energy = st.property[energy_name]
    for i in range(max_min_steps):
        minimize_structure(st, ffld_version=OPLS_2005, max_steps=1)
        if i % min_step_report == 0:
            minimized_energy = st.property[energy_name]
            delta_energy = initial_energy - minimized_energy
            print("The minimized delta energy is {} kcal/mol for step {}.".format(delta_energy, i+1))
            initial_energy = st.property[energy_name]
        if delta_energy <= min_conv_threshold:
            print('Minimization finished.')
            break

    # Save structure
    minimized_file = 'ligand_minimized.mae'
    st.write(minimized_file)

if run_conformers:

    ### Conformer search ###
    print('Calculating ligand conformers')

    if not os.path.exists(conformer_folder):
        os.mkdir(conformer_folder)

    # Save structure
    minimized_file = '../ligand_minimized.mae'

    os.chdir(conformer_folder)
    command = '${SCHRODINGER}/confgenx '
    command += minimized_file+' '
    command += '-jobname confgen_1 '
    command += '-max_num_conformers '+str(n_conformers)+' '
    command += ' -optimize -HOST localhost'
    command += ' -WAIT'

    p = subprocess.run(command, shell=True)
    print('Finished conformer generation.')
    os.chdir('..')

# Get the number of conformers generated
generated_conformers = 0
with open(conformer_folder+'/confgen_1.log') as cgl:
    for l in cgl:
        if 'structures stored' in l:
            generated_conformers = int(l.split()[0])

if generated_conformers < n_conformers:
    print('Only %s of %s conformers were generated!' % (generated_conformers, n_conformers))

if run_optimization:
    ### Set up Jaguar calculations ###
    print('DFT optimization')
    if not os.path.exists(optimization_folder):
        os.mkdir(optimization_folder)
    os.chdir(optimization_folder)

    command = '${SCHRODINGER}/jaguar run canonical.py '
    command += '-jobname=jag_batch_opt_'+functional+'_'+basis_set+' '
    if not gas_phase:
        command += '-keyword=isolv=7 '
    command += '-keyword=basis='+basis_set+' '
    command += '-keyword=ip172=2 '
    command += '-keyword=gcharge=-2 '
    command += '-keyword=igeopt=1 '
    command += '-keyword=icfit=1 '
    command += '-keyword=molchg='+str(system_charge)+' '
    command += '-keyword=dftname='+functional+' '
    command += '-keyword=nogas=0 '
    command += '../'+conformer_folder+'/confgen_1-out.maegz '
    command += '-HOST localhost:'+str(simultaneous_jobs)+' '
    command += '-PARALLEL '+str(cpus)+' '
    command += '-TMPLAUNCHDIR '
    if generated_conformers == 1:
        command += '-WAIT '
        p = subprocess.run(command, shell=True)
    else:
        p = subprocess.Popen(command, shell=True)
        # Check until all optimizations start
        log_files = {}
        previous= len(log_files)
        current= len(log_files)
        while current < generated_conformers:
            log_files = {}
            # Get log file paths
            for f in os.listdir():
                if 'isomer' in f and f.endswith('.log'):
                    index = int(f.split('isomer')[-1].split('.')[0])
                    log_files[index] = f
            current = len(log_files)
            time.sleep(1)

        # Check until all optimizations are finished
        finished = []
        while False in finished or len(finished) < generated_conformers:
            finished = []
            for isomer in log_files:
                ended = False
                with open(log_files[isomer]) as isof:
                    for l in isof:
                        if 'Finished' in l:
                            ended = True
                if ended:
                    finished.append(True)
                else:
                    finished.append(False)
            time.sleep(3)

    print('Finished DFT optimization.')
    os.chdir('..')

if os.path.exists(optimization_folder):
    # Read log files
    log_files = {}
    # Get log file paths
    for f in os.listdir(optimization_folder):
        if generated_conformers == 1:
            if 'loner' in f and f.endswith('.log'):
                log_files[1] = '../'+optimization_folder+'/'+f
        else:
            if 'isomer' in f and f.endswith('.log'):
                index = int(f.split('isomer')[-1].split('.')[0])
                log_files[index] = '../'+optimization_folder+'/'+f

    optimized_conformers = len(log_files)
else:
    if run_resp:
        print('Optimization folder not found. Cannot calculate RESP')
        exit()

# Generate RESP charges
def readMol2Charges(mol2_file):
    """
    Read and return charges from a mol2 file

    Parameters
    ==========
    mol2_file : str
        Path to a mol2 file

    Returns
    =======
    charges : dict
        Dictionary containing the atom names as keys and charges as values.
    """
    charges = {}
    with open(mol2_file) as mf:
        c = False
        for l in mf:
            if '@<TRIPOS>ATOM' in l:
                c = True
                continue
            elif '@<TRIPOS>BOND' in l:
                c = False
                continue
            if c:
                charge = float(l.strip().split()[-1])
                atom = l.strip().split()[1]
                charges[atom] = charge

    return charges

# Calculate MC RESP charges
def changeRespinParameters(respin_file, weights, system_charge, ligand_name='UNK'):
    """
    given a list of weights it updates the weights in the RESP input file.

    Parameters
    ==========
    respin_file : str
        File path to the respin file generated by the respgen program of antechamber.
    weights : list
        A list with the weights to add. The weights should match the order of the
        multiconfiguration resp file.
    system_charge : int
        Target system charge for the fitting.
    """
    with open(respin_file+'.tmp', 'w') as trif:
        with open(respin_file) as rif:
            nmol = False
            probability = False
            count = 0
            for l in rif:
                if 'nmol' in l:
                    nmol = int(l.split()[-1].replace(',',''))
                    if len(weights) != nmol:
                        raise ValueError('Wrong number of weights given! There is %s optimized configurations' % nmol)
                if '&end' in l:
                    nmol = True
                    trif.write(l)
                    continue
                if nmol and '1.0' in l:
                    trif.write(l.replace('1.0', str(weights[count])))
                    count += 1
                    continue
                if 'Resp charges for organic molecule' in l:
                    probability = True
                    trif.write(l.replace('organic molecule', ligand_name))
                    continue
                if probability and l!= '\n' and '    1.0' in l:
                    trif.write(l.replace(l.split()[0], str(system_charge)))
                    probability = False
                    continue
                trif.write(l)
    shutil.move(respin_file+'.tmp', respin_file)

if run_resp:

    if optimized_conformers == 1:
        print('Calculating RESP charges fitting')
    else:
        print('Calculating multi-configuration RESP charges')

    if not os.path.exists(resp_folder):
        os.mkdir(resp_folder)
    os.chdir(resp_folder)

    ## Read optimization energies
    optimization_energies = {}
    failed_optimizations = []
    for isomer in sorted(log_files):
        optimization_energies[isomer] = []
        with open(log_files[isomer]) as isof:
            for l in isof:
                if 'Total energy' in l:
                    optimization_energies[isomer].append(float(l.split()[2]))
                elif 'Error' in l:
                    if isomer not in failed_optimizations:
                        print('Optimization for isomer %s failed with:' % isomer)
                        failed_optimizations.append(isomer)
                    print(l) # Print error lines

    # Remove from log files failed conformers
    for isomer in failed_optimizations:
        optimized_conformers -= 1 # Update the number of optimized conformers
        del log_files[isomer]

    # Create MultiConfiguration resp file for antechamber
    with open('MC.resp', 'w') as iso:
        for f in os.listdir('../'+optimization_folder):
            if ('isomer' in f and f.endswith('.resp')) or ('loner' in f and f.endswith('.resp')):
                isomer = int(f.split('isomer')[-1].split('.')[0])
                if isomer not in failed_optimizations: # Remove resp files from failed optimzations
                    with open('../'+optimization_folder+'/'+f) as isoi:
                        for i,l in enumerate(isoi):
                            if i == 0:
                                ls = l.split()
                                iso.write('%4s %-8s\n' % (ls[0], ls[1]))
                            else:
                                iso.write(l)

    # Select lowest energy conformer
    lowest_energy_isomer = 0
    for isomer in optimization_energies:
        if isomer not in failed_optimizations:
            final_energy = optimization_energies[isomer][-1]
            if lowest_energy_isomer == 0:
                lowest_energy_isomer = isomer
                lowest_energy = final_energy
            elif final_energy < lowest_energy:
                lowest_energy_isomer = isomer
                lowest_energy = final_energy
    print('Lowest energy isomer is %s' % lowest_energy_isomer)

    # Create mol2 for fitting # move
    for f in sorted(os.listdir('../'+optimization_folder)):
        if (f.endswith('01.mae') and 'isomer' in f) or (f.endswith('01.mae') and 'loner' in f):
            isomer = int(f.split('isomer')[-1].split('.')[0])
            if isomer == lowest_energy_isomer:
                for st in structure.StructureReader('../'+optimization_folder+'/'+f):
                    st.write(ligand_name+'.mol2', format='mol2')

    # Calcualte Boltzmann probabilities
    KT = 0.593 # kcal/mol
    hartree_to_kcal_mol = 627.503
    final_energies = [optimization_energies[isomer][-1] for isomer in sorted(log_files)]
    final_energies = np.array(final_energies)*hartree_to_kcal_mol # Convert to kcal/mol
    relative_energies = final_energies - np.min(final_energies) # relative energies
    partition_coefficient = np.sum(np.exp(-relative_energies/KT))
    probabilities = np.exp(-relative_energies/KT)/partition_coefficient

    command = 'antechamber -fi mol2 -fo ac -i '+ligand_name+'.mol2'+' -o '+ligand_name+'.ac -nc '+str(system_charge)+'\n'
    command += 'respgen -i '+ligand_name+'.ac -o '+ligand_name+'.respin1 -f resp1 -n '+str(optimized_conformers)+'\n'
    command += 'respgen -i '+ligand_name+'.ac -o '+ligand_name+'.respin2 -f resp2 -n '+str(optimized_conformers)
    subprocess.call(command, shell=True)
    if optimized_conformers > 1:
        changeRespinParameters(ligand_name+'.respin1', probabilities, system_charge, ligand_name=ligand_name)
        changeRespinParameters(ligand_name+'.respin2', probabilities, system_charge, ligand_name=ligand_name)
    command = 'resp -O -i '+ligand_name+'.respin1 -o '+ligand_name+'.respout1 -e MC.resp -t '+ligand_name+'_qout_stage1\n'
    command += 'resp -O -i '+ligand_name+'.respin2 -o '+ligand_name+'.respout2 -e MC.resp -q '+ligand_name+'_qout_stage1 -t '+ligand_name+'_qout_stage2\n'
    command += 'antechamber -i '+ligand_name+'.mol2 -fi mol2 -o '+ligand_name+'_resp.mol2 -fo mol2 -c rc -cf '+ligand_name+'_qout_stage2\n'
    subprocess.call(command, shell=True)
    os.chdir('..')
    print('RESP fitting finished.')

    # Assign residue name
    for residue in st.residue:
        residue.pdbres = ligand_name

    charges = readMol2Charges(resp_folder+'/'+ligand_name+'_resp.mol2')

    for atom in st.atom:
        atom_name = atom.pdbname.strip()
        atom.partial_charge = charges[atom_name]

    st.write(resp_folder+'/'+ligand_name+'_resp.mae')

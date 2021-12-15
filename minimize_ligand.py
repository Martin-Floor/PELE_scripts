from schrodinger import structure
from schrodinger.structutils.minimize import minimize_structure
import schrodinger
import argparse

# ## Define input variables
parser = argparse.ArgumentParser()
parser.add_argument('input_file', help='Input PDB file contatining ligand.')
parser.add_argument('output_file', help='Output PDB file contatining the minimized ligand.')
parser.add_argument('--max_min_steps', default=10000, help='Maximum ligand minimization steps.')
parser.add_argument('--min_conv_threshold', default=0.0001, help='Threshold to stop ligand minimization before total number of steps are reached.')
parser.add_argument('--min_step_report', default=10, help='Number of steps to report the change in minimized energy.')
args=parser.parse_args()

input_file = args.input_file
output_file = args.output_file
max_min_steps = args.max_min_steps
min_conv_threshold = args.min_conv_threshold
min_step_report = args.min_step_report

### Read input ligand ###
print('Reading ligand file: %s' % input_file)
for st in structure.StructureReader(input_file):
    st
    break

### Ligand minimization ###

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
st.write(output_file, format='PDB')

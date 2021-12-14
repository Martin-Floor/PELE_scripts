# PELE_scripts
Scripts for PELE:

-ligand_setup.py
  Calculates multiconfigurational RESP charges:
  Usage:
    $SCHRODINGER/run python3 ligand_setup.py ligand_pdb_file three_letter_residue_name 
  For other options run only:
    $SCHRODINGER/run python3 ligand_setup.py
    
  The final output file is found in folder 03_resp_charges (the only .mae file there).

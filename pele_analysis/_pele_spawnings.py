import os
import shutil
import json

def formatPELESpawnings(pele_folder, separator='-', proteins=None, ligands=None, combinations=None, keep_original=False, output_folder=None,
                        skip_0_epochs=True):
    """
    Format the spawning PELE folder with the standard PELE format for easier analysis
    with peleAnalysis.
    """

    if keep_original and not output_folder:
        raise ValueError('You must provide an output folder if you want to preserve the original folder.')

    elif output_folder and not keep_original:
        keep_original = True

    if proteins and isinstance(proteins, str):
        proteins = [proteins]

    if ligands and isinstance(ligands, str):
        ligands = [ligands]

    if combinations and isinstance(combinations, tuple):
        combinations = [combinations]

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    for pele in os.listdir(pele_folder):

        # Skip files
        if not os.path.isdir(pele_folder+'/'+pele):
            continue

        # Check given separator
        if separator not in pele:
            raise ValueError(f'The separator was not found in folder {pele}')
        elif pele.count(separator) > 1:
            raise ValueError(f'The separator was found too many times in {pele}')

        # Get protein and ligand name
        protein, ligand = pele.split(separator)
        pele_path = pele_folder+'/'+pele

        if proteins and protein not in proteins:
            continue

        if ligands and ligand not in ligands:
            continue

        if combinations and (protein, ligand) not in combinations:
            continue

        # Get spawnings if found
        spawnings = []
        for spawning in os.listdir(pele_path):
            try:
                spawnings.append(int(spawning))
            except:
                continue

        # Check if the given folder has the spawning format
        if not spawnings:

            if os.path.exists(pele_path+'/._spawning_mapping.json'):
                print(f'PELE folder {pele_path} was already converted.')
            else:
                print(f'PELE folder {pele_path} has not a spawning format. Skiping {protein} and {ligand}.')
            continue

        spawnings = sorted(spawnings)

        if 0 not in spawnings:
            raise ValueError(f'First spawning folder was not found at {pele_path}. Please check your folder for {protein} and {ligand}!')

        if not os.path.exists(pele_path+'/0/output/output'):
            print(f'There is no output folder for the first spawning folder. Skipping {protein} and {ligand}.')
            continue

        if not os.path.exists(pele+'/0/output/output'):
            print(f'There is no output folder for the first spawning folder. Skipping {protein} and {ligand}.')
            continue

        # Copy full spawning-zero folder to tmp
        if keep_original:
            tmp = output_folder+'/'+pele
        else:
            tmp = pele_path+'_tmp'
        shutil.copytree(pele_path+'/0', tmp, symlinks=True)

        # Copy metrics dictionaries
        if os.path.exists(pele_path+'/metrics.json'):
            shutil.copy(pele_path+'/metrics.json', tmp+'/metrics.json')
        if os.path.exists(pele_path+'/metrics_thresholds.json'):
            shutil.copy(pele_path+'/metrics_thresholds.json', tmp+'/metrics_thresholds.json')

        # Get epochs at first spawning
        epochs = []
        for epoch in os.listdir(tmp+'/output/output'):
            try:
                epochs.append(int(epoch))
            except:
                continue

        if not epochs:
            print(f'There are no epochs for the first spawning folder. Skipping {protein} and {ligand}.')
            shutil.rmtree(tmp)
            continue

        epochs = sorted(epochs)

        current_epoch = epochs[-1] # Last epoch from first spawning
        epoch_spawning = {} # Map each new epoch assignment to its original spawning
        for epoch in epochs:
            epoch_spawning[epoch] = 0

        # Copy epochs from the remaining spawning folders
        for spawning in spawnings:

            if spawning == 0:
                continue

            if not os.path.exists(pele_path+'/'+str(spawning)+'/output/output'):
                print(f'There is no output folder for spawning {str(spawning)}. Skipping spawning for {protein} and {ligand}.')
                continue

            # Get epochs' folders list
            epochs = []
            for epoch in os.listdir(pele_path+'/'+str(spawning)+'/output/output'):
                try:
                    epochs.append(int(epoch))
                except:
                    continue

            if skip_0_epochs:
                epochs = [e for e in epochs if e !=0]
            epochs = sorted(epochs)

            if not epochs:
                print(f'There are no epochs for spawning {str(spawning)}. Skipping spawning for {protein} and {ligand}.')
                continue

            for epoch in epochs:
                current_epoch += 1
                epoch_folder = pele_path+'/'+str(spawning)+'/output/output/'+str(epoch)
                shutil.copytree(epoch_folder, tmp+'/output/output/'+str(current_epoch))
                epoch_spawning[current_epoch] = spawning

        # Write epoch-spawning mapping
        with open(tmp+'/._spawning_mapping.json', 'w') as jf:
            json.dump(epoch_spawning, jf)

        if not keep_original:
            shutil.rmtree(pele_path)
            shutil.move(tmp, pele_path)

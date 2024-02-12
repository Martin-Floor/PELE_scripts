import os
import shutil
import json

def formatPELESpawnings(pele_folder, separator='-'):
    """
    Format the spawning PELE folder with the standard PELE format for easier analysis
    with peleAnalysis.
    """

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
        pele = pele_folder+'/'+pele

        # Get spawnings if found
        spawnings = []
        for spawning in os.listdir(pele):
            try:
                spawnings.append(int(spawning))
            except:
                continue

        # Check if the given folder has the spawning format
        if not spawnings:

            if os.path.exists(pele+'/._spawning_mapping.json'):
                print(f'PELE folder {pele} was already converted.')
            else:
                print(f'PELE folder {pele} has not a spawning format. Skiping {protein} and {ligand}.')
            continue

        spawnings = sorted(spawnings)

        if 0 not in spawnings:
            raise ValueError(f'First spawning folder was not found at {pele}. Please check your folder!')

        # Copy full spawning-zero folder to tmp
        tmp = pele+'_tmp'
        shutil.copytree(pele+'/0', tmp, symlinks=True)

        # Copy metrics dictionaries
        if os.path.exists(pele+'/metrics.json'):
            shutil.copy(pele+'/metrics.json', tmp+'/metrics.json')
        if os.path.exists(pele+'/metrics_thresholds.json'):
            shutil.copy(pele+'/metrics.json', tmp+'/metrics_thresholds.json')

        # Get latest epoch at first spawning
        epochs = []
        for epoch in os.listdir(tmp+'/output/output'):
            try:
                epochs.append(int(epoch))
            except:
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

            # Get epochs' folders list
            epochs = []
            for epoch in os.listdir(pele+'/'+str(spawning)+'/output/output'):
                try:
                    epochs.append(int(epoch))
                except:
                    continue
            epochs = sorted(epochs)

            for epoch in epochs:
                current_epoch += 1
                epoch_folder = pele+'/'+str(spawning)+'/output/output/'+str(epoch)
                shutil.copytree(epoch_folder, tmp+'/output/output/'+str(current_epoch))
                epoch_spawning[current_epoch] = spawning

        # Write epoch-spawning mapping
        with open(tmp+'/._spawning_mapping.json', 'w') as jf:
            json.dump(epoch_spawning, jf)

        shutil.rmtree(pele)
        shutil.move(tmp, pele)

import mdtraj as md
import numpy as np
import pandas as pd
import argparse
import json
import os
import re
import copy


parser = argparse.ArgumentParser()
parser.add_argument('--protein')
parser.add_argument('--ligand')
parser.add_argument('--threshold')

args = parser.parse_args()
ligand = args.ligand
protein = args.protein
threshold = float(args.threshold)


with open(protein+'/'+ligand+'/index_dic.json') as f:
    index_dic = json.load(f)
index_dic_swap = {tuple(v): k for k, v in index_dic.items()}


topology_file = protein+'/topology.pdb'

traj_paths = sorted(os.listdir(protein+'/'+ligand))
traj_paths = [protein+'/'+ligand+'/'+i for i in traj_paths if i.endswith('.xtc')]
traj = md.load(traj_paths,top=topology_file)

with open(protein+'/'+ligand+'/be.json') as f:
    be = json.load(f)

ite_traj = copy.copy(traj)
clusters = {}
cluster_num = 0


while not len(be) == 0:
    ## Get best energy traj from remaining data
    #data = pele_data['Binding Energy'].idxmin()
    print(len(be))

    data = (min(be, key=be.get)).split('_')
    data[2] = int(data[2])
    data[3] = int(data[3])

    best_energy_traj = ite_traj[int(index_dic_swap[tuple(data)])]

    #best_energy_traj = md.load_frame(protein+'/'+ligand+'/traj_'+data[0]+'_'+data[1]+'_'+data[2],data[3],top=topology_file)

    #traj.superpose(best_energy_traj,atom_indices=protein_atoms)

    ## Compute rmsd
    #rmsd = md.rmsd(traj,best_energy_traj,atom_indices=lig_bb_atoms)

    rmsd = np.array((np.sqrt(3*np.mean((ite_traj.xyz - best_energy_traj.xyz)**2, axis=(1,2)))))
    assert(rmsd[int(index_dic_swap[tuple(data)])]==0.0)


    cluster = (np.where(rmsd<threshold)[0])

    #not_cluster = (np.where(rmsd>threshold)[0])
    mask_cluster = np.array([(i in cluster) for i in range(len(ite_traj))])

    #not_cluster = [i for i in range(len(rmsd)) if i not in cluster]

    ## Get cluster data and remove trajectories from data
    cluster_data = [index_dic[str(index)] for index in cluster]
    clusters[cluster_num] = cluster_data

    for i in cluster_data:
        be.pop(str(i[0])+'_'+str(i[1])+'_'+str(i[2])+'_'+str(i[3]))
    #pele_data = pele_data[~pele_data.index.isin(cluster_data)]

    cluster_num += 1

    ## Remove frames from traj

    ite_traj = ite_traj[~mask_cluster]

    ## Get new indexes
    new_ind = 0
    new_ind_dic = {}
    for i in index_dic:
        if int(i) not in cluster:
            #print(i)
            #print(new_ind)
            new_ind_dic[str(new_ind)] = index_dic[i]
            new_ind += 1
    index_dic = new_ind_dic
    index_dic_swap = {tuple(v): k for k, v in index_dic.items()}

with open(protein+'/'+ligand+'/clusters.json','w') as f:
    json.dump(clusters,f)

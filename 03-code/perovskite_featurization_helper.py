import numpy as np
import pandas as pd
import os
import glob
import json
from config import COLUMNS_DICT, PROJECT_ROOT_DIRECTORY
from perovskite_toolkit import ElectronicStructureParser

from pymatgen.util.coord import pbc_diff, get_angle
from pymatgen.io.cif import CifWriter
from pymatgen.core import Structure


organic_genome_path = PROJECT_ROOT_DIRECTORY + '02-metadata/06-csv-files/01-organic-genome.csv'
vasp_organic_xyz_path = PROJECT_ROOT_DIRECTORY + '01-rawdata/05-vasp-organic-xyz/'
initial_perovskite_cif_path = PROJECT_ROOT_DIRECTORY + '01-rawdata/06-initial-perovskite-cif/'
organic_dimension_path = PROJECT_ROOT_DIRECTORY + '02-metadata/02-organic-dimension/'
relax_set_path = PROJECT_ROOT_DIRECTORY + '01-rawdata/07-relax-set/'
refine_set_path = PROJECT_ROOT_DIRECTORY + '01-rawdata/08-refine-set/'
hse_set_path = PROJECT_ROOT_DIRECTORY + '01-rawdata/09-hse-set/'
hse_frontier_json_path = PROJECT_ROOT_DIRECTORY + '02-metadata/05-hse-frontier/'
hse_frontier_dataframe_path = PROJECT_ROOT_DIRECTORY + '02-metadata/06-csv-files/08-hse-frontier.csv'

final_perovskite_cif_path = PROJECT_ROOT_DIRECTORY + "02-metadata/03-final-perovskite-cif/"
struc_info_path = PROJECT_ROOT_DIRECTORY + "02-metadata/04-structure-info/"
structure_info_dataframe_path = PROJECT_ROOT_DIRECTORY + '02-metadata/06-csv-files/07-structure-info.csv'


def get_angle_between_closest_sites(
    structure: Structure, 
    i: int, 
    j: int, 
    k: int
) -> float:
    """
    Returns angle specified by three sites where the closest image
    of the sites i and k to the central site, j, is used.

    Args:
        structure: A structure object.
        i: Index of first site.
        j: Index of second site.
        k: Index of third site.

    Returns:
        Angle in degrees.
    """
    v1 = pbc_diff(structure[i].frac_coords, structure[j].frac_coords)
    v2 = pbc_diff(structure[k].frac_coords, structure[j].frac_coords)
    
    cart_v1 = structure.lattice.get_cartesian_coords(v1)
    cart_v2 = structure.lattice.get_cartesian_coords(v2)

    return get_angle(cart_v1, cart_v2, units="degrees")


def get_structure_info_json(identifier, save_cif=False, save_json=False):

    init_struc_path = os.path.join(relax_set_path, str(identifier), "POSCAR")
    init_struc = Structure.from_file(filename=init_struc_path)
    
    Pb_I_bond = float(3.2)

    scaling_factor = {}

    scaling_factor['x'] = np.around(init_struc.lattice.a/(2*Pb_I_bond)).astype('int')
    scaling_factor['y'] = np.around(init_struc.lattice.b/(2*Pb_I_bond)).astype('int')
    scaling_factor['z'] = int(len(init_struc.indices_from_symbol("Pb"))/(scaling_factor['x']*scaling_factor['y']))


    inorganic_matrix = np.full((scaling_factor['x']*2, scaling_factor['y']*2, 2*scaling_factor['z']+1), None)

    indicies_framework = np.concatenate((init_struc.indices_from_symbol("Pb"), init_struc.indices_from_symbol("I")))

    for index in indicies_framework:

        ind = np.around(init_struc.cart_coords[index]/Pb_I_bond).astype('int')
        
        if ind[2] >= 2*scaling_factor['z']:
            ind[2] = 0   #ind[2] - 2*scaling_factor['z'] - 1
        else:
            ind[2] += 1

        inorganic_matrix[ind[0]][ind[1]][ind[2]]=index

    #print(inorganic_matrix.T)

    final_struc_path = os.path.join(refine_set_path, str(identifier), "CONTCAR")
    struc = Structure.from_file(filename=final_struc_path)
    if save_cif:
        w = CifWriter(struc)
        w.write_file(os.path.join(final_perovskite_cif_path, str(identifier).zfill(5)+'.cif'))
        print('Final perovskite structure saved to cif file, identifier #', str(identifier))

    bond = {'x': [], 'y': [], 'z': []}
    angle_XMX = {'x': [], 'y': [], 'z': []}
    angle_MXM = {'x': [], 'y': [], 'z': []}

    for c in np.arange(1, 2*scaling_factor['z']):
        if (c % 2) != 0:
            for a in range(scaling_factor['x']*2):
                for b in range(scaling_factor['y']*2):
                    if (a % 2) == 0 and (b % 2) == 0:  # Pb

                        angle_XMX['x'].append(get_angle_between_closest_sites(
                            structure=struc, i=inorganic_matrix[(a-1)%(scaling_factor['x']*2)][b][c], j=inorganic_matrix[a][b][c], k=inorganic_matrix[(a+1)%(scaling_factor['x']*2)][b][c]))

                        angle_XMX['y'].append(get_angle_between_closest_sites(
                            structure=struc, i=inorganic_matrix[a][(b-1)%(scaling_factor['y']*2)][c], j=inorganic_matrix[a][b][c], k=inorganic_matrix[a][(b+1)%(scaling_factor['y']*2)][c]))
                    
                        angle_XMX['z'].append(get_angle_between_closest_sites(
                            structure=struc, i=inorganic_matrix[a][b][c-1], j=inorganic_matrix[a][b][c], k=inorganic_matrix[a][b][c+1]))

                        bond['x'].append(struc.get_distance(inorganic_matrix[a][b][c], inorganic_matrix[(a-1)%(scaling_factor['x']*2)][b][c]))
                        bond['x'].append(struc.get_distance(inorganic_matrix[a][b][c], inorganic_matrix[(a+1)%(scaling_factor['x']*2)][b][c]))

                        bond['y'].append(struc.get_distance(inorganic_matrix[a][b][c], inorganic_matrix[a][(b-1)%(scaling_factor['y']*2)][c]))
                        bond['y'].append(struc.get_distance(inorganic_matrix[a][b][c], inorganic_matrix[a][(b+1)%(scaling_factor['y']*2)][c]))

                        bond['z'].append(struc.get_distance(inorganic_matrix[a][b][c], inorganic_matrix[a][b][c-1]))
                        bond['z'].append(struc.get_distance(inorganic_matrix[a][b][c], inorganic_matrix[a][b][c+1]))

                    elif (a % 2) != 0 and (b % 2) == 0:  # I_xdirection
                        angle_MXM['x'].append(get_angle_between_closest_sites(
                            structure=struc, i=inorganic_matrix[(a-1)%(scaling_factor['x']*2)][b][c], j=inorganic_matrix[a][b][c], k=inorganic_matrix[(a+1)%(scaling_factor['x']*2)][b][c]))
                    elif (a % 2) == 0 and (b % 2) != 0:  # I_ydirection
                        angle_MXM['y'].append(get_angle_between_closest_sites(
                            structure=struc, i=inorganic_matrix[a][(b-1)%(scaling_factor['y']*2)][c], j=inorganic_matrix[a][b][c], k=inorganic_matrix[a][(b+1)%(scaling_factor['y']*2)][c]))

        elif (c % 2) == 0:
            for a in range(scaling_factor['x']*2):
                for b in range(scaling_factor['y']*2):
                    if (a % 2) == 0 and (b % 2) == 0:  # I_zdirection
                        angle_MXM['z'].append(get_angle_between_closest_sites(
                            structure=struc, i=inorganic_matrix[a][b][c-1], j=inorganic_matrix[a][b][c], k=inorganic_matrix[a][b][c+1]))

    for matrix in [bond, angle_XMX, angle_MXM]:
        matrix['x_average'] = np.average(matrix['x']) 
        matrix['y_average'] = np.average(matrix['y'])
        matrix['z_average'] = np.average(matrix['z'])
        matrix['xy_average'] = np.average(np.concatenate((matrix['x'], matrix['y']),axis=None))
        matrix['average'] = np.average(np.concatenate((matrix['x'], matrix['y'], matrix['z']), axis=None))
    
    n_layer = scaling_factor['z']
    d_interlayer = struc.lattice.c - bond['z_average']*2*scaling_factor['z']

    struc_info = {'n_layer': n_layer, 'd_interlayer': d_interlayer, 'bond': bond, 'angle_XMX': angle_XMX, 'angle_MXM': angle_MXM}
    if save_json:
        json_file = os.path.join(struc_info_path, str(identifier).zfill(5)+'.json')
        with open(json_file, "w") as outfile:
            json.dump(struc_info, outfile)
        print('Perovskite structure info saved to json file, identifier #', str(identifier))

    return struc_info

def gather_structure_info_dataframe(save=False):

    dataframe = pd.DataFrame(columns=COLUMNS_DICT['structure_info'])
    dataframe.index.name = 'identifier'
    for filepath in sorted(glob.glob(struc_info_path + '*.json')):
        filename = os.path.basename(filepath)
        identifier = int(filename.split('.')[0])
        with open(filepath, 'r') as file:
            structure_info = json.load(file)
        dataframe.loc[identifier] = [
            structure_info['d_interlayer'], structure_info['bond']['average'], structure_info['bond']['xy_average'], structure_info['bond']['z_average'],
            structure_info['angle_XMX']['average'], structure_info['angle_XMX']['xy_average'], structure_info['angle_XMX']['z_average'],
            structure_info['angle_MXM']['average']
            ]
    if save:
        dataframe.to_csv(structure_info_dataframe_path)
    return dataframe 

def get_hse_frontier_json(identifier, save_json=False):
    electronic_structure_parser = ElectronicStructureParser(identifier=identifier)
    electronic_structure_parser.get_projection_on_components()
    hse_frontier = electronic_structure_parser.get_energy_level_alignment(cutoff=0.25)
    if save_json:
        filename = os.path.join(hse_frontier_json_path, str(identifier).zfill(5)+'.json')
        with open(filename, "w") as outfile:
            json.dump(hse_frontier, outfile)
    return hse_frontier

def gather_hse_frontier_dataframe(save=False):
    dataframe = pd.DataFrame(columns=COLUMNS_DICT['hse_frontier'])
    dataframe.index.name = 'identifier'
    for filepath in sorted(glob.glob(hse_frontier_json_path + '*.json')):
        filename = os.path.basename(filepath)
        identifier = int(filename.split('.')[0])
        with open(filepath, 'r') as file:
            hse_frontier = json.load(file)
        dataframe.loc[identifier] = [
            hse_frontier['inorganic_cbm_gamma'], hse_frontier['inorganic_cbm_z'],
            hse_frontier['inorganic_vbm_gamma'], hse_frontier['inorganic_vbm_z'],
            hse_frontier['organic_LUMO'], hse_frontier['organic_HOMO'], hse_frontier['alignment_type']
        ]

    # calibrate all energy levels by setting average of VBM to zero
    reference_energylevel = dataframe['inorganic_vbm_gamma'].mean()
    dataframe['inorganic_cbm_gamma'] = dataframe['inorganic_cbm_gamma'] - reference_energylevel
    dataframe['inorganic_cbm_z'] = dataframe['inorganic_cbm_z'] - reference_energylevel
    dataframe['inorganic_vbm_gamma'] = dataframe['inorganic_vbm_gamma'] - reference_energylevel
    dataframe['inorganic_vbm_z'] = dataframe['inorganic_vbm_z'] - reference_energylevel  
    dataframe['organic_HOMO'] = dataframe['organic_HOMO'] - reference_energylevel
    dataframe['organic_LUMO'] = dataframe['organic_LUMO'] - reference_energylevel 

    if save:
        dataframe.to_csv(hse_frontier_dataframe_path)
    return dataframe 
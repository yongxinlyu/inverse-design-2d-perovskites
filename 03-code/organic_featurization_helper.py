import pandas as pd
import os
import glob
import json
import numpy as np
import re

from rdkit import Chem
from rdkit.Chem import Descriptors
from config import COLUMNS_DICT, PROJECT_ROOT_DIRECTORY
import pubchempy

mo_energetics_path = PROJECT_ROOT_DIRECTORY + '02-metadata/01-mo-energetics/'
mo_energetics_dataframe_path = PROJECT_ROOT_DIRECTORY + '02-metadata/06-csv-files/04-mo-energetics.csv'
organic_dimension_path = PROJECT_ROOT_DIRECTORY + '02-metadata/02-organic-dimension/'
organic_dimension_dataframe_path = PROJECT_ROOT_DIRECTORY + '02-metadata/06-csv-files/06-organic-dimension.csv'


def calculate_linker_position_descriptor(mol, linker_length, primaryamine):
    amat = Chem.GetDistanceMatrix(mol)
    aromatic_atoms_idx= []
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            aromatic_atoms_idx.append(atom.GetIdx())
    aromatic_distance_matrix = amat[np.ix_(aromatic_atoms_idx, aromatic_atoms_idx)]

    charged_atoms_idx= []
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            charged_atoms_idx.append(atom.GetIdx())
    charged_atom_1 = charged_atoms_idx[0]
    charged_atom_2 = charged_atoms_idx[1]
    disNN = amat[charged_atom_1][charged_atom_2]

    real_distance = disNN - linker_length - primaryamine
    max_distance = aromatic_distance_matrix.max()
    linker_position_descriptor = real_distance/max_distance
    return linker_position_descriptor


'''
get organic structure descriptors
'''

def get_organic_structure_descriptors(smiles_canonical):

    #smiles_neutral = smiles_canonical.replace('[NH3+]','N').replace('[nH+]','n') 
    mol = Chem.MolFromSmiles(smiles_canonical)
    
    # conjugated backbone
    ringcount = int(Descriptors.NumAromaticRings(mol))
    patt_biphenyl = Chem.MolFromSmarts('c-c')
    linkage = len(mol.GetSubstructMatches(patt_biphenyl))
    fusion = ringcount - 1 - linkage
    if ringcount==1:
        linkage_p = 0
    else:
        linkage_p = linkage/(ringcount-1)

    patt_furan = Chem.MolFromSmarts('[o&r5]')
    furan = len(mol.GetSubstructMatches(patt_furan))
    patt_thiophene = Chem.MolFromSmarts('[s&r5]')
    thiophene = len(mol.GetSubstructMatches(patt_thiophene))
    patt_pyrrole = Chem.MolFromSmarts('[nH&r5]')
    patt_not_pyrrole = Chem.MolFromSmarts('[nH+&r5]')
    pyrrole = len(mol.GetSubstructMatches(patt_pyrrole)) - len(mol.GetSubstructMatches(patt_not_pyrrole))
    six_ring = ringcount - furan - thiophene - pyrrole
    five_ring = thiophene + furan + pyrrole
    six_ring_p = six_ring/ringcount

    # tethering ammonium and side chain
    patt_primary_amine = Chem.MolFromSmarts('[NH3+]')
    primaryamine = len(mol.GetSubstructMatches(patt_primary_amine))

    patt_aliphatic_carbon = Chem.MolFromSmarts('C')
    aliphatic_carbon = len(mol.GetSubstructMatches(patt_aliphatic_carbon))
    patt_sidechain_on_linker = Chem.MolFromSmarts('C-[CH3]')
    sidechain_on_linker = len(mol.GetSubstructMatches(patt_sidechain_on_linker))
    patt_sidechain_on_backbone = Chem.MolFromSmarts('c-[CH3]')
    sidechain_on_backbone = len(mol.GetSubstructMatches(patt_sidechain_on_backbone))
    linker_length = aliphatic_carbon - sidechain_on_linker - sidechain_on_backbone
    linker_length_p = linker_length/primaryamine
    linker_position = calculate_linker_position_descriptor(mol, linker_length, primaryamine)

    # heteroatom substitution
    patt_hetero_nitrogen = Chem.MolFromSmarts('[nH0]')
    hetero_nitrogen = len(mol.GetSubstructMatches(patt_hetero_nitrogen))
    patt_fluorination = Chem.MolFromSmarts('F')
    fluorination = len(mol.GetSubstructMatches(patt_fluorination))

    organic_structure_descriptors_dict = {
        'ringcount': ringcount,
        'six_ring' : six_ring,
        'six_ring_p': six_ring_p,
        'five_ring': five_ring,
        'thiophene': thiophene,
        'linkage': linkage,
        'linkage_p': linkage_p,
        'fusion': fusion,
        'primaryamine': primaryamine,
        'linker_length': linker_length,
        'linker_length_p': linker_length_p,
        'linker_position': linker_position,
        'hetero_nitrogen': hetero_nitrogen,
        'fluorination': fluorination,
        'furan': furan,
        'pyrrole': pyrrole,
        'sidechain_on_linker': sidechain_on_linker,
        'sidechain_on_backbone': sidechain_on_backbone,
    }
    return organic_structure_descriptors_dict

def get_neutral_smiles(smiles_canonical):
    smiles_neutral = smiles_canonical.replace('[NH3+]','N').replace('[nH+]','n') 
    return smiles_neutral

def get_organic_existence(smiles_neutral):
    compounds = pubchempy.get_compounds(smiles_neutral, namespace='smiles')
    iupac_name = compounds[0].iupac_name
    cid = compounds[0].cid
    organic_existence_dict = {
        'smiles_neutral': smiles_neutral,
        'iupac_name': iupac_name,
        'cid': cid,
    }
    return organic_existence_dict

def get_backbone(smiles):

    # remove all H unless it has n in front of it
    smiles_backbone = re.sub(r'(?<![n])H', '', smiles)

    # remove all uppercase letters other than H and the H or numbers following them
    smiles_backbone = re.sub(r'[A-GI-Z]\d*H?', '', smiles_backbone)
    
    # remove all square brackets and parentheses if there is nothing inside them do it iteratively until there are no more

    while re.search(r'\[\]|\(\)', smiles_backbone):
        smiles_backbone = re.sub(r'\[\]|\(\)', '', smiles_backbone)

    molecule_backbone = Chem.MolFromSmiles(smiles_backbone)
    smiles_backbone = Chem.MolToSmiles(molecule_backbone)
    return smiles_backbone


def gather_mo_energetics_dataframe(save=False):

    dataframe = pd.DataFrame(columns=COLUMNS_DICT['mo_energetics'])
    dataframe.index.name = 'identifier'
    for filepath in sorted(glob.glob(mo_energetics_path + '*.json')):
        filename = os.path.basename(filepath)
        identifier = int(filename.split('.')[0])
        with open(filepath, 'r') as file:
            mo_energetics = json.load(file)
     
        dataframe.loc[identifier] = [
            mo_energetics['HOMO'], mo_energetics['LUMO'], mo_energetics['LUMO'] - mo_energetics['HOMO']
        ]
    if save:
        dataframe.to_csv(mo_energetics_dataframe_path)
    return dataframe 


def gather_organic_dimension_dataframe(save=False):

    dataframe = pd.DataFrame(columns=COLUMNS_DICT['organic_dimension'])
    dataframe.index.name = 'identifier'
    for filepath in sorted(glob.glob(organic_dimension_path + '*.json')):
        filename = os.path.basename(filepath)
        identifier = int(filename.split('.')[0])
        with open(filepath, 'r') as file:
            organic_dimension = json.load(file)
        dataframe.loc[identifier] = [
            organic_dimension['height'], organic_dimension['width'], organic_dimension['length'],
        ]
    if save:
        dataframe.to_csv(organic_dimension_dataframe_path)
    return dataframe




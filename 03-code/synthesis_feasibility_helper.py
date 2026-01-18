import pandas as pd
from config import PROJECT_ROOT_DIRECTORY, COLUMNS_DICT
from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem import rdMolDescriptors
import numpy as np
import itertools
import json
from scipy.special import expit

def calculate_topological_formability_descriptors(mol):
    amat = Chem.GetDistanceMatrix(mol)

    # Add explicit hydrogens to the molecule
    mol_with_h = Chem.AddHs(mol)

    # List to store indices of nitrogen atoms with attached hydrogen(s)
    nitrogen_with_h_indices = []

    # Iterate over all atoms in the molecule
    for atom in mol_with_h.GetAtoms():
        # Check if the atom is nitrogen
        if atom.GetAtomicNum() == 7:  # 7 is the atomic number of nitrogen
            # Iterate through the atom's neighbors
            for neighbor in atom.GetNeighbors():
                # Check if the neighbor is a hydrogen atom
                if neighbor.GetAtomicNum() == 1:  # 1 is the atomic number of hydrogen
                    # If a hydrogen is attached, add the nitrogen atom's index
                    nitrogen_with_h_indices.append(atom.GetIdx())
                    break  # No need to check further neighbors for this nitrogen atom

    STEI_list, eccentricity_list = [], []
    for nitrogen_index in nitrogen_with_h_indices:
        STEI_nitrogen = (1/(np.delete(amat[nitrogen_index], nitrogen_index)**3)).sum()
        STEI_list.append(STEI_nitrogen)

        eccentricity_nitrogen = max(np.delete(amat[nitrogen_index], nitrogen_index))
        eccentricity_list.append(eccentricity_nitrogen)
    
    # the STEI is the second smallest value in STEI list (as used in Jinlan's paper, maximum)
    
    STEI = sorted(STEI_list)[1]

    # the eccentricity is the second largest value in eccentricity list
    #eccentricity = sorted(eccentricity_list)[1]
    eccentricity = sorted(eccentricity_list)[-2]

    disNN_list = []
    # calculate the distance between any two nitrogen atoms in the nitrogen_with_h_indices
    for nitrogen_pair in list(itertools.combinations(nitrogen_with_h_indices, 2)):
        disNN = 1/(amat[nitrogen_pair[0]][nitrogen_pair[1]])**2
        disNN_list.append(disNN)
    disNN = min(disNN_list)

    charged_atoms_idx= []
    for atom in mol.GetAtoms():
        if atom.GetFormalCharge() != 0:
            charged_atoms_idx.append(atom.GetIdx())
    ring_atoms_idx = []
    for atom in mol.GetAtoms():
        if atom.GetIsAromatic():
            ring_atoms_idx.append(atom.GetIdx())

    # average of number of rotatable bonds in the tail
    num_rot_tail_1 = min(amat[charged_atoms_idx[0]][ring_atoms_idx])
    num_rot_tail_2 = min(amat[charged_atoms_idx[1]][ring_atoms_idx])
    num_rot_tail = 0.5*(num_rot_tail_1 + num_rot_tail_2)

    #NumN = len(mol.GetSubstructMatches(Chem.MolFromSmarts('[#7]')))
    NumN = len(charged_atoms_idx) # in Jinlan's paper, this is defined as number of nitrogen contributing to forming hydrogen bond

    return {
        'STEI': STEI, 
        'NumRot_tail': num_rot_tail,
        'eccentricity': eccentricity,
        'disNN': disNN, 
        'NumN': NumN,
        }

def calculate_general_formability_descriptors(mol):
    ## Calculate molecular descriptors from Jinlan's paper from RDKit
    descriptors_dict = {}
    descriptors_dict['MolWt'] = Descriptors.MolWt(mol)
    descriptors_dict['NumHeteroatoms'] = rdMolDescriptors.CalcNumHeteroatoms(mol)
    descriptors_dict['NumRotatableBonds'] = rdMolDescriptors.CalcNumRotatableBonds(mol)
    descriptors_dict['FractionCSP3'] = rdMolDescriptors.CalcFractionCSP3(mol)
    descriptors_dict['Kappa1'] = rdMolDescriptors.CalcKappa1(mol)
    descriptors_dict['Kappa2'] = rdMolDescriptors.CalcKappa2(mol)
    descriptors_dict['Kappa3'] = rdMolDescriptors.CalcKappa3(mol)
    descriptors_dict['NumAromaticCarbocycles'] = rdMolDescriptors.CalcNumAromaticCarbocycles(mol)
    descriptors_dict['NumAromaticRings'] = rdMolDescriptors.CalcNumAromaticRings(mol)
    descriptors_dict['NumAmideBonds'] = rdMolDescriptors.CalcNumAmideBonds(mol)
    descriptors_dict['NumAtomStereoCenters'] = rdMolDescriptors.CalcNumAtomStereoCenters(mol)
    descriptors_dict['NumBridgeheadAtoms'] = rdMolDescriptors.CalcNumBridgeheadAtoms(mol)
    descriptors_dict['NumSaturatedCarbocycles'] = rdMolDescriptors.CalcNumSaturatedCarbocycles(mol)
    descriptors_dict['NumAliphaticCarbocycles'] = rdMolDescriptors.CalcNumAliphaticCarbocycles(mol)
    descriptors_dict['NumAromaticHeterocycles'] = rdMolDescriptors.CalcNumAromaticHeterocycles(mol)
    descriptors_dict['NumAliphaticRings'] = rdMolDescriptors.CalcNumAliphaticRings(mol)
    descriptors_dict['NumLipinskiHBA'] = rdMolDescriptors.CalcNumLipinskiHBA(mol)
    descriptors_dict['NumLipinskiHBD'] = rdMolDescriptors.CalcNumLipinskiHBD(mol)
    descriptors_dict['NumRings'] = rdMolDescriptors.CalcNumRings(mol)
    descriptors_dict['NumSaturatedHeterocycles'] = rdMolDescriptors.CalcNumSaturatedHeterocycles(mol)
    descriptors_dict['NumSaturatedRings'] = rdMolDescriptors.CalcNumSaturatedRings(mol)
    descriptors_dict['NumSpiroAtoms'] = rdMolDescriptors.CalcNumSpiroAtoms(mol)
    descriptors_dict['NumUnspecifiedAtomStereoCenters'] = rdMolDescriptors.CalcNumUnspecifiedAtomStereoCenters(mol)
    descriptors_dict['NumHeterocycles'] = rdMolDescriptors.CalcNumHeterocycles(mol)

    ## Calculate molecular descriptors from Yiying's paper
    ring_info = mol.GetRingInfo()
    ring_sizes = [len(ring) for ring in ring_info.AtomRings()]
    descriptors_dict['LargestRingSize'] = max(ring_sizes) if ring_sizes else 0

    return descriptors_dict



# load parameters_dict from file
with open(PROJECT_ROOT_DIRECTORY+'01-rawdata/13-existing-molecule/parameters_dict.json', 'r') as f:
    parameters_dict = json.load(f)

def logistic(x, a, b):
    return expit(a * (x + b))  # equivalent to 1 / (1 + exp(-a*(x + b)))


def calculate_formability_score(formability_descriptors_dict, smearing_factor=1.0):

    # step 1: scaling the descriptors
    scaled_descriptors_dict = {}
    for feature, params in parameters_dict.items():
        mean = params["mean"]
        scale = params["scale"]
        scaled_descriptors_dict[feature] = (formability_descriptors_dict[feature] - mean) / scale

    # step 2: calculating the formability score using the logistic function

    formability_score = {}
    for descriptor, params in parameters_dict.items():
        a = params['a'] * smearing_factor
        b = params['b']
        x = scaled_descriptors_dict[descriptor]
        y = logistic(x, a, b)
        formability_score[descriptor+'_score'] = y
    # Calculate the overall formability score
    formability_overall_score = np.mean(list(formability_score.values()))
    formability_score['formability_score'] = formability_overall_score
    
    return formability_score

'''Abondoned
formability_descriptor_criteria_df = pd.read_csv(PROJECT_ROOT_DIRECTORY+'01-rawdata/11-formability/formability_descriptor_criteria.csv')
def calculate_formability_decision(formability_descriptors_dict):
    formability_decision_dict = {}
    for i in formability_descriptor_criteria_df.index:
        descriptor = formability_descriptor_criteria_df.loc[i, 'formability_descriptors']
        if formability_descriptor_criteria_df.loc[i, 'criteria'] == 'min':
            if formability_descriptors_dict[descriptor] < formability_descriptor_criteria_df.loc[i, 'boundary_value']:
                formability_decision_dict[descriptor+'_decision'] = False
            else:
                formability_decision_dict[descriptor+'_decision'] = True
        elif formability_descriptor_criteria_df.loc[i, 'criteria'] == 'max':
            if formability_descriptors_dict[descriptor] > formability_descriptor_criteria_df.loc[i, 'boundary_value']:
                formability_decision_dict[descriptor+'_decision'] = False
            else:
                formability_decision_dict[descriptor+'_decision'] = True
    
    # according to all the formability descriptors decision, make the final decision and add to formability_decision_dict
    if all(formability_decision_dict.values()):
        formability_decision_dict['formability_decision'] = True
    else:
        formability_decision_dict['formability_decision'] = False

    return formability_decision_dict
'''
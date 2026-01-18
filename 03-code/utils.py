from config import PROJECT_ROOT_DIRECTORY, COLUMNS_DICT
from rdkit import Chem
from rdkit.Chem import Draw
import pandas as pd

def visualize_smiles_list(smiles_list, mols_per_row=5, sub_img_size=(200, 200), save_path=False, legends=None):
    """
    Visualizes a list of SMILES strings in a grid format.
    
    Parameters:
    - smiles_list: List of SMILES strings to visualize.
    - mols_per_row: Number of molecules per row in the grid.
    - sub_img_size: Size of each molecule image in the grid.
    
    Returns:
    - Displays the grid image of molecules.
    """
    # Convert each SMILES to an RDKit Mol object
    mol_list = [Chem.MolFromSmiles(smiles) for smiles in smiles_list]
    
    if legends == None:
        legends = [f"{smiles}" for smiles in smiles_list]
    # Generate the grid image
    img = Draw.MolsToGridImage(mol_list, molsPerRow=mols_per_row, subImgSize=sub_img_size, 
                               returnPNG = False,
                               legends=legends)
    if save_path:
        img.save(save_path)
    return img



# MO prediction
unnormalized_coefficient_df = pd.read_csv(PROJECT_ROOT_DIRECTORY + "01-rawdata/10-machine-learning/unnormalized_coefficient_lasso.csv", index_col=0)

def predict_mo_lasso(fingerprint_dict, target):
    intercept = unnormalized_coefficient_df.loc[target,'intercept']
    feature_contribution_dict = {
        key: unnormalized_coefficient_df.loc[target,key] * fingerprint_dict[key]
        for key in COLUMNS_DICT['machine_learning_features']
    }
    mo_prediction = sum(feature_contribution_dict.values()) + intercept
    return mo_prediction


# Define the cutoff for HOMO_predicted
HOMO_CUTOFF = -11.1

# Define the cutoffs for LUMO_predicted based on the ringcount
LUMO_CUTOFFS_dict = {
    1: -11,
    2: -10.6,
    3: -9.6,
    4: -7.3,
    5: -6.1,
    6: -6,
    7: -6,
    8: -6,
    9: -6,
    10: -6,
    11: -6,
}

def get_alignment_type_prediction(ringcount, HOMO_prediction, LUMO_prediction):
    # need HOMO_prediction, LUMO_prediction, and ringcount

    # Get the LUMO cutoff based on the ring count
    LUMO_CUTOFF = LUMO_CUTOFFS_dict.get(ringcount, None)
    
    alignment_type = 'Unknown'
    
    # Determine the type based on the cutoffs
    if HOMO_prediction <= HOMO_CUTOFF and LUMO_prediction >= LUMO_CUTOFF:
        alignment_type = 'Ia'
    elif HOMO_prediction < HOMO_CUTOFF and LUMO_prediction < LUMO_CUTOFF:
        alignment_type = 'IIb'
    elif HOMO_prediction > HOMO_CUTOFF and LUMO_prediction > LUMO_CUTOFF:
        alignment_type = 'IIa'
    elif HOMO_prediction > HOMO_CUTOFF and LUMO_prediction < LUMO_CUTOFF:
        alignment_type = 'Ib'

    return alignment_type

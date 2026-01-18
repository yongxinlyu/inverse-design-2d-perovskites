from rdkit import Chem

def convert_smiles_to_canonical(smiles_dataframe):
    "this functiona takes a dataframe containing smiles and add a new column containing canonical smiles"
    for identifier in smiles_dataframe.index:
        smiles = smiles_dataframe.at[identifier, 'smiles']
        mol = Chem.MolFromSmiles(smiles)
        smiles_canonical = Chem.MolToSmiles(mol)
        smiles_dataframe.at[identifier,'smiles_canonical'] = smiles_canonical
    return smiles_dataframe

def convert_smiles_to_mols(smiles_dataframe):
    # this function take a dataframe containing smiles and return a list of mol object
    mols = []
    for identifier in smiles_dataframe.index:
        smiles = smiles_dataframe.at[identifier, 'smiles_canonical']
        mol = Chem.MolFromSmiles(smiles)
        mol.SetProp("_Name", str(identifier))
        mols.append(mol)
    return mols


def convert_mols_to_gridimage(mols_list):
    gridimage = Chem.Draw.MolsToGridImage(
        mols=mols_list, 
        legends=[x.GetProp("_Name") for x in mols_list],
        returnPNG = False,
        molsPerRow=6, subImgSize=(150, 150))
    return gridimage

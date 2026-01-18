import os
from rdkit import Chem
from rdkit.Chem import AllChem, Draw

def dataframe_to_image_file(dataframe, grid_image_path):
    # this function take a dataframe containing identifier as index, canonical smiles, return gridimage
    mols_list = []
    identifier_list = []
    for identifier in dataframe.index:
        smiles_canonical = dataframe.at[identifier, 'smiles_canonical']
        mol = Chem.MolFromSmiles(smiles_canonical)
        mols_list.append(mol)
        identifier_list.append(identifier)

    n_image = int(len(mols_list)/50)

    for i in range(n_image):
        mols_list_in_this_image = mols_list[50*i:50*(i+1)]
        identifier_list_in_this_image = identifier_list[50*i:50*(i+1)]
        gridimage = Draw.MolsToGridImage(
            mols = mols_list_in_this_image, 
            legends = [str(identifier) for identifier in identifier_list_in_this_image],
            returnPNG = False,
            molsPerRow=5, subImgSize=(150, 150))  

        gridimage.save(grid_image_path + str(identifier_list_in_this_image[0]).zfill(5) + '-' + str(identifier_list_in_this_image[-1]).zfill(5)+'.png',quality=200)

def dataframe_to_image_show(dataframe):
    # this function take a dataframe containing identifier as index, canonical smiles, return gridimage
    mols_list = []
    for identifier in dataframe.index:
        smiles_canonical = dataframe.at[identifier, 'smiles_canonical']
        mol = Chem.MolFromSmiles(smiles_canonical)
        mol.SetProp("_Name", str(identifier))

        mols_list.append(mol)

    gridimage = Draw.MolsToGridImage(
        mols = mols_list, 
        legends = [mol.GetProp("_Name") for mol in mols_list],
        returnPNG = False,
        molsPerRow=5, subImgSize=(150, 150)) 
    return gridimage 
          


def dataframe_to_xyz_file(dataframe, output_directory):
    # this function takes dataframe containing organic genome dataframe, write xyz file
    for identifier in dataframe.index:
        smiles_canonical = dataframe.at[identifier, 'smiles_canonical']
        mol = Chem.MolFromSmiles(smiles_canonical)

        # Add Hs, essential for generating good structure 
        mol = Chem.AddHs(mol)

        # Generate conformer with random seeds for reproducibility
        AllChem.EmbedMolecule(mol, randomSeed=2)
        AllChem.MMFFOptimizeMolecule(mol)

        # write 3d structure to xyz file

        xyz_path = os.path.join(output_directory, str(identifier) + '.xyz')
        Chem.rdmolfiles.MolToXYZFile(mol, xyz_path)


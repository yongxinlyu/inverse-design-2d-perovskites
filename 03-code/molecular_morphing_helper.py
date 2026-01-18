import pandas as pd
import numpy as np
import math


def update_organic_genome_dataframe(previous_organic_genome_dataframe, dataframe_to_merge):

    #replace generation value for overlapping organic molecules
    overlapping_organic_molecules = pd.merge(previous_organic_genome_dataframe[['smiles_canonical']].reset_index(), dataframe_to_merge, on='smiles_canonical',how='inner')
    identifier_list_overlapping = overlapping_organic_molecules.identifier
    for identifier in identifier_list_overlapping:
        if math.isnan(previous_organic_genome_dataframe.at[identifier, 'generation']):
            previous_organic_genome_dataframe.at[identifier, 'generation'] = overlapping_organic_molecules.set_index('identifier').at[identifier, 'generation']

    #append new organic molecules
    number_of_existing_organic_molecule = previous_organic_genome_dataframe.shape[0]
    appending_organic_molecules = dataframe_to_merge[~dataframe_to_merge.smiles_canonical.isin(previous_organic_genome_dataframe.smiles_canonical)]
    number_of_appending_organic_molecule = appending_organic_molecules.shape[0]
    appending_organic_molecules.loc[:,'identifier'] = np.arange(number_of_existing_organic_molecule+1, number_of_existing_organic_molecule+number_of_appending_organic_molecule+1)
    new_organic_genome_dataframe = pd.concat([previous_organic_genome_dataframe.reset_index(), appending_organic_molecules]).set_index('identifier')
    return new_organic_genome_dataframe
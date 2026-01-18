import os
import numpy as np
import pandas as pd
import json
import matplotlib.pyplot as plt
import seaborn as sns
from config import PROJECT_ROOT_DIRECTORY, VASP_USER_SETTING_DICT

from pymatgen.io.vasp.outputs import Vasprun, Outcar
from pymatgen.electronic_structure.core import Spin
from pymatgen.electronic_structure.plotter import DosPlotter

from rdkit import Chem
from rdkit.Chem import AllChem

from pymatgen.core import Lattice, Structure, Molecule
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from pymatgen.io.vasp.sets import MPRelaxSet, MPHSEBSSet

organic_genome_path = PROJECT_ROOT_DIRECTORY + '02-metadata/06-csv-files/01-organic-genome.csv'
vasp_organic_xyz_path = PROJECT_ROOT_DIRECTORY + '01-rawdata/05-vasp-organic-xyz/'
organic_dimension_path = PROJECT_ROOT_DIRECTORY + '02-metadata/02-organic-dimension/'
initial_perovskite_cif_path = PROJECT_ROOT_DIRECTORY + '01-rawdata/06-initial-perovskite-cif/'
relax_set_path = PROJECT_ROOT_DIRECTORY + '01-rawdata/07-relax-set/'
refine_set_path = PROJECT_ROOT_DIRECTORY + '01-rawdata/08-refine-set/'
hse_set_path = PROJECT_ROOT_DIRECTORY + '01-rawdata/09-hse-set/'
hse_frontier_path = PROJECT_ROOT_DIRECTORY + '02-metadata/05-hse-frontier/'
hse_frontier_dataframe_path = PROJECT_ROOT_DIRECTORY + '02-metadata/06-csv-files/08-hse-frontier.csv'


class PerovskiteStructureBuilder:

    def __init__(self, identifier):
        self.identifier = identifier
        self.Pb_I_bond = float(3.2)  #initial bond length
        self.n_layers = 1
        organic_genome_dataframe = pd.read_csv(organic_genome_path,index_col='identifier')
        self.smiles = organic_genome_dataframe.at[self.identifier, 'smiles_canonical']

    def generate_conformer(self, write_xyz_file=False):

        print('Generating conformer for molecule #', str(self.identifier), ' ...')
        mol = Chem.MolFromSmiles(self.smiles)
        mol = Chem.AddHs(mol)
        AllChem.EmbedMolecule(mol, randomSeed=2)
        AllChem.MMFFOptimizeMolecule(mol)
        if write_xyz_file:
            xyz_path = os.path.join(vasp_organic_xyz_path, str(self.identifier)+'.xyz')
            Chem.rdmolfiles.MolToXYZFile(mol, xyz_path) 
            print('Organic molecule structure written to ', str(xyz_path), ', please check and modify conformer')     
    
    def find_calibration_vector(self):
        xyz_path = os.path.join(vasp_organic_xyz_path, str(self.identifier)+'.xyz')
        self.mol = Molecule.from_file(filename=xyz_path)

        # Identify index of charged atoms and aromatic atoms using rdkit
        mol_rdkit = Chem.MolFromSmiles(self.smiles)
        charged_atoms_idx, aromatic_atoms_idx = [], []
        for atom in mol_rdkit.GetAtoms():
            if atom.GetFormalCharge() != 0:
                charged_atoms_idx.append(atom.GetIdx())
            if atom.GetIsAromatic():
                aromatic_atoms_idx.append(atom.GetIdx())            
        
        #print(charged_atoms_idx)
        #charged_atoms_idx = [1, 12] # 186
        
        #charged_atoms_idx = [2, 13] # 36236-1
        #charged_atoms_idx = [3, 15] # 36236

        #charged_atoms_idx = [28, 35] # 197-1
        #charged_atoms_idx = [1, 12] # 197-2

        # Find Major axis along the two charged nitrogen
        major_axis = self.mol.cart_coords[charged_atoms_idx[0]] - self.mol.cart_coords[charged_atoms_idx[1]]
        major_axis_norm = major_axis/np.linalg.norm(major_axis)

        # Find minor axis perpendicular the conjugated rings
        aromatic_atoms_coords = []
        for idx in aromatic_atoms_idx:
            coords = self.mol.cart_coords[idx]
            aromatic_atoms_coords.append(coords)
        aromatic_atoms_coords = np.array(aromatic_atoms_coords)

        centroid = np.mean(aromatic_atoms_coords, axis=0)
        centered_points = aromatic_atoms_coords - centroid
        cov_matrix = np.cov(centered_points, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        min_eigenvalue_index = np.argmin(eigenvalues)
        minor_axis = eigenvectors[:, min_eigenvalue_index]
        minor_axis_norm = minor_axis/np.linalg.norm(minor_axis)

        # Get xyz vectors to rotate the molecule
        y_axis = np.cross(major_axis_norm, minor_axis_norm)
        x_axis = np.cross(y_axis, major_axis_norm)
        z_axis = major_axis_norm

        self.calibration_vector = np.stack((x_axis, y_axis, z_axis), axis=0)
        self.calibration_origin = centroid        

    def calibrate_molecule(self, save_dimension=False):

        # Rotate the molecule according to calibration vector around calibration origin
        mol_coords = self.mol.cart_coords
        mol_species = self.mol.species
        mol_coords_calibrated = np.empty((0,3))

        for coord in mol_coords:
            coord_calibrated = np.matmul(self.calibration_vector, coord - self.calibration_origin)
            mol_coords_calibrated = np.vstack((mol_coords_calibrated, coord_calibrated))

        mol_calibrated = Molecule(species=mol_species, coords=mol_coords_calibrated)

        # Obtain dimensions of the molecule
        max_x = max(coord[0] for coord in mol_coords_calibrated)
        min_x = min(coord[0] for coord in mol_coords_calibrated)
        max_y = max(coord[1] for coord in mol_coords_calibrated)
        min_y = min(coord[1] for coord in mol_coords_calibrated)
        max_z = max(coord[2] for coord in mol_coords_calibrated)
        min_z = min(coord[2] for coord in mol_coords_calibrated)
        height = max_z - min_z
        width = max_x - min_x
        length = max_y - min_y
        center = np.array([(max_x+min_x)*0.5, (max_y+min_y)*0.5, (max_z+min_z)*0.5])

        self.mol_calibrated = mol_calibrated
        self.mol_dimension = {'height': height, 'width': width, 'length': length, 'center':center,}
        if save_dimension:
            filename = os.path.join(organic_dimension_path, str(self.identifier).zfill(5)+'.json')
            with open(filename, "w") as outfile:
                json.dump({'height': height, 'width': width, 'length': length}, outfile)

    def build_inorganic_framework(self, penetration_depth=0.2): #NH3+ penetration depth

        # Build 3d perovskite framework with one additional layer for slicing
        coord_Pb = np.array([[0.0, 0.0, 0.0]])
        coord_I = np.array([[0.5, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, 0.0, 0.5]])
        coord_Cs = np.array([[0.5, 0.5, 0.5]])
        coords = np.concatenate((coord_Pb, coord_I, coord_Cs))
        species = ["Pb", "I", "I", "I", "Cs"]
        lattice = Lattice.from_parameters(
            a=self.Pb_I_bond*2, b=self.Pb_I_bond*2, c=self.Pb_I_bond*2,
            alpha=90.0, beta=90.0, gamma=90.0)
        structure_3d_inorganic = Structure(lattice=lattice, 
                                            species=species,
                                            coords=coords)
        #scaling_matrix_2unit = np.array([[-1, 1, 0], [1, 1, 0], [0, 0, 1]])
        scaling_matrix = np.array([[2, 0, 0], [0, 2, 0], [0, 0, self.n_layers+1]])
        structure_3d_inorganic.make_supercell(scaling_matrix=scaling_matrix)

        # Slice 3d perovskite framework into 2d perovskite framework
        indices_to_remove=[]
        for i in structure_3d_inorganic.indices_from_symbol("Pb"):
            coords = structure_3d_inorganic.frac_coords[i]
            if -0.25/self.n_layers < coords[2] < 0.25/self.n_layers:
                indices_to_remove.append(i)
        for i in structure_3d_inorganic.indices_from_symbol("I"):
            coords = structure_3d_inorganic.frac_coords[i]
            if -0.25/self.n_layers < coords[2] < 0.25/self.n_layers:
                indices_to_remove.append(i)
        for i in structure_3d_inorganic.indices_from_symbol("Cs"):
            coords = structure_3d_inorganic.frac_coords[i]
            if coords[2] < 0.75/self.n_layers or 1-coords[2] < 0.75/self.n_layers:
                indices_to_remove.append(i)
        structure_3d_inorganic.remove_sites(indices=indices_to_remove)

        # Get 2d perovskite framework
        lattice_3d = structure_3d_inorganic.lattice
        species_2d = structure_3d_inorganic.species
        cart_coords_2d = structure_3d_inorganic.cart_coords
        d_interlayer = self.mol_dimension['height'] - penetration_depth*2 - 1.025*2 #N-H bond length 1.025

        lattice_2d = Lattice.from_parameters(
            a=lattice_3d.a, b=lattice_3d.b, c=lattice_3d.c+d_interlayer-2*self.Pb_I_bond,
            alpha=lattice_3d.alpha, beta=lattice_3d.beta, gamma=lattice_3d.gamma)

        structure_2d_inorganic = Structure(
            lattice=lattice_2d, species=species_2d,
            coords=cart_coords_2d, coords_are_cartesian=True)

        structure_2d_inorganic.translate_sites(
            indices=list(range(structure_2d_inorganic.num_sites)), 
            vector=(0, 0, -2*self.Pb_I_bond), frac_coords=False)
        
        self.inorganic_framework = structure_2d_inorganic

    def build_perovskite(self, save_cif=False):

        # Rotate the organic spacers and insert into cavities
        inorganic_framework = self.inorganic_framework
        cavity_c = 0.5 + (self.n_layers - 1)*self.Pb_I_bond/(inorganic_framework.lattice.c)
        packing_herringbone = {
            'cavity1': {"vector": np.array([[1, 0, 0],[0, 1, 0],[0, 0, 1]]), "displacement": np.array([0.25, 0.25, cavity_c])},
            'cavity2': {"vector": np.array([[0, 1, 0],[-1, 0, 0],[0, 0, 1]]), "displacement": np.array([0.25, 0.75, cavity_c])},
            'cavity3': {"vector": np.array([[0, -1, 0],[-1, 0, 0],[0, 0, -1]]), "displacement": np.array([0.75, 0.25, cavity_c])},
            'cavity4': {"vector": np.array([[1, 0, 0],[0, -1, 0],[0, 0, -1]]), "displacement": np.array([0.75, 0.75, cavity_c])},
        }
        for i in range(4):
            cavity_idx = 'cavity'+str(i+1)
            mol_coords_cavity = np.empty((0,3))
            for coord in self.mol_calibrated.cart_coords:
                coord_cavity = np.matmul(packing_herringbone[cavity_idx]['vector'], coord-self.mol_dimension['center'])
                mol_coords_cavity = np.vstack((mol_coords_cavity, coord_cavity))
            mol_cavity = Molecule(species=self.mol_calibrated.species,
                                  coords=mol_coords_cavity)
            
            displacement = packing_herringbone[cavity_idx]['displacement'] * np.array(inorganic_framework.lattice.abc)

            for site_idx in range(mol_cavity.num_sites):
                species = mol_cavity.species[site_idx]
                site_coord = mol_cavity.cart_coords[site_idx]
                
                inorganic_framework.append(species=species, coords=site_coord+displacement, coords_are_cartesian=True)

        inorganic_framework.sort() # sort sequence of species according to electronegativity
        self.perovskite_structure = inorganic_framework
        if save_cif:
            filename = os.path.join(initial_perovskite_cif_path, str(self.identifier)+'.cif')
            self.perovskite_structure.to(filename=filename)
            print('perovskite structure written to ', str(filename),', please check crystal structure') 

    def get_space_group_info(self):
        spacegroup = SpacegroupAnalyzer(structure=self.perovskite_structure)
        print(f'space group info: {spacegroup.get_crystal_system()}, point group {spacegroup.get_point_group_symbol()}, space group {spacegroup.get_space_group_symbol()}, {spacegroup.get_space_group_number()}')



class VaspInputGenerator:
    def __init__(self):
        self.relax_set_incar_settings = VASP_USER_SETTING_DICT['relax_set_incar_settings']
        self.refine_set_incar_settings = VASP_USER_SETTING_DICT['relax_set_incar_settings']  # same as relax set
        self.relax_set_kpoints_settings = VASP_USER_SETTING_DICT['relax_set_kpoints_settings']
        self.refine_set_kpoints_settings = VASP_USER_SETTING_DICT['refine_set_kpoints_settings']
        self.hse_set_incar_settings = VASP_USER_SETTING_DICT['hse_set_incar_settings']
        self.hse_set_kpoints_settings = VASP_USER_SETTING_DICT['hse_set_kpoints_settings']

    def make_relax_set(self, identifier_list):

        for identifier in identifier_list:

            filename = os.path.join(initial_perovskite_cif_path, str(identifier)+'.cif')
            structure = Structure.from_file(filename=filename)
            
            relax_set = MPRelaxSet(
                structure=structure, 
                user_incar_settings=self.relax_set_incar_settings, 
                user_kpoints_settings=self.relax_set_kpoints_settings,
                user_potcar_functional="PBE_54",
            )

            relax_set.write_input(output_dir=os.path.join(relax_set_path, str(identifier)))

    def make_refine_set(self, identifier_list):

        for identifier in identifier_list:

            filename = os.path.join(relax_set_path, str(identifier), 'CONTCAR')
            structure = Structure.from_file(filename=filename)

            refine_set = MPRelaxSet(
                structure = structure,
                user_incar_settings = self.refine_set_incar_settings,
                user_kpoints_settings = self.refine_set_kpoints_settings,
                user_potcar_functional = 'PBE_54',        
            )

            refine_set.write_input(output_dir=os.path.join(refine_set_path, str(identifier)))

    def make_hse_set(self, identifier_list):

        for identifier in identifier_list:

            filename = os.path.join(refine_set_path,str(identifier),'CONTCAR')
            structure = Structure.from_file(filename = filename)
            hse_set = MPHSEBSSet(
                structure = structure,
                user_incar_settings = self.hse_set_incar_settings,
                user_kpoints_settings = self.hse_set_kpoints_settings,
                user_potcar_functional = 'PBE_54',
            )

            hse_set.write_input(output_dir=os.path.join(hse_set_path, str(identifier)))


class ElectronicStructureParser:

    def __init__(self, identifier):
        self.identifier = identifier
        vasprun_xml_path = os.path.join(hse_set_path, str(self.identifier), 'vasprun.xml')
        vasprun = Vasprun(filename = vasprun_xml_path, parse_projected_eigen=True)
        indices_from_Pb = vasprun.final_structure.indices_from_symbol('Pb')

        outcar_path = os.path.join(hse_set_path, str(self.identifier),'OUTCAR')
        outcar = Outcar(filename=outcar_path)
        eigenvalues = outcar.read_core_state_eigen()
        Pb_1s_eigenvalue = []
        for atom_index in indices_from_Pb:
            Pb_1s_eigenvalue.append(eigenvalues[atom_index]['1s'][0]) 
        self.core_level = np.average(Pb_1s_eigenvalue) + 88086.5  #this value is a reference value to make fermi level close to zero, should be same across all compounds

        self.nb_kpoints = len(vasprun.actual_kpoints)
        self.efermi = vasprun.efermi
        self.band_structure = vasprun.get_band_structure()
        self.complete_dos = vasprun.complete_dos
        self.nb_bands = self.band_structure.nb_bands
        self.projection_on_elements = self.band_structure.get_projection_on_elements()[Spin.up] #return a array with band_index and kpoint_index
        self.bands = self.band_structure.bands[Spin.up]
        self.cbm_band_index = self.band_structure.get_cbm()['band_index'][Spin.up][0] #first band in the band pair is cbm
        self.vbm_band_index = self.band_structure.get_vbm()['band_index'][Spin.up][-1] #second band in the band pair is vbm

    def get_organic_inorganic_projection(self, band_index, kpoint_index):
        organic_inorganic_projection = {'inorganic': 0, 'organic': 0}

        #projection_on_elements on particular band and kpoint
        projection = self.projection_on_elements[band_index][kpoint_index]
        for element, projection_value in projection.items():
            if element in ['Pb', 'I']:
                organic_inorganic_projection['inorganic'] += projection_value
            else:
                organic_inorganic_projection['organic'] += projection_value
        return organic_inorganic_projection

    def get_projection_on_components(self):
        # helper function to get projection on organic and inorganic components for all bands and kpoints, returns a dataframe containing all information

        projection_on_components = []
        for band_index in range(self.nb_bands):
            for kpoint_index in range(self.nb_kpoints):
                organic_inorganic_projection = self.get_organic_inorganic_projection(band_index,kpoint_index)
                projection_on_components.append([band_index,kpoint_index,
                                                  organic_inorganic_projection['inorganic'],
                                                  organic_inorganic_projection['organic'],
                                                  self.bands[band_index][kpoint_index]])
        
        projection_on_components = pd.DataFrame(data=np.array(projection_on_components),
                                                           columns=['band_index','kpoint_index','inorganic_projection','organic_projection','bands'])
        projection_on_components['inorganic_ratio'] = projection_on_components['inorganic_projection']/(projection_on_components['inorganic_projection']+projection_on_components['organic_projection'])
        projection_on_components['organic_ratio'] = projection_on_components['organic_projection']/(projection_on_components['inorganic_projection']+projection_on_components['organic_projection'])
        self.projection_on_components = projection_on_components

    def get_energy_level_alignment(self, cutoff = 0.25):
        dataframe = self.projection_on_components
        kpoint_index_list = np.arange(self.nb_kpoints)

        inorganic_cbm_gamma = min(dataframe[(dataframe.inorganic_ratio >= cutoff) & (dataframe.kpoint_index == kpoint_index_list[0]) & (dataframe.bands >= self.efermi)].bands)
        inorganic_cbm_z = min(dataframe[(dataframe.inorganic_ratio >= cutoff) & (dataframe.kpoint_index == kpoint_index_list[-1]) & (dataframe.bands >= self.efermi)].bands)
        inorganic_vbm_gamma = max(dataframe[(dataframe.inorganic_ratio >= cutoff) & (dataframe.kpoint_index == kpoint_index_list[0]) & (dataframe.bands <= self.efermi)].bands)
        inorganic_vbm_z = max(dataframe[(dataframe.inorganic_ratio >= cutoff) & (dataframe.kpoint_index == kpoint_index_list[-1]) & (dataframe.bands <= self.efermi)].bands)

        organic_LUMO = min(dataframe[(dataframe.organic_ratio >= cutoff) & (dataframe.kpoint_index == kpoint_index_list[0]) & (dataframe.bands >= self.efermi)].bands)
        organic_HOMO = max(dataframe[(dataframe.organic_ratio >= cutoff) & (dataframe.kpoint_index == kpoint_index_list[0]) & (dataframe.bands <= self.efermi)].bands)

        if inorganic_vbm_z >= organic_HOMO and inorganic_cbm_z <= organic_LUMO:
            alignment_type = 'Ia'
        elif  inorganic_vbm_z < organic_HOMO and inorganic_cbm_z <= organic_LUMO:
            alignment_type = 'IIa'
        elif inorganic_vbm_z >= organic_HOMO and inorganic_cbm_z > organic_LUMO:
            alignment_type = 'IIb'
        elif inorganic_vbm_z < organic_HOMO and inorganic_cbm_z > organic_LUMO:
            alignment_type = 'Ib'

        #band edge are calibrated by core level as reference
        energy_level_alignment = {'inorganic_cbm_gamma': inorganic_cbm_gamma - self.core_level,
                                       'inorganic_cbm_z': inorganic_cbm_z - self.core_level,
                                       'inorganic_vbm_gamma': inorganic_vbm_gamma - self.core_level,
                                       'inorganic_vbm_z': inorganic_vbm_z - self.core_level,
                                       'organic_LUMO': organic_LUMO - self.core_level,
                                       'organic_HOMO': organic_HOMO - self.core_level,
                                       'alignment_type': alignment_type}
        self.energy_level_alignment = energy_level_alignment
        return energy_level_alignment

    def get_projection_plot(self, band_index_range=100): #range is bands on each side of cbm and vbm

        first_band_index = self.vbm_band_index - band_index_range
        last_band_index = self.cbm_band_index + band_index_range
        dataframe = self.projection_on_components
        dataframe = dataframe[(dataframe.band_index > first_band_index) & (dataframe.band_index < last_band_index)]

        fig, axs = plt.subplots(1,4, sharey=True)
        sns.scatterplot(data=dataframe.query('kpoint_index == 0'),x='inorganic_ratio',y='bands', ax=axs[0],color='blue',alpha=0.3)
        sns.scatterplot(data=dataframe.query('kpoint_index == 1'),x='inorganic_ratio',y='bands', ax=axs[1],color='blue',alpha=0.3)
        plt.axhline(y=self.efermi)
        sns.scatterplot(data=dataframe.query('kpoint_index == 0'),x='organic_ratio',y='bands', ax=axs[2],color='red',alpha=0.3)
        sns.scatterplot(data=dataframe.query('kpoint_index == 1'),x='organic_ratio',y='bands', ax=axs[3],color='red',alpha=0.3)

        return fig
    
    def plot_projected_dos(self):
        pdos = self.complete_dos.get_element_dos()
        pdos_inorganic_list = []
        pdos_organic_list = []

        for element in pdos.keys():
            if str(element) in {'Pb', 'I'}:
                pdos_inorganic_list.append(pdos[element])
            else:
                pdos_organic_list.append(pdos[element])

        pdos_inorganic = pdos_inorganic_list[0]
        for i in range(len(pdos_inorganic_list)-1):
            pdos_inorganic = pdos_inorganic + pdos_inorganic_list[i+1]

        pdos_organic = pdos_organic_list[0]
        for i in range(len(pdos_organic_list)-1):
            pdos_organic = pdos_organic + pdos_organic_list[i+1]

        ax = DosPlotter()
        ax.add_dos("inorganic", pdos_inorganic)
        ax.add_dos("organic", pdos_organic)
        ax.show(xlim=[-5,5])

        return ax 

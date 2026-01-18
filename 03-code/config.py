# config.py

# ==================== Project Paths ====================
PROJECT_ROOT_DIRECTORY = "/Users/yongxinlyu/OneDrive - UNSW/01-bench/DJperovskite/"

# ==================== Data Columns Configuration ====================
COLUMNS_DICT = {
    'organic_genome': ['smiles_canonical','generation'],
    'organic_structure_descriptors': [
        'ringcount','six_ring','six_ring_p','five_ring','thiophene','linkage','linkage_p','fusion',
        'primaryamine','linker_length','linker_length_p','linker_position',
        'hetero_nitrogen','fluorination','furan','pyrrole',
        'sidechain_on_linker','sidechain_on_backbone'
        ],
    'organic_property_descriptors': ['LogP', 'Hacceptor', 'Hdonor', 'STEI', 'disNN', 'eccentricity'],
    'molecular_fingerprint': [
        'ringcount','linkage_p','six_ring_p', 
        'primaryamine','linker_length','linker_position',
        'hetero_nitrogen','fluorination','furan','pyrrole', 
        'sidechain_on_linker','sidechain_on_backbone',
        ],
    'pca_descriptors': [
        'ringcount','six_ring','five_ring','thiophene','linkage','fusion',
        'primaryamine','linker_length','linker_position',
        'hetero_nitrogen','fluorination','furan','pyrrole',
        'sidechain_on_linker','sidechain_on_backbone'
        ],
    'pca': ['d1','d2','d1_jitter','d2_jitter'],
    'mo_energetics': ['HOMO','LUMO','HOMO_LUMO_gap'],
    'mo_prediction':['HOMO_predicted','LUMO_predicted'],
    'organic_dimension': ['height','width','length'],
    'structure_info': ['d_interlayer', 'bond_average','bond_equatorial','bond_axial','angleXMX_average', 'angleXMX_equatorial', 'angleXMX_axial','angleMXM_average'],
    'hse_frontier': ['inorganic_cbm_gamma', 'inorganic_cbm_z','inorganic_vbm_gamma', 'inorganic_vbm_z','organic_LUMO', 'organic_HOMO','alignment_type'],
    'other': ['mp_id','cid'],
    'machine_learning_features': [
        'ringcount','linkage_p','six_ring_p', 
        'primaryamine','linker_length','linker_position',
        'hetero_nitrogen','fluorination','furan','pyrrole', 
        'sidechain_on_linker','sidechain_on_backbone',
        ],
    '2D_formability_descriptors': ['STEI', 'NumRot_tail', 'eccentricity','disNN'],
    '2D_formability_general_descriptors': [
    'MolWt',
    'NumHeteroatoms',
    'NumRotatableBonds',
    'FractionCSP3',
    'Kappa1',
    'Kappa2',
    'Kappa3',
    'NumAromaticCarbocycles',
    'NumAromaticRings',
    'NumAmideBonds',
    'NumAtomStereoCenters',
    'NumBridgeheadAtoms',
    'NumSaturatedCarbocycles',
    'NumAliphaticCarbocycles',
    'NumAromaticHeterocycles',
    'NumAliphaticRings',
    'NumLipinskiHBA',
    'NumLipinskiHBD',
    'NumRings',
    'NumSaturatedHeterocycles',
    'NumSaturatedRings',
    'NumSpiroAtoms',
    'NumUnspecifiedAtomStereoCenters',
    'NumHeterocycles',
    'LargestRingSize'],
    'molecular_fingerprint_full_names':[
        'no. rings', '% ring linkage', '% 6-membered rings',
        'no. primary ammonium', 'linker length', 'ammonium position', 
        'no. N (pyridine)', 'no. F', 'no. O (furan)', 'no. N (pyrrole)', 
        'no. side chain (linker)', 'no. side chain (backbone)']
    # Additional categories here...
}

# ==================== Machine Learning Models Configuration ====================

# Regressor parameter grids
REGRESSOR_PARAM_GRIDS = {
    "lasso": {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
    "ridge": {'alpha': [10, 5, 1, 0.1, 0.01, 0.001]},
    "elastic_net": {'alpha': [1, 0.1, 0.01, 0.001, 0.0001, 0.0001]},
    "svr_linear": {'C': [20, 10, 1, 0.1], 'epsilon': [1, 0.1, 0.01]},
    "svr_rbf": {'C': [20, 10, 1, 0.1], 'epsilon': [1, 0.1, 0.01]},
    "svr_poly": {'C': [20, 10, 1, 0.1], 'epsilon': [1, 0.1, 0.01]},
    "knn_regressor": {'n_neighbors': [3, 4, 5, 6, 7], 'weights': ['uniform', 'distance']},
    "random_forest_regressor": {'n_estimators': [10, 50, 100, 150, 200]},
}

# Classifier parameter grids
CLASSIFIER_PARAM_GRIDS = {
    "svc_linear": {'C': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
    "linear_svc": {'C': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
    "knn_classifier": {'n_neighbors': [1, 2, 3, 4, 5]},
    "decision_tree_classifier": {'max_depth': [5, 6, 7, 8, 9, 10]},
    "svc_rbf": {'C': [1, 0.1, 0.01, 0.001, 0.0001, 0.00001]},
    "logistic_regression": {'C': [10, 5, 2, 1, 0.1]},
}

# ==================== Identifier Configuration ====================
IDENTIFIER_DICT = {
    'gen_0': [26],
    'gen_1': [4,7,22,25,28,30,34,37,58,105,121,211,220,269,270],
    'gen_2': [1,5,6,8,9,10,11,17,18,21,24,27,29,31,33,35,36,39,40,41,42,43,44,45,46,47,48,49,50,51,52,56,57,62,63,64,68,85,89,90,97,100,103,106,108,112,113,114,115,116,118,120,122,131,133,141,161,168,175,183,191,193,194,195,196,203,204,212,216,217,218,221,222,223,224,228,229,234,258,260,262,267,271,272,273,274,275,276,277,278,279,280,281,282,283,284,285,286,287,288,289,290,291,292,293,294,295,296,297,298,299,300,301,302,303,304,305,306,307,308,309,310,311,312,313,314,315,316,317,318,319,320,321,322,323,324,325,326,327,328,329,330,331,332,333,334,335,336,337,338,339,340,341,342,343,344,345,346,347,348,349,350,351,352,353,354,355,356,357,358,359,360,361,362,363,364,365,366,367,368,369,370,371,372,373,374,375,376,377,378,379,380,381,382,383,384,385,386,387,388,389,390,391,392,393,394,395,396,397,398,399,400,401,1822,1823,1824,1825,1826,1827,1828,1829,1830,1831,1832,1833,1834,1835,1836,1837,1838,1839,1840,1841,1842,1843,1844],
    'typeII_a':[102,205,239,189,206,36228,35575,179,180,181,182,27437],
    #'typeII_a': [205,768,979,1190,3410,4302,4680,5127,5665,19521,19522,25887,25888,202,25881,26061,26060,19710,25870,1212,19514,26099,1204,4330,33135,29466],
    #'typeII_a_new': [36222,36223,36224,36225,36226,36227,36228,36232,511,205,101,102],
    'typeII_b': [39,376,440,441,634,635,640,641,110,38,1672,1673,111,1674,1675,1676,1680,3787,3883,3884,7971,7972,7979,8401,8412,34225,34226,34227,34235,34236,34244,34289,34291,34293,34296,34301,36192,36193,36194,36195,40541,40542,40543,40544,40545,40546,40547,40548,40549,40550,40551,40552,40553,40554,40555,40556,40557,40558],
    #'typeII_b': [111,1674,1675,1680,3787,7971,8014,34225,34227,34235,34236,34244,34289,34290,34293,34294,34295,34299,34300,34322,110,1672,38,1673,634,440,635,640,641,441,1681,1678],
    #'typeII_b_final': [167,439,638,168,3782,3783,7959,7960,7998,7999,1667,1668,440,441,634,635,640,641,39,376,],
    #'typeII_b_new': [38,39,110,111,376,440,441,634,635,640,641,1672,1673,1674,1675,1676,1680],
    'typeI_b': [36236,186], # 200, 201
    #'typeI_b': [200,104,232,233,239,243,201,103,209,27256,27279,27280,27281],
    'existing': [1,4,11,26,37,56,59,60,100,105,163,175,188,189,190,203,204,205,206,213,380,],
    'pi_conjugation': [101,102,104,107,145,146,147,148,149,150,151,152,153,154,200],
    'acene':[103,27437,104,184,200,201,36236,186], #197
    'oligothiophene': [101,102,205,239,189,206,36228,35575,179,180,181,182],
    'diazine': [],
    }

# ==================== VASP input Configuration ====================
VASP_USER_SETTING_DICT = {
    'relax_set_incar_settings': {
        'SYSTEM': 'DJ_perovskite',
        'ALGO': 'Fast',
        'EDIFF_PER_ATOM': 5e-06, #(default is 5e-05)
        'ENCUT': 480,
        'IBRION': 1,  #converge faster when close to minimum
        #'ISIF': 3,
        'ISYM': 0,  #turn off symmetry
        #'ISMEAR': -5,   # automatically change to 0 for 1 or 2 kpoints
        'ISPIN': None,
        #'LASPH': True,    # non-spherical charge density
        #'LORBIT': 11,   # projected orbitals
        #'LREAL': 'Auto',
        'LWAVE': False,
        'NELM': 30,  #if does not converge in 30 electron step, it will not converge
        'NSW': 1000,   # more ionic steps for convergence
        #'PREC': 'Accurate',
        #'SIGMA': 0.05,
        'MAGMOM': None,
        #'LMAXMIX': 6,  # for ALGO=Fast
        'IVDW': 11,
        'LCHARG': False,
        'KPAR': 1,
        'NCORE': 48,
    },
    'relax_set_kpoints_settings': {
        'reciprocal_density': 64    # default to 64
    },
    'refine_set_kpoints_settings': {
        'reciprocal_density': 300    # default to 64
    },
    'hse_set_incar_settings':{
        'SYSTEM': 'DJ_perovskite',
        'ALGO': None,
        #'EDIFF': 1e-5,
        'EDIFF_PER_ATOM': 5e-06, #default 5e-05
        'ENCUT': 480,
        'ICHARG': None,
        'IBRION': None,  
        'ISIF': None,
        'ISMEAR': 0,
        'LASPH': True,
        'LWAVE': True,
        'NELM': 30,  #if does not converge in 30 electron step, restart from CHGCAR and WAVECAR, change ALGO to ALL
        'LCHARG': True,
        #'LAECHG': None,
        #'LVHAR': None,
        'ISPIN': None,
        'MAGMOM': None,
        'PREC': None,
        'IVDW': 11,
        'KPAR': 1,
        'NCORE': 48,
        'LHFCALC': True,
        'GGA': 'PE',
        'HFSCREEN': 0.2,
        'AEXX': 0.4,
        'LSORBIT': True,
        'ICORELEVEL': 1,
        'PRECFOCK': None,
    },
    'hse_set_kpoints_settings': {
        'reciprocal_density': 50,
    },
    'bs_set_incar_settings': {
        'SYSTEM': 'DJ_perovskite',
        'EDIFF_PER_ATOM': 5e-06, #default 5e-05
        'ENCUT': 480,
        #'IBRION': None,
        'ICHARG': 11,
        #'ISIF': 3,
        'ISMEAR': 0,
        'ISPIN': None,
        'LASPH': True,
        'LORBIT': 11,
        'LSORBIT': True,
        'LREAL': 'Auto',
        'LWAVE': False,
        'NSW': 0,
        'PREC': 'Accurate',
        'SIGMA': 0.05,
        #'MAGMOM': None,
        'LMAXMIX': 6,
        'IVDW': 11,
        'LCHARG': False,
        'ICORELEVEL': 1,
        'KPAR': 1,
        'NCORE': 48,
    },
}

COLOR_PALETTE = {
    'warm000': '#d95f0e',
    'warm002': '#fddbc7',
    'cold000': '#2166ac',
    'cold001': '#92c5de',
    'cold002': '#d1e5f0', 
}
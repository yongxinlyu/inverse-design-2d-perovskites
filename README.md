## Inverse Design of 2D Hybrid Perovskites

This repository contains the code and data supporting the manuscript:

“Fingerprinting Organic Molecules for the Inverse Design of Two-Dimensional Hybrid Perovskites with Target Energetics”
(submitted to Science Advances)

The workflow integrates molecular fingerprinting, high-throughput first-principles data, interpretable machine learning, and synthesis feasibility screening to enable inverse design of Dion–Jacobson (DJ) phase two-dimensional hybrid perovskites with targeted energy level alignment.

### Repository Overview

The repository implements the full inverse-design pipeline described in the manuscript, including:
- Generation of invertible 12-digit molecular fingerprints
- Expansion of chemical space via molecular morphing
- Machine-learning prediction of HOMO/LUMO energy levels
- Screening for synthetic accessibility and 2D formability
- Identification of candidate organic spacers with Type Ib, IIa, and IIb energy level alignment

### Directory Structure
```
├── README.md     
│    
├── 02-metadata/   
│   ├── 01-mo-energetics/       
│   ├── 02-organic-dimension/        
│   ├── 03-final-perovskite-cif/
│   ├── 04-structure-info/
│   ├── 05-hse-frontier/
│   └── 06-csv-files/
│
├── 03-code/
│   ├── molecular_morphing_helper.py
│   ├── organic_featurization_helper.py
│   ├── perovskite_featurization_helper.py
│   ├── model_training_helper.py
    ├── ...
│   └── utils.py
│
└── 04-notebooks/
    ├── 01-molecular-morphing.ipynb 
    ├── 02-organic-energy-level.ipynb 
    ├── 04-build-perovskite.ipynb
    ├── 05-perovskite-featurization.ipynb
    ├── 06-machine-learning.ipynb
    ├── ...
    └── 21-final-candidates-visualization.ipynb


```

### Contact Information

- Principal Investigator Information
    - Name: Tom Wu
    - Institution: The Hong Kong Polytechnic University 
    - Email: tom-tao.wu@polyu.edu.hk

- First Author Information
    - Name: Yongxin Lyu
    - Institution: University of New South Wales
    - Email: yongxin.lyu@unsw.edu.au

This readme file was generated on 2026-01-18 by Yongxin Lyu

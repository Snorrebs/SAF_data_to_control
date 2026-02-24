# Conda Environment Setup

## 1. Install Conda
Install **Miniconda**.

## 2. Create environment from file
From the repository root:

```bash
conda env create -f environment.yml
conda activate plsr-metamodel
```

## 3. Verify
```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib, joblib"
```
Should not give error.

## 4. Run Project
See `readme_VRFT_simulation.md` for the simulation and VRFT workflow.

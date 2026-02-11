# Conda Environment Setup

## 1. Install Conda
Install **Miniconda**.

## 2. Create environment
```bash
conda create -n ENVIRONMENT_NAME python=3.11
conda activate ENVIRONMENT_NAME
```
Change ENVIRONMENT_NAME to desired name

## 3. Install Packages
```bash
conda install numpy pandas scipy scikit-learn matplotlib joblib
```

## 4. Verify
```bash
python -c "import numpy, pandas, scipy, sklearn, matplotlib, joblib"
```
Should not give error!

## 5. Run Project
Look at `SAF_data_to_control/readme_VRFT_simulation.md` to run the simulator and/or the VRFT.
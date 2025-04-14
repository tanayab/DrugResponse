Project Title
Advancing Precision Oncology: A Deep Learning Framework for Predicting Drug Sensitivity
1. Overview of Codebase
This repository implements a modular, deep learning-based drug response prediction framework using a hybrid GCN-CNN model enhanced with cross-attention fusion. The model predicts IC50 drug sensitivity based on drug molecular structures and gene expression profiles.

Key components:
- Preprocessing SMILES into molecular graphs using RDKit
- Normalising gene expression data
- Combining data using PyTorch Geometric
- Training hybrid GCN + CNN models with attention
- Evaluating and deploying using Gradio


2. Setup Instructions
   For running the Web Application locally:
```bash
git clone https://huggingface.co/spaces/tanaya-jb/Drug_Response_WebApp
cd WebApp

pip install -r requirements.txt
pip install torch-geometric

# run the web app
python app.py
```

For executing the entire pipeline (Data Loading -> Preprocessing -> Training -> Testing -> Deployment):
Run these jupyter notebooks -> 1_Preprocessing_and_Training.ipynb and 2_Testing_and_Deployment.ipynb




3. Data Sources:

Complete data link :
https://drive.google.com/file/d/1cwBgLi-QLPtOS_hoehR3POlPLlWCs6eV/view?usp=drive_link 

Test data link:
https://drive.google.com/file/d/1haMLs_F3o9OozvFQu5za1LKyeD9CzzCW/view?usp=sharing

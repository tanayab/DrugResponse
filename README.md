# Advancing Precision Oncology  
## A Deep Learning Framework for Predicting Drug Sensitivity

---

## 1. Overview of Codebase

This repository presents a modular, deep learning-based framework for **predicting drug response (IC50)** values using a hybrid **Graph Convolutional Network (GCN)** and **Convolutional Neural Network (CNN)** architecture, enhanced with **cross-attention fusion**.

The model integrates **drug molecular structure** (via SMILES graphs) with **cancer cell gene expression profiles** to predict drug sensitivity — enabling advancements in precision oncology.

### Key Components:
- Preprocessing SMILES into molecular graphs using **RDKit**
- Normalising and formatting gene expression data
- Combining drug and gene data using **PyTorch Geometric**
- Training a hybrid **GCN + CNN** model with **multi-head cross-attention**
- Evaluating and deploying predictions through an interactive **Gradio web app**

---

## 2. Setup Instructions

### Run the Web Application Locally

```bash
# Clone the repo
git clone https://github.com/tanayab/DrugResponse.git
cd DrugResponse/WebApp/

#(Optional) Create a Python Virtual Environment:
python3 -m  venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
pip install torch-geometric

# Launch the Gradio app
python app.py
```

### For executing the entire pipeline (Data Loading -> Preprocessing -> Training -> Testing -> Deployment):
Run these jupyter notebooks -> 1_Preprocessing_and_Training.ipynb and 2_Testing_and_Deployment.ipynb

## 3. Data Sources

### Complete Raw Data link :
https://drive.google.com/file/d/1cwBgLi-QLPtOS_hoehR3POlPLlWCs6eV/view?usp=drive_link 

### Preprocessed Test data link:
https://drive.google.com/file/d/1haMLs_F3o9OozvFQu5za1LKyeD9CzzCW/view?usp=sharing

## 4. Code Documentation 
A detailed code explanation for the entire pipeline is present in a document under the `\documentation` folder

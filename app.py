
import gradio as gr
import torch
import json
import pickle
import numpy as np
from torch_geometric import data as DATA
from model import DrugResponseModel  # Assumes your model class is in model.py

# Load model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model2 = DrugResponseModel(drug_feat_dim=78).to(device)
model2.load_state_dict(torch.load("baseline.model", map_location=device))
model2.eval()

# Load required mappings and features
with open("drug_name_to_smiles.json") as f:
    drug_name_to_smiles = json.load(f)

with open("cell_name_to_vector.pkl", "rb") as f:
    cell_name_to_vector = pickle.load(f)

with open("valid_drug_cell_pairs.pkl", "rb") as f:
    valid_pairs = pickle.load(f)

with open("smiles_graph_map.pkl", "rb") as f:
    smiles_graph_map = pickle.load(f)

with open("cell_label_to_id.json") as f:
    label_to_cosmic_id = json.load(f)

def smiles_to_pyg(smiles):
    graph_data = smiles_graph_map.get(smiles)
    if graph_data is None:
        raise ValueError("Graph for this SMILES not found!")

    num_atoms, node_features, edge_index, edge_features, _ = graph_data

    x = torch.tensor(node_features, dtype=torch.float32)
    edge_index = torch.tensor(edge_index, dtype=torch.long).T
    edge_attr = torch.tensor(edge_features, dtype=torch.float32)
    batch = torch.zeros(x.shape[0], dtype=torch.long)

    return DATA.Data(x=x, edge_index=edge_index, edge_attr=edge_attr, batch=batch)

cell_dropdown_labels = sorted(label_to_cosmic_id.keys())

def predict_ic50_dropdown(drug_name, cell_label):
    try:
        cell_id = label_to_cosmic_id[cell_label]
        if (drug_name, cell_id) not in valid_pairs:
            return "⚠️ This combination is not supported by training data. Please select a different cell line for reliable prediction."

        smiles = drug_name_to_smiles[drug_name]
        drug_data = smiles_to_pyg(smiles)

        x = drug_data.x.to(device)
        edge_index = drug_data.edge_index.to(device)
        edge_attr = drug_data.edge_attr.to(device)
        batch = drug_data.batch.to(device)

        cell_vector = cell_name_to_vector[cell_id]
        cell_tensor = torch.tensor(cell_vector, dtype=torch.float32).view(1, 1, -1).to(device)

        with torch.no_grad():
            output = model2(x, edge_index, batch, cell_tensor, edge_attr)
            ic50 = output.item()

        if ic50 < 0.5337153:
            interpretation = "🟢 Strong Response"
        elif ic50 < 0.5839771:
            interpretation = "🟡 Moderate Response"
        else:
            interpretation = "🔴 Weak Response"

        return f"{ic50:.7f}\n{interpretation}"

    except Exception as e:
        return f"❌ Error during prediction: {str(e)}"

iface = gr.Interface(
    fn=predict_ic50_dropdown,
    inputs=[
        gr.Dropdown(choices=sorted(drug_name_to_smiles.keys()), label="Select Drug"),
        gr.Dropdown(choices=cell_dropdown_labels, label="Select Cell Line")
    ],
    outputs=[gr.Textbox(label="Predicted IC50")],
    title="Cancer Drug Response Predictor (IC50)",
    description="""
    This app predicts the **IC50 (half maximal inhibitory concentration)** value for a given cancer drug and cell line combination.
    The **IC50** indicates how much of a drug is needed to inhibit 50% of the cancer cell activity — **lower IC50 implies stronger drug efficacy**.

    🔬 **Use Case**: Predict the sensitivity of cancer cell lines to targeted drugs using molecular features (SMILES for drugs, gene expression for cell lines).

    🧾 **Inputs**:
    - **Drug**: Select a drug from the list (converted from SMILES structure into graph features).
    - **Cell Line**: Choose a cancer cell line (represented by its gene expression profile).

    The app uses a trained deep learning model combining **GCNs (Graph Convolutional Networks)** and **CNNs** to estimate how effective a drug would be against a specific cancer cell line.

    **IC50 Response Guide:**
    🟢 **Strong**: High sensitivity to drug (effective at low concentration)
    🟡 **Moderate**: Partial response — drug shows some effectiveness
    🔴 **Weak**: Likely resistance — requires higher concentration to inhibit cells
    *(Lower IC50 = better drug efficacy)*  
    **Note:** Relative sensitivity is calculated based on the IC50 distribution in this dataset.
    """,
    flagging_mode="never"
)

iface.launch(share=True)

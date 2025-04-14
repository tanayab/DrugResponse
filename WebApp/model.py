import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv, global_max_pool as gmp

class DrugResponseModel(torch.nn.Module):
    def __init__(self, output_size=1, conv_filters=32, embedding_dim=128, drug_feat_dim=334, output_dim=128, dropout_rate=0.5):  # qwe

        super(DrugResponseModel, self).__init__()

        self.output_size = output_size

        # Graph Convolutional Network layers for drug SMILES representation
        self.gcn_layer1 = GCNConv(drug_feat_dim, drug_feat_dim)
        self.gcn_layer2 = GCNConv(drug_feat_dim, drug_feat_dim*2)
        self.gcn_layer3 = GCNConv(drug_feat_dim*2, drug_feat_dim * 4)

        self.fc_gcn1 = torch.nn.Linear(drug_feat_dim*4, 1024)
        self.fc_gcn2 = torch.nn.Linear(1024, output_dim)

        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)

        # 1D CNN layers for cell line feature extraction
        self.cell_conv1 = nn.Conv1d(
            in_channels=1, out_channels=conv_filters, kernel_size=8)
        self.cell_pool1 = nn.MaxPool1d(3)

        self.cell_conv2 = nn.Conv1d(
            in_channels=conv_filters, out_channels=conv_filters*2, kernel_size=8)
        self.cell_pool2 = nn.MaxPool1d(3)

        self.cell_conv3 = nn.Conv1d(
            in_channels=conv_filters*2, out_channels=conv_filters*4, kernel_size=8)
        self.cell_pool3 = nn.MaxPool1d(3)

        self.fc_cell = nn.Linear(4096, output_dim)

        # Cross-attention layers to fuse drug and cell features
        self.cross_attention1 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout_rate)
        self.cross_attention2 = nn.MultiheadAttention(output_dim, num_heads=8, dropout=dropout_rate)

        self.norm_attn1 = nn.LayerNorm(output_dim)
        self.norm_attn2 = nn.LayerNorm(output_dim)

        # Fully connected layers for final prediction
        self.fc_combined = nn.Linear(2*output_dim, 128)
        self.output_layer = nn.Linear(128, self.output_size)

    def forward(self, x, edge_idx, batch, cell_x, edge_weight=None):

        # GCN branch for drug input
        x = self.gcn_layer1(x, edge_idx, edge_weight)
        x = self.relu(x)

        x = self.gcn_layer2(x, edge_idx, edge_weight)
        x = self.relu(x)

        x = self.gcn_layer3(x, edge_idx, edge_weight)
        x = self.relu(x)

        # Global Max Pooling across graph nodes
        x = gmp(x, batch)

        # Fully connected layers for drug features
        x = self.relu(self.fc_gcn1(x))
        x = self.dropout(x)
        x = self.fc_gcn2(x)
        x = self.dropout(x)


        # 1D CNN branch for cell line features
        cell_feat = self.cell_conv1(cell_x)
        cell_feat = F.relu(cell_feat)
        cell_feat = self.cell_pool1(cell_feat)

        cell_feat = self.cell_conv2(cell_feat)
        cell_feat = F.relu(cell_feat)
        cell_feat = self.cell_pool2(cell_feat)

        cell_feat = self.cell_conv3(cell_feat)
        cell_feat = F.relu(cell_feat)
        cell_feat = self.cell_pool3(cell_feat)

        # Flatten and apply fully connected layer
        xt = cell_feat.view(-1, cell_feat.shape[1] * cell_feat.shape[2])
        xt = self.fc_cell(xt)

        print(f"x shape: {x.shape}")
        print(f"xt shape: {xt.shape}")


        # Cross-attention mechanism
        attn_out1, _ = self.cross_attention1(x, xt, xt)
        attn_out1 = attn_out1 + x
        attn_out1 = self.norm_attn1(attn_out1)

        attn_out2, _ = self.cross_attention2(xt, x, x)
        attn_out2 = attn_out2 + xt
        attn_out2 = self.norm_attn2(attn_out2)

        # Concatenate and classify
        combined = torch.cat((attn_out1, attn_out2), 1)
        combined = self.relu(combined)
        combined = self.dropout(combined)

        combined = self.fc_combined(combined)
        combined = self.relu(combined)
        combined = self.dropout(combined)

        output = self.output_layer(combined)
        # No activation function as it is regression problem

        return output

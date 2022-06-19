
from turtle import forward
import torch
from dgl.nn.pytorch.glob import SumPooling, AvgPooling, MaxPooling

from src.model import GNN

class GNN_graphpred(torch.nn.Module):
    def __init__(self, gnn, n_layer, dim, num_tasks, node_feature_len, bond_feature_len):
        super(GNN_graphpred, self).__init__()
        self.gnn = gnn
        self.n_layer = n_layer
        self.dim = dim
        self.bond_feature_len = bond_feature_len
        self.node_feature_len = node_feature_len
        self.num_tasks = num_tasks
        self.relu = torch.nn.ReLU()
        self.GNN_model = GNN(self.gnn, self.n_layer, self.node_feature_len, self.dim, self.bond_feature_len)
        self.graph_pred_linear = torch.nn.Linear(2*self.dim, self.num_tasks)


    def from_pretrained(self, model_file):
        self.GNN_model.load_state_dict(torch.load(model_file))

    def forward(self, react, prod):
        react_representation = self.GNN_model(react)
        prod_representation = self.GNN_model(prod)
        graph_representation = torch.cat((prod_representation, react_representation), 1)
        graph_representation = self.relu(graph_representation)
        return self.graph_pred_linear(graph_representation)


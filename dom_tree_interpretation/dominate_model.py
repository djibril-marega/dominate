import torch
import torch.nn as nn
from dom_tree_interpretation.bert import TextualEncoderModel
from dom_tree_interpretation.gnn import GCNConvModel, gnn_inputs_prepared
from dom_tree_interpretation.decoder import DecoderModel

class DominateModel(nn.Module):
    def __init__(self, in_channels=2304, out_channels=768, D=768, S=16, d_model=768):
        super().__init__()
        self.textualEncoderModel = TextualEncoderModel()
        self.GNNModel = GCNConvModel(in_channels, out_channels)
        self.decoder = DecoderModel(D, S, d_model)
    
    def forward(self, texts, serialized_attributes, serialized_tag, j_dom_file_path):
        outputs = self.textualEncoderModel(texts, serialized_attributes, serialized_tag)
        gnn_inputs_data = gnn_inputs_prepared(outputs, j_dom_file_path)
        outputs = self.GNNModel(**gnn_inputs_data)
        outputs = self.decoder(outputs)
        return outputs

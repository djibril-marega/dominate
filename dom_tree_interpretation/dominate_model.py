import torch
import torch.nn as nn
from dom_tree_interpretation.bert import TextualEncoderModel
from dom_tree_interpretation.gnn import GCNConvModel, gnn_inputs_prepared
from dom_tree_interpretation.decoder import DecoderModel

class DominateModel(nn.Module):
    def __init__(self, in_channels=2304, out_channels=768, D=768, S=16, d_model=768):
        super().__init__()
        self.textual_encoder_model = TextualEncoderModel()
        self.GNN_model = GCNConvModel(in_channels, out_channels)
        self.decoder_model = DecoderModel(D, S, d_model)
    
    def forward(self, texts, serialized_attributes, serialized_tag, j_dom_file_path, stop_at='None', hidden_states=False, get_gnc_inputs=False):
        outputs_enc = self.textual_encoder_model(texts, serialized_attributes, serialized_tag)
        gnn_inputs_data = gnn_inputs_prepared(outputs_enc, j_dom_file_path)
        outputs_gnn = self.GNN_model(**gnn_inputs_data)
        if stop_at == 'gnc':
            if hidden_states:
                if get_gnc_inputs:
                    return {'encoder': outputs_enc, 'gnc': outputs_gnn, 'gnc_inputs': gnn_inputs_data}
                else:
                    return {'encoder': outputs_enc, 'gnc': outputs_gnn}
            else:
                return outputs_gnn
        outputs_dec = self.decoder_model(outputs_gnn)
        if hidden_states:
            return {'encoder': outputs_enc, 'gnc': outputs_gnn, 'decoder': outputs_dec}
        else:
            return outputs_dec

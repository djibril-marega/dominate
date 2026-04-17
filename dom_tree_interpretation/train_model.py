import torch.nn as nn
from dom_tree_interpretation.gnn import gnn_pipeline
from dom_tree_interpretation.decoder import decoder_t5_pipeline
from dom_tree_interpretation.bert import generate_textual_embeddings

class DominateModal(nn.Module):
    def __init__(self):
        super().__init__()
        j_dom_file_path = None
        bert_embedding_file_path = None
        gnn_outputs_file_path = None
        decoder_outputs_file_path = None
        self.modelEncoder = generate_textual_embeddings(
            bert_embedding_file_path=bert_embedding_file_path, 
            j_dom_file_path=j_dom_file_path
        )
        self.modelGraph = gnn_pipeline(j_dom_file_path, gnn_outputs_file_path)
        self.modelDecoder = decoder_t5_pipeline(gnn_outputs_file_path, decoder_outputs_file_path)
    
    def forward(self, outputs):
        outputs = self.modelEncoder()
        self.modelGraph(outputs)
        self.modelDecoder()
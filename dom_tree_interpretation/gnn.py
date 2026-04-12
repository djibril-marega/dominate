from torch_geometric.data import Data
import torch
from file_manager.json_file_manager import find_key, get_json_data
import torch
from torch_geometric.nn import MessagePassing
from torch.nn import Linear, Parameter, ReLU
from torch_geometric.utils import add_self_loops, degree


def get_textual_features(dict_textual_embeddings):
    textual_embeddings_list = []
    for t_textual_embeddings in zip(*dict_textual_embeddings.values()):
        textual_embeddings = [t[1] for t in t_textual_embeddings]
        textual_embeddings_feat = torch.concat(textual_embeddings, dim=1)
        textual_embeddings_list.append(textual_embeddings_feat)

    textual_embeddings_sample = torch.concat(textual_embeddings_list, dim=0)
    return textual_embeddings_sample


def extract_structure(json_dom, depth=0, index_child=0, result=None):
    if result is None:
        result = []

    # Ajouter le noeud courant
    node_id = json_dom.get("index")
    result.append((node_id, depth, index_child))

    # Parcourir les enfants
    children = json_dom.get("children", [])
    for i, child in enumerate(children):
        extract_structure(child, depth + 1, i, result)

    return result


# textual features
def pipeline_get_textual_embeddings(textual_embedding_file_path):
    dict_textual_embeddings = torch.load(textual_embedding_file_path)
    return get_textual_features(dict_textual_embeddings)
    

# structural embeddings features
def pipeline_get_structural_embeddings(json_dom):
    dom_structure = extract_structure(json_dom)
    list_structure_dom = []
    for elem_html_structure in dom_structure:
        t_html_structure = torch.tensor(elem_html_structure[1:])
        list_structure_dom.append(t_html_structure)

    return torch.stack(list_structure_dom, dim=0)


def pipeline_get_gnn_nodes_feat(json_dom, textual_embedding_file_path):
    textual_feats = pipeline_get_textual_embeddings(textual_embedding_file_path)
    structural_feats = pipeline_get_structural_embeddings(json_dom)
    return torch.concat([textual_feats, structural_feats], dim=1)


def build_edges_from_dom(dom_json):
    edges = []

    def traverse(node):
        parent_id = node["index"] - 1  # optionnel (0-based)

        for child in node.get("children", []):
            child_id = child["index"] - 1

            # parent -> children
            edges.append([parent_id, child_id])

            # children -> parent (bidirectionnel)
            edges.append([child_id, parent_id])

            # récursion
            traverse(child)

    traverse(dom_json)

    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return edge_index


class GCNConvModel(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super().__init__(aggr='add')
        self.lin1 = Linear(in_channels, 256, bias=True)
        self.relu1 = ReLU()

        self.lin2 = Linear(256, 64, bias=True)
        self.relu2 = ReLU()

        self.lin3 = Linear(64, out_channels, bias=True)
        self.relu3 = ReLU()

        self.reset_parameters()
    
    def reset_parameters(self):
        self.lin1.reset_parameters()
        self.lin2.reset_parameters()
        self.lin3.reset_parameters()
    
    def forward(self, x, edge_index):
        # Step 1: Add self-loops to the adjacency matrix.
        edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))

        # Step 2 : Transformation of nodes features
        x = self.lin1(x)
        x = self.relu1(x)
        x = self.lin2(x)
        x = self.relu2(x)
        x = self.lin3(x)
        x = self.relu3(x)

        # Step 3: Compute normalization.
        row, col = edge_index
        deg = degree(col, x.size(0), dtype=x.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        deg_inv_sqrt[deg_inv_sqrt == float('inf')] = 0
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]

        # Step 4-5: Start propagating messages.
        out = self.propagate(edge_index, x=x, norm=norm)
        return out
    
    def message(self, x_j, norm):
        # x_j has shape [E, out_channels]
        # Step 4: Normalize node features.
        return norm.view(-1, 1) * x_j



def gnn_pipeline(j_dom_file_path, gnn_outp_file_path, textual_embedding_file_path):
    json_dom = get_json_data(j_dom_file_path)
    nodes_feats = pipeline_get_gnn_nodes_feat(json_dom, textual_embedding_file_path)
    edge_index = build_edges_from_dom(json_dom)
    data = Data(x=nodes_feats, edge_index=edge_index)
    model = GCNConvModel(in_channels=nodes_feats.size(1), out_channels=16)
    outputs = model(**data)
    torch.save(outputs, gnn_outp_file_path)
    print("GNC outputs generated")





# Create a node for each html element in DOM
#data = HeteroData()


# create an edge type graph connectivity

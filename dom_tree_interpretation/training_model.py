from file_manager.json_file_manager import find_key, get_json_data, atributes_similarity_scores
from dom_tree_interpretation.bert import serialize_attributs, serialize_tags, serialize_tuple
from dom_tree_interpretation.dominate_model import DominateModel
import torch.nn.functional as F
import torch
from collections import defaultdict, deque

def prepared_encoder_outputs(list_outputs):
    list_samples_embeddings = [t[1] for t in list_outputs]
    return torch.cat(list_samples_embeddings, dim=0)

def compute_embedding_similarity_matrix(embed_st_gen, embed_nd_gen):
    """
    Calcule une matrice de similarité cosinus entre deux ensembles d'embeddings.
    Supporte :
    - (N, D)
    - (D,)
    - (B, N, D) -> aplati automatiquement
    """

    def normalize(x):
        # Si vecteur seul -> (D,) -> (1, D)
        if x.dim() == 1:
            return x.unsqueeze(0)

        # Si batch de noeuds -> (B, N, D) -> flatten (B*N, D)
        if x.dim() == 3:
            return x.reshape(-1, x.size(-1))

        return x  # déjà (N, D)

    embed_st_gen = normalize(embed_st_gen)
    embed_nd_gen = normalize(embed_nd_gen)

    list_rows = []

    for emb_a in embed_st_gen:
        row = []

        for emb_b in embed_nd_gen:
            # cosine similarity entre deux vecteurs 1D
            sim = F.cosine_similarity(emb_a, emb_b, dim=0)
            row.append(sim)

        list_rows.append(torch.stack(row))

    return torch.stack(list_rows)

def norm_cos(t_cos):
    return (t_cos+1)/2


def structural_score_matrix(edge_index, alpha=1.0):
    """
    edge_index: Tensor (2, E)
    return: Tensor (N, N)
    """

    src = edge_index[0]
    dst = edge_index[1]

    n_nodes = int(torch.max(torch.cat([src, dst]))) + 1

    # adjacency list
    graph = defaultdict(list)
    for s, d in zip(src.tolist(), dst.tolist()):
        graph[s].append(d)

    # distance matrix
    dist = torch.full((n_nodes, n_nodes), float("inf"))
    dist.fill_diagonal_(0)

    # BFS from each node
    for start in range(n_nodes):
        queue = deque([(start, 0)])
        visited = set()

        while queue:
            node, d = queue.popleft()

            if node in visited:
                continue
            visited.add(node)

            dist[start, node] = min(dist[start, node], d)

            for neigh in graph[node]:
                if neigh not in visited:
                    queue.append((neigh, d + 1))

    # convert distance to score
    scores = torch.exp(-alpha * dist)

    # unreachable nodes → 0
    scores[dist == float("inf")] = 0.0

    return scores

class TrainModel():
    def __init__(self, show_w_diff=False, node_lr=0.01, func_lr=0.01, struc_lr=0.01, dec_lr=0.01, show_in_loss=False):
        self.model = DominateModel().train()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=0.001, momentum=0.9)
        self.node_lr = node_lr
        self.func_lr = func_lr
        self.struc_lr = struc_lr
        self.show_w_diff = show_w_diff
        self.dec_lr = dec_lr
        self.show_in_loss = show_in_loss

    
    def train_step(self, texts, serialized_attributes, serialized_tag, j_dom_file_path):
        outputs_fist_generation = self.model(texts, serialized_attributes, serialized_tag, j_dom_file_path, hidden_states=True)

        self.optimizer.zero_grad()

        # training node build
        outputs_second_generation = self.model(
            outputs_fist_generation["decoder"]["texts"], 
            serialized_attributes, 
            serialized_tag, 
            j_dom_file_path,
            hidden_states=True,
            get_gnc_inputs=True
        )
        # node objective : get node similarity rebuild score
        scoresNodes = F.cosine_similarity(outputs_fist_generation["gnc"] , outputs_second_generation["gnc"], dim=1)
        node_rebuild_score = 1-norm_cos(scoresNodes.mean())

        # functional objective : get node functional similarity rebuild score by attributes
        embed_st_gen = prepared_encoder_outputs(outputs_fist_generation["encoder"][1])
        embed_nd_gen = prepared_encoder_outputs(outputs_second_generation["encoder"][1])

        attrs_matrice_scores = compute_embedding_similarity_matrix(embed_st_gen, embed_nd_gen)
        norm_attrs_matrice_scores = norm_cos(attrs_matrice_scores)

        json_dom = get_json_data(j_dom_file_path)
        brut_attrs_matrice_scores = atributes_similarity_scores(json_dom)

        diff_attrs_scores = torch.abs(norm_attrs_matrice_scores - brut_attrs_matrice_scores)
        func_rebuild_score = diff_attrs_scores.mean()

        # structural objective :
        struc_brut_score = structural_score_matrix(outputs_second_generation["gnc_inputs"]["edge_index"], alpha=1.0)
        embed_struc_score = compute_embedding_similarity_matrix(outputs_fist_generation["gnc"], outputs_second_generation["gnc"])
        diff_struct_scores = torch.abs(struc_brut_score - embed_struc_score)
        struct_rebuild_score = diff_struct_scores.mean()

        # decoder objective : 
        #print(f"decoder hidden state : {outputs_fist_generation['decoder']['hidden_states'].shape}")
        embed_decoder_score = F.cosine_similarity(
            outputs_fist_generation["decoder"]["hidden_states"], 
            outputs_second_generation["decoder"]["hidden_states"], dim=2)
        decoder_rebuild_score = 1-norm_cos(embed_decoder_score.mean())

        # compute loss
        loss = (self.node_lr*node_rebuild_score + 
                self.func_lr*func_rebuild_score + 
                self.struc_lr*struct_rebuild_score + 
                self.dec_lr*decoder_rebuild_score
        )
        if self.show_in_loss:
            print("node_loss :", self.node_lr * node_rebuild_score)
            print("func_loss :", self.func_lr * func_rebuild_score)
            print("struct_loss :", self.struc_lr * struct_rebuild_score)
            print("decoder_loss :", self.dec_lr * decoder_rebuild_score)

        loss.backward()

        if self.show_w_diff:
            before = {}
            for name, param in self.model.named_parameters():
                before[name] = param.clone().detach()

        # Adjust learning weights
        self.optimizer.step()
        
        if self.show_w_diff:
            for name, param in self.model.named_parameters():
                diff = (param - before[name]).abs().mean()
                print(name, diff.item())


        return loss
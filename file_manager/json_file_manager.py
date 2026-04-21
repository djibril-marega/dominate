import json
from difflib import SequenceMatcher
import torch

def find_key(data, target_key):
    results = []
 
    if isinstance(data, dict):
        index = data.get("index")  # get index if exist

        for key, value in data.items():
            if key == target_key:
                results.append((index, value))

            results.extend(find_key(value, target_key))

    elif isinstance(data, list):
        for item in data:
            results.extend(find_key(item, target_key))

    return results

def get_json_data(json_path_file):
    with open(json_path_file) as f:
        json_dom = json.load(f)
    f.closed

    return json_dom


def string_similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def value_similarity(v1, v2):
    if isinstance(v1, list) and isinstance(v2, list):
        if not v1 or not v2:
            return 0

        scores = []
        for x in v1:
            best = max(string_similarity(str(x), str(y)) for y in v2)
            scores.append(best)

        return sum(scores) / len(scores)

    return string_similarity(str(v1), str(v2))


def attribute_similarity(attr1, attr2):
    all_keys = set(attr1.keys()) | set(attr2.keys())

    if not all_keys:
        return 1.0

    scores = []

    for key in all_keys:
        if key in attr1 and key in attr2:
            sim = value_similarity(attr1[key], attr2[key])
        else:
            sim = 0
        scores.append(sim)

    return sum(scores) / len(scores)


def rank_by_similarity(data, reference_index):
    ref_attr = None

    for idx, attrs in data:
        if idx == reference_index:
            ref_attr = attrs
            break

    if ref_attr is None:
        raise ValueError("Index de référence introuvable")

    results = []

    for idx, attrs in data:
        score = attribute_similarity(ref_attr, attrs)
        results.append((idx, round(score, 3)))

    results.sort(key=lambda x: x[0])

    return results

def atributes_similarity_scores(json_dom):
    attributes = find_key(json_dom, 'attributes')
    attributes_sorted_with_score = rank_by_similarity(attributes, 1)
    index_list = [t[0] for t in attributes_sorted_with_score]
    list_t_scores_nodes = []
    for index in index_list:
        outputs = rank_by_similarity(attributes, index)
        list_t_scores_nodes.append(torch.tensor([t[1] for t in outputs]).unsqueeze(1))
    
    return torch.cat(list_t_scores_nodes, dim=1)
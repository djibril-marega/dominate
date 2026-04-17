from dom_tree_builder.test_get_dom import get_dom_from_website
from dom_tree_builder.tree_structure_dom import structure_dom
from file_manager.json_file_manager import find_key, get_json_data
from dom_tree_interpretation.bert import serialize_attributs, serialize_tags, serialize_tuple
from dom_tree_interpretation.dominate_model import DominateModel

dom_path_file = "saved_file/dom_website.txt"
j_dom_file_path = "saved_file/dataset.json"
bert_embedding_file_path = "saved_file/textual_embeddings.pt"
website_link = "https://playwright.dev"
gnn_outputs_file_path = "saved_file/gnc_outputs.pt"
decoder_outputs_file_path = "saved_file/decoder_t5_outputs.pt"
"""
get_dom_from_website(website_link, dom_path_file)
structure_dom(dom_path_file, j_dom_file_path)
"""

dominate_model = DominateModel()

json_dom = get_json_data(j_dom_file_path)

texts = find_key(json_dom, "text")
attributes = find_key(json_dom, "attributes")
tags = find_key(json_dom, "tag")

serialized_attributes = [serialize_tuple(x, serialize_attributs) for x in attributes]
serialized_tag = serialize_tags(tags)

outputs = dominate_model(texts, serialized_attributes, serialized_tag, j_dom_file_path)







from dom_tree_interpretation.bert import generate_textual_embeddings
from dom_tree_interpretation.gnn import gnn_pipeline
from dom_tree_builder.test_get_dom import get_dom_from_website
from dom_tree_builder.tree_structure_dom import structure_dom


dom_path_file = "saved_file/dom_website.txt"
j_dom_file_path = "saved_file/dataset.json"
bert_embedding_file_path = "saved_file/textual_embeddings.pt"
website_link = "https://playwright.dev"
gnn_outputs_file_path = "saved_file/gnc_outputs.pt"

get_dom_from_website(website_link, dom_path_file)
structure_dom(dom_path_file, j_dom_file_path)
generate_textual_embeddings(bert_embedding_file_path, j_dom_file_path)
gnn_pipeline(j_dom_file_path, gnn_outputs_file_path, bert_embedding_file_path)





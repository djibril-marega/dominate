from dom_tree_interpretation.training_model import TrainModel
from file_manager.json_file_manager import find_key, get_json_data
from dom_tree_interpretation.bert import serialize_attributs, serialize_tags, serialize_tuple


j_dom_file_path = "saved_file/dataset.json"
json_dom = get_json_data(j_dom_file_path)

texts = find_key(json_dom, "text")
attributes = find_key(json_dom, "attributes")
tags = find_key(json_dom, "tag")

serialized_attributes = [serialize_tuple(x, serialize_attributs) for x in attributes]
serialized_tag = serialize_tags(tags)


model_training = TrainModel()
outputs = model_training.train_step(texts, serialized_attributes, serialized_tag, j_dom_file_path)
print(outputs)
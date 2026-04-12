from bs4 import BeautifulSoup
import json
from lxml import etree

def get_direct_text(element):
    return "".join(
        child.strip()
        for child in element.children
        if child.name is None
    )

def element_to_dict(element, counter={"i": 0}):
    if element.name is None:
        return None

    counter["i"] += 1
    current_index = counter["i"]

    text = get_direct_text(element)

    return {
        "index": current_index,
        "tag": element.name,
        "attributes": dict(element.attrs),
        "text": text if text else None,
        "children": [
            child_dict
            for child in element.children
            if child.name is not None
            and (child_dict := element_to_dict(child, counter)) is not None
        ]
    }

def structure_dom(dom_file_path, j_dom_file_path):
    with open(dom_file_path) as f:
        read_DOM = f.read()
    f.closed

    soup = BeautifulSoup(read_DOM, "lxml")

    dataset = element_to_dict(soup)

    with open(j_dom_file_path, "w", encoding="utf-8") as f:
        json.dump(dataset, f, ensure_ascii=False, indent=4)

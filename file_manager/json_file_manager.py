import json

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
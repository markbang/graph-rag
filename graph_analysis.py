import os

import json
from lightrag.utils import xml_to_json

WORKING_DIR = "./dickens"


def convert_xml_to_json(xml_path, output_path):
    """Converts XML file to JSON and saves the output."""
    if not os.path.exists(xml_path):
        print(f"Error: File not found - {xml_path}")
        return None

    json_data = xml_to_json(xml_path)
    if json_data:
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(json_data, f, ensure_ascii=False, indent=2)
        print(f"JSON file created: {output_path}")
        return json_data
    else:
        print("Failed to create JSON data")
        return None


def process_in_batches(tx, query, data, batch_size):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        tx.run(query, {"nodes": batch} if "nodes" in query else {"edges": batch})


def get_info():
    # Paths
    xml_file = os.path.join(WORKING_DIR, "graph_chunk_entity_relation.graphml")
    json_file = os.path.join(WORKING_DIR, "graph_data.json")

    # Convert XML to JSON
    json_data = convert_xml_to_json(xml_file, json_file)
    if json_data is None:
        print("Exiting due to conversion failure.")
    return {
        "nodes": len(json_data.get("nodes", [])),
        "edges": len(json_data.get("edges", [])),
    }


if __name__ == "__main__":
    print(get_info())

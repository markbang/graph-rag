import pipmaster as pm

if not pm.is_installed("pyvis"):
    pm.install("pyvis")
if not pm.is_installed("networkx"):
    pm.install("networkx")

import networkx as nx
from pyvis.network import Network
import random
import os

# Load the GraphML file
# G = nx.read_graphml("./dickens/graph_chunk_entity_relation.graphml")

# Create a Pyvis network
# net = Network(height="100vh", notebook=True)

# Convert NetworkX graph to Pyvis network
# net.from_nx(G)


# Add colors and title to nodes
# for node in net.nodes:
#     node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
#     if "description" in node:
#         node["title"] = node["description"]

# Add title to edges
# for edge in net.edges:
#     if "description" in edge:
#         edge["title"] = edge["description"]

# Save and display the network
# net.show("knowledge_graph.html")


def generate_visualization(graph_path, output_path):
    """Generates a visualization for a given graph file."""
    G = nx.read_graphml(graph_path)
    net = Network(
        height="100vh", notebook=True, cdn_resources="remote"
    )  # Use remote CDN resources
    net.from_nx(G)

    for node in net.nodes:
        node["color"] = "#{:06x}".format(random.randint(0, 0xFFFFFF))
        if "description" in node:
            node["title"] = node["description"]

    for edge in net.edges:
        if "description" in edge:
            edge["title"] = edge["description"]

    net.show(output_path)
    print(f"Visualization saved to {output_path}")


# Process the main dickens graph
# generate_visualization("./dickens/graph_chunk_entity_relation.graphml", "./dickens/knowledge_graph.html")


# Process graphs in subdirectories of 'tobe'
tobe_dir = "./tobe"
for subdir in os.listdir(tobe_dir):
    subdir_path = os.path.join(tobe_dir, subdir)
    if os.path.isdir(subdir_path):
        graph_file = os.path.join(subdir_path, "graph_chunk_entity_relation.graphml")
        if os.path.exists(graph_file):
            output_html = os.path.join(subdir_path, "knowledge_graph.html")
            generate_visualization(graph_file, output_html)
        else:
            print(f"GraphML file not found in {subdir_path}")

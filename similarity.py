import pipmaster

if not pipmaster.is_installed("networkx"):
    pm_result = pipmaster.install("networkx")
    print(f"networkx installation result: {pm_result}")
if not pipmaster.is_installed("scikit-learn"):
    pm_result = pipmaster.install("scikit-learn")
    print(f"scikit-learn installation result: {pm_result}")

import networkx as nx
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import os


def get_edge_descriptions(graph):
    """Extracts edge descriptions from a NetworkX graph.
    Assumes edges are uniquely identified by (source_id, target_id, relation_label).
    The description is stored in the 'description' attribute of the edge.
    """
    edge_data = {}
    for u, v, data in graph.edges(data=True):
        # Node IDs are usually strings in GraphML if read directly.
        # Ensure consistent ordering for undirected graphs if necessary,
        # but relations are often directed. Let's assume directed or order matters.
        relation_label = data.get(
            "label", data.get("relation", "DEFAULT_RELATION")
        )  # Try 'label', then 'relation', then default
        description = data.get("description")

        # Ensure u and v are strings for consistent key creation
        u_str = str(u)
        v_str = str(v)

        if description:
            # To handle potential variations in GraphML (e.g. 'relation_type' vs 'label')
            # We prioritize 'label', then 'relation', then a default.
            # The key for an edge will be a tuple (source_node_id, target_node_id, type_of_relation)
            edge_key = (u_str, v_str, str(relation_label))
            edge_data[edge_key] = str(
                description
            )  # Ensure description is also a string
    return edge_data


def compare_graph_relations(
    graphml_file1, graphml_file2, file1_name="Graph 1", file2_name="Graph 2"
):
    """
    Compares relationship descriptions between two graphs from GraphML files.
    """
    if not os.path.exists(graphml_file1):
        print(f"Error: File not found - {graphml_file1}")
        return
    if not os.path.exists(graphml_file2):
        print(f"Error: File not found - {graphml_file2}")
        return

    try:
        # node_type=str ensures node IDs are read as strings, aiding consistency
        g1 = nx.read_graphml(graphml_file1, node_type=str)
        g2 = nx.read_graphml(graphml_file2, node_type=str)
    except Exception as e:
        print(f"Error reading GraphML files: {e}")
        return

    edges1 = get_edge_descriptions(g1)
    edges2 = get_edge_descriptions(g2)

    common_relations_keys = set(edges1.keys()).intersection(set(edges2.keys()))

    if not common_relations_keys:
        print("No common relations with descriptions found between the two graphs.")
        return

    vectorizer = TfidfVectorizer()

    print(
        f"Comparing descriptions for {len(common_relations_keys)} common relations found between '{file1_name}' and '{file2_name}':\n"
    )

    results = []
    total_similarity = 0.0
    processed_relations_count = 0

    for rel_key in common_relations_keys:
        desc1 = edges1[rel_key]
        desc2 = edges2[rel_key]

        # Ensure descriptions are not empty or just whitespace
        if not desc1.strip() or not desc2.strip():
            similarity = 0.0
            # print(f"Skipping relation {rel_key} due to empty description(s).")
        else:
            try:
                # Fit and transform the two descriptions
                # This creates a matrix where rows are documents and columns are features (terms)
                tfidf_matrix = vectorizer.fit_transform([desc1, desc2])
                similarity_score = cosine_similarity(
                    tfidf_matrix[0:1], tfidf_matrix[1:2]
                )

                # The result is a 2D array (e.g., [[value]]), so we extract the single value.
                similarity = similarity_score[0][0]
            except ValueError:  # Can happen if vocabulary is empty
                similarity = 0.0
                # print(f"Could not compute similarity for {rel_key} due to TF-IDF error. Assigning 0.")

        source_node, target_node, relation_label = rel_key

        result_entry = {
            "relation": f"({source_node}) -[{relation_label}]-> ({target_node})",
            "description1": desc1,
            "description2": desc2,
            "similarity": similarity,
        }
        results.append(result_entry)

        print(f"Relation: ({source_node}) -[{relation_label}]-> ({target_node})")
        # print(f"  Description ({file1_name}): {desc1[:100] + '...' if len(desc1) > 100 else desc1}")
        # print(f"  Description ({file2_name}): {desc2[:100] + '...' if len(desc2) > 100 else desc2}")
        print(f"  Similarity Score: {similarity:.4f}\n")

        # Only include relations with similarity > 0 in the average calculation
        if similarity > 0:
            total_similarity += similarity
            processed_relations_count += 1

    # You can add code here to save results to a file if needed
    # For example, to a CSV or JSON file.
    if processed_relations_count > 0:
        average_similarity = total_similarity / processed_relations_count
        print(
            f"Average Similarity Score for {processed_relations_count} common relations: {average_similarity:.4f}"
        )
    else:
        print("No relations were processed to calculate an average similarity.")


if __name__ == "__main__":
    base_path = "./tobe"

    # Paths to the GraphML files
    graphml_path_expert = os.path.join(
        base_path, "专家分析师版", "graph_chunk_entity_relation.graphml"
    )
    graphml_path_benchmark = os.path.join(
        base_path, "基准版", "graph_chunk_entity_relation.graphml"
    )

    print("Starting comparison between:")  # Removed f-string
    print(f"1. 专家分析师版: {graphml_path_expert}")
    print(f"2. 基准版: {graphml_path_benchmark}")
    print("-" * 30)

    compare_graph_relations(
        graphml_path_expert,
        graphml_path_benchmark,
        file1_name="专家分析师版",
        file2_name="基准版",
    )
    print("-" * 30)
    print("Comparison finished.")

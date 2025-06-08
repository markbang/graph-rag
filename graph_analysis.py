import os
import json
from lightrag.utils import xml_to_json


# --- Evaluation Configuration ---
DEFAULT_PROTAGONIST_ID = "萧炎"  # Default protagonist if not in config

# --- Unsupervised Evaluation Weights ---
# Top-Level Weights
W_FOCUS_UNSUPERVISED = 1.0 / 3.0
W_STRUCTURE_UNSUPERVISED = 1.0 / 3.0
W_RICHNESS_UNSUPERVISED = 1.0 / 3.0

# Structure Score Component Weights
W_CONN_UNSUPERVISED_WEIGHT = 0.5
W_SCOPE_UNSUPERVISED_WEIGHT = 0.5

# Richness Score Component Weights
W_DETAIL_UNSUPERVISED_WEIGHT = 1.0 / 3.0
W_DIVERSITY_UNSUPERVISED_WEIGHT = 1.0 / 3.0
W_STRENGTH_UNSUPERVISED_WEIGHT = 1.0 / 3.0


def convert_xml_to_json(xml_path, output_path):
    """Converts XML file to JSON and saves the output."""
    if not os.path.exists(xml_path):
        print(f"Error: XML File not found - {xml_path}")
        return None
    try:
        json_data = xml_to_json(xml_path)  # This is the call to the imported function
        if json_data:
            # Ensure 'nodes' and 'edges' keys exist, defaulting to empty lists
            if "nodes" not in json_data:
                json_data["nodes"] = []
            if "edges" not in json_data:
                json_data["edges"] = []

            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(json_data, f, ensure_ascii=False, indent=2)
            # print(f"JSON file created: {output_path}") # Reduced verbosity
            return json_data
        else:
            print(f"Failed to convert XML to JSON (empty result): {xml_path}")
            return None
    except Exception as e:
        print(f"Error during XML to JSON conversion for {xml_path}: {e}")
        return None


def process_in_batches(tx, query, data, batch_size):
    """Process data in batches and execute the given query."""
    for i in range(0, len(data), batch_size):
        batch = data[i : i + batch_size]
        tx.run(query, {"nodes": batch} if "nodes" in query else {"edges": batch})


# --- Unsupervised Evaluation Helper Functions ---


def get_graph_data_and_config(experiment_dir_path):
    """Loads graph data from JSON and eval config for an experiment."""
    graph_json_path = os.path.join(experiment_dir_path, "graph_data.json")
    graphml_path = os.path.join(
        experiment_dir_path, "graph_chunk_entity_relation.graphml"
    )
    eval_config_path = os.path.join(experiment_dir_path, "eval_config.json")

    graph_data = None
    # Prefer JSON, but convert from GraphML if JSON is missing or older
    if os.path.exists(graph_json_path):
        try:
            # Check if GraphML is newer than JSON; if so, reconvert
            if os.path.exists(graphml_path) and os.path.getmtime(
                graphml_path
            ) > os.path.getmtime(graph_json_path):
                print(f"Re-converting newer GraphML for {experiment_dir_path}")
                graph_data = convert_xml_to_json(graphml_path, graph_json_path)
            else:
                with open(graph_json_path, "r", encoding="utf-8") as f:
                    graph_data = json.load(f)
                # Basic validation
                if "nodes" not in graph_data or "edges" not in graph_data:
                    print(
                        f"Warning: JSON graph data in {graph_json_path} is missing 'nodes' or 'edges'. Attempting GraphML conversion."
                    )
                    if os.path.exists(graphml_path):
                        graph_data = convert_xml_to_json(graphml_path, graph_json_path)
                    else:
                        graph_data = None  # Cannot recover
        except json.JSONDecodeError:
            print(
                f"Error decoding JSON from {graph_json_path}. Attempting GraphML conversion."
            )
            if os.path.exists(graphml_path):
                graph_data = convert_xml_to_json(graphml_path, graph_json_path)
            else:
                graph_data = None  # Cannot recover
        except Exception as e:
            print(
                f"Error loading graph data from {graph_json_path}: {e}. Attempting GraphML conversion."
            )
            if os.path.exists(graphml_path):
                graph_data = convert_xml_to_json(graphml_path, graph_json_path)
            else:
                graph_data = None

    elif os.path.exists(graphml_path):
        print(
            f"JSON graph data not found for {experiment_dir_path}, converting from GraphML."
        )
        graph_data = convert_xml_to_json(graphml_path, graph_json_path)
    else:
        return None, DEFAULT_PROTAGONIST_ID, {"error": "GraphML file not found"}

    if not graph_data:  # If conversion or loading failed
        return (
            None,
            DEFAULT_PROTAGONIST_ID,
            {"error": "Failed to load or convert graph data"},
        )

    # Ensure nodes and edges are lists, even if empty
    if "nodes" not in graph_data:
        graph_data["nodes"] = []
    if "edges" not in graph_data:
        graph_data["edges"] = []

    protagonist_id = DEFAULT_PROTAGONIST_ID
    if os.path.exists(eval_config_path):
        try:
            with open(eval_config_path, "r", encoding="utf-8") as f:
                eval_config = json.load(f)
            protagonist_id = eval_config.get("protagonist_id", DEFAULT_PROTAGONIST_ID)
        except Exception as e:
            print(
                f"Error loading eval_config.json from {eval_config_path}: {e}. Using default protagonist."
            )

    return graph_data, protagonist_id, {}  # Empty dict for no error


def calculate_protagonist_centrality_unsupervised(graph_data, protagonist_id):
    """Calculates Protagonist Centrality for unsupervised evaluation."""
    if not graph_data or not graph_data.get("nodes"):
        return 0.0

    nodes = graph_data["nodes"]
    edges = graph_data.get("edges", [])
    num_total_entities = len(nodes)

    if not protagonist_id or not any(
        node.get("id") == protagonist_id for node in nodes
    ):
        # print(f"Warning: Protagonist '{protagonist_id}' not found or not specified.")
        return 0.0

    if num_total_entities <= 1:  # If only protagonist or no other nodes
        return 0.0

    connected_to_protagonist = set()
    for edge in edges:
        source = edge.get("source")
        target = edge.get("target")
        if source == protagonist_id and target:
            connected_to_protagonist.add(target)
        elif target == protagonist_id and source:
            connected_to_protagonist.add(source)

    # Denominator is (Total entities - 1) as per formula
    denominator = num_total_entities - 1
    if (
        denominator <= 0
    ):  # Should be caught by num_total_entities <= 1, but as a safeguard
        return 0.0

    return len(connected_to_protagonist) / denominator


def calculate_connectivity_score_unsupervised(graph_data):
    """Calculates Connectivity Score for unsupervised evaluation."""
    if not graph_data or not graph_data.get("nodes"):
        return 0.0  # Or 1.0 if interpreting "no orphans" in an empty graph? Formula implies 0.

    nodes = graph_data["nodes"]
    edges = graph_data.get("edges", [])
    num_total_entities = len(nodes)

    if num_total_entities == 0:
        return 0.0  # No nodes, no connections.

    nodes_with_connections = set()
    for edge in edges:
        if edge.get("source"):
            nodes_with_connections.add(edge.get("source"))
        if edge.get("target"):
            nodes_with_connections.add(edge.get("target"))

    # Formula: 1 - (orphan_nodes / total_entities) which is num_connected_nodes / total_entities
    return len(nodes_with_connections) / num_total_entities


def calculate_raw_scope_score_unsupervised(graph_data):
    """Calculates the raw Scope Score (total number of entities)."""
    if not graph_data or not graph_data.get("nodes"):
        return 0
    return len(graph_data["nodes"])


def calculate_average_description_length_unsupervised(graph_data):
    """Calculates the average length of descriptions for entities and relationships."""
    if not graph_data:
        return 0.0

    nodes = graph_data.get("nodes", [])
    edges = graph_data.get("edges", [])

    total_desc_length = 0
    item_count = 0

    for node in nodes:
        description = node.get("description", "")
        if isinstance(description, str):
            total_desc_length += len(description)
        item_count += 1  # Count node even if no description, for average base

    for edge in edges:
        description = edge.get("description", "")
        # Fallback to 'label' for edges if 'description' is missing, as per typical graph structures
        if not description and isinstance(edge.get("label"), str):
            description = edge.get("label", "")
        if isinstance(description, str):
            total_desc_length += len(description)
        item_count += 1  # Count edge even if no description

    if item_count == 0:
        return 0.0
    return total_desc_length / item_count


def calculate_relation_diversity_unsupervised(graph_data):
    """Calculates Relationship Diversity."""
    if not graph_data or not graph_data.get("edges"):
        return 0.0

    edges = graph_data["edges"]
    num_total_relations = len(edges)

    if num_total_relations == 0:
        return 0.0

    # Assuming relationship types are in the 'label' field of edges
    relation_types = {edge.get("label") for edge in edges if edge.get("label")}

    return len(relation_types) / num_total_relations


def calculate_average_relation_strength_unsupervised(graph_data):
    """Calculates the average strength of relationships."""
    if not graph_data or not graph_data.get("edges"):
        return 0.0

    edges = graph_data["edges"]
    if not edges:
        return 0.0

    total_strength = 0
    strength_count = 0
    for edge in edges:
        # Assuming strength is in 'weight' or 'strength'. Prioritize 'strength'.
        strength = edge.get("strength", edge.get("weight"))
        if isinstance(strength, (int, float)):
            total_strength += strength
            strength_count += 1

    if strength_count == 0:
        return 0.0  # No edges with valid strength values
    return total_strength / strength_count


# --- Helper for empty results ---
def _create_empty_unsupervised_results(error_message=""):
    """Creates a dictionary with all unsupervised evaluation metrics set to 0 or default."""
    return {
        "error": error_message,
        "protagonist_centrality": 0.0,
        "focus_score": 0.0,
        "connectivity_score": 0.0,
        "raw_scope_score": 0,
        "normalized_scope_score": 0.0,
        "structure_score": 0.0,
        "raw_avg_desc_length": 0.0,
        "normalized_avg_desc_length": 0.0,
        "relationship_diversity": 0.0,
        "raw_avg_rel_strength": 0.0,
        "normalized_avg_rel_strength": 0.0,
        "richness_score": 0.0,
        "novel_graph_score": 0.0,
        "nodes_count": 0,
        "edges_count": 0,
    }


# --- Main Unsupervised Evaluation Function ---
def evaluate_graph_unsupervised(
    graph_data,
    protagonist_id,
    raw_scope_for_this_experiment,  # Pass the already calculated raw scope
    raw_avg_desc_len_for_this_experiment,  # Pass the already calculated raw avg desc len
    raw_avg_rel_strength_for_this_experiment,  # Pass the already calculated raw avg rel strength
    max_overall_raw_scope,
    max_overall_raw_avg_desc_len,
    max_overall_raw_avg_rel_strength,
):
    """
    Calculates all unsupervised evaluation scores for a single graph.
    """
    if not graph_data:
        return _create_empty_unsupervised_results("Graph data is None")

    nodes_count = len(graph_data.get("nodes", []))
    edges_count = len(graph_data.get("edges", []))

    # Focus Score
    protagonist_centrality = calculate_protagonist_centrality_unsupervised(
        graph_data, protagonist_id
    )
    focus_score = protagonist_centrality  # Focus_Score = Protagonist_Centrality

    # Structure Score
    connectivity_score = calculate_connectivity_score_unsupervised(graph_data)
    # raw_scope_score is already calculated in Pass 1 (raw_scope_for_this_experiment)

    normalized_scope_score = 0.0
    if max_overall_raw_scope > 0:
        normalized_scope_score = raw_scope_for_this_experiment / max_overall_raw_scope

    structure_score = (W_CONN_UNSUPERVISED_WEIGHT * connectivity_score) + (
        W_SCOPE_UNSUPERVISED_WEIGHT * normalized_scope_score
    )

    # Richness Score
    # raw_avg_desc_length is already calculated in Pass 1
    normalized_avg_desc_length = 0.0
    if max_overall_raw_avg_desc_len > 0:
        normalized_avg_desc_length = (
            raw_avg_desc_len_for_this_experiment / max_overall_raw_avg_desc_len
        )

    relationship_diversity = calculate_relation_diversity_unsupervised(graph_data)

    # raw_avg_rel_strength is already calculated in Pass 1
    normalized_avg_rel_strength = 0.0
    if max_overall_raw_avg_rel_strength > 0:
        normalized_avg_rel_strength = (
            raw_avg_rel_strength_for_this_experiment / max_overall_raw_avg_rel_strength
        )

    richness_score = (
        (W_DETAIL_UNSUPERVISED_WEIGHT * normalized_avg_desc_length)
        + (W_DIVERSITY_UNSUPERVISED_WEIGHT * relationship_diversity)
        + (
            W_STRENGTH_UNSUPERVISED_WEIGHT * normalized_avg_rel_strength
        )  # Corrected variable name here
    )

    # Final Novel Graph Score
    novel_graph_score = (
        (W_FOCUS_UNSUPERVISED * focus_score)
        + (W_STRUCTURE_UNSUPERVISED * structure_score)
        + (W_RICHNESS_UNSUPERVISED * richness_score)
    )

    return {
        "error": None,
        "nodes_count": nodes_count,
        "edges_count": edges_count,
        "protagonist_centrality": protagonist_centrality,
        "focus_score": focus_score,
        "connectivity_score": connectivity_score,
        "raw_scope_score": raw_scope_for_this_experiment,  # Include for reference
        "normalized_scope_score": normalized_scope_score,
        "structure_score": structure_score,
        "raw_avg_desc_length": raw_avg_desc_len_for_this_experiment,  # Include for reference
        "normalized_avg_desc_length": normalized_avg_desc_length,
        "relationship_diversity": relationship_diversity,
        "raw_avg_rel_strength": raw_avg_rel_strength_for_this_experiment,  # Include for reference
        "normalized_avg_rel_strength": normalized_avg_rel_strength,
        "richness_score": richness_score,
        "novel_graph_score": novel_graph_score,
    }


# --- Main Execution Logic ---
if __name__ == "__main__":
    tobe_dir = "./tobe"
    all_experiment_data_for_pass1 = []

    if not os.path.exists(tobe_dir) or not os.path.isdir(tobe_dir):
        print(f"Error: Directory '{tobe_dir}' not found or is not a directory.")
        exit()

    print("--- Starting Unsupervised Evaluation: Pass 1 (Data Collection) ---")
    for experiment_dir_name in sorted(
        os.listdir(tobe_dir)
    ):  # Sort for consistent order
        experiment_path = os.path.join(tobe_dir, experiment_dir_name)
        if not os.path.isdir(experiment_path):
            continue

        print(f"Processing (Pass 1): {experiment_dir_name}")
        graph_data, protagonist_id, error_info = get_graph_data_and_config(
            experiment_path
        )

        if error_info:
            print(
                f"  Skipping {experiment_dir_name} due to error: {error_info.get('error')}"
            )
            all_experiment_data_for_pass1.append(
                {
                    "dir_name": experiment_dir_name,
                    "error": error_info.get("error"),
                    "graph_data": None,  # No data to process
                    "protagonist_id": None,
                    "raw_scope": 0,
                    "raw_avg_desc_len": 0.0,
                    "raw_avg_rel_strength": 0.0,
                    "nodes_count_initial": 0,
                    "edges_count_initial": 0,
                }
            )
            continue

        if not graph_data:  # Should be caught by error_info, but as a safeguard
            print(
                f"  Skipping {experiment_dir_name}: No graph data loaded (should have been caught by error_info)."
            )
            all_experiment_data_for_pass1.append(
                {
                    "dir_name": experiment_dir_name,
                    "error": "No graph data loaded after get_graph_data_and_config (safeguard)",
                    "graph_data": None,
                    "protagonist_id": None,
                    "raw_scope": 0,
                    "raw_avg_desc_len": 0.0,
                    "raw_avg_rel_strength": 0.0,
                    "nodes_count_initial": 0,
                    "edges_count_initial": 0,
                }
            )
            continue

        raw_scope = calculate_raw_scope_score_unsupervised(graph_data)
        raw_avg_desc_len = calculate_average_description_length_unsupervised(graph_data)
        raw_avg_rel_strength = calculate_average_relation_strength_unsupervised(
            graph_data
        )
        nodes_count_initial = len(graph_data.get("nodes", []))
        edges_count_initial = len(graph_data.get("edges", []))

        all_experiment_data_for_pass1.append(
            {
                "dir_name": experiment_dir_name,
                "graph_data": graph_data,
                "protagonist_id": protagonist_id,
                "raw_scope": raw_scope,
                "raw_avg_desc_len": raw_avg_desc_len,
                "raw_avg_rel_strength": raw_avg_rel_strength,
                "nodes_count_initial": nodes_count_initial,
                "edges_count_initial": edges_count_initial,
                "error": None,  # Explicitly set no error for successful pass 1
            }
        )
        print(
            f"  Collected: Nodes={nodes_count_initial}, Edges={edges_count_initial}, Scope={raw_scope}, AvgDescLen={raw_avg_desc_len:.2f}, AvgRelStr={raw_avg_rel_strength:.2f}"
        )

    # Calculate overall maximums for normalization after Pass 1
    max_overall_raw_scope = 0
    max_overall_raw_avg_desc_len = 0.0
    max_overall_raw_avg_rel_strength = 0.0

    # Filter out entries with errors before calculating max values
    valid_pass1_data = [
        d
        for d in all_experiment_data_for_pass1
        if d.get("error") is None and d.get("graph_data") is not None
    ]

    if valid_pass1_data:
        valid_scopes = [
            d["raw_scope"] for d in valid_pass1_data if d["raw_scope"] is not None
        ]
        if valid_scopes:
            max_overall_raw_scope = max(valid_scopes) if valid_scopes else 0

        valid_desc_lens = [
            d["raw_avg_desc_len"]
            for d in valid_pass1_data
            if d["raw_avg_desc_len"] is not None
        ]
        if valid_desc_lens:
            max_overall_raw_avg_desc_len = (
                max(valid_desc_lens) if valid_desc_lens else 0.0
            )

        valid_rel_strengths = [
            d["raw_avg_rel_strength"]
            for d in valid_pass1_data
            if d["raw_avg_rel_strength"] is not None
        ]
        if valid_rel_strengths:
            max_overall_raw_avg_rel_strength = (
                max(valid_rel_strengths) if valid_rel_strengths else 0.0
            )

    print("\\n--- Normalization Factors ---")
    print(f"Max Overall Raw Scope: {max_overall_raw_scope}")
    print(f"Max Overall Raw Avg Desc Length: {max_overall_raw_avg_desc_len:.2f}")
    print(f"Max Overall Raw Avg Rel Strength: {max_overall_raw_avg_rel_strength:.2f}")

    print("\\n--- Starting Unsupervised Evaluation: Pass 2 (Final Scoring) ---")
    final_results_unsupervised = {}

    for experiment_data in all_experiment_data_for_pass1:
        dir_name = experiment_data["dir_name"]
        print(f"Processing (Pass 2): {dir_name}")

        if experiment_data.get("error") or not experiment_data.get("graph_data"):
            print(
                f"  Skipping {dir_name} due to error in Pass 1: {experiment_data.get('error', 'No graph data')}"
            )
            final_results_unsupervised[dir_name] = _create_empty_unsupervised_results(
                experiment_data.get("error", "No graph data from Pass 1")
            )
            # Populate with initial counts if available, even on error
            final_results_unsupervised[dir_name]["nodes_count"] = experiment_data.get(
                "nodes_count_initial", 0
            )
            final_results_unsupervised[dir_name]["edges_count"] = experiment_data.get(
                "edges_count_initial", 0
            )
            final_results_unsupervised[dir_name]["raw_scope_score"] = (
                experiment_data.get("raw_scope", 0)
            )
            final_results_unsupervised[dir_name]["raw_avg_desc_length"] = (
                experiment_data.get("raw_avg_desc_len", 0.0)
            )
            final_results_unsupervised[dir_name]["raw_avg_rel_strength"] = (
                experiment_data.get("raw_avg_rel_strength", 0.0)
            )

            continue

        eval_scores = evaluate_graph_unsupervised(
            graph_data=experiment_data["graph_data"],
            protagonist_id=experiment_data["protagonist_id"],
            raw_scope_for_this_experiment=experiment_data["raw_scope"],
            raw_avg_desc_len_for_this_experiment=experiment_data["raw_avg_desc_len"],
            raw_avg_rel_strength_for_this_experiment=experiment_data[
                "raw_avg_rel_strength"
            ],
            max_overall_raw_scope=max_overall_raw_scope,
            max_overall_raw_avg_desc_len=max_overall_raw_avg_desc_len,
            max_overall_raw_avg_rel_strength=max_overall_raw_avg_rel_strength,
        )
        final_results_unsupervised[dir_name] = eval_scores
        print(f"  Calculated Scores for {dir_name}:")
        for score_name, score_value in eval_scores.items():
            if score_name != "error":  # Don't print error if None
                if isinstance(score_value, float):
                    print(f"    {score_name}: {score_value:.4f}")
                else:
                    print(f"    {score_name}: {score_value}")

    print("\\n--- Final Unsupervised Evaluation Summary ---")
    # Sort results by Novel_Graph_Score in descending order for better comparison
    sorted_results = sorted(
        final_results_unsupervised.items(),
        key=lambda item: item[1].get("novel_graph_score", 0.0)
        if item[1] and item[1].get("error") is None
        else -1,  # place errors at the end
        reverse=True,
    )

    for dir_name, scores in sorted_results:
        print(f"\\nExperiment: {dir_name}")
        if scores.get("error"):
            print(f"  Error: {scores['error']}")
            print(
                f"  Initial Nodes: {scores.get('nodes_count', 'N/A')}, Initial Edges: {scores.get('edges_count', 'N/A')}"
            )
            print(f"  Raw Scope: {scores.get('raw_scope_score', 'N/A')}")
            print(f"  Raw Avg Desc Len: {scores.get('raw_avg_desc_length', 'N/A')}")
            print(
                f"  Raw Avg Rel Strength: {scores.get('raw_avg_rel_strength', 'N/A')}"
            )

        else:
            print(f"  Novel Graph Score: {scores['novel_graph_score']:.4f}")
            print(f"  Nodes: {scores['nodes_count']}, Edges: {scores['edges_count']}")
            print(
                f"  Focus Score (Protagonist Centrality): {scores['focus_score']:.4f} (Centrality: {scores['protagonist_centrality']:.4f})"
            )
            print(
                f"  Structure Score: {scores['structure_score']:.4f} (Connectivity: {scores['connectivity_score']:.4f}, Norm. Scope: {scores['normalized_scope_score']:.4f} [Raw: {scores['raw_scope_score']}])"
            )
            print(
                f"  Richness Score: {scores['richness_score']:.4f} (Norm. Desc Len: {scores['normalized_avg_desc_length']:.4f} [Raw: {scores['raw_avg_desc_length']:.2f}], Diversity: {scores['relationship_diversity']:.4f}, Norm. Strength: {scores['normalized_avg_rel_strength']:.4f} [Raw: {scores['raw_avg_rel_strength']:.2f}])"
            )

    print("\\nScript execution finished.")

# convert_to_graph_embeddings.py
import os
import json
import pickle
import numpy as np
import torch
from transformers import RobertaTokenizer, RobertaModel
from tqdm import tqdm
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Cấu hình
DATASET_DIR = "dataset"
OUTPUT_DIR = "processed"
OUTPUT_PKL = os.path.join(OUTPUT_DIR, "graph_data.pkl")

# Label mapping
LABELS = {
    "sqli": 0,
    "command_injection": 1,
    "path_traversal": 2,
    "buffer_overflow": 3,
    "safe": 4
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {device}")

# Load CodeBERT/GraphCodeBERT
print("[+] Loading GraphCodeBERT model...")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")
model = RobertaModel.from_pretrained("microsoft/graphcodebert-base").to(device)
model.eval()


@torch.no_grad()
def get_code_embedding(code_text, max_length=128):
    """
    Lấy embedding từ GraphCodeBERT
    """
    if not code_text or len(code_text.strip()) == 0:
        return np.zeros(768, dtype=np.float32)

    try:
        # Tokenize
        inputs = tokenizer(
            code_text,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding='max_length'
        ).to(device)

        # Get embeddings
        outputs = model(**inputs)

        # Mean pooling
        embedding = outputs.last_hidden_state.mean(dim=1).squeeze()

        return embedding.cpu().numpy().astype(np.float32)

    except Exception as e:
        print(f"[!] Error encoding: {str(e)[:100]}")
        return np.zeros(768, dtype=np.float32)


def extract_vulnerability_features(func_data, vuln_type):
    """
    Trích xuất các features đặc trưng cho từng loại vulnerability
    """
    features = {
        # Common features
        'has_user_input': 0,
        'has_dangerous_func': 0,
        'has_sanitization': 0,

        # SQL Injection specific
        'has_sql_concat': 0,
        'has_sql_exec': 0,
        'has_prepared_stmt': 0,

        # Command Injection specific
        'has_system_call': 0,
        'has_shell_exec': 0,

        # Path Traversal specific
        'has_file_operation': 0,
        'has_path_concat': 0,

        # Buffer Overflow specific
        'has_unsafe_copy': 0,
        'has_buffer_operation': 0,
    }

    # Keywords for detection
    sql_keywords = ['SELECT', 'INSERT', 'UPDATE', 'DELETE', 'WHERE', 'FROM']
    dangerous_funcs = ['system', 'exec', 'eval', 'Runtime.getRuntime', 'ProcessBuilder']
    file_ops = ['readFile', 'writeFile', 'open', 'FileInputStream', 'FileOutputStream']
    unsafe_funcs = ['strcpy', 'strcat', 'sprintf', 'gets', 'memcpy']
    sanitization = ['sanitize', 'escape', 'validate', 'filter', 'prepared']

    for gtype in ["AST", "CFG", "PDG"]:
        for node in func_data.get(gtype, []):
            props = node.get("properties", {})
            code = props.get("CODE", "").upper()
            name = props.get("NAME", "")
            method_name = props.get("METHOD_FULL_NAME", "")

            # User input detection
            if any(keyword in name.lower() for keyword in ['input', 'request', 'param', 'user']):
                features['has_user_input'] = 1

            # Dangerous function detection
            for func in dangerous_funcs:
                if func in code or func in name:
                    features['has_dangerous_func'] = 1
                    if func in ['system', 'exec', 'Runtime']:
                        features['has_system_call'] = 1
                    if 'exec' in func.lower():
                        features['has_shell_exec'] = 1

            # Sanitization detection
            for san in sanitization:
                if san in code.lower() or san in name.lower():
                    features['has_sanitization'] = 1

            # SQL Injection features
            if vuln_type == "sqli":
                if '<operator>.addition' in method_name:
                    for keyword in sql_keywords:
                        if keyword in code:
                            features['has_sql_concat'] = 1
                            break

                if 'executeQuery' in name or 'executeUpdate' in name:
                    features['has_sql_exec'] = 1

                if 'prepareStatement' in code.lower() or 'PreparedStatement' in code:
                    features['has_prepared_stmt'] = 1

            # File operation features
            for fop in file_ops:
                if fop in code or fop in name:
                    features['has_file_operation'] = 1
                    if '<operator>.addition' in method_name or '+' in code:
                        features['has_path_concat'] = 1

            # Buffer overflow features
            for ufunc in unsafe_funcs:
                if ufunc in code or ufunc in name:
                    features['has_unsafe_copy'] = 1
                    features['has_buffer_operation'] = 1

    return np.array(list(features.values()), dtype=np.float32)


def build_graph_from_json(filepath, label_id, label_name):
    """
    Xây dựng graph structure từ JSON file
    """
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except Exception as e:
        print(f"[!] Error reading {filepath}: {e}")
        return None

    # Containers
    all_nodes = {}  # node_id -> node_info
    all_edges = []  # (src_id, dst_id, edge_type)

    # Process all functions
    for func_idx, func in enumerate(data.get("functions", [])):

        # Collect nodes from AST, CFG, PDG
        for gtype in ["AST", "CFG", "PDG"]:
            for node in func.get(gtype, []):
                nid = node["id"]

                if nid not in all_nodes:
                    props = node.get("properties", {})

                    # Lấy code hoặc label
                    code = props.get("CODE", "")
                    label = node.get("label", "")
                    node_type = props.get("TYPE_FULL_NAME", label)

                    # Tạo text representation cho node
                    if code:
                        node_text = f"{label}: {code}"
                    else:
                        node_text = f"{label}"

                    all_nodes[nid] = {
                        'text': node_text,
                        'label': label,
                        'code': code,
                        'type': node_type
                    }

        # Collect edges
        edge_type_map = {"AST": 0, "CFG": 1, "PDG": 2}

        for gtype in ["AST", "CFG", "PDG"]:
            etype = edge_type_map[gtype]

            for node in func.get(gtype, []):
                for edge in node.get("edges", []):
                    src = edge.get("out")
                    dst = edge.get("in")

                    if src and dst and src in all_nodes and dst in all_nodes:
                        all_edges.append((src, dst, etype))

        # Extract vulnerability features for this function
        vuln_features = extract_vulnerability_features(func, label_name)

    if len(all_nodes) == 0:
        return None

    # Create node index mapping
    node_ids = list(all_nodes.keys())
    node_id_to_idx = {nid: idx for idx, nid in enumerate(node_ids)}

    # Generate embeddings for all nodes
    node_embeddings = []
    node_texts = []
    node_types = []

    for nid in node_ids:
        node_info = all_nodes[nid]
        text = node_info['text']

        node_texts.append(text)
        node_types.append(node_info['label'])

        # Get embedding
        emb = get_code_embedding(text)
        node_embeddings.append(emb)

    # Convert edges to index-based format
    edge_index = []
    edge_types = []

    for src_id, dst_id, etype in all_edges:
        if src_id in node_id_to_idx and dst_id in node_id_to_idx:
            src_idx = node_id_to_idx[src_id]
            dst_idx = node_id_to_idx[dst_id]
            edge_index.append([src_idx, dst_idx])
            edge_types.append(etype)

    # Convert to numpy arrays
    x = np.array(node_embeddings, dtype=np.float32)  # [num_nodes, 768]
    edge_index = np.array(edge_index, dtype=np.int64).T  # [2, num_edges]
    edge_types = np.array(edge_types, dtype=np.int64)  # [num_edges]

    # Create graph data structure
    graph_data = {
        'x': x,
        'edge_index': edge_index,
        'edge_types': edge_types,
        'vuln_features': vuln_features,
        'y': label_id,
        'label_name': label_name,
        'num_nodes': len(node_ids),
        'num_edges': len(edge_index[0]) if len(edge_index) > 0 else 0,
        'node_texts': node_texts,
        'node_types': node_types,
        'filename': os.path.basename(filepath)
    }

    return graph_data


def process_dataset():
    """
    Process toàn bộ dataset
    """
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    all_graphs = []
    stats = defaultdict(int)
    failed_files = []

    print("\n" + "=" * 70)
    print("PROCESSING DATASET")
    print("=" * 70)

    # Process each label
    for label_name, label_id in LABELS.items():
        folder = os.path.join(DATASET_DIR, label_name)

        if not os.path.isdir(folder):
            print(f"\n[!] Folder not found: {folder}")
            continue

        # Get all JSON files
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]

        if len(json_files) == 0:
            print(f"\n[!] No JSON files found in: {folder}")
            continue

        print(f"\n{'=' * 70}")
        print(f"Processing: {label_name.upper()} (Label ID: {label_id})")
        print(f"{'=' * 70}")
        print(f"Found {len(json_files)} JSON files")

        # Process each file
        for fname in tqdm(json_files, desc=f"  Encoding {label_name}"):
            filepath = os.path.join(folder, fname)

            try:
                graph_data = build_graph_from_json(filepath, label_id, label_name)

                if graph_data is not None:
                    all_graphs.append(graph_data)
                    stats[label_name] += 1
                    stats['total_nodes'] += graph_data['num_nodes']
                    stats['total_edges'] += graph_data['num_edges']
                else:
                    failed_files.append((fname, "No data extracted"))

            except Exception as e:
                failed_files.append((fname, str(e)))
                print(f"\n[!] Failed to process {fname}: {e}")

    # Print statistics
    print("\n" + "=" * 70)
    print("DATASET STATISTICS")
    print("=" * 70)
    print(f"Total graphs successfully processed: {len(all_graphs)}")
    print(f"\nLabel distribution:")
    for label_name in LABELS.keys():
        count = stats[label_name]
        if count > 0:
            pct = (count / len(all_graphs)) * 100
            print(f"  {label_name:20s}: {count:5d} graphs ({pct:5.1f}%)")

    if len(all_graphs) > 0:
        avg_nodes = stats['total_nodes'] / len(all_graphs)
        avg_edges = stats['total_edges'] / len(all_graphs)
        print(f"\nGraph statistics:")
        print(f"  Average nodes per graph: {avg_nodes:.1f}")
        print(f"  Average edges per graph: {avg_edges:.1f}")

    if failed_files:
        print(f"\n[!] Failed to process {len(failed_files)} files:")
        for fname, error in failed_files[:10]:
            print(f"  - {fname}: {error[:50]}")
        if len(failed_files) > 10:
            print(f"  ... and {len(failed_files) - 10} more")

    # Save processed data
    print(f"\n[+] Saving processed data to {OUTPUT_PKL}...")

    output_data = {
        'graphs': all_graphs,
        'labels': LABELS,
        'stats': dict(stats),
        'embedding_dim': 768,
        'vuln_features_dim': 12,
        'num_edge_types': 3
    }

    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output_data, f)

    # Save metadata
    metadata_file = os.path.join(OUTPUT_DIR, "metadata.json")
    metadata = {
        'num_graphs': len(all_graphs),
        'label_distribution': {k: stats[k] for k in LABELS.keys()},
        'avg_nodes': avg_nodes if len(all_graphs) > 0 else 0,
        'avg_edges': avg_edges if len(all_graphs) > 0 else 0,
        'embedding_dim': 768,
        'vuln_features_dim': 12,
        'num_edge_types': 3,
        'failed_files': len(failed_files)
    }

    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"✅ Saved metadata to {metadata_file}")
    print("=" * 70 + "\n")

    return all_graphs, stats


if __name__ == "__main__":
    all_graphs, stats = process_dataset()

    if len(all_graphs) > 0:
        print(f"\n✅ SUCCESS! Processed {len(all_graphs)} graphs")
        print(f"   Output: {OUTPUT_PKL}")
        print(f"   Ready for training!\n")
    else:
        print("\n[!] ERROR: No graphs were processed successfully")
        print("   Please check your dataset and try again.\n")
"""
GraphCodeBERT Data Tokenizer & Processor
Cải tiến tokenization phục vụ train GraphCodeBERT cho vulnerability detection
"""

import os
import json
import pickle
import numpy as np
import torch
from transformers import RobertaTokenizer
from tqdm import tqdm
from collections import defaultdict
import warnings

warnings.filterwarnings('ignore')

# Cấu hình
OUTPUT_DIR = "output"
OUTPUT_SAFE_DIR = "output_safe"
PROCESSED_DIR = "processed_graphcodebert"
OUTPUT_PKL = os.path.join(PROCESSED_DIR, "tokenized_data.pkl")

# Label mapping (vulnerable vs safe)
VULNERABILITY_TYPES = {
    "Buffer_Overflow": 0,
    "Command_Injection": 1,
    "Path_Traversal": 2,
    "SQL_Injection": 3
}

# Binary classification: 1=vulnerable, 0=safe
BINARY_LABELS = {
    "vulnerable": 1,
    "safe": 0
}

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[+] Using device: {device}")

# Load GraphCodeBERT tokenizer
print("[+] Loading GraphCodeBERT tokenizer...")
tokenizer = RobertaTokenizer.from_pretrained("microsoft/graphcodebert-base")

# Special tokens
CLS_TOKEN = tokenizer.cls_token  # <s>
SEP_TOKEN = tokenizer.sep_token  # </s>
PAD_TOKEN = tokenizer.pad_token  # <pad>
UNK_TOKEN = tokenizer.unk_token  # <unk>

# Max lengths
MAX_CODE_LENGTH = 256
MAX_DFG_LENGTH = 64


class GraphCodeBERTDataTokenizer:
    """Tokenizer phù hợp với GraphCodeBERT architecture"""
    
    def __init__(self, max_code_len=256, max_dfg_len=64):
        self.max_code_len = max_code_len
        self.max_dfg_len = max_dfg_len
        self.tokenizer = tokenizer
        
    def extract_code_tokens(self, nodes):
        """Extract và tokenize code từ AST nodes"""
        code_tokens = []
        token_to_node_index = {}  # Map token position to node
        node_positions = []  # Track which node each token belongs to
        
        for node_idx, node in enumerate(nodes):
            props = node.get("properties", {})
            
            # Lấy code hoặc các thuộc tính quan trọng
            code = props.get("CODE", "")
            name = props.get("NAME", "")
            label = node.get("label", "")
            
            # Combine thông tin để tạo token text
            if code and code.strip():
                text = code
            elif name:
                text = name
            else:
                text = label
            
            if text and text.strip():
                # Tokenize
                tokens = self.tokenizer.tokenize(text)
                
                # Track node positions
                start_pos = len(code_tokens)
                code_tokens.extend(tokens)
                
                # Map tokens to nodes
                for i in range(start_pos, len(code_tokens)):
                    token_to_node_index[i] = node_idx
                    node_positions.append(node_idx)
        
        return code_tokens, token_to_node_index, node_positions
    
    def extract_dfg_edges(self, func_data):
        """Extract Data Flow Graph edges từ Joern output"""
        dfg_edges = []
        node_id_map = {}  # Map node IDs to sequential indices
        
        # Build node ID mapping
        all_nodes = []
        for gtype in ["AST", "CFG", "PDG"]:
            for node in func_data.get(gtype, []):
                nid = node.get("id")
                if nid and nid not in node_id_map:
                    node_id_map[nid] = len(node_id_map)
                    all_nodes.append(node)
        
        # Extract edges from all graph types
        for gtype in ["AST", "CFG", "PDG"]:
            edge_type_name = gtype
            
            for node in func_data.get(gtype, []):
                for edge in node.get("edges", []):
                    src_id = edge.get("out")
                    dst_id = edge.get("in")
                    etype = edge.get("edgeType", edge_type_name)
                    
                    if src_id in node_id_map and dst_id in node_id_map:
                        src_idx = node_id_map[src_id]
                        dst_idx = node_id_map[dst_id]
                        dfg_edges.append({
                            'source': src_idx,
                            'target': dst_idx,
                            'type': etype
                        })
        
        return dfg_edges, all_nodes, node_id_map
    
    def build_variable_mapping(self, nodes):
        """Build mapping từ variables đến token positions"""
        var_to_positions = defaultdict(list)
        
        for node_idx, node in enumerate(nodes):
            props = node.get("properties", {})
            label = node.get("label", "")
            
            # Identify variables (IDENTIFIER nodes)
            if label == "IDENTIFIER" or label == "LOCAL":
                var_name = props.get("NAME", "")
                if var_name:
                    var_to_positions[var_name].append(node_idx)
        
        return var_to_positions
    
    def tokenize_function(self, func_data, label, vuln_type=None):
        """
        Tokenize một function theo format GraphCodeBERT
        
        Returns:
            dict: {
                'input_ids': token IDs của code,
                'attention_mask': attention mask,
                'position_idx': position embeddings,
                'dfg_to_code': mapping từ DFG nodes tới code tokens,
                'dfg_edges': DFG edge index,
                'label': label,
                'vuln_type': vulnerability type (nếu có)
            }
        """
        # Extract DFG và nodes
        dfg_edges, all_nodes, node_id_map = self.extract_dfg_edges(func_data)
        
        # Extract code tokens
        code_tokens, token_to_node, node_positions = self.extract_code_tokens(all_nodes)
        
        # Truncate nếu quá dài
        if len(code_tokens) > self.max_code_len - 2:  # -2 for CLS and SEP
            code_tokens = code_tokens[:self.max_code_len - 2]
            node_positions = node_positions[:self.max_code_len - 2]
        
        # Add special tokens
        tokens = [CLS_TOKEN] + code_tokens + [SEP_TOKEN]
        
        # Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # Attention mask (1 for real tokens, 0 for padding)
        attention_mask = [1] * len(input_ids)
        
        # Position indices
        position_idx = list(range(len(input_ids)))
        
        # Padding
        padding_length = self.max_code_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            position_idx = position_idx + [0] * padding_length
        
        # Build DFG to code mapping
        # Adjust node_positions for CLS token (+1 offset)
        adjusted_positions = [0] + [p + 1 for p in node_positions] + [len(node_positions) + 1]
        
        dfg_to_code = []
        for node_idx in range(len(all_nodes)):
            # Find token positions for this node
            token_positions = [i for i, pos in enumerate(adjusted_positions) if pos == node_idx]
            if token_positions:
                dfg_to_code.append(token_positions)
            else:
                dfg_to_code.append([0])  # Map to CLS if no tokens found
        
        # Build DFG adjacency matrix
        max_nodes = min(len(all_nodes), self.max_dfg_len)
        dfg_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        
        for edge in dfg_edges:
            src = edge['source']
            dst = edge['target']
            if src < max_nodes and dst < max_nodes:
                dfg_matrix[src][dst] = 1.0
                # Add reverse edge for undirected graph
                dfg_matrix[dst][src] = 1.0
        
        # Build edge index (COO format for GNN)
        edge_index = []
        for edge in dfg_edges[:self.max_dfg_len]:
            src = edge['source']
            dst = edge['target']
            if src < max_nodes and dst < max_nodes:
                edge_index.append([src, dst])
        
        if len(edge_index) == 0:
            edge_index = [[0, 0]]  # Dummy edge
        
        return {
            'input_ids': np.array(input_ids, dtype=np.int64),
            'attention_mask': np.array(attention_mask, dtype=np.int64),
            'position_idx': np.array(position_idx, dtype=np.int64),
            'dfg_to_code': dfg_to_code,
            'dfg_matrix': dfg_matrix,
            'edge_index': np.array(edge_index, dtype=np.int64).T,  # [2, num_edges]
            'num_nodes': max_nodes,
            'num_edges': len(edge_index),
            'label': label,
            'vuln_type': vuln_type,
            'code_tokens': tokens,
            'original_length': len(code_tokens)
        }
    
    def process_json_file(self, filepath, label, vuln_type=None):
        """Process một JSON file từ Joern output"""
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Error reading {filepath}: {e}")
            return []
        
        tokenized_functions = []
        
        # Process each function trong file
        for func in data.get("functions", []):
            try:
                tokenized = self.tokenize_function(func, label, vuln_type)
                tokenized['function_name'] = func.get('function', 'unknown')
                tokenized['file'] = func.get('file', '')
                tokenized['source_file'] = os.path.basename(filepath)
                
                tokenized_functions.append(tokenized)
            except Exception as e:
                print(f"[!] Error processing function in {filepath}: {e}")
                continue
        
        return tokenized_functions


def process_all_data():
    """Process toàn bộ dataset (vulnerable + safe)"""
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    tokenizer_obj = GraphCodeBERTDataTokenizer(
        max_code_len=MAX_CODE_LENGTH,
        max_dfg_len=MAX_DFG_LENGTH
    )
    
    all_data = []
    stats = defaultdict(int)
    
    print("\n" + "=" * 70)
    print("TOKENIZING DATASET FOR GRAPHCODEBERT")
    print("=" * 70)
    
    # Process vulnerable data
    print("\n[+] Processing VULNERABLE samples...")
    for vuln_name, vuln_id in VULNERABILITY_TYPES.items():
        folder = os.path.join(OUTPUT_DIR, vuln_name)
        
        if not os.path.isdir(folder):
            print(f"[!] Folder not found: {folder}")
            continue
        
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        print(f"\n  Processing {vuln_name}: {len(json_files)} files")
        
        for fname in tqdm(json_files, desc=f"    {vuln_name}"):
            filepath = os.path.join(folder, fname)
            
            tokenized_funcs = tokenizer_obj.process_json_file(
                filepath,
                label=BINARY_LABELS['vulnerable'],
                vuln_type=vuln_id
            )
            
            all_data.extend(tokenized_funcs)
            stats[f'vulnerable_{vuln_name}'] += len(tokenized_funcs)
            stats['total_vulnerable'] += len(tokenized_funcs)
    
    # Process safe data
    print("\n[+] Processing SAFE samples...")
    for vuln_name in VULNERABILITY_TYPES.keys():
        folder = os.path.join(OUTPUT_SAFE_DIR, vuln_name)
        
        if not os.path.isdir(folder):
            print(f"[!] Folder not found: {folder}")
            continue
        
        json_files = [f for f in os.listdir(folder) if f.endswith('.json')]
        print(f"\n  Processing safe {vuln_name}: {len(json_files)} files")
        
        for fname in tqdm(json_files, desc=f"    safe_{vuln_name}"):
            filepath = os.path.join(folder, fname)
            
            tokenized_funcs = tokenizer_obj.process_json_file(
                filepath,
                label=BINARY_LABELS['safe'],
                vuln_type=None
            )
            
            all_data.extend(tokenized_funcs)
            stats[f'safe_{vuln_name}'] += len(tokenized_funcs)
            stats['total_safe'] += len(tokenized_funcs)
    
    # Statistics
    print("\n" + "=" * 70)
    print("TOKENIZATION STATISTICS")
    print("=" * 70)
    print(f"Total samples: {len(all_data)}")
    print(f"  Vulnerable: {stats['total_vulnerable']}")
    print(f"  Safe: {stats['total_safe']}")
    
    print("\nDetailed breakdown:")
    for key, count in sorted(stats.items()):
        if key not in ['total_vulnerable', 'total_safe']:
            print(f"  {key:30s}: {count:5d}")
    
    # Calculate average lengths
    if len(all_data) > 0:
        avg_tokens = np.mean([d['original_length'] for d in all_data])
        avg_nodes = np.mean([d['num_nodes'] for d in all_data])
        avg_edges = np.mean([d['num_edges'] for d in all_data])
        
        print(f"\nAverage statistics:")
        print(f"  Tokens per sample: {avg_tokens:.1f}")
        print(f"  Nodes per sample: {avg_nodes:.1f}")
        print(f"  Edges per sample: {avg_edges:.1f}")
    
    # Save tokenized data
    print(f"\n[+] Saving tokenized data to {OUTPUT_PKL}...")
    
    output_data = {
        'data': all_data,
        'stats': dict(stats),
        'config': {
            'max_code_length': MAX_CODE_LENGTH,
            'max_dfg_length': MAX_DFG_LENGTH,
            'tokenizer': 'microsoft/graphcodebert-base',
            'num_labels': 2,  # Binary classification
            'vulnerability_types': VULNERABILITY_TYPES,
            'label_mapping': BINARY_LABELS
        }
    }
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    print(f"✅ Saved {len(all_data)} samples")
    
    # Save metadata JSON
    metadata_file = os.path.join(PROCESSED_DIR, "metadata.json")
    metadata = {
        'total_samples': len(all_data),
        'vulnerable_samples': stats['total_vulnerable'],
        'safe_samples': stats['total_safe'],
        'config': output_data['config'],
        'statistics': dict(stats)
    }
    
    with open(metadata_file, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print(f"✅ Saved metadata to {metadata_file}")
    print("=" * 70 + "\n")
    
    return all_data, stats


def create_train_val_test_split(all_data, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    """Split data thành train/val/test sets"""
    
    print("\n[+] Splitting data into train/val/test...")
    
    # Shuffle data
    np.random.seed(42)
    indices = np.random.permutation(len(all_data))
    
    n_train = int(len(all_data) * train_ratio)
    n_val = int(len(all_data) * val_ratio)
    
    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]
    
    train_data = [all_data[i] for i in train_indices]
    val_data = [all_data[i] for i in val_indices]
    test_data = [all_data[i] for i in test_indices]
    
    # Save splits
    splits = {
        'train': train_data,
        'val': val_data,
        'test': test_data
    }
    
    for split_name, split_data in splits.items():
        split_file = os.path.join(PROCESSED_DIR, f"{split_name}.pkl")
        with open(split_file, 'wb') as f:
            pickle.dump(split_data, f, protocol=pickle.HIGHEST_PROTOCOL)
        
        # Count labels
        n_vuln = sum(1 for d in split_data if d['label'] == 1)
        n_safe = len(split_data) - n_vuln
        
        print(f"  {split_name:5s}: {len(split_data):5d} samples ({n_vuln} vulnerable, {n_safe} safe)")
    
    print(f"\n✅ Saved train/val/test splits to {PROCESSED_DIR}/")
    
    return splits


if __name__ == "__main__":
    # Process all data
    all_data, stats = process_all_data()
    
    if len(all_data) > 0:
        # Create splits
        splits = create_train_val_test_split(all_data)
        
        print(f"\n{'=' * 70}")
        print("✅ SUCCESS! Tokenization complete!")
        print(f"{'=' * 70}")
        print(f"Total samples processed: {len(all_data)}")
        print(f"Output directory: {PROCESSED_DIR}/")
        print(f"\nReady for GraphCodeBERT training!")
        print(f"{'=' * 70}\n")
    else:
        print("\n[!] ERROR: No data processed successfully")
        print("   Please check your dataset and try again.\n")

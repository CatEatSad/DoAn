"""
GraphCodeBERT Data Tokenizer & Processor - Enhanced Version
Trích xuất toàn bộ đặc trưng từ Joern CPG JSON
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
import re
import csv

warnings.filterwarnings('ignore')

# Tăng giới hạn CSV field size để đọc được JSON lớn
csv.field_size_limit(10**7)

# Cấu hình - ĐỌC TỪ CSV FILES
VULNERABLE_CSV = "vulnerable_data_filtered.csv"
SAFE_CSV = "safe_data_filtered.csv"
PROCESSED_DIR = "processed_graphcodebert_enhanced"
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
MAX_CODE_LENGTH = 512  # Tăng lên để chứa nhiều thông tin hơn
MAX_DFG_LENGTH = 128    # Tăng lên để chứa nhiều node hơn

# Node labels có ý nghĩa quan trọng
CRITICAL_NODE_LABELS = {
    'CALL', 'METHOD', 'METHOD_PARAMETER_IN', 'METHOD_PARAMETER_OUT',
    'IDENTIFIER', 'LOCAL', 'LITERAL', 'BLOCK', 'RETURN',
    'CONTROL_STRUCTURE', 'FIELD_IDENTIFIER', 'METHOD_RETURN',
    'MODIFIER'
}

# Edge types quan trọng
EDGE_TYPE_MAPPING = {
    'AST': 0,
    'CFG': 1, 
    'PDG': 2,
    'REACHING_DEF': 3,
    'CALL': 4,
    'REF': 5,
    'EVAL_TYPE': 6
}


class GraphCodeBERTDataTokenizer:
    """
    Enhanced Tokenizer - Trích xuất TẤT CẢ đặc trưng từ Joern CPG JSON
    Đảm bảo dữ liệu đầu vào được rõ ràng và đầy đủ nhất
    """
    
    def __init__(self, max_code_len=512, max_dfg_len=128):
        self.max_code_len = max_code_len
        self.max_dfg_len = max_dfg_len
        self.tokenizer = tokenizer
        
    def extract_node_properties(self, node):
        """
        Trích xuất TẤT CẢ properties quan trọng từ một node
        Returns: dict chứa tất cả thông tin của node
        """
        props = node.get("properties", {})
        label = node.get("label", "")
        node_id = node.get("id", "")
        
        # Trích xuất tất cả properties có thể có
        node_features = {
            'id': node_id,
            'label': label,
            'code': props.get("CODE", ""),
            'name': props.get("NAME", ""),
            'type_full_name': props.get("TYPE_FULL_NAME", ""),
            'method_full_name': props.get("METHOD_FULL_NAME", ""),
            'signature': props.get("SIGNATURE", ""),
            'dispatch_type': props.get("DISPATCH_TYPE", ""),
            'line_number': props.get("LINE_NUMBER", ""),
            'column_number': props.get("COLUMN_NUMBER", ""),
            'order': props.get("ORDER", ""),
            'argument_index': props.get("ARGUMENT_INDEX", ""),
            'evaluation_strategy': props.get("EVALUATION_STRATEGY", ""),
            'modifier_type': props.get("MODIFIER_TYPE", ""),
            'canonical_name': props.get("CANONICAL_NAME", ""),
            'generic_signature': props.get("GENERIC_SIGNATURE", ""),
            'index': props.get("INDEX", ""),
            'dynamic_type_hint': props.get("DYNAMIC_TYPE_HINT_FULL_NAME", ""),
        }
        
        return node_features
    
    def build_comprehensive_code_sequence(self, nodes):
        """
        Xây dựng chuỗi code đầy đủ từ tất cả nodes
        Kết hợp nhiều loại thông tin để tạo representation phong phú
        """
        code_tokens = []
        node_features_list = []
        node_type_ids = []  # Track loại node
        node_positions = []  # Track vị trí trong code sequence
        
        for node_idx, node in enumerate(nodes):
            features = self.extract_node_properties(node)
            node_features_list.append(features)
            
            label = features['label']
            
            # Xây dựng text representation cho node
            text_parts = []
            
            # 1. CODE là quan trọng nhất
            if features['code']:
                text_parts.append(features['code'])
            
            # 2. NAME cho identifiers và methods
            if features['name'] and features['name'] not in str(features['code']):
                text_parts.append(features['name'])
            
            # 3. Label cho context
            if label in CRITICAL_NODE_LABELS:
                text_parts.append(f"[{label}]")
            
            # 4. Type information
            if features['type_full_name'] and features['type_full_name'] != 'ANY':
                type_name = features['type_full_name'].split('.')[-1]  # Lấy tên ngắn
                text_parts.append(f":{type_name}")
            
            # 5. Method signatures cho CALL nodes
            if label == 'CALL' and features['method_full_name']:
                method_name = features['method_full_name'].split('.')[-1]
                if method_name not in str(features['code']):
                    text_parts.append(f"@{method_name}")
            
            # 6. Dispatch type cho dynamic calls
            if features['dispatch_type']:
                text_parts.append(f"#{features['dispatch_type']}")
            
            # Kết hợp tất cả parts
            text = " ".join(text_parts)
            
            if text and text.strip():
                # Tokenize
                tokens = self.tokenizer.tokenize(text)
                
                # Track positions
                start_pos = len(code_tokens)
                code_tokens.extend(tokens)
                
                # Track node info for each token
                for _ in range(start_pos, len(code_tokens)):
                    node_positions.append(node_idx)
                    # Encode node type as ID
                    if label in CRITICAL_NODE_LABELS:
                        type_id = list(CRITICAL_NODE_LABELS).index(label)
                    else:
                        type_id = len(CRITICAL_NODE_LABELS)
                    node_type_ids.append(type_id)
        
        return code_tokens, node_features_list, node_type_ids, node_positions
    
    def extract_comprehensive_graph(self, func_data):
        """
        Trích xuất TOÀN BỘ thông tin graph từ AST, CFG, PDG
        Returns: 
            - all_nodes: danh sách tất cả nodes
            - all_edges: danh sách tất cả edges với type
            - node_id_map: mapping từ ID gốc sang index
            - graph_structure: cấu trúc graph cho từng loại
        """
        node_id_map = {}  # Map node IDs to sequential indices
        all_nodes = []
        all_edges = []
        
        # Thu thập TẤT CẢ nodes từ cả 3 graph types
        for gtype in ["AST", "CFG", "PDG"]:
            for node in func_data.get(gtype, []):
                nid = node.get("id")
                if nid and nid not in node_id_map:
                    node_id_map[nid] = len(node_id_map)
                    all_nodes.append(node)
        
        # Trích xuất TẤT CẢ edges từ cả 3 graph types
        graph_structure = {
            'AST': {'nodes': [], 'edges': []},
            'CFG': {'nodes': [], 'edges': []},
            'PDG': {'nodes': [], 'edges': []}
        }
        
        for gtype in ["AST", "CFG", "PDG"]:
            for node in func_data.get(gtype, []):
                nid = node.get("id")
                if nid in node_id_map:
                    src_idx = node_id_map[nid]
                    graph_structure[gtype]['nodes'].append(src_idx)
                    
                    # Trích xuất tất cả edges của node này
                    for edge in node.get("edges", []):
                        src_id = edge.get("out")
                        dst_id = edge.get("in")
                        edge_type = edge.get("edgeType", gtype)
                        edge_id = edge.get("id", "")
                        
                        if src_id in node_id_map and dst_id in node_id_map:
                            src_idx = node_id_map[src_id]
                            dst_idx = node_id_map[dst_id]
                            
                            # Map edge type to ID
                            edge_type_id = EDGE_TYPE_MAPPING.get(edge_type, len(EDGE_TYPE_MAPPING))
                            
                            edge_info = {
                                'source': src_idx,
                                'target': dst_idx,
                                'type': edge_type,
                                'type_id': edge_type_id,
                                'graph_type': gtype,
                                'edge_id': edge_id
                            }
                            
                            all_edges.append(edge_info)
                            graph_structure[gtype]['edges'].append(edge_info)
        
        return all_nodes, all_edges, node_id_map, graph_structure
    
    def extract_data_flow_features(self, nodes, edges):
        """
        Trích xuất các đặc trưng data flow:
        - Variable definitions và uses
        - Control dependencies
        - Data dependencies
        """
        # Variable tracking
        var_definitions = defaultdict(list)  # var_name -> [def nodes]
        var_uses = defaultdict(list)  # var_name -> [use nodes]
        
        # Method calls tracking
        method_calls = []
        
        # Control flow features
        control_flow_nodes = []
        
        for node_idx, node in enumerate(nodes):
            props = node.get("properties", {})
            label = node.get("label", "")
            
            # Track variables
            if label == "IDENTIFIER" or label == "LOCAL":
                var_name = props.get("NAME", "")
                if var_name:
                    # Phân biệt definition vs use
                    eval_strategy = props.get("EVALUATION_STRATEGY", "")
                    if label == "LOCAL" or "OUT" in label:
                        var_definitions[var_name].append(node_idx)
                    else:
                        var_uses[var_name].append(node_idx)
            
            # Track method calls (quan trọng cho vulnerability)
            if label == "CALL":
                method_name = props.get("METHOD_FULL_NAME", "")
                dispatch_type = props.get("DISPATCH_TYPE", "")
                method_calls.append({
                    'node_idx': node_idx,
                    'method': method_name,
                    'dispatch': dispatch_type,
                    'name': props.get("NAME", "")
                })
            
            # Track control flow nodes
            if label in ["CONTROL_STRUCTURE", "IF", "WHILE", "FOR", "RETURN"]:
                control_flow_nodes.append(node_idx)
        
        # Build reaching definitions từ PDG edges
        reaching_defs = defaultdict(set)
        for edge in edges:
            if edge.get('type') == 'REACHING_DEF':
                src = edge['source']
                dst = edge['target']
                reaching_defs[dst].add(src)
        
        return {
            'var_definitions': dict(var_definitions),
            'var_uses': dict(var_uses),
            'method_calls': method_calls,
            'control_flow_nodes': control_flow_nodes,
            'reaching_defs': dict(reaching_defs)
        }
    
    def tokenize_function(self, func_data, label, vuln_type=None):
        """
        ENHANCED: Tokenize một function với TẤT CẢ đặc trưng
        
        Returns:
            dict: Comprehensive feature dictionary bao gồm:
                - Code sequence features (tokens, IDs, attention)
                - Graph structure features (AST, CFG, PDG)
                - Data flow features (variables, reaching defs)
                - Control flow features
                - Node-level features
                - Edge-level features
        """
        # 1. Trích xuất TOÀN BỘ graph structure
        all_nodes, all_edges, node_id_map, graph_structure = self.extract_comprehensive_graph(func_data)
        
        # 2. Trích xuất code sequence với đầy đủ features
        code_tokens, node_features_list, node_type_ids, node_positions = \
            self.build_comprehensive_code_sequence(all_nodes)
        
        # 3. Trích xuất data flow features
        df_features = self.extract_data_flow_features(all_nodes, all_edges)
        
        # 4. Truncate nếu cần (nhưng cố gắng giữ nhiều info nhất)
        original_length = len(code_tokens)
        if len(code_tokens) > self.max_code_len - 2:  # -2 for CLS and SEP
            code_tokens = code_tokens[:self.max_code_len - 2]
            node_positions = node_positions[:self.max_code_len - 2]
            node_type_ids = node_type_ids[:self.max_code_len - 2]
        
        # 5. Add special tokens
        tokens = [CLS_TOKEN] + code_tokens + [SEP_TOKEN]
        
        # 6. Convert to IDs
        input_ids = self.tokenizer.convert_tokens_to_ids(tokens)
        
        # 7. Attention mask
        attention_mask = [1] * len(input_ids)
        
        # 8. Position indices
        position_idx = list(range(len(input_ids)))
        
        # 9. Node type IDs (for tracking node types)
        node_type_sequence = [len(CRITICAL_NODE_LABELS)] + node_type_ids + [len(CRITICAL_NODE_LABELS)]
        
        # 10. Padding
        padding_length = self.max_code_len - len(input_ids)
        if padding_length > 0:
            input_ids = input_ids + [self.tokenizer.pad_token_id] * padding_length
            attention_mask = attention_mask + [0] * padding_length
            position_idx = position_idx + [0] * padding_length
            node_type_sequence = node_type_sequence + [len(CRITICAL_NODE_LABELS)] * padding_length
        
        # 11. Build mapping từ nodes to code tokens
        adjusted_positions = [0] + [p + 1 for p in node_positions] + [len(node_positions) + 1]
        
        node_to_token_map = defaultdict(list)
        for token_idx, node_idx in enumerate(adjusted_positions):
            if token_idx < len(input_ids):
                node_to_token_map[node_idx].append(token_idx)
        
        dfg_to_code = []
        for node_idx in range(len(all_nodes)):
            if node_idx in node_to_token_map:
                dfg_to_code.append(node_to_token_map[node_idx])
            else:
                dfg_to_code.append([0])  # Map to CLS
        
        # 12. Build multi-type adjacency matrices
        max_nodes = min(len(all_nodes), self.max_dfg_len)
        
        # Separate matrices cho từng loại graph
        ast_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        cfg_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        pdg_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        
        # Combined matrix
        combined_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        
        for edge in all_edges:
            src = edge['source']
            dst = edge['target']
            gtype = edge['graph_type']
            
            if src < max_nodes and dst < max_nodes:
                combined_matrix[src][dst] = 1.0
                
                if gtype == 'AST':
                    ast_matrix[src][dst] = 1.0
                elif gtype == 'CFG':
                    cfg_matrix[src][dst] = 1.0
                elif gtype == 'PDG':
                    pdg_matrix[src][dst] = 1.0
        
        # 13. Build edge index cho GNN (separate by type)
        edge_index_by_type = defaultdict(list)
        edge_attr_list = []
        
        for edge in all_edges:
            src = edge['source']
            dst = edge['target']
            if src < max_nodes and dst < max_nodes:
                edge_type_id = edge['type_id']
                edge_index_by_type[edge['type']].append([src, dst])
                edge_attr_list.append(edge_type_id)
        
        # Combined edge index
        all_edge_indices = []
        for edges in edge_index_by_type.values():
            all_edge_indices.extend(edges)
        
        if len(all_edge_indices) == 0:
            all_edge_indices = [[0, 0]]
            edge_attr_list = [0]
        
        edge_index = np.array(all_edge_indices, dtype=np.int64).T  # [2, num_edges]
        edge_attr = np.array(edge_attr_list, dtype=np.int64)
        
        # 14. Extract function metadata
        func_name = func_data.get('function', 'unknown')
        func_file = func_data.get('file', '')
        func_id = func_data.get('id', '')
        
        # 15. Return comprehensive feature dict
        return {
            # Basic features
            'input_ids': np.array(input_ids, dtype=np.int64),
            'attention_mask': np.array(attention_mask, dtype=np.int64),
            'position_idx': np.array(position_idx, dtype=np.int64),
            'node_type_ids': np.array(node_type_sequence, dtype=np.int64),
            
            # Graph structure features
            'ast_matrix': ast_matrix,
            'cfg_matrix': cfg_matrix,
            'pdg_matrix': pdg_matrix,
            'combined_matrix': combined_matrix,
            'dfg_matrix': combined_matrix,  # Alias for backward compatibility
            'edge_index': edge_index,
            'edge_attr': edge_attr,
            
            # Node-code mapping
            'dfg_to_code': dfg_to_code,
            'node_to_token_map': dict(node_to_token_map),
            
            # Data flow features
            'var_definitions': df_features['var_definitions'],
            'var_uses': df_features['var_uses'],
            'method_calls': df_features['method_calls'],
            'control_flow_nodes': df_features['control_flow_nodes'],
            'reaching_defs': df_features['reaching_defs'],
            
            # Statistics
            'num_nodes': max_nodes,
            'num_edges': len(all_edge_indices),
            'num_ast_edges': len(edge_index_by_type.get('AST', [])),
            'num_cfg_edges': len(edge_index_by_type.get('CFG', [])),
            'num_pdg_edges': len(edge_index_by_type.get('PDG', [])),
            'original_length': original_length,
            'num_tokens': len(code_tokens),
            
            # Labels
            'label': label,
            'vuln_type': vuln_type,
            
            # Metadata
            'function_name': func_name,
            'file': func_file,
            'function_id': func_id,
            'code_tokens': tokens[:50],  # Giữ một phần để debug
        }
    
    def process_json_file(self, filepath, label, vuln_type=None):
        """
        Process một JSON file từ Joern output
        Trích xuất TẤT CẢ functions với đầy đủ features
        """
        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            print(f"[!] Error reading {filepath}: {e}")
            return []
        
        tokenized_functions = []
        
        # Process each function trong file
        for func_idx, func in enumerate(data.get("functions", [])):
            try:
                # Tokenize với enhanced features
                tokenized = self.tokenize_function(func, label, vuln_type)
                
                # Add file-level metadata
                tokenized['source_file'] = os.path.basename(filepath)
                tokenized['func_index'] = func_idx
                tokenized['total_functions'] = len(data.get("functions", []))
                
                # Add statistics
                tokenized['num_ast_nodes'] = len(func.get('AST', []))
                tokenized['num_cfg_nodes'] = len(func.get('CFG', []))
                tokenized['num_pdg_nodes'] = len(func.get('PDG', []))
                
                tokenized_functions.append(tokenized)
                
            except Exception as e:
                print(f"[!] Error processing function {func_idx} in {filepath}: {e}")
                import traceback
                traceback.print_exc()
                continue
        
        return tokenized_functions


def process_csv_file(csv_file, label, tokenizer_obj, is_vulnerable=True):
    """
    Process một CSV file chứa filtered functions
    CSV format: vul_type, vul_json
    """
    if not os.path.exists(csv_file):
        print(f"[!] CSV file not found: {csv_file}")
        return [], {}
    
    print(f"\n[+] Reading from CSV: {csv_file}")
    
    tokenized_functions = []
    stats = defaultdict(int)
    
    try:
        with open(csv_file, 'r', encoding='utf-8') as f:
            csv_reader = csv.DictReader(f)
            
            # Get total count for progress bar
            rows = list(csv_reader)
            print(f"[+] Found {len(rows)} samples in CSV")
            
            for row_idx, row in enumerate(tqdm(rows, desc=f"    Processing {os.path.basename(csv_file)}")):
                try:
                    vul_type_name = row['vul_type']
                    vul_json_str = row['vul_json']
                    
                    # Parse JSON string with better error handling
                    try:
                        func_data = json.loads(vul_json_str)
                    except json.JSONDecodeError as json_err:
                        # Skip malformed JSON rows
                        stats['malformed_json'] += 1
                        if stats['malformed_json'] <= 5:  # Only show first 5 errors
                            print(f"\n[!] Skipping row {row_idx} - Malformed JSON: {str(json_err)[:100]}")
                        continue
                    
                    # Get vulnerability type ID
                    if is_vulnerable:
                        vuln_type_id = VULNERABILITY_TYPES.get(vul_type_name, -1)
                    else:
                        vuln_type_id = None
                    
                    # Tokenize function
                    tokenized = tokenizer_obj.tokenize_function(
                        func_data,
                        label=label,
                        vuln_type=vuln_type_id
                    )
                    
                    # Add CSV metadata
                    tokenized['csv_source'] = os.path.basename(csv_file)
                    tokenized['csv_row'] = row_idx
                    tokenized['vul_type_name'] = vul_type_name
                    
                    tokenized_functions.append(tokenized)
                    
                    # Update stats
                    if is_vulnerable:
                        stats[f'vulnerable_{vul_type_name}'] += 1
                        stats['total_vulnerable'] += 1
                    else:
                        stats[f'safe_{vul_type_name}'] += 1
                        stats['total_safe'] += 1
                    
                except Exception as e:
                    stats['processing_errors'] += 1
                    if stats['processing_errors'] <= 5:  # Only show first 5 errors
                        print(f"\n[!] Error processing row {row_idx}: {str(e)[:100]}")
                    continue
    
    except Exception as e:
        print(f"[!] Error reading CSV file {csv_file}: {e}")
        import traceback
        traceback.print_exc()
        return [], {}
    
    # Report errors if any
    if stats.get('malformed_json', 0) > 0 or stats.get('processing_errors', 0) > 0:
        print(f"\n⚠️  Processing warnings:")
        if stats.get('malformed_json', 0) > 0:
            print(f"   - Skipped {stats['malformed_json']} rows with malformed JSON")
        if stats.get('processing_errors', 0) > 0:
            print(f"   - Skipped {stats['processing_errors']} rows with processing errors")
        print(f"   - Successfully processed: {len(tokenized_functions)} functions")
    
    return tokenized_functions, stats


def process_all_data():
    """Process toàn bộ dataset từ CSV files (vulnerable + safe)"""
    
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    
    tokenizer_obj = GraphCodeBERTDataTokenizer(
        max_code_len=MAX_CODE_LENGTH,
        max_dfg_len=MAX_DFG_LENGTH
    )
    
    all_data = []
    stats = defaultdict(int)
    
    print("\n" + "=" * 70)
    print("TOKENIZING DATASET FROM CSV FILES")
    print("=" * 70)
    print(f"[+] Using GraphCodeBERT tokenizer")
    print(f"[+] Max code length: {MAX_CODE_LENGTH}")
    print(f"[+] Max graph nodes: {MAX_DFG_LENGTH}")
    
    # Process vulnerable data từ CSV
    print("\n" + "=" * 70)
    print("PROCESSING VULNERABLE SAMPLES FROM CSV")
    print("=" * 70)
    
    vulnerable_funcs, vuln_stats = process_csv_file(
        VULNERABLE_CSV,
        label=BINARY_LABELS['vulnerable'],
        tokenizer_obj=tokenizer_obj,
        is_vulnerable=True
    )
    
    all_data.extend(vulnerable_funcs)
    for key, value in vuln_stats.items():
        stats[key] += value
    
    # Process safe data từ CSV
    print("\n" + "=" * 70)
    print("PROCESSING SAFE SAMPLES FROM CSV")
    print("=" * 70)
    
    safe_funcs, safe_stats = process_csv_file(
        SAFE_CSV,
        label=BINARY_LABELS['safe'],
        tokenizer_obj=tokenizer_obj,
        is_vulnerable=False
    )
    
    all_data.extend(safe_funcs)
    for key, value in safe_stats.items():
        stats[key] += value
    
    # Statistics
    print("\n" + "=" * 70)
    print("ENHANCED TOKENIZATION STATISTICS")
    print("=" * 70)
    print(f"Total samples: {len(all_data)}")
    print(f"  Vulnerable: {stats['total_vulnerable']}")
    print(f"  Safe: {stats['total_safe']}")
    
    print("\nDetailed breakdown:")
    for key, count in sorted(stats.items()):
        if key not in ['total_vulnerable', 'total_safe']:
            print(f"  {key:30s}: {count:5d}")
    
    # Calculate comprehensive statistics
    if len(all_data) > 0:
        avg_tokens = np.mean([d['num_tokens'] for d in all_data])
        avg_nodes = np.mean([d['num_nodes'] for d in all_data])
        avg_edges = np.mean([d['num_edges'] for d in all_data])
        avg_ast_edges = np.mean([d['num_ast_edges'] for d in all_data])
        avg_cfg_edges = np.mean([d['num_cfg_edges'] for d in all_data])
        avg_pdg_edges = np.mean([d['num_pdg_edges'] for d in all_data])
        avg_methods = np.mean([len(d['method_calls']) for d in all_data])
        avg_vars = np.mean([len(d['var_definitions']) + len(d['var_uses']) for d in all_data])
        
        print(f"\n{'=' * 70}")
        print("ENHANCED FEATURES STATISTICS")
        print(f"{'=' * 70}")
        print(f"Code Sequence:")
        print(f"  Avg tokens per sample: {avg_tokens:.1f}")
        print(f"\nGraph Structure:")
        print(f"  Avg nodes per sample: {avg_nodes:.1f}")
        print(f"  Avg total edges: {avg_edges:.1f}")
        print(f"    - AST edges: {avg_ast_edges:.1f}")
        print(f"    - CFG edges: {avg_cfg_edges:.1f}")
        print(f"    - PDG edges: {avg_pdg_edges:.1f}")
        print(f"\nData Flow:")
        print(f"  Avg method calls: {avg_methods:.1f}")
        print(f"  Avg variables tracked: {avg_vars:.1f}")
        
        # Feature richness
        features_per_sample = avg_tokens + avg_edges + avg_methods + avg_vars
        print(f"\nFeature Richness:")
        print(f"  Total features per sample: {features_per_sample:.1f}")
        print(f"  Feature density: {features_per_sample / max(avg_tokens, 1):.2f}x")
    
    # Save enhanced tokenized data
    print(f"\n[+] Saving ENHANCED tokenized data to {OUTPUT_PKL}...")
    
    output_data = {
        'data': all_data,
        'stats': dict(stats),
        'config': {
            'max_code_length': MAX_CODE_LENGTH,
            'max_dfg_length': MAX_DFG_LENGTH,
            'tokenizer': 'microsoft/graphcodebert-base',
            'num_labels': 2,  # Binary classification
            'vulnerability_types': VULNERABILITY_TYPES,
            'label_mapping': BINARY_LABELS,
            'edge_types': EDGE_TYPE_MAPPING,
            'node_types': list(CRITICAL_NODE_LABELS),
            'enhanced_features': [
                'ast_matrix', 'cfg_matrix', 'pdg_matrix',
                'node_type_ids', 'edge_attr',
                'var_definitions', 'var_uses', 'method_calls',
                'reaching_defs', 'control_flow_nodes'
            ]
        },
        'feature_description': {
            'code_sequence': 'Comprehensive token sequence with node type info',
            'graph_structure': 'Separate AST/CFG/PDG adjacency matrices',
            'data_flow': 'Variable definitions, uses, and reaching definitions',
            'control_flow': 'Control structure nodes and CFG edges',
            'method_calls': 'All method invocations with dispatch types',
            'node_mapping': 'Bidirectional mapping between nodes and tokens'
        }
    }
    
    with open(OUTPUT_PKL, 'wb') as f:
        pickle.dump(output_data, f, protocol=pickle.HIGHEST_PROTOCOL)
    
    file_size_mb = os.path.getsize(OUTPUT_PKL) / (1024 * 1024)
    print(f"✅ Saved {len(all_data)} samples ({file_size_mb:.2f} MB)")
    
    # Save comprehensive metadata JSON
    metadata_file = os.path.join(PROCESSED_DIR, "metadata.json")
    
    # Calculate feature statistics
    if len(all_data) > 0:
        sample_features = all_data[0]
        feature_keys = list(sample_features.keys())
    else:
        feature_keys = []
    
    metadata = {
        'total_samples': len(all_data),
        'vulnerable_samples': stats['total_vulnerable'],
        'safe_samples': stats['total_safe'],
        'config': output_data['config'],
        'statistics': dict(stats),
        'feature_keys': feature_keys,
        'enhanced_features': output_data['config']['enhanced_features'],
        'feature_description': output_data['feature_description']
    }
    
    with open(metadata_file, 'w', encoding='utf-8') as f:
        json.dump(metadata, f, indent=2, ensure_ascii=False)
    
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
    print("\n" + "=" * 70)
    print("ENHANCED GRAPHCODEBERT TOKENIZER - CSV INPUT")
    print("=" * 70)
    print("Data Source:")
    print(f"  ✓ Vulnerable: {VULNERABLE_CSV}")
    print(f"  ✓ Safe: {SAFE_CSV}")
    print("\nFeatures extracted:")
    print("  ✓ Code sequence với node type information")
    print("  ✓ AST, CFG, PDG graphs riêng biệt")
    print("  ✓ Data flow: variable definitions, uses, reaching defs")
    print("  ✓ Control flow: control structures và CFG edges")
    print("  ✓ Method calls với dispatch types")
    print("  ✓ Node properties: code, name, type, signature, ...")
    print("  ✓ Edge types: AST, CFG, PDG, REACHING_DEF, ...")
    print("=" * 70 + "\n")
    
    # Process all data
    all_data, stats = process_all_data()
    
    if len(all_data) > 0:
        # Create splits
        splits = create_train_val_test_split(all_data)
        
        print(f"\n{'=' * 70}")
        print("✅ SUCCESS! ENHANCED Tokenization complete!")
        print(f"{'=' * 70}")
        print(f"Total samples processed: {len(all_data)}")
        print(f"  Source: Filtered CSV files with critical functions only")
        print(f"  Output directory: {PROCESSED_DIR}/")
        print(f"\nEnhanced features available:")
        print(f"  - Code sequence features: input_ids, attention_mask, node_type_ids")
        print(f"  - Graph matrices: ast_matrix, cfg_matrix, pdg_matrix")
        print(f"  - Data flow: var_definitions, var_uses, reaching_defs")
        print(f"  - Control flow: control_flow_nodes")
        print(f"  - Method calls: method_calls with full context")
        print(f"  - Node properties: 18+ properties per node")
        print(f"  - Edge types: 7+ classified edge types")
        print(f"\nData quality:")
        print(f"  ✓ Functions: Only critical functions from filtered CSV")
        print(f"  ✓ Features: Full detail extraction from JSON")
        print(f"  ✓ Graphs: AST/CFG/PDG preserved separately")
        print(f"\nReady for ENHANCED GraphCodeBERT training!")
        print(f"{'=' * 70}\n")
    else:
        print("\n[!] ERROR: No data processed successfully")
        print("   Please check your CSV files and try again.\n")
        print(f"   Expected files:")
        print(f"     - {VULNERABLE_CSV}")
        print(f"     - {SAFE_CSV}\n")

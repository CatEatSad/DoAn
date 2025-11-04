"""
PyTorch Dataset và DataLoader cho GraphCodeBERT
"""

import torch
from torch.utils.data import Dataset, DataLoader
import pickle
import numpy as np
from typing import Dict, List


class GraphCodeBERTDataset(Dataset):
    """Dataset cho GraphCodeBERT vulnerability detection"""
    
    def __init__(self, data_file: str):
        """
        Args:
            data_file: Path đến file pickle chứa tokenized data
        """
        with open(data_file, 'rb') as f:
            self.data = pickle.load(f)
        
        if isinstance(self.data, dict):
            # Nếu load từ file split (train/val/test)
            self.samples = self.data
        elif isinstance(self.data, list):
            # Nếu load trực tiếp list
            self.samples = self.data
        else:
            raise ValueError("Invalid data format")
        
        print(f"[+] Loaded {len(self.samples)} samples from {data_file}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        """
        Returns:
            dict với các keys:
                - input_ids: [max_code_length]
                - attention_mask: [max_code_length]
                - position_idx: [max_code_length]
                - dfg_matrix: [max_dfg_len, max_dfg_len]
                - edge_index: [2, num_edges]
                - label: scalar
                - vuln_type: scalar (optional)
        """
        sample = self.samples[idx]
        
        # Handle vuln_type - convert None to -1
        vuln_type = sample.get('vuln_type', -1)
        if vuln_type is None:
            vuln_type = -1
        
        return {
            'input_ids': torch.tensor(sample['input_ids'], dtype=torch.long),
            'attention_mask': torch.tensor(sample['attention_mask'], dtype=torch.long),
            'position_idx': torch.tensor(sample['position_idx'], dtype=torch.long),
            'dfg_matrix': torch.tensor(sample['dfg_matrix'], dtype=torch.float32),
            'edge_index': torch.tensor(sample['edge_index'], dtype=torch.long),
            'label': torch.tensor(sample['label'], dtype=torch.long),
            'vuln_type': torch.tensor(vuln_type, dtype=torch.long),
            'num_nodes': sample['num_nodes'],
            'num_edges': sample['num_edges']
        }


def collate_batch(batch):
    """
    Custom collate function để xử lý variable-length graphs
    """
    # Stack fixed-size tensors
    input_ids = torch.stack([item['input_ids'] for item in batch])
    attention_mask = torch.stack([item['attention_mask'] for item in batch])
    position_idx = torch.stack([item['position_idx'] for item in batch])
    labels = torch.stack([item['label'] for item in batch])
    vuln_types = torch.stack([item['vuln_type'] for item in batch])
    
    # Handle variable-size DFG matrices
    # Get max dimensions in batch
    max_nodes = max([item['dfg_matrix'].shape[0] for item in batch])
    
    # Pad DFG matrices
    dfg_matrices = []
    for item in batch:
        matrix = item['dfg_matrix']
        if matrix.shape[0] < max_nodes:
            # Pad to max_nodes
            padded = torch.zeros(max_nodes, max_nodes, dtype=torch.float32)
            padded[:matrix.shape[0], :matrix.shape[1]] = matrix
            dfg_matrices.append(padded)
        else:
            dfg_matrices.append(matrix)
    
    dfg_matrices = torch.stack(dfg_matrices)
    
    # Handle edge indices (keep as list since they have different sizes)
    edge_indices = [item['edge_index'] for item in batch]
    
    return {
        'input_ids': input_ids,
        'attention_mask': attention_mask,
        'position_idx': position_idx,
        'dfg_matrix': dfg_matrices,
        'edge_index': edge_indices,
        'label': labels,
        'vuln_type': vuln_types
    }


def create_dataloader(data_file: str, batch_size: int = 32, shuffle: bool = True, num_workers: int = 0):
    """
    Tạo DataLoader từ tokenized data file
    
    Args:
        data_file: Path đến file pickle
        batch_size: Batch size
        shuffle: Có shuffle data không
        num_workers: Số workers cho DataLoader
    
    Returns:
        DataLoader object
    """
    dataset = GraphCodeBERTDataset(data_file)
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        collate_fn=collate_batch,
        pin_memory=True if torch.cuda.is_available() else False
    )
    
    return dataloader


def load_datasets(processed_dir: str, batch_size: int = 32):
    """
    Load train/val/test datasets
    
    Args:
        processed_dir: Directory chứa train.pkl, val.pkl, test.pkl
        batch_size: Batch size
    
    Returns:
        dict với keys: 'train', 'val', 'test' dataloaders
    """
    import os
    
    dataloaders = {}
    
    for split in ['train', 'val', 'test']:
        data_file = os.path.join(processed_dir, f"{split}.pkl")
        
        if os.path.exists(data_file):
            shuffle = (split == 'train')  # Only shuffle training data
            dataloaders[split] = create_dataloader(
                data_file,
                batch_size=batch_size,
                shuffle=shuffle,
                num_workers=0
            )
            print(f"[+] Created {split} dataloader")
        else:
            print(f"[!] Warning: {data_file} not found")
    
    return dataloaders


if __name__ == "__main__":
    # Test loading
    import os
    
    processed_dir = "processed_graphcodebert"
    
    if os.path.exists(processed_dir):
        print("Testing dataloader...")
        dataloaders = load_datasets(processed_dir, batch_size=8)
        
        if 'train' in dataloaders:
            print("\nTesting train dataloader:")
            train_loader = dataloaders['train']
            
            # Get one batch
            batch = next(iter(train_loader))
            
            print(f"  Batch keys: {batch.keys()}")
            print(f"  input_ids shape: {batch['input_ids'].shape}")
            print(f"  attention_mask shape: {batch['attention_mask'].shape}")
            print(f"  dfg_matrix shape: {batch['dfg_matrix'].shape}")
            print(f"  labels shape: {batch['label'].shape}")
            print(f"  Labels in batch: {batch['label'].tolist()}")
            
            print("\n✅ Dataset loading works correctly!")
    else:
        print(f"[!] Please run graphcodebert_tokenizer.py first to create {processed_dir}/")

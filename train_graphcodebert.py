"""
Training Script cho GraphCodeBERT Vulnerability Detection
"""

import os
import json
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix, roc_auc_score
import warnings
warnings.filterwarnings('ignore')

from graphcodebert_model import create_model
from graphcodebert_dataset import load_datasets


# Configuration
CONFIG = {
    'model_type': 'simple',  # 'simple' or 'gnn'
    'model_name': 'microsoft/graphcodebert-base',
    'num_labels': 2,
    'use_dfg': True,
    
    # Training
    'batch_size': 16,
    'learning_rate': 2e-5,
    'num_epochs': 10,
    'warmup_steps': 100,
    'max_grad_norm': 1.0,
    'weight_decay': 0.01,
    
    # Paths
    'processed_dir': 'processed_graphcodebert',
    'output_dir': 'models/graphcodebert_vuln_detector',
    'log_dir': 'logs/graphcodebert',
    
    # Device
    'device': 'cuda' if torch.cuda.is_available() else 'cpu',
    'num_workers': 0
}


class Trainer:
    """Trainer class cho GraphCodeBERT"""
    
    def __init__(self, config):
        self.config = config
        self.device = torch.device(config['device'])
        
        print(f"\n{'='*70}")
        print(f"GRAPHCODEBERT VULNERABILITY DETECTION - TRAINING")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        
        # Create directories
        os.makedirs(config['output_dir'], exist_ok=True)
        os.makedirs(config['log_dir'], exist_ok=True)
        
        # Load datasets
        print(f"\n[+] Loading datasets from {config['processed_dir']}...")
        self.dataloaders = load_datasets(
            config['processed_dir'],
            batch_size=config['batch_size']
        )
        
        if 'train' not in self.dataloaders:
            raise ValueError("Training data not found!")
        
        # Create model
        print(f"\n[+] Creating {config['model_type']} model...")
        self.model = create_model(
            model_type=config['model_type'],
            model_name=config['model_name'],
            num_labels=config['num_labels'],
            use_dfg=config['use_dfg']
        )
        self.model.to(self.device)
        
        # Loss function
        self.criterion = nn.CrossEntropyLoss()
        
        # Optimizer
        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=config['weight_decay']
        )
        
        # Learning rate scheduler
        num_training_steps = len(self.dataloaders['train']) * config['num_epochs']
        self.scheduler = optim.lr_scheduler.LinearLR(
            self.optimizer,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=num_training_steps
        )
        
        # TensorBoard
        self.writer = SummaryWriter(config['log_dir'])
        
        # Tracking
        self.global_step = 0
        self.best_val_acc = 0.0
        self.best_val_f1 = 0.0
        
        print(f"\n[+] Trainer initialized")
        print(f"    Training samples: {len(self.dataloaders['train'].dataset)}")
        if 'val' in self.dataloaders:
            print(f"    Validation samples: {len(self.dataloaders['val'].dataset)}")
        if 'test' in self.dataloaders:
            print(f"    Test samples: {len(self.dataloaders['test'].dataset)}")
    
    def train_epoch(self, epoch):
        """Train one epoch"""
        self.model.train()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        
        pbar = tqdm(self.dataloaders['train'], desc=f"Epoch {epoch+1}/{self.config['num_epochs']}")
        
        for batch_idx, batch in enumerate(pbar):
            # Move to device
            input_ids = batch['input_ids'].to(self.device)
            attention_mask = batch['attention_mask'].to(self.device)
            labels = batch['label'].to(self.device)
            dfg_matrix = batch['dfg_matrix'].to(self.device) if self.config['use_dfg'] else None
            
            # Forward pass
            logits = self.model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                dfg_matrix=dfg_matrix
            )
            
            # Compute loss
            loss = self.criterion(logits, labels)
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config['max_grad_norm']
            )
            
            # Update weights
            self.optimizer.step()
            self.scheduler.step()
            
            # Track metrics
            total_loss += loss.item()
            preds = torch.argmax(logits, dim=-1)
            
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
            # Update progress bar
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'lr': f'{self.scheduler.get_last_lr()[0]:.2e}'
            })
            
            # Log to tensorboard
            if self.global_step % 10 == 0:
                self.writer.add_scalar('train/loss', loss.item(), self.global_step)
                self.writer.add_scalar('train/lr', self.scheduler.get_last_lr()[0], self.global_step)
            
            self.global_step += 1
        
        # Epoch metrics
        avg_loss = total_loss / len(self.dataloaders['train'])
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }
        
        return metrics
    
    def evaluate(self, split='val'):
        """Evaluate on validation or test set"""
        if split not in self.dataloaders:
            return None
        
        self.model.eval()
        
        total_loss = 0
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for batch in tqdm(self.dataloaders[split], desc=f"Evaluating {split}"):
                # Move to device
                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)
                labels = batch['label'].to(self.device)
                dfg_matrix = batch['dfg_matrix'].to(self.device) if self.config['use_dfg'] else None
                
                # Forward pass
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    dfg_matrix=dfg_matrix
                )
                
                # Compute loss
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                
                # Predictions
                probs = torch.softmax(logits, dim=-1)
                preds = torch.argmax(logits, dim=-1)
                
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs[:, 1].cpu().numpy())  # Probability of class 1
        
        # Compute metrics
        avg_loss = total_loss / len(self.dataloaders[split])
        accuracy = accuracy_score(all_labels, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            all_labels, all_preds, average='binary'
        )
        
        # AUC-ROC
        try:
            auc = roc_auc_score(all_labels, all_probs)
        except:
            auc = 0.0
        
        # Confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        metrics = {
            'loss': avg_loss,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'auc': auc,
            'confusion_matrix': cm.tolist()
        }
        
        return metrics
    
    def train(self):
        """Main training loop"""
        print(f"\n{'='*70}")
        print(f"STARTING TRAINING")
        print(f"{'='*70}\n")
        
        for epoch in range(self.config['num_epochs']):
            # Train
            train_metrics = self.train_epoch(epoch)
            
            # Evaluate
            val_metrics = None
            if 'val' in self.dataloaders:
                val_metrics = self.evaluate('val')
            
            # Log metrics
            print(f"\nEpoch {epoch+1}/{self.config['num_epochs']}:")
            print(f"  Train - Loss: {train_metrics['loss']:.4f}, "
                  f"Acc: {train_metrics['accuracy']:.4f}, "
                  f"F1: {train_metrics['f1']:.4f}")
            
            if val_metrics:
                print(f"  Val   - Loss: {val_metrics['loss']:.4f}, "
                      f"Acc: {val_metrics['accuracy']:.4f}, "
                      f"F1: {val_metrics['f1']:.4f}, "
                      f"AUC: {val_metrics['auc']:.4f}")
                
                # TensorBoard
                self.writer.add_scalar('val/loss', val_metrics['loss'], epoch)
                self.writer.add_scalar('val/accuracy', val_metrics['accuracy'], epoch)
                self.writer.add_scalar('val/f1', val_metrics['f1'], epoch)
                self.writer.add_scalar('val/auc', val_metrics['auc'], epoch)
                
                # Save best model
                if val_metrics['f1'] > self.best_val_f1:
                    self.best_val_f1 = val_metrics['f1']
                    self.best_val_acc = val_metrics['accuracy']
                    
                    save_path = os.path.join(self.config['output_dir'], 'best_model.pt')
                    torch.save({
                        'epoch': epoch,
                        'model_state_dict': self.model.state_dict(),
                        'optimizer_state_dict': self.optimizer.state_dict(),
                        'val_metrics': val_metrics,
                        'config': self.config
                    }, save_path)
                    
                    print(f"  ✅ Saved best model (F1: {val_metrics['f1']:.4f})")
            
            # Save checkpoint
            checkpoint_path = os.path.join(self.config['output_dir'], f'checkpoint_epoch_{epoch+1}.pt')
            torch.save({
                'epoch': epoch,
                'model_state_dict': self.model.state_dict(),
                'optimizer_state_dict': self.optimizer.state_dict(),
                'train_metrics': train_metrics,
                'val_metrics': val_metrics,
                'config': self.config
            }, checkpoint_path)
        
        print(f"\n{'='*70}")
        print(f"TRAINING COMPLETED")
        print(f"{'='*70}")
        print(f"Best Validation F1: {self.best_val_f1:.4f}")
        print(f"Best Validation Acc: {self.best_val_acc:.4f}")
        
        # Test evaluation
        if 'test' in self.dataloaders:
            print(f"\n[+] Evaluating on test set...")
            
            # Load best model
            best_model_path = os.path.join(self.config['output_dir'], 'best_model.pt')
            checkpoint = torch.load(best_model_path)
            self.model.load_state_dict(checkpoint['model_state_dict'])
            
            test_metrics = self.evaluate('test')
            
            print(f"\nTest Results:")
            print(f"  Accuracy:  {test_metrics['accuracy']:.4f}")
            print(f"  Precision: {test_metrics['precision']:.4f}")
            print(f"  Recall:    {test_metrics['recall']:.4f}")
            print(f"  F1 Score:  {test_metrics['f1']:.4f}")
            print(f"  AUC-ROC:   {test_metrics['auc']:.4f}")
            print(f"\nConfusion Matrix:")
            print(f"  {test_metrics['confusion_matrix']}")
            
            # Save test results
            results_path = os.path.join(self.config['output_dir'], 'test_results.json')
            with open(results_path, 'w') as f:
                json.dump(test_metrics, f, indent=2)
            
            print(f"\n✅ Saved test results to {results_path}")
        
        self.writer.close()


def main():
    """Main entry point"""
    trainer = Trainer(CONFIG)
    trainer.train()


if __name__ == "__main__":
    main()

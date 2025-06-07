#!/usr/bin/env python3
"""
BERT Fine-tuning Training Script for Vietnamese Fake News Detection
with Multimodal Fusion and Domain Regularization
"""

import os
# Set PyTorch CUDA memory allocation configuration for better memory management
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import os
import json
from datetime import datetime
from torch.cuda.amp import autocast, GradScaler

from bert_fine_tuner import (
    create_multimodal_bert_model, 
    prepare_bert_optimizer
)
from bert_dataset import (
    create_bert_data_splits,
    create_bert_data_loaders,
    balance_bert_dataset
)
from data_loader import load_and_preprocess_real_data
from data_balancer import apply_smotetomek_balancing
from results_saver import save_comprehensive_results
from config import DEVICE

# Enable cuDNN autotuner
torch.backends.cudnn.benchmark = True

def print_gpu_utilization():
    """Print GPU memory usage"""
    print(f"GPU Memory Used: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    print(f"GPU Memory Cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

def train_bert_model(model, train_loader, val_loader, 
                    num_epochs=3, learning_rate=2e-5,
                    gradient_accumulation_steps=8,
                    warmup_ratio=0.1, weight_decay=0.01,
                    domain_penalty_weight=0.5,
                    save_dir='bert_models',
                    model_name='multimodal_bert'):
    """
    Fine-tune BERT model for fake news detection
    
    Args:
        model: MultimodalBERTFusionModel
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        num_epochs: Number of training epochs (3-5 for BERT)
        learning_rate: Learning rate (2e-5 typical for BERT)
        warmup_ratio: Warmup ratio for scheduler
        weight_decay: Weight decay
        domain_penalty_weight: Weight for domain regularization penalty
        save_dir: Directory to save models
        model_name: Name for saving models
    """
    
    print(f"\nSTARTING BERT FINE-TUNING")
    print("="*60)
    print(f"Model: {model.__class__.__name__}")
    print(f"Device: {DEVICE}")
    print(f"Epochs: {num_epochs}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Batch Size: {train_loader.batch_size}")
    print(f"Gradient Accumulation Steps: {gradient_accumulation_steps}")
    print(f"Effective Batch Size: {train_loader.batch_size * gradient_accumulation_steps}")
    print(f"Training Samples: {len(train_loader.dataset)}")
    print(f"Validation Samples: {len(val_loader.dataset)}")
    print(f"Training Steps per Epoch: {len(train_loader)}")
    print(f"Optimizer Updates per Epoch: {len(train_loader) // gradient_accumulation_steps}")
    
    # Print initial GPU utilization
    if torch.cuda.is_available():
        print("\nInitial GPU Memory Usage:")
        print_gpu_utilization()
    
    # Create save directory
    os.makedirs(save_dir, exist_ok=True)
    
    # Setup optimizer and scheduler
    # Calculate actual number of optimizer updates (considering gradient accumulation)
    updates_per_epoch = len(train_loader) // gradient_accumulation_steps
    total_steps = updates_per_epoch * num_epochs
    warmup_steps = int(total_steps * warmup_ratio)
    
    print(f"Scheduler Configuration:")
    print(f"  Updates per epoch: {updates_per_epoch}")
    print(f"  Total optimizer updates: {total_steps}")
    print(f"  Warmup steps: {warmup_steps}")
    
    optimizer, scheduler = prepare_bert_optimizer(
        model, learning_rate, warmup_steps, total_steps
    )
    
    # Initialize gradient scaler for mixed precision training
    scaler = GradScaler()
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Training tracking
    train_losses = []
    val_accuracies = []
    val_f1_scores = []
    domain_penalties = []
    best_val_f1 = 0.0
    best_epoch = 0
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\n📅 EPOCH {epoch+1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        total_train_loss = 0
        total_domain_penalty = 0
        train_predictions = []
        train_labels = []
        
        # Gradient accumulation tracking
        accumulated_loss = 0.0
        accumulated_domain_penalty = 0.0
        optimizer.zero_grad()  # Initialize gradients to zero at start of epoch
        
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch+1}")
        
        for batch_idx, batch in enumerate(train_pbar):
            # Get batch data
            titles = batch['titles']
            summaries = batch['summaries']
            contents = batch['contents']
            labels = batch['labels'].to(DEVICE)
            domains = batch.get('domains', None)
            if domains is not None:
                domains = domains.to(DEVICE)
            
            # Forward pass with mixed precision
            with autocast():
                outputs = model(titles, summaries, contents, domains)
                main_loss = criterion(outputs, labels)
                
                # Domain regularization penalty
                domain_penalty = 0.0
                if hasattr(model, 'get_current_penalty'):
                    domain_penalty = model.get_current_penalty()
                    if isinstance(domain_penalty, torch.Tensor):
                        domain_penalty = domain_penalty.item()
                
                # Combined loss
                total_loss = main_loss + domain_penalty_weight * domain_penalty
                
                # ⬆️ GRADIENT ACCUMULATION: Divide loss by accumulation steps
                total_loss = total_loss / gradient_accumulation_steps
            
            # Backward pass with gradient scaling
            scaler.scale(total_loss).backward()
            
            # Accumulate loss for tracking
            accumulated_loss += total_loss.item()
            accumulated_domain_penalty += domain_penalty / gradient_accumulation_steps
            
            # Track predictions for metrics
            _, predicted = torch.max(outputs, 1)
            train_predictions.extend(predicted.cpu().numpy())
            train_labels.extend(labels.cpu().numpy())
            
            # ⬆️ GRADIENT ACCUMULATION: Only update optimizer after accumulating enough gradients
            if (batch_idx + 1) % gradient_accumulation_steps == 0:
                # Gradient clipping
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # Optimizer step with gradient scaling
                scaler.step(optimizer)
                scaler.update()
                scheduler.step()
                
                # Reset gradients for next accumulation
                optimizer.zero_grad()
                
                # Track accumulated metrics
                total_train_loss += accumulated_loss
                total_domain_penalty += accumulated_domain_penalty
                
                # Reset accumulation trackers
                accumulated_loss = 0.0
                accumulated_domain_penalty = 0.0
            
            # Update progress bar with current batch metrics
            current_loss = total_loss.item() * gradient_accumulation_steps  # Show original scale
            train_pbar.set_postfix({
                'Loss': f'{current_loss:.4f}',
                'Domain_Penalty': f'{domain_penalty:.6f}',
                'LR': f'{scheduler.get_last_lr()[0]:.2e}',
                'Step': f'{(batch_idx + 1) % gradient_accumulation_steps}/{gradient_accumulation_steps}'
            })
            
            # ⬆️ Clear GPU cache periodically to prevent memory buildup
            if batch_idx % 10 == 0:
                torch.cuda.empty_cache()
            
            # Print GPU utilization every 100 batches
            if batch_idx % 100 == 0 and torch.cuda.is_available():
                print(f"\nBatch {batch_idx}: GPU Memory Usage:")
                print_gpu_utilization()
        
        # Handle remaining accumulated gradients at end of epoch
        if (len(train_loader)) % gradient_accumulation_steps != 0:
            # Gradient clipping
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            # Final optimizer step
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            
            # Track remaining accumulated metrics
            total_train_loss += accumulated_loss
            total_domain_penalty += accumulated_domain_penalty
            
            optimizer.zero_grad()
        
        # Calculate training metrics
        # Adjust for gradient accumulation - divide by number of actual updates, not batches
        num_updates = len(train_loader) // gradient_accumulation_steps
        if len(train_loader) % gradient_accumulation_steps != 0:
            num_updates += 1  # Account for final partial accumulation
            
        avg_train_loss = total_train_loss / num_updates if num_updates > 0 else 0
        avg_domain_penalty = total_domain_penalty / num_updates if num_updates > 0 else 0
        train_accuracy = accuracy_score(train_labels, train_predictions)
        train_f1 = f1_score(train_labels, train_predictions, average='weighted')
        
        train_losses.append(avg_train_loss)
        domain_penalties.append(avg_domain_penalty)
        
        print(f"Training - Loss: {avg_train_loss:.4f}, Accuracy: {train_accuracy:.4f}, F1: {train_f1:.4f}")
        print(f"Domain Penalty: {avg_domain_penalty:.6f}")
        
        # Validation phase
        model.eval()
        val_predictions = []
        val_labels_list = []
        val_loss = 0.0
        
        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc=f"Validation Epoch {epoch+1}")
            
            for batch in val_pbar:
                titles = batch['titles']
                summaries = batch['summaries']
                contents = batch['contents']
                labels = batch['labels'].to(DEVICE)
                domains = batch.get('domains', None)
                if domains is not None:
                    domains = domains.to(DEVICE)
                
                # Use mixed precision for inference too
                with autocast():
                    outputs = model(titles, summaries, contents, domains)
                    loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                val_predictions.extend(predicted.cpu().numpy())
                val_labels_list.extend(labels.cpu().numpy())
        
        # Calculate validation metrics
        avg_val_loss = val_loss / len(val_loader)
        val_accuracy = accuracy_score(val_labels_list, val_predictions)
        val_f1 = f1_score(val_labels_list, val_predictions, average='weighted')
        val_f1_fake = f1_score(val_labels_list, val_predictions, pos_label=1)
        
        val_accuracies.append(val_accuracy)
        val_f1_scores.append(val_f1)
        
        print(f"Validation - Loss: {avg_val_loss:.4f}, Accuracy: {val_accuracy:.4f}")
        print(f"F1 (Weighted): {val_f1:.4f}, F1 (Fake): {val_f1_fake:.4f}")
        
        # Save best model
        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            best_epoch = epoch
            
            # Save model
            model_save_path = os.path.join(save_dir, f"{model_name}_best.pt")
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'scaler_state_dict': scaler.state_dict(),  # Save scaler state
                'val_f1': val_f1,
                'val_accuracy': val_accuracy,
                'config': {
                    'learning_rate': learning_rate,
                    'batch_size': train_loader.batch_size,
                    'num_epochs': num_epochs,
                    'warmup_ratio': warmup_ratio,
                    'weight_decay': weight_decay,
                    'domain_penalty_weight': domain_penalty_weight
                }
            }, model_save_path)
            print(f"Best model saved: {model_save_path}")
        
        # Print GPU utilization at end of epoch
        if torch.cuda.is_available():
            print("\nEnd of Epoch GPU Memory Usage:")
            print_gpu_utilization()
    
    print(f"\nTraining completed!")
    print(f"Best validation F1: {best_val_f1:.4f} (Epoch {best_epoch+1})")
    
    return {
        'train_losses': train_losses,
        'val_accuracies': val_accuracies,
        'val_f1_scores': val_f1_scores,
        'domain_penalties': domain_penalties,
        'best_val_f1': best_val_f1,
        'best_epoch': best_epoch
    }

def evaluate_bert_model(model, test_loader, model_path=None):
    """Evaluate BERT model on test set with memory optimization"""
    import gc
    
    # 🛠️ MEMORY OPTIMIZATION: Clear GPU cache before evaluation
    if torch.cuda.is_available():
        print("💾 Clearing GPU cache before evaluation...")
        torch.cuda.empty_cache()
        gc.collect()
        print("GPU Memory before evaluation:")
        print_gpu_utilization()
    
    if model_path:
        print(f"Loading model from {model_path}")
        try:
            # 🛠️ MEMORY OPTIMIZATION: Load checkpoint to CPU first to avoid GPU memory issues
            print("Loading checkpoint to CPU first...")
            checkpoint = torch.load(model_path, map_location='cpu')
            
            # Load state dict to model
            model.load_state_dict(checkpoint['model_state_dict'])
            
            # Move model to device after loading
            model = model.to(DEVICE)
            
            # Clear checkpoint from memory
            del checkpoint
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()
            
            print("✅ Model loaded successfully with memory optimization")
            
        except torch.cuda.OutOfMemoryError as e:
            print(f"❌ CUDA Out of Memory Error: {str(e)}")
            print("🔄 Attempting CPU-only evaluation...")
            
            # Load model on CPU and switch to CPU evaluation
            checkpoint = torch.load(model_path, map_location='cpu')
            model = model.to('cpu')
            model.load_state_dict(checkpoint['model_state_dict'])
            
            print("⚠️  Switching to CPU evaluation due to memory constraints")
            
        except Exception as e:
            print(f"❌ Error loading model: {str(e)}")
            print("🔄 Attempting alternative loading method...")
            
            # Alternative loading with weights_only
            try:
                checkpoint = torch.load(model_path, map_location='cpu', weights_only=True)
                model.load_state_dict(checkpoint)
                model = model.to(DEVICE)
                print("✅ Model loaded with alternative method")
            except Exception as e2:
                print(f"❌ Failed with alternative method: {str(e2)}")
                raise e2
    
    model.eval()
    predictions = []
    labels_list = []
    probabilities = []
    
    print("🔍 Evaluating on test set...")
    
    # 🛠️ MEMORY OPTIMIZATION: Use gradient checkpointing for evaluation if available
    if hasattr(model, 'gradient_checkpointing_enable'):
        model.gradient_checkpointing_enable()
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(tqdm(test_loader, desc="Testing")):
            try:
                titles = batch['titles']
                summaries = batch['summaries']
                contents = batch['contents']
                
                # Determine device based on model's current device
                current_device = next(model.parameters()).device
                labels = batch['labels'].to(current_device)
                domains = batch.get('domains', None)
                if domains is not None:
                    domains = domains.to(current_device)
                
                outputs = model(titles, summaries, contents, domains)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                predictions.extend(predicted.cpu().numpy())
                labels_list.extend(labels.cpu().numpy())
                probabilities.extend(probs.cpu().numpy())
                
                # 🛠️ MEMORY OPTIMIZATION: Periodic cache clearing during evaluation
                if batch_idx % 100 == 0 and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    
            except torch.cuda.OutOfMemoryError as e:
                print(f"⚠️  Memory error at batch {batch_idx}, clearing cache and retrying...")
                torch.cuda.empty_cache()
                gc.collect()
                
                # Retry the batch
                try:
                    outputs = model(titles, summaries, contents, domains)
                    probs = F.softmax(outputs, dim=1)
                    _, predicted = torch.max(outputs, 1)
                    
                    predictions.extend(predicted.cpu().numpy())
                    labels_list.extend(labels.cpu().numpy())
                    probabilities.extend(probs.cpu().numpy())
                except Exception as retry_error:
                    print(f"❌ Failed to process batch {batch_idx}: {str(retry_error)}")
                    continue
    
    # Calculate metrics
    accuracy = accuracy_score(labels_list, predictions)
    f1_macro = f1_score(labels_list, predictions, average='macro')
    f1_weighted = f1_score(labels_list, predictions, average='weighted')
    f1_fake = f1_score(labels_list, predictions, pos_label=1)
    
    # Classification report
    report = classification_report(
        labels_list, predictions, 
        target_names=['Real News', 'Fake News'],
        output_dict=True
    )
    
    print(f"\nTEST RESULTS:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"F1-Macro: {f1_macro:.4f}")
    print(f"F1-Weighted: {f1_weighted:.4f}")
    print(f"F1-Fake News: {f1_fake:.4f}")
    
    print(f"\nDetailed Classification Report:")
    print(classification_report(
        labels_list, predictions,
        target_names=['Real News', 'Fake News']
    ))
    
    return {
        'accuracy': accuracy,
        'f1_macro': f1_macro,
        'f1_weighted': f1_weighted,
        'f1_fake': f1_fake,
        'predictions': predictions,
        'labels': labels_list,
        'probabilities': probabilities,
        'classification_report': report
    }

def plot_training_history(history, save_path=None):
    """Plot training history"""
    
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    
    # Training loss
    axes[0, 0].plot(history['train_losses'], 'b-', label='Training Loss', linewidth=2)
    axes[0, 0].set_title('Training Loss', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Loss')
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].legend()
    
    # Validation accuracy
    axes[0, 1].plot(history['val_accuracies'], 'g-', label='Validation Accuracy', linewidth=2)
    axes[0, 1].set_title('Validation Accuracy', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Accuracy')
    axes[0, 1].grid(True, alpha=0.3)
    axes[0, 1].legend()
    
    # Validation F1
    axes[1, 0].plot(history['val_f1_scores'], 'r-', label='Validation F1', linewidth=2)
    axes[1, 0].set_title('Validation F1-Score', fontsize=14, fontweight='bold')
    axes[1, 0].set_xlabel('Epoch')
    axes[1, 0].set_ylabel('F1-Score')
    axes[1, 0].grid(True, alpha=0.3)
    axes[1, 0].legend()
    
    # Domain penalties
    axes[1, 1].plot(history['domain_penalties'], 'm-', label='Domain Penalty', linewidth=2)
    axes[1, 1].set_title('Domain Regularization Penalty', fontsize=14, fontweight='bold')
    axes[1, 1].set_xlabel('Epoch')
    axes[1, 1].set_ylabel('Penalty')
    axes[1, 1].grid(True, alpha=0.3)
    axes[1, 1].legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"Training history plot saved to: {save_path}")
    
    plt.show()

def main_bert_training(real_file_path, fake_file_path,
                      bert_model='vinai/phobert-base',
                      fusion_type='attention',
                      use_domain=True,
                      num_epochs=3,
                      batch_size=8,
                      learning_rate=2e-5,
                      freeze_bert_layers=0,
                      balance_strategy='smotetomek',
                      balance_target_ratio=0.65,
                      save_dir='bert_experiments'):
    """
    Main function for BERT fine-tuning experiment
    """
    
    print("STARTING BERT FINE-TUNING EXPERIMENT")
    print("="*80)
    
    # Create experiment directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(save_dir, f"bert_{fusion_type}_{timestamp}")
    os.makedirs(experiment_dir, exist_ok=True)
    
    # 1. Load and preprocess data
    print("\n1. Loading and preprocessing data...")
    df, domain_features, domain_dim = load_and_preprocess_real_data(
        real_file_path, fake_file_path
    )
    
    # 1.5. Apply SMOTETomek balancing if requested
    balance_info = None
    if balance_strategy == 'smotetomek':
        print("\n1.5. Applying SMOTETomek balancing...")
        df_balanced, balance_info = apply_smotetomek_balancing(
            df, target_ratio=balance_target_ratio, random_state=42
        )
        
        # Reprocess domain features for balanced data
        if use_domain:
            from data_loader import RealDataLoader
            temp_loader = RealDataLoader()
            temp_loader.df = df_balanced
            domain_features, domain_dim = temp_loader.process_domains()
        
        df = df_balanced
        print(f"Data balanced: {len(df)} samples")
    
    # 2. Create datasets
    print("\n2. Creating BERT datasets...")
    train_dataset, val_dataset, test_dataset = create_bert_data_splits(
        df, domain_features if use_domain else None, include_domain=use_domain
    )
    
    # 3. Balance training data (if not already balanced with SMOTETomek)
    class_weights = None
    if balance_strategy == 'undersample':
        print("\n3. Balancing training data...")
        train_dataset = balance_bert_dataset(train_dataset, balance_strategy)
    elif balance_strategy == 'weighted':
        class_weights = balance_bert_dataset(train_dataset, balance_strategy)
    elif balance_strategy == 'smotetomek':
        print("\n3. Using SMOTETomek balanced data (no additional balancing)")
    else:
        print("\n3. No additional data balancing applied")
    
    # 4. Create data loaders
    print("\n4. Creating data loaders...")
    train_loader, val_loader, test_loader = create_bert_data_loaders(
        train_dataset, val_dataset, test_dataset, batch_size=batch_size
    )
    
    # 5. Create model
    print("\n5. Creating BERT fusion model...")
    model = create_multimodal_bert_model(
        domain_dim=domain_dim if use_domain else None,
        bert_model=bert_model,
        fusion_type=fusion_type,
        freeze_bert_layers=freeze_bert_layers
    )
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # 6. Train model
    print("\n6. Training model...")
    
    # Import gradient accumulation steps from config
    from config import TRAINING_CONFIG
    gradient_accumulation_steps = TRAINING_CONFIG.get('gradient_accumulation_steps', 8)
    
    history = train_bert_model(
        model, train_loader, val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        gradient_accumulation_steps=gradient_accumulation_steps,
        save_dir=experiment_dir,
        model_name=f'bert_{fusion_type}'
    )
    
    # 7. Evaluate on test set
    print("\n7. Evaluating on test set...")
    
    # 🛠️ MEMORY OPTIMIZATION: Comprehensive cleanup before evaluation
    print("💾 Performing comprehensive memory cleanup before evaluation...")
    import gc
    
    # Move model to CPU to free GPU memory
    model = model.cpu()
    
    # Clear all CUDA caches
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
    
    # Force garbage collection
    gc.collect()
    
    # Print memory status
    if torch.cuda.is_available():
        print("Memory status after cleanup:")
        print_gpu_utilization()
    
    best_model_path = os.path.join(experiment_dir, f'bert_{fusion_type}_best.pt')
    test_results = evaluate_bert_model(model, test_loader, best_model_path)
    
    # 7.5. Save comprehensive results
    print("\n7.5. Saving comprehensive results...")
    experiment_config = {
        'bert_model': bert_model,
        'fusion_type': fusion_type,
        'use_domain': use_domain,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'freeze_bert_layers': freeze_bert_layers,
        'balance_strategy': balance_strategy,
        'balance_target_ratio': balance_target_ratio if balance_strategy == 'smotetomek' else None,
        'dataset_size': len(df),
        'real_file_path': real_file_path,
        'fake_file_path': fake_file_path
    }
    
    detailed_metrics = save_comprehensive_results(
        test_results=test_results,
        save_dir=experiment_dir,
        model_name=f'bert_{fusion_type}',
        history=history,
        balance_info=balance_info,
        experiment_config=experiment_config
    )
    
    # 8. Save results and plots
    print("\n8. Saving results...")
    
    # Plot training history
    plot_path = os.path.join(experiment_dir, 'training_history.png')
    plot_training_history(history, plot_path)
    
    # Save test results
    results_path = os.path.join(experiment_dir, 'test_results.json')
    # Convert numpy arrays to lists for JSON serialization
    test_results_json = {
        'accuracy': test_results['accuracy'],
        'f1_macro': test_results['f1_macro'],
        'f1_weighted': test_results['f1_weighted'],
        'f1_fake': test_results['f1_fake'],
        'classification_report': test_results['classification_report']
    }
    
    with open(results_path, 'w') as f:
        json.dump(test_results_json, f, indent=2)
    
    # Save experiment config
    config = {
        'bert_model': bert_model,
        'fusion_type': fusion_type,
        'use_domain': use_domain,
        'num_epochs': num_epochs,
        'batch_size': batch_size,
        'learning_rate': learning_rate,
        'freeze_bert_layers': freeze_bert_layers,
        'balance_strategy': balance_strategy,
        'dataset_size': len(df),
        'real_file_path': real_file_path,
        'fake_file_path': fake_file_path
    }
    
    config_path = os.path.join(experiment_dir, 'experiment_config.json')
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    
    print(f"\nExperiment completed! Results saved to: {experiment_dir}")
    
    return test_results, history

if __name__ == "__main__":
    # Example usage
    REAL_FILE_PATH = "path/to/your/real_articles.csv"
    FAKE_FILE_PATH = "path/to/your/fake_articles.csv"
    
    # Run BERT fine-tuning experiment
    test_results, history = main_bert_training(
        real_file_path=REAL_FILE_PATH,
        fake_file_path=FAKE_FILE_PATH,
        bert_model='vinai/phobert-base',  
        fusion_type='attention',  # 'concat', 'attention', 'gated'
        use_domain=True,
        num_epochs=3,
        batch_size=8,
        learning_rate=2e-5,
        freeze_bert_layers=0,  # 0 = train all layers, 6 = freeze first 6 layers
        balance_strategy='weighted'  # 'undersample', 'weighted', None
    ) 
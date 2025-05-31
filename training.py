import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from config import DEVICE, MODEL_CONFIG

def train_model(model, train_loader, val_loader=None, num_epochs=None, learning_rate=None):
    """Train the model"""
    
    # Use config defaults if not provided
    num_epochs = num_epochs or MODEL_CONFIG['num_epochs']
    learning_rate = learning_rate or MODEL_CONFIG['learning_rate']
    
    # Setup training components
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr=learning_rate, 
        weight_decay=MODEL_CONFIG['weight_decay']
    )
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # Move model to device
    model.to(DEVICE)
    
    # Training tracking
    train_losses = []
    val_accuracies = []
    
    print(f"Training on {DEVICE}")
    print(f"Epochs: {num_epochs}, Learning Rate: {learning_rate}")
    
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in train_loader:
            optimizer.zero_grad()
            
            # Move data to device
            title = batch['title'].to(DEVICE)
            summary = batch['summary'].to(DEVICE)
            content = batch['content'].to(DEVICE)
            domain = batch['domain'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            # Forward pass
            outputs = model(title, summary, content, domain)
            loss = criterion(outputs, labels)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        avg_loss = total_loss / num_batches
        train_losses.append(avg_loss)
        
        # Validation phase
        val_accuracy = 0
        if val_loader:
            val_accuracy = evaluate_accuracy(model, val_loader)
            val_accuracies.append(val_accuracy)
        
        scheduler.step()
        
        # Print progress
        if epoch % 5 == 0 or epoch == num_epochs - 1:
            print(f'Epoch [{epoch+1}/{num_epochs}]')
            print(f'  Loss: {avg_loss:.4f}')
            if val_loader:
                print(f'  Val Accuracy: {val_accuracy:.2f}%')
    
    return train_losses, val_accuracies

def evaluate_accuracy(model, data_loader):
    """Evaluate model accuracy"""
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in data_loader:
            title = batch['title'].to(DEVICE)
            summary = batch['summary'].to(DEVICE)
            content = batch['content'].to(DEVICE)
            domain = batch['domain'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(title, summary, content, domain)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    return 100 * correct / total

def evaluate_model(model, test_loader):
    """Comprehensive model evaluation"""
    model.eval()
    all_predictions = []
    all_labels = []
    all_probabilities = []
    
    with torch.no_grad():
        for batch in test_loader:
            title = batch['title'].to(DEVICE)
            summary = batch['summary'].to(DEVICE)
            content = batch['content'].to(DEVICE)
            domain = batch['domain'].to(DEVICE)
            labels = batch['label'].to(DEVICE)
            
            outputs = model(title, summary, content, domain)
            probabilities = F.softmax(outputs, dim=1)
            _, predicted = torch.max(outputs, 1)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probabilities.extend(probabilities.cpu().numpy())
    
    return np.array(all_predictions), np.array(all_labels), np.array(all_probabilities)

def calculate_metrics(predictions, labels, probabilities):
    """Calculate evaluation metrics"""
    accuracy = accuracy_score(labels, predictions)
    auc_roc = roc_auc_score(labels, probabilities[:, 1])
    
    metrics = {
        'accuracy': accuracy,
        'auc_roc': auc_roc,
        'classification_report': classification_report(
            labels, predictions, target_names=['Real', 'Fake'], output_dict=True
        ),
        'confusion_matrix': confusion_matrix(labels, predictions)
    }
    
    return metrics

def print_evaluation_results(model_name, metrics):
    """Print evaluation results"""
    print(f"\n{'='*50}")
    print(f"{model_name} Results:")
    print(f"{'='*50}")
    print(f"Accuracy: {metrics['accuracy']:.4f}")
    print(f"AUC-ROC: {metrics['auc_roc']:.4f}")
    print("\nClassification Report:")
    
    # Print classification report in readable format
    report = metrics['classification_report']
    print(f"{'Class':<10} {'Precision':<10} {'Recall':<10} {'F1-Score':<10}")
    print("-" * 45)
    for class_name in ['Real', 'Fake']:
        class_key = '0' if class_name == 'Real' else '1'
        if class_key in report:
            precision = report[class_key]['precision']
            recall = report[class_key]['recall']
            f1 = report[class_key]['f1-score']
            print(f"{class_name:<10} {precision:<10.3f} {recall:<10.3f} {f1:<10.3f}")
    
    print(f"\nOverall F1-Score: {report['macro avg']['f1-score']:.4f}")

if __name__ == "__main__":
    print("Training utilities imported successfully!")
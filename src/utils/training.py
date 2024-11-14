# src/utils/training.py
import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle

class EarlyStopping:
    def __init__(self, patience=12, min_delta=0.0008):  
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = None
        self.best_acc = None  
        self.early_stop = False
        
    def step(self, val_loss, val_acc): 
        if self.best_loss is None:
            self.best_loss = val_loss
            self.best_acc = val_acc
        elif val_loss > self.best_loss - self.min_delta and val_acc <= self.best_acc:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = min(val_loss, self.best_loss)
            self.best_acc = max(val_acc, self.best_acc)
            self.counter = 0
        return self.early_stop

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler, max_grad_norm=1.0):
    model.train()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    for batch in tqdm(train_loader, desc='Training'):
        x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
        if len(x2.shape) == 3:
            x2 = x2.squeeze(1)
            
        y1 = batch['y1'].to(device).squeeze()
        y2 = batch['y2'].to(device).squeeze()
        
        y1 = y1.long()
        y2 = y2.long()
        
       
        emb1, emb2, logits1, logits2 = model(x1, x2)
        
        
        loss = criterion(emb1, emb2, logits1, logits2, y1, y2)
        
     
        optimizer.zero_grad()
        loss.backward()
        
        
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        scheduler.step()
        
        total_loss += loss.item()
        
        _, preds = torch.max(logits1, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(y1.cpu().numpy())
    
    return total_loss / len(train_loader), all_preds, all_labels

def evaluate(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in data_loader:
            x1, x2 = batch['x1'].to(device), batch['x2'].to(device)
           
            if len(x2.shape) == 3:
                x2 = x2.squeeze(1)
                
    
            y1 = batch['y1'].to(device).squeeze()  
            y2 = batch['y2'].to(device).squeeze()  
            
        
            y1 = y1.long()
            y2 = y2.long()
            
            emb1, emb2, logits1, logits2 = model(x1, x2)
            loss = criterion(emb1, emb2, logits1, logits2, y1, y2)
            
            total_loss += loss.item()
            
            _, preds = torch.max(logits1, 1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y1.cpu().numpy())
    
    return total_loss / len(data_loader), all_preds, all_labels

def plot_metrics(train_losses, val_losses, train_accs, val_accs):
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.title('Loss over epochs')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.title('Accuracy over epochs')
    plt.legend()
    
    plt.tight_layout()
    plt.show()

def plot_confusion_matrix(y_true, y_pred, classes):
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.show()

def train_model(processed_data, num_epochs=100, batch_size=128, lr=0.001):
    from src.models.snn import SNN, CombinedLoss
    from src.data.dataset import create_dataloaders
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Data dimension check
    print("\nData dimension check:")
    print(f"Feature shape: {processed_data['X_train'].shape}")
    print(f"Label shape: {processed_data['y_train'].shape}")
    print(f"Number of classes: {len(processed_data['classes'])}")

    train_loader, val_loader, test_loader = create_dataloaders(processed_data, batch_size)

    # Check first batch dimensions
    for batch in train_loader:
        print("\nFirst batch dimensions:")
        print(f"x1 shape: {batch['x1'].shape}")
        print(f"x2 shape: {batch['x2'].shape}")
        print(f"y1 shape: {batch['y1'].shape}")
        print(f"y2 shape: {batch['y2'].shape}")
        break
    
    # Create model
    input_size = processed_data['feature_shape'][0]
    num_classes = len(processed_data['classes'])
    print(f"\nModel configuration:")
    print(f"Input dimension: {input_size}")
    print(f"Number of classes: {num_classes}")
    
    model = SNN(input_size=input_size,
                hidden_size=192,
                num_classes=num_classes).to(device)
    
    # Define loss function and optimizer
    criterion = CombinedLoss(
        margin=2.0, 
        alpha=0.65
    )
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=lr, 
        weight_decay=0.02,
        betas=(0.9, 0.999)
    )
    
    # Use OneCycleLR scheduler
    steps_per_epoch = len(train_loader)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        epochs=num_epochs,
        steps_per_epoch=steps_per_epoch,
        pct_start=0.25,
        div_factor=20,
        final_div_factor=1e3
    )
    
    # Enhanced early stopping strategy
    early_stopping = EarlyStopping(patience=15, min_delta=0.0005)
    
    # Add gradient clipping parameter
    max_grad_norm = 1.0
    
    # Training history
    train_losses, val_losses = [], []
    train_accs, val_accs = [], []
    best_val_acc = 0.0
    best_val_loss = float('inf')
    
    # Training loop
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # Train and validate
        model.train()
        train_loss, train_preds, train_labels = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler, max_grad_norm)
        train_acc = np.mean(np.array(train_preds) == np.array(train_labels))
        
        model.eval()
        val_loss, val_preds, val_labels = evaluate(
            model, val_loader, criterion, device)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        
        # Improved model saving condition
        if (val_acc > best_val_acc) or \
           (val_acc >= best_val_acc and val_loss < best_val_loss * 0.995):
            best_val_acc = val_acc
            best_val_loss = val_loss
            
            # Get model configuration
            model_config = {
                'input_size': model.input_size if hasattr(model, 'input_size') else input_size,
                'hidden_size': model.hidden_size if hasattr(model, 'hidden_size') else 192,
                'num_classes': model.num_classes if hasattr(model, 'num_classes') else num_classes,
                'alpha': getattr(criterion, 'alpha', 0.65),
                'margin': getattr(criterion, 'margin', 2.0)
            }
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'train_acc': train_acc,
                'train_loss': train_loss,
                'model_config': model_config
            }, 'best_model.pth')
            print(f"Saved new best model with val_acc: {val_acc:.4f}, val_loss: {val_loss:.4f}")
        
        # Record metrics
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # Print current status
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current LR: {current_lr:.6f}")
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
        
        # Check early stopping
        if early_stopping.step(val_loss, val_acc):
            print(f"Early stopping triggered at epoch {epoch+1}")
            break
    
    # Plot training process
    plot_metrics(train_losses, val_losses, train_accs, val_accs)
    
    # Load best model for testing
    checkpoint = torch.load('best_model.pth')
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"\nLoaded best model from epoch {checkpoint['epoch']+1}")
    
    # Test set evaluation
    test_loss, test_preds, test_labels = evaluate(
        model, test_loader, criterion, device)
    test_acc = np.mean(np.array(test_preds) == np.array(test_labels))
    
    print("\nTest Results:")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print("\nClassification Report:")
    print(classification_report(test_labels, test_preds, 
                              target_names=processed_data['classes']))
    
    # Plot confusion matrix
    plot_confusion_matrix(test_labels, test_preds, processed_data['classes'])
    
    return model, {
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'best_epoch': checkpoint['epoch'],
        'best_val_acc': checkpoint['val_acc'],
        'test_acc': test_acc,
        'test_loss': test_loss
    }
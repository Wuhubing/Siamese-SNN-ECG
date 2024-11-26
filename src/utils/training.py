import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import pickle
from datetime import datetime
import os

from src.models.snn import SNN, CombinedLoss
from src.data.dataset import create_dataloaders

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

def train_epoch(model, train_loader, criterion, optimizer, device, scheduler=None, max_grad_norm=1.0):
    """训练一个epoch"""
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
        
        # 前向传播
        embedding1, embedding2, logits1, logits2 = model(x1, x2)
        
        # 计算损失
        loss = criterion(embedding1, embedding2, logits1, logits2, y1, y2)
        
        # 反向传播
        optimizer.zero_grad()
        loss.backward()
        
        # 梯度裁剪
        if max_grad_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
        
        optimizer.step()
        if scheduler is not None:
            scheduler.step()
        
        # 记录损失和预测
        total_loss += loss.item()
        _, preds1 = torch.max(logits1, 1)
        _, preds2 = torch.max(logits2, 1)
        all_preds.extend(preds1.cpu().numpy())
        all_preds.extend(preds2.cpu().numpy())
        all_labels.extend(y1.cpu().numpy())
        all_labels.extend(y2.cpu().numpy())
    
    # 计算平均损失和准确率
    avg_loss = total_loss / len(train_loader)
    accuracy = np.mean(np.array(all_preds) == np.array(all_labels))
    
    return avg_loss, accuracy, all_preds, all_labels

def evaluate(model, data_loader, criterion, device):
    """评估模型"""
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
            
            # 前向传播
            embedding1, embedding2, logits1, logits2 = model(x1, x2)
            
            # 计算损失
            loss = criterion(embedding1, embedding2, logits1, logits2, y1, y2)
            
            # 记录损失和预测
            total_loss += loss.item()
            _, preds1 = torch.max(logits1, 1)
            _, preds2 = torch.max(logits2, 1)
            all_preds.extend(preds1.cpu().numpy())
            all_preds.extend(preds2.cpu().numpy())
            all_labels.extend(y1.cpu().numpy())
            all_labels.extend(y2.cpu().numpy())
    
    # 计算平均损失
    avg_loss = total_loss / len(data_loader)
    
    return avg_loss, all_preds, all_labels

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

def train_model(processed_data, model_name='default', num_epochs=100, batch_size=128, lr=0.001):
    """训练模型并返回结果"""
    # 设置设备和超参数
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    max_grad_norm = 1.0
    print(f"\nUsing device: {device}")
    
    # 获取输入维度
    X_train = processed_data['X_train']
    input_size = X_train.shape[1] * X_train.shape[2] if len(X_train.shape) == 3 else X_train.shape[1]
    num_classes = len(processed_data['classes'])
    
    print("\nData dimension check:")
    print(f"Feature shape: {tuple(X_train.shape)}")
    print(f"Label shape: {processed_data['y_train'].shape}")
    print(f"Number of classes: {num_classes}")
    print(f"Input size after flattening: {input_size}")
    
    # 创建模型
    model = SNN(input_size=input_size, hidden_size=192, num_classes=num_classes).to(device)
    criterion = CombinedLoss(margin=2.0, alpha=0.65)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    
    # 创建学习率调度器
    num_training_steps = len(processed_data['X_train']) // batch_size * num_epochs
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=lr,
        total_steps=num_training_steps,
        pct_start=0.3,
        anneal_strategy='cos'
    )
    
    # 创建数据加载器
    train_loader, val_loader, test_loader = create_dataloaders(
        processed_data, batch_size=batch_size
    )
    
    # 打印第一个batch的维度
    print("\nFirst batch dimensions:")
    first_batch = next(iter(train_loader))
    print(f"x1 shape: {first_batch['x1'].shape}")
    print(f"x2 shape: {first_batch['x2'].shape}")
    print(f"y1 shape: {first_batch['y1'].shape}")
    print(f"y2 shape: {first_batch['y2'].shape}")
    
    # 初始化训练变量
    best_val_acc = 0
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []
    early_stopping = EarlyStopping(patience=12, min_delta=0.0008)
    start_time = datetime.now()
    
    # 训练循环
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        # 训练一个epoch
        train_loss, train_acc, _, _ = train_epoch(
            model, train_loader, criterion, optimizer, device, scheduler, max_grad_norm)
        
        # 验证
        val_loss, val_preds, val_labels = evaluate(model, val_loader, criterion, device)
        val_acc = np.mean(np.array(val_preds) == np.array(val_labels))
        
        # 记录指标
        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)
        
        # 打印当前epoch的结果
        print(f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}")
        print(f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        
        # 保存最佳模型
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, 'best_model.pth')
        
        # 早停检查
        if early_stopping.step(val_loss, val_acc):
            print("\nEarly stopping triggered!")
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
    
    # 保存训练结果
    results = {
        'model_name': model_name,
        'train_losses': train_losses,
        'val_losses': val_losses,
        'train_accs': train_accs,
        'val_accs': val_accs,
        'test_acc': test_acc,
        'test_loss': test_loss,
        'best_epoch': checkpoint['epoch'],
        'training_time': (datetime.now() - start_time).total_seconds() / 60,
        'model_config': {
            'input_size': model.input_size,
            'hidden_size': model.hidden_size,
            'num_classes': model.num_classes,
            'alpha': criterion.alpha,
            'margin': criterion.margin
        }
    }
    
    # 保存结果到文件
    with open(f'results/{model_name}_results.pkl', 'wb') as f:
        pickle.dump(results, f)
        
    return model, results
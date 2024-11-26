import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, X, y, augment=True):
        self.X = torch.FloatTensor(X)
        self.y = torch.LongTensor(y)
        self.augment = augment
        
    def augment_signal(self, signal):
        """简单的数据增强"""
        if np.random.random() < 0.5:
            # 添加随机噪声
            noise = torch.randn_like(signal) * 0.01
            signal = signal + noise
        if np.random.random() < 0.5:
            # 随机时间偏移
            shift = np.random.randint(-20, 20)
            signal = torch.roll(signal, shift, dims=0)
        return signal
    
    def __len__(self):
        return len(self.X)
    
    def __getitem__(self, idx):
        x1 = self.X[idx]
        if self.augment and self.y[idx] != 0:  # 只对非Normal类进行增强
            x1 = self.augment_signal(x1)
            
        # 随机选择另一个样本
        idx2 = np.random.randint(len(self.X))
        x2 = self.X[idx2]
        if self.augment and self.y[idx2] != 0:
            x2 = self.augment_signal(x2)
            
        return {
            'x1': x1,
            'x2': x2,
            'y1': self.y[idx],
            'y2': self.y[idx2]
        }

def create_dataloaders(processed_data, batch_size=128):
    train_dataset = ECGDataset(processed_data['X_train'], processed_data['y_train'])
    val_dataset = ECGDataset(processed_data['X_val'], processed_data['y_val'])
    test_dataset = ECGDataset(processed_data['X_test'], processed_data['y_test'])
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True,
        num_workers=0
    )
    
    val_loader = torch.utils.data.DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    test_loader = torch.utils.data.DataLoader(
        test_dataset, 
        batch_size=batch_size, 
        shuffle=False,
        num_workers=0
    )
    
    return train_loader, val_loader, test_loader
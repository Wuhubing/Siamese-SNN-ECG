# src/data/dataset.py
import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np

class ECGDataset(Dataset):
    def __init__(self, features, labels, augment=False):
        self.features = torch.FloatTensor(features)

        self.labels = torch.LongTensor(labels).squeeze()
        self.augment = augment
        
    def augment_signal(self, x):
        if torch.rand(1) < 0.5:
            x = x + torch.randn_like(x) * 0.01
        if torch.rand(1) < 0.5:
            scale = torch.randn(1) * 0.1 + 1
            x = x * scale
        return x
    
    def __len__(self):
        return len(self.features)
    
    def __getitem__(self, idx):
        x1 = self.features[idx]
        y1 = self.labels[idx]
        
        if self.augment:
            x1 = self.augment_signal(x1)
        
        if torch.rand(1) < 0.7:
       
            same_class_indices = torch.where(self.labels == y1)[0]
            idx2 = same_class_indices[torch.randint(len(same_class_indices), (1,))]
        else:
            
            idx2 = torch.randint(len(self.features), (1,))
            
        x2 = self.features[idx2]
        y2 = self.labels[idx2]
        
        if self.augment:
            x2 = self.augment_signal(x2)
            
        return {
            'x1': x1,
            'x2': x2,
            'y1': y1.long(), 
            'y2': y2.long() 
        }

def create_dataloaders(processed_data, batch_size=128):
    train_dataset = ECGDataset(processed_data['X_train'], processed_data['y_train'], augment=True)
    val_dataset = ECGDataset(processed_data['X_val'], processed_data['y_val'], augment=False)
    test_dataset = ECGDataset(processed_data['X_test'], processed_data['y_test'], augment=False)
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size)
    test_loader = DataLoader(test_dataset, batch_size=batch_size)
    
    return train_loader, val_loader, test_loader
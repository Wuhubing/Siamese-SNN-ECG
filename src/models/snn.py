# src/models/snn.py
import torch
import torch.nn as nn
import torch.nn.functional as F
class SNN(nn.Module):
    def __init__(self, input_size=222, hidden_size=128, num_classes=5):
        super(SNN, self).__init__()
        
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_classes = num_classes
        
        self.feature_extractor = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),  
            nn.Dropout(0.3),  
            ResidualBlock(hidden_size), 
            nn.Linear(hidden_size, hidden_size//2),
            nn.LayerNorm(hidden_size//2),
            nn.GELU(),
            nn.Dropout(0.3)
        )
        
        self.attention = nn.MultiheadAttention(hidden_size//2, num_heads=4, batch_first=True)
        self.metric_layer = nn.Sequential(
            nn.Linear(hidden_size//2, hidden_size//4),
            nn.LayerNorm(hidden_size//4),
            nn.GELU(),
            nn.Linear(hidden_size//4, hidden_size//8)
        )
        
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size//8, num_classes)
        )
        
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
    
    def forward_one(self, x):
 
        features = self.feature_extractor(x)
       
        attn_output, _ = self.attention(features.unsqueeze(1), 
                                      features.unsqueeze(1), 
                                      features.unsqueeze(1))
        features = attn_output.squeeze(1)
        
        embeddings = self.metric_layer(features)
        logits = self.classifier(embeddings)
        return embeddings, logits
                
    def forward(self, x1, x2=None):
    
        embedding1, logits1 = self.forward_one(x1)
        
        if x2 is not None:
           
            embedding2, logits2 = self.forward_one(x2)
            return embedding1, embedding2, logits1, logits2
        
        return embedding1, logits1

class ResidualBlock(nn.Module):
    def __init__(self, hidden_size):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Linear(hidden_size, hidden_size)
        )
        
    def forward(self, x):
        return x + self.block(x)
class CombinedLoss(nn.Module):
    def __init__(self, margin=1.0, alpha=0.7):
        super().__init__()
        self.margin = margin
        self.alpha = alpha 
        self.ce_loss = nn.CrossEntropyLoss(label_smoothing=0.1) 
        
    def forward(self, embeddings1, embeddings2, logits1, logits2, labels1, labels2):
      
        ce_loss = self.ce_loss(logits1, labels1) + self.ce_loss(logits2, labels2)
        
        
        distance = F.pairwise_distance(embeddings1, embeddings2)
        same_class = (labels1 == labels2).float()
       
        pt = torch.exp(-distance)
        contrastive_loss = same_class * distance.pow(2) * (1 - pt).pow(2) + \
                          (1 - same_class) * F.relu(self.margin - distance).pow(2) * pt.pow(2)
        
        return self.alpha * ce_loss + (1 - self.alpha) * contrastive_loss.mean()
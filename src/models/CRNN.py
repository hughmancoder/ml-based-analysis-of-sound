import torch
import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, num_classes, p_drop=0.3, in_ch=2, freq_bins: int = 128):
        super().__init__()
        if freq_bins <= 0:
            raise ValueError(f"freq_bins must be > 0, got {freq_bins}")
        # 1. Feature Extraction (CNN)
        self.conv_block = nn.Sequential(
            nn.Conv2d(in_ch, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # Reduce freq/time
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)
        )
        
        # 2. Sequence Learning (RNN)
        # Two MaxPools (2x2) reduce H by factor of 4 (floor).
        h_out = max(1, int(freq_bins) // 4)
        self.gru = nn.GRU(input_size=h_out * 64, hidden_size=128, 
                          num_layers=2, batch_first=True, bidirectional=True)
        
        # 3. Classification
        self.fc = nn.Linear(128 * 2, num_classes) # *2 for bidirectional
        self.dropout = nn.Dropout(p_drop)

    def forward(self, x):
        # x: (B, 2, 128, W)
        x = self.conv_block(x) # (B, 64, 32, W')
        
        # Prepare for RNN: (Batch, Time, Features)
        B, C, H, W = x.shape
        x = x.permute(0, 3, 1, 2).contiguous() 
        x = x.view(B, W, C * H)
        
        # GRU returns (output, hidden)
        x, _ = self.gru(x) 
        
        # Take the last time step or use Global Average Pooling over time
        x = torch.mean(x, dim=1) 
        x = self.dropout(x)
        return self.fc(x)

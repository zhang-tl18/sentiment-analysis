import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, embedding_size=50, feature_size=2, max_text_len=679, num_class=2, dropout_rate=0.5) -> None:
        super().__init__()
        self.name = 'MLP'
        self.fc1 = nn.Sequential(nn.Linear(embedding_size*max_text_len, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, num_class), nn.ReLU())
        self.droupout = nn.Dropout(dropout_rate)

    def forward(self, x):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.droupout(x)
        x = self.fc2(x)
        return x


class TextCNN(nn.Module):
    def __init__(self, embedding_size=50, feature_size=4, max_text_len=679, window_sizes=[2,3,4], num_class=2, dropout_rate=0.5) -> None:
        super().__init__()
        self.name = 'TextCNN'
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(embedding_size, feature_size, h),
                          nn.ReLU(),
                          nn.MaxPool1d(max_text_len-h+1))
            for h in window_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(feature_size*len(window_sizes), num_class)
    
    def forward(self, x):
        x = x.permute(0,2,1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out

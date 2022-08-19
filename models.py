import torch
import torch.nn as nn
import torch.nn.functional as F

class MLP(nn.Module):
    def __init__(self, embedding_size=50, max_text_len=679, num_class=2, dropout_rate=0.5) -> None:
        super().__init__()
        self.name = 'MLP'
        self.fc1 = nn.Sequential(nn.Linear(embedding_size*max_text_len, 64), nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(64, num_class), nn.ReLU())
        self.droupout = nn.Dropout(dropout_rate)

    def forward(self, x, x_len=None):
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        x = self.droupout(x)
        x = self.fc2(x)
        return x


class CNN(nn.Module):
    def __init__(self, embedding_size=50, feature_size=2, max_text_len=679, window_sizes=[2,3,4], num_class=2, dropout_rate=0.5) -> None:
        super().__init__()
        self.name = 'CNN'
        self.convs = nn.ModuleList([
            nn.Sequential(nn.Conv1d(embedding_size, feature_size, h),
                          nn.ReLU(),
                          nn.MaxPool1d(max_text_len-h+1))
            for h in window_sizes
        ])
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(feature_size*len(window_sizes), num_class)
    
    def forward(self, x, x_len=None):
        x = x.permute(0,2,1)
        out = [conv(x) for conv in self.convs]
        out = torch.cat(out, dim=1)
        out = out.view(out.size(0), -1)
        out = self.dropout(out)
        out = self.fc(out)
        return out


class RNN(nn.Module):
    def __init__(self, embedding_size=50, hidden_size=16, num_class=2, dropout_rate=0.5, num_layers=2) -> None:
        super().__init__()
        self.name = 'RNN'
        self.lstm = nn.LSTM(input_size = embedding_size,
                            hidden_size = hidden_size,
                            num_layers = num_layers,
                            dropout = dropout_rate,
                            bidirectional =True)
        self.fc = nn.Linear(hidden_size*2, num_class)
    
    def forward(self, x, x_len):
        x_len_des, idx = torch.sort(x_len, dim=0, descending=True)
        _, un_idx = torch.sort(idx, dim=0)
        x = x[idx]
        x_packed = torch.nn.utils.rnn.pack_padded_sequence(x, x_len_des, batch_first=True)
        out, (hn, cn) = self.lstm(x_packed)
        output = torch.cat([hn[-2], hn[-1]], dim=1)
        output = output[un_idx]
        output = self.fc(output)
        return output

'''
Author: Hugo
Date: 2025-11-15 13:50:29
LastEditors: shen.lan123@gmail.com
LastEditTime: 2025-11-15 13:50:32
Description: 
'''
import torch
import torch.nn as nn

class DeltaLag(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, dropout, l_max, top_k):
        super(DeltaLag, self).__init__()
        self.gru = nn.GRU(input_size, hidden_size, num_layers=num_layers, batch_first=True, dropout=dropout)
        self.Wk = nn.Linear(hidden_size, hidden_size, bias=False)
        self.Wq = nn.Linear(hidden_size, hidden_size, bias=False)
        self.softmax = nn.Softmax(dim=1)
        self.prediction_head = nn.Sequential(
            nn.Linear(input_size, input_size),
            nn.LeakyReLU(),
            nn.Linear(input_size, 1)
        )

        self.l_max = l_max
        self.top_k = top_k

    def forward(self, x):
        # x: (N, T, F)
        N, T, F = x.size()
        h, _ = self.gru(x)  # out: (N, T, H)

        last_h = h[:, -1, :] # (N, H)
        queries = self.Wq(last_h) # (N, H)

        key_h = h[:, -self.l_max:, :]  # (N, l_max, H)
        keys = self.Wk(key_h)  # (N, l_max, H)

        queries_expanded = queries.unsqueeze(1).unsqueeze(2)  # (N, 1, 1, H)
        keys_expanded = keys.unsqueeze(0)      # (1, N, l_max, H)
        attention_scores = torch.sum(queries_expanded * keys_expanded, dim=-1)  # (N, N, l_max)

        mask = torch.eye(N, dtype=torch.bool, device=x.device).unsqueeze(-1)  # (N, N, 1)
        attention_scores = attention_scores.masked_fill(mask, -1e9)

        flat_scores = attention_scores.view(N, -1)  # (N, N * l_max)
        topk_values, topk_indices = torch.topk(flat_scores, self.top_k, dim=1, sorted=False)  # (N, k)

        leader_indices = topk_indices // self.l_max  # (N, k)
        lag_indices = topk_indices % self.l_max      # (N, k)
        lag_values = self.l_max - lag_indices        # (N, k)

        time_indices = T - lag_values # (N, k)
        leader_features = x[leader_indices, time_indices]  # (N, k, F)

        weighted_features = torch.sum(leader_features * self.softmax(topk_values).unsqueeze(-1), dim=1)  # (N, F)
        out = self.prediction_head(weighted_features).reshape(-1)  # (N,)
        return out
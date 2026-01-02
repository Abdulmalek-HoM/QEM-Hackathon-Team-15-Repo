import torch
import torch.nn as nn

class QEM_LSTM(nn.Module):
    def __init__(self, vocab_size, embedding_dim=16, hidden_size=32):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        # batch_first=True -> (Batch, Seq, Feature)
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, 1)
        
    def forward(self, x):
        embeds = self.embedding(x)
        lstm_out, _ = self.lstm(embeds)
        # Take last time step
        last_out = lstm_out[:, -1, :] 
        return self.fc(last_out)

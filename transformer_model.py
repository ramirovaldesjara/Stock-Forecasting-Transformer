import torch
import torch.nn as nn



class LearnablePositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=10000):
        super().__init__()
        self.position_embedding = nn.Parameter(torch.zeros(max_len, d_model))
        nn.init.normal_(self.position_embedding, mean=0.0, std=0.1)

    def forward(self, x):  # x: [B, S, d]
        S = x.size(1)
        return x + self.position_embedding[:S, :].unsqueeze(0)


class TransformerTS(nn.Module):
    """
    Univariate, multi-step forecaster:
      input:  [B, S, 1]
      output: [B, H]
    """
    def __init__(self, input_dim: int, d_model: int, nhead: int, num_layers: int, dropout: float, horizon: int):
        super().__init__()
        self.horizon = horizon
        self.encoder = nn.Linear(input_dim, d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model, nhead=nhead, dim_feedforward=d_model * 4, dropout=dropout, batch_first=True
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # readout from the last token (S-1); produce H outputs
        self.decoder = nn.Linear(d_model, horizon)

    def forward(self, x):  # x: [B, S, 1]
        z = self.encoder(x)               # [B,S,d]
        z = self.pos_encoder(z)           # [B,S,d]
        z = self.backbone(z)              # [B,S,d]
        last = z[:, -1, :]                # [B,d]
        y = self.decoder(last)            # [B,H]
        return y



class TransformerTS_Classification(nn.Module):
    def __init__(self, input_dim: int, d_model: int, nhead: int,
                 num_layers: int, dropout: float, out_dim: int):
        super().__init__()
        self.encoder = nn.Linear(input_dim, d_model)
        self.in_norm = nn.LayerNorm(d_model)
        self.pos_encoder = LearnablePositionalEncoding(d_model)

        enc_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.backbone = nn.TransformerEncoder(enc_layer, num_layers=num_layers)

        # out_dim = 1 if BCEWithLogitsLoss (binary); or 2 if CE
        self.decoder = nn.Linear(d_model, out_dim)

    def forward(self, x):  # x: [B, S, C]
        z = self.encoder(x)          # [B, S, D]
        z = self.in_norm(z)
        z = self.pos_encoder(z)      # [B, S, D]
        z = self.backbone(z)         # [B, S, D]

        feat = z.mean(dim=1)         # ---- mean pooling over time: [B, D]
        y = self.decoder(feat)       # [B, out_dim]
        return y


"""
EDITABLE — Spatial transformer over per-player trajectory summaries, on MPS.
Same architecture as the 0.390 run (summary features: last/mean/delta for s/x/y).
Key fix: enable_nested_tensor=False disables the nested tensor codepath that
blocked MPS. With MPS, 60 epochs should run in ~80s (vs 320s on CPU).
Fixed seed for reproducibility.
"""
import numpy as np
import torch
import torch.nn as nn
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import LabelEncoder

SEED = 42
DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
N_DEF = 11
N_OFF = 11
N_PLAYERS = N_DEF + N_OFF
N_FRAMES = 40
N_RAW_FEATS = 3
N_SUMMARY = 9
TRACKING_PREFIX = ("def_p", "off_p")
SIDES = [0] * N_DEF + [1] * N_OFF


def _extract_summaries(arr):
    t = arr.reshape(len(arr), N_PLAYERS, N_FRAMES, N_RAW_FEATS)
    return np.concatenate([t[:, :, -1, :], t.mean(2), t[:, :, -1, :] - t[:, :, 0, :]], axis=2)


class _SpatialTransformer(nn.Module):
    def __init__(self, in_feats, hidden_dim, n_heads, n_layers, dropout, n_classes):
        super().__init__()
        self.proj     = nn.Linear(in_feats, hidden_dim)
        self.side_emb = nn.Embedding(2, hidden_dim)
        enc_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim, nhead=n_heads,
            dim_feedforward=hidden_dim * 4, dropout=dropout,
            batch_first=True,
        )
        # enable_nested_tensor=False disables the nested tensor codepath → MPS-safe
        self.encoder = nn.TransformerEncoder(
            enc_layer, num_layers=n_layers, enable_nested_tensor=False
        )
        self.head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, n_classes),
        )

    def forward(self, x, mask=None):
        sides = torch.tensor(SIDES, dtype=torch.long, device=x.device)
        x = self.proj(x) + self.side_emb(sides)
        x = self.encoder(x, src_key_padding_mask=mask)
        if mask is not None:
            x = x.masked_fill(mask.unsqueeze(-1), 0.0)
            valid = (~mask).float().sum(1, keepdim=True).clamp(min=1)
            x = x.sum(1) / valid
        else:
            x = x.mean(1)
        return self.head(x)


class CoverageTransformer(BaseEstimator, ClassifierMixin):
    def __init__(self, hidden_dim=64, n_heads=4, n_layers=3, dropout=0.15,
                 lr=1e-3, n_epochs=60, batch_size=128, weight_decay=1e-4):
        self.hidden_dim   = hidden_dim
        self.n_heads      = n_heads
        self.n_layers     = n_layers
        self.dropout      = dropout
        self.lr           = lr
        self.n_epochs     = n_epochs
        self.batch_size   = batch_size
        self.weight_decay = weight_decay

    def _tracking_cols(self, X):
        return sorted(c for c in X.columns if c.startswith(TRACKING_PREFIX))

    def _to_tensors(self, X):
        arr = X[self._tracking_cols(X)].fillna(0).values.astype(np.float32)
        s = _extract_summaries(arr)
        s = (s - self.mean_) / (self.std_ + 1e-6)
        missing = (arr.reshape(len(arr), N_PLAYERS, -1).sum(2) == 0)
        return (torch.tensor(s, dtype=torch.float32),
                torch.tensor(missing, dtype=torch.bool))

    def fit(self, X, y):
        torch.manual_seed(SEED)

        self.le_ = LabelEncoder()
        y_enc = self.le_.fit_transform(y)
        self.classes_ = self.le_.classes_
        n_classes = len(self.classes_)

        arr = X[self._tracking_cols(X)].fillna(0).values.astype(np.float32)
        raw = _extract_summaries(arr)
        self.mean_ = raw.mean(axis=(0, 1), keepdims=True)
        self.std_  = raw.std(axis=(0, 1), keepdims=True)

        X_t, mask_t = self._to_tensors(X)
        y_t = torch.tensor(y_enc, dtype=torch.long)

        counts  = np.bincount(y_enc, minlength=n_classes).astype(np.float32)
        weights = torch.tensor(1.0 / counts.clip(1))
        weights = weights / weights.mean()

        self.model_ = _SpatialTransformer(
            in_feats=N_SUMMARY, hidden_dim=self.hidden_dim,
            n_heads=self.n_heads, n_layers=self.n_layers,
            dropout=self.dropout, n_classes=n_classes,
        ).to(DEVICE)

        # Verify weights actually moved to DEVICE
        actual = next(self.model_.parameters()).device.type
        print(f"  model on: {actual}", flush=True)

        weights = weights.to(DEVICE)
        X_t     = X_t.to(DEVICE)
        mask_t  = mask_t.to(DEVICE)
        y_t     = y_t.to(DEVICE)

        optimizer = torch.optim.AdamW(
            self.model_.parameters(), lr=self.lr, weight_decay=self.weight_decay
        )
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=75, T_mult=1, eta_min=1e-5
        )
        criterion = nn.CrossEntropyLoss(weight=weights)

        N = len(X_t)
        self.model_.train()
        for epoch in range(self.n_epochs):
            perm   = torch.randperm(N, device=DEVICE)
            X_shuf = X_t[perm]; mask_shuf = mask_t[perm]; y_shuf = y_t[perm]
            for i in range(0, N, self.batch_size):
                optimizer.zero_grad()
                logits = self.model_(X_shuf[i:i+self.batch_size],
                                     mask_shuf[i:i+self.batch_size])
                loss = criterion(logits, y_shuf[i:i+self.batch_size])
                loss.backward()
                nn.utils.clip_grad_norm_(self.model_.parameters(), 1.0)
                optimizer.step()
            scheduler.step()
            if (epoch + 1) % 10 == 0:
                print(f"  epoch {epoch+1}/{self.n_epochs}", flush=True)
        return self

    def predict(self, X):
        X_t, mask_t = self._to_tensors(X)
        X_t    = X_t.to(DEVICE)
        mask_t = mask_t.to(DEVICE)
        self.model_.eval()
        preds = []
        with torch.no_grad():
            for i in range(0, len(X_t), self.batch_size):
                logits = self.model_(X_t[i:i+self.batch_size],
                                     mask_t[i:i+self.batch_size])
                preds.append(logits.argmax(1).cpu().numpy())
        return self.le_.inverse_transform(np.concatenate(preds))


def build_model():
    return CoverageTransformer(
        hidden_dim=64,
        n_heads=4,
        n_layers=3,
        dropout=0.15,
        lr=1e-3,
        n_epochs=150,
        batch_size=128,
        weight_decay=1e-4,
    )

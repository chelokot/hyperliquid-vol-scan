"""Fast small-scale sequence-model probe: does a GRU over raw per-second series
beat the tabular GBDT/MLP correlation? Kept tiny for ~2-minute iterations."""

from __future__ import annotations

import argparse
import time

import numpy as np
import torch
import torch.nn as nn

from features_v2 import FEATURE_NAMES, SYMBOLS, WARMUP, build_episode

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
PREMIUM_IDX = FEATURE_NAMES.index("premium")
FORWARD = 60


def build_sequences(dates: list[str], pairs: list[str], seq_len: int, within_stride: int):
    seqs, tabs, targets = [], [], []
    for date in dates:
        for symbol in pairs:
            episode = build_episode(symbol, date)
            if episode is None:
                continue
            perp = episode["perp"]
            feats = episode["features"]
            premium = feats[:, PREMIUM_IDX]
            logret = np.zeros_like(perp)
            logret[1:] = np.log(perp[1:] / perp[:-1])
            prem_chg = np.zeros_like(premium)
            prem_chg[1:] = premium[1:] - premium[:-1]
            step = np.stack([logret * 100, premium, prem_chg], axis=1).astype(np.float32)  # [T, 3]
            end = len(perp) - FORWARD - 1
            for t in range(max(WARMUP, seq_len), end, within_stride):
                seqs.append(step[t - seq_len : t])
                tabs.append(feats[t])
                targets.append((perp[t + FORWARD] / perp[t] - 1) * 1e4)
    return np.stack(seqs), np.stack(tabs).astype(np.float32), np.array(targets, dtype=np.float32)


class GRUHead(nn.Module):
    def __init__(self, in_dim: int = 3, hidden: int = 64) -> None:
        super().__init__()
        self.gru = nn.GRU(in_dim, hidden, num_layers=1, batch_first=True)
        self.head = nn.Sequential(nn.Linear(hidden, 32), nn.GELU(), nn.Linear(32, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, h = self.gru(x)
        return self.head(h[-1]).squeeze(-1)


class TabMLP(nn.Module):
    def __init__(self, in_dim: int, hidden: int = 128) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(), nn.Dropout(0.2),
            nn.Linear(hidden, 64), nn.GELU(), nn.Linear(64, 1)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.clamp(-6, 6)).squeeze(-1)


def train_eval(model, tx, ty, vx, vy_real, ystd, ymean, epochs, batch):
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    loss_fn = nn.MSELoss()
    n = len(tx)
    best = -1.0
    for epoch in range(epochs):
        model.train()
        perm = torch.randperm(n, device=DEVICE)
        for s in range(0, n, batch):
            idx = perm[s : s + batch]
            opt.zero_grad()
            loss_fn(model(tx[idx]), ty[idx]).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            pred = model(vx).cpu().numpy() * ystd + ymean
        best = max(best, float(np.corrcoef(pred, vy_real)[0, 1]))
    return best


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-days", type=int, default=8)
    parser.add_argument("--val-days", type=int, default=3)
    parser.add_argument("--pairs", default="HOOD,NVDA,INTC,MSTR,CRWV,TSM")
    parser.add_argument("--seq", type=int, default=120)
    parser.add_argument("--within-stride", type=int, default=20)
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--hidden", type=int, default=64)
    args = parser.parse_args()
    import os

    dates = sorted(os.listdir("out/rl_research/hl_archive/trades"))
    val_dates = dates[-args.val_days :]
    train_dates = dates[-args.val_days - args.train_days : -args.val_days]
    pairs = args.pairs.split(",")
    print(f"device {DEVICE} | seq {args.seq} | pairs {pairs} | train {len(train_dates)}d val {len(val_dates)}d", flush=True)

    started = time.time()
    tx_np, ttab_np, ty_np = build_sequences(train_dates, pairs, args.seq, args.within_stride)
    vx_np, vtab_np, vy_np = build_sequences(val_dates, pairs, args.seq, args.within_stride)
    print(f"windows: train {len(tx_np)}, val {len(vx_np)} (built in {time.time() - started:.0f}s)", flush=True)

    fmean = tx_np.reshape(-1, 3).mean(0)
    fstd = tx_np.reshape(-1, 3).std(0) + 1e-6
    tabmean = ttab_np.mean(0)
    tabstd = ttab_np.std(0) + 1e-6
    ymean, ystd = ty_np.mean(), ty_np.std() + 1e-6
    tx = torch.from_numpy(((tx_np - fmean) / fstd).astype(np.float32)).to(DEVICE)
    ty = torch.from_numpy(((ty_np - ymean) / ystd).astype(np.float32)).to(DEVICE)
    vx = torch.from_numpy(((vx_np - fmean) / fstd).astype(np.float32)).to(DEVICE)
    ttab = torch.from_numpy(((ttab_np - tabmean) / tabstd).astype(np.float32)).to(DEVICE)
    vtab = torch.from_numpy(((vtab_np - tabmean) / tabstd).astype(np.float32)).to(DEVICE)
    vy_real = vy_np

    seq_corr = train_eval(GRUHead(3, args.hidden).to(DEVICE), tx, ty, vx, vy_real, ystd, ymean, args.epochs, 4096)
    tab_corr = train_eval(TabMLP(ttab.shape[1]).to(DEVICE), ttab, ty, vtab, vy_real, ystd, ymean, args.epochs, 4096)
    print(f"\n=== ЧЕСТНОЕ СРАВНЕНИЕ на одних точках ({len(vx_np)} val окон) ===", flush=True)
    print(f"seq-GRU (сырая последовательность): val corr {seq_corr:.4f}", flush=True)
    print(f"tab-MLP (47 фич в момент t):        val corr {tab_corr:.4f}", flush=True)
    print(f"→ последовательность {'+%.1f%% выше' % ((seq_corr/tab_corr-1)*100) if seq_corr>tab_corr else 'НЕ помогла'} | {time.time() - started:.0f}с", flush=True)


if __name__ == "__main__":
    main()

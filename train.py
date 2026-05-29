"""
train.py  —  train ReconNet to improve FBP reconstructions
"""

import os, torch, torch.nn as nn
from torch.utils.data import DataLoader, random_split
from dataset import ReconDataset
from models.recon_net import ReconNet


if __name__ == "__main__":
    os.makedirs("checkpoints", exist_ok=True)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    ds    = ReconDataset(n_samples=5000)
    n_val = 500
    train_ds, val_ds = random_split(
        ds, [len(ds) - n_val, n_val])

    train_dl = DataLoader(train_ds, batch_size=16,
                          shuffle=True, num_workers=2)
    val_dl   = DataLoader(val_ds,   batch_size=16,
                          num_workers=2)

    model     = ReconNet().to(device)
    optimiser = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimiser, T_max=30)
    criterion = nn.MSELoss()

    best = float("inf")
    for epoch in range(1, 31):
        model.train()
        tr = 0.0
        for x, y in train_dl:
            x, y = x.to(device), y.to(device)
            optimiser.zero_grad()
            loss = criterion(model(x), y)
            loss.backward()
            optimiser.step()
            tr += loss.item()

        model.eval()
        vl = 0.0
        with torch.no_grad():
            for x, y in val_dl:
                x, y = x.to(device), y.to(device)
                vl += criterion(model(x), y).item()

        tr /= len(train_dl)
        vl /= len(val_dl)
        scheduler.step()
        print(f"Epoch {epoch:02d} | Train: {tr:.5f} "
              f"| Val: {vl:.5f}")

        if vl < best:
            best = vl
            torch.save(model.state_dict(),
                       "checkpoints/reconnet_best.pt")

    print(f"Done. Best val loss: {best:.5f}")

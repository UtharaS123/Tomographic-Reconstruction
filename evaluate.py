"""
evaluate.py
Compare FBP vs ReconNet. Saves figure + prints metrics.
"""

import numpy as np, torch
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.metrics import (structural_similarity as ssim,
                              peak_signal_noise_ratio as psnr)

from dataset import make_fbp_pair
from models.recon_net import ReconNet


if __name__ == "__main__":
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu")

    model = ReconNet().to(device)
    model.load_state_dict(torch.load(
        "checkpoints/reconnet_best.pt", map_location=device))
    model.eval()

    phantom = shepp_logan_phantom().astype(np.float32)
    phantom = (phantom - phantom.min()) / (
               phantom.max() - phantom.min())

    fbp_deg, gt = make_fbp_pair(phantom)

    x = torch.tensor(fbp_deg[None, None]).to(device)
    with torch.no_grad():
        recon = model(x).cpu().numpy()[0, 0]

    for name, img in [("FBP (degraded)", fbp_deg),
                      ("ReconNet",       recon)]:
        s = ssim(gt, img, data_range=1.0)
        p = psnr(gt, img, data_range=1.0)
        print(f"{name:<22} SSIM: {s:.4f}  PSNR: {p:.2f} dB")

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    for ax, (title, img) in zip(axes, [
        ("Ground truth",        gt),
        ("FBP — sparse angles", fbp_deg),
        ("ReconNet output",     recon),
    ]):
        ax.imshow(img, cmap="gray", vmin=0, vmax=1)
        ax.set_title(title); ax.axis("off")

    plt.tight_layout()
    plt.savefig("reconstruction_comparison.png", dpi=150)
    print("Saved reconstruction_comparison.png")

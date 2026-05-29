"""
dataset.py
Generate (degraded_FBP, ground_truth) pairs for training.
Uses random ellipse phantoms for varied training data.
"""

import numpy as np
import torch
from torch.utils.data import Dataset
from skimage.draw import ellipse
from skimage.transform import radon, iradon


def random_ellipse_phantom(size=128, n_ellipses=5):
    """Generate a random ellipse phantom."""
    image = np.zeros((size, size), dtype=np.float32)
    for _ in range(n_ellipses):
        r_r = np.random.randint(10, size // 4)
        r_c = np.random.randint(10, size // 4)
        c_r = np.random.randint(r_r, size - r_r)
        c_c = np.random.randint(r_c, size - r_c)
        val = np.random.uniform(0.3, 1.0)
        rr, cc = ellipse(c_r, c_c, r_r, r_c,
                         shape=(size, size))
        image[rr, cc] += val
    return np.clip(image, 0, 1)


def make_fbp_pair(phantom, n_angles_sparse=30,
                  noise_std=0.01):
    """
    Returns:
        fbp_degraded : sparse-angle + noisy FBP (input)
        phantom      : ground truth (target)
    """
    angles = np.linspace(0, 180, n_angles_sparse, endpoint=False)
    sino   = radon(phantom, theta=angles, circle=True)
    sino  += np.random.normal(0, noise_std, sino.shape)
    fbp    = iradon(sino, theta=angles, circle=True,
                    filter_name="ramp")
    fbp    = (fbp - fbp.min()) / (fbp.max() - fbp.min() + 1e-8)
    return fbp.astype(np.float32), phantom.astype(np.float32)


class ReconDataset(Dataset):
    def __init__(self, n_samples=5000, size=128, seed=0):
        np.random.seed(seed)
        self.pairs = [
            make_fbp_pair(random_ellipse_phantom(size))
            for _ in range(n_samples)
        ]

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, idx):
        fbp, gt = self.pairs[idx]
        return (torch.tensor(fbp[np.newaxis]),
                torch.tensor(gt[np.newaxis]))

"""
radon.py
From-scratch implementation of:
  1. Radon transform  (image -> sinogram)
  2. Ramp-filtered backprojection  (sinogram -> image)

Run directly for a Shepp-Logan demo:
    python radon.py
"""

import numpy as np
import matplotlib.pyplot as plt
from skimage.data import shepp_logan_phantom
from skimage.transform import resize


def radon_transform(image, angles_deg):
    """
    Compute the Radon transform of a 2D image.

    Parameters
    ----------
    image      : (N, N) ndarray
    angles_deg : array of projection angles in degrees

    Returns
    -------
    sinogram : (len(angles_deg), N) ndarray
    """
    N        = image.shape[0]
    n_ang    = len(angles_deg)
    sinogram = np.zeros((n_ang, N))
    cx, cy   = N // 2, N // 2
    x_idx    = np.arange(N) - cx
    y_idx    = np.arange(N) - cy
    X, Y     = np.meshgrid(x_idx, y_idx, indexing="ij")

    for i, theta in enumerate(np.deg2rad(angles_deg)):
        x_rot = X * np.cos(theta) + Y * np.sin(theta)
        for d in range(N):
            d_coord = d - cx
            mask = np.abs(x_rot - d_coord) < 0.5
            sinogram[i, d] = np.sum(image[mask])

    return sinogram


def ramp_filter(sinogram):
    """Apply ramp (Ram-Lak) filter to each projection."""
    n_det  = sinogram.shape[1]
    freqs  = np.fft.fftfreq(n_det)
    ramp   = np.abs(freqs)

    filtered = np.zeros_like(sinogram)
    for i in range(sinogram.shape[0]):
        F            = np.fft.fft(sinogram[i])
        filtered[i]  = np.real(np.fft.ifft(F * ramp))

    return filtered


def filtered_backprojection(sinogram, angles_deg):
    """
    Reconstruct image from sinogram via filtered backprojection.
    """
    N        = sinogram.shape[1]
    filtered = ramp_filter(sinogram)
    recon    = np.zeros((N, N))
    cx, cy   = N // 2, N // 2
    x_idx    = np.arange(N) - cx
    y_idx    = np.arange(N) - cy
    X, Y     = np.meshgrid(x_idx, y_idx, indexing="ij")

    for i, theta in enumerate(np.deg2rad(angles_deg)):
        t     = X * np.cos(theta) + Y * np.sin(theta)
        t_idx = np.round(t + cx).astype(int)
        t_idx = np.clip(t_idx, 0, N - 1)
        recon += filtered[i, t_idx]

    return recon * (np.pi / (2 * len(angles_deg)))


if __name__ == "__main__":
    phantom = shepp_logan_phantom()
    phantom = resize(phantom, (128, 128), anti_aliasing=True)

    angles_full   = np.linspace(0, 180, 180, endpoint=False)
    angles_sparse = np.linspace(0, 180, 30,  endpoint=False)

    sino_full   = radon_transform(phantom, angles_full)
    sino_sparse = radon_transform(phantom, angles_sparse)
    fbp_full    = filtered_backprojection(sino_full,   angles_full)
    fbp_sparse  = filtered_backprojection(sino_sparse, angles_sparse)

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    # ... (plotting code)
    plt.savefig("fbp_demo.png", dpi=150)
    print("Saved fbp_demo.png")

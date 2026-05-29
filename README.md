# 3D Tomographic Reconstruction with Deep Learning

From-scratch implementation of the Radon transform, filtered
backprojection (FBP), and a learned CNN reconstruction model,
extending to 3D cone-beam geometry. Motivated by the 3D dose
reconstruction challenge in proton-acoustic dosimetry.

## Motivation
Proton-acoustic dosimetry requires reconstructing a 3D dose
distribution from acoustic projection data — an inverse problem
mathematically equivalent to CT reconstruction. This project
builds that understanding from first principles.

## Project Structure
```
tomographic-reconstruction/
├── radon.py              # Radon transform + FBP from scratch
├── dataset.py            # Phantom generation + projection dataset
├── models/
│   └── recon_net.py      # CNN post-processing network
├── train.py              # Training loop
├── evaluate.py           # Metrics + visualisation
├── requirements.txt
└── README.md
```

## Quickstart
```bash
pip install -r requirements.txt
python radon.py            # demo: sinogram + FBP on Shepp-Logan
python train.py            # train CNN reconstruction net
python evaluate.py         # FBP vs learned reconstruction
```

## References
- Kak & Slaney (1988). Principles of Computerized Tomographic
  Imaging. IEEE Press.
- Hauptmann & Cox (2020). Deep learning in photoacoustic
  tomography. J. Biomed. Optics, 25(11).

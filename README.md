# AEROLAB - A Flexible Experimentation Framework for AEROBLADE
## Setup
Create a virtual environment using a Python >=3.10 interpreter.

**CUDA Support:**
If your system doesn't support *NVIDIA CUDA*, make sure to comment out `-e ./aeroblade[cuda]` in the beginning of the `requirements.txt`. 
Instead, require `-e ./aeroblade` by uncommenting it.

Then run
```
pip install -r requirements.txt
```
## References
- *AEROBLADE* is adapted from:
  - Ricker, Jonas, Denis Lukovnikov, und Asja Fischer. „AEROBLADE: Training-Free Detection of Latent Diffusion Images Using Autoencoder Reconstruction Error“. arXiv, 27. März 2024. https://doi.org/10.48550/arXiv.2401.17879.
- `meaningful_complexity.py` is adapted from: 
  - Mahon, Louis, und Thomas Lukasiewicz. „Minimum description length clustering to measure meaningful image complexity“. Pattern Recognition 145 (1. Januar 2024): 109889. https://doi.org/10.1016/j.patcog.2023.109889.

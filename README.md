# Explainability in Conditional Diffusion for Vision AI
### A Markov Process Perspective for High-Fidelity Visual Reconstruction

**Author:** Surjo Dey (Roll No. 22CS3072)  
**Degree:** Bachelor of Technology — Computer Science and Engineering  
**Institution:** Rajiv Gandhi Institute of Petroleum Technology, Amethi, UP  
**Supervisor:** Dr. Pallabi Saikia, Assistant Professor, CSE Department  
**Submitted:** April 2026

---

## Overview

This project studies **explainability in conditional diffusion models** for image reconstruction. Rather than only evaluating final output quality, this work analyzes the *entire denoising trajectory* — examining how and when the model forms its reconstruction at each step.

A U-Net based diffusion model was trained on 800 high-resolution images from the **DIV2K** dataset and evaluated under three input conditions:

| Condition | Description |
|-----------|-------------|
| **Clean** | Original high-resolution images |
| **Noisy** | Images with Gaussian noise (σ = 0.1, i.e., 10% intensity) |
| **Patched** | Images with 5% random white patches (spatial occlusion) |

Three novel **quantitative explainability metrics** are proposed and studied alongside standard metrics (FID, PSNR, LPIPS).

---

## Key Findings

- **Clean images** achieved best perceptual quality at **epoch 400** (FID = 7.583), with stable PSNR ≈ 22.5 dB throughout training.
- **Noisy images** converged *faster*, reaching best FID of **6.953 at epoch 200** — suggesting mild stochastic corruption acts as a regularizer in early training.
- **Patched images** were the hardest case: FID stayed above 11, PSNR dropped to ~16 dB — nearly **6.5 dB below the clean baseline** — throughout all epochs.
- Trajectory visualizations confirmed that patch-perturbed inputs produce **more chaotic and inconsistent denoising paths**, indicating the model struggles when large regions are entirely missing.

---

## Explainability Framework

### Three Interpretability Tools

1. **Denoising Trajectory Visualization** — Tracks how the model's estimate of the clean image (`X̂₀⁽ᵗ⁾`) evolves across timesteps `t = 999 → 0`.
2. **Noise Prediction Score Maps** — Pixel-wise maps showing where the model focuses its denoising effort at each timestep.
3. **High-Activation Patch Analysis** — Identifies spatial regions of peak model activity during reconstruction.

### Three Quantitative Metrics

| Metric | What It Measures |
|--------|-----------------|
| **Trajectory Consistency (TC)** | How smoothly the reconstruction evolves over time (based on SSIM between consecutive estimates). Grounded in the martingale property of the posterior mean sequence. |
| **Attribution Localization Accuracy (ALA)** | IoU between the model's gradient-based attention map and semantically meaningful regions. Connects to faithfulness criteria in interpretability literature. |
| **Information Gain per Step (IGS)** | Mutual information gained about the original image at each reverse diffusion step. Identifies which timesteps resolve the most uncertainty. |

---

## Model Architecture & Training

- **Architecture:** U-Net based conditional diffusion model
- **Conditioning:** Degraded input image concatenated with noisy state at every timestep: `ϵθ([c, xₜ], t)`
- **Dataset:** DIV2K — 800 training images, 100 test images
- **Learning Rate:** 2e-4
- **Diffusion Timesteps:** 1000
- **Batch Size:** 4
- **Training Epochs Evaluated:** 100, 200, 300, 400, 500
- **Hardware:** 2× NVIDIA L40S GPUs

---

## Ablation Study Results

### Case 1: Clean High-Resolution Images

| Epoch | Timesteps | LR | FID ↓ | PSNR ↑ | LPIPS ↓ |
|-------|-----------|-----|-------|--------|---------|
| 100 | 1000 | 2e-4 | 10.934 | 22.318 | 0.264 |
| 200 | 1000 | 2e-4 | 8.764 | 22.590 | 0.199 |
| 300 | 1000 | 2e-4 | 8.441 | 22.530 | 0.208 |
| **400** | **1000** | **2e-4** | **7.583** | **22.507** | **0.212** |
| 500 | 1000 | 2e-4 | 8.808 | 22.450 | 0.214 |

### Case 2: Gaussian-Noised Images (σ = 0.1)

| Epoch | Timesteps | LR | FID ↓ | PSNR ↑ | LPIPS ↓ |
|-------|-----------|-----|-------|--------|---------|
| 100 | 1000 | 2e-4 | 10.902 | 22.208 | 0.280 |
| **200** | **1000** | **2e-4** | **6.953** | **22.504** | **0.217** |
| 300 | 1000 | 2e-4 | 9.768 | 22.406 | 0.228 |
| 400 | 1000 | 2e-4 | 8.527 | 22.409 | 0.225 |
| 500 | 1000 | 2e-4 | 7.360 | 22.400 | 0.232 |

### Case 3: Patch-Perturbed Images (5% white patches)

| Epoch | Timesteps | LR | FID ↓ | PSNR ↑ | LPIPS ↓ |
|-------|-----------|-----|-------|--------|---------|
| 100 | 1000 | 2e-4 | 14.475 | 15.971 | 0.347 |
| 200 | 1000 | 2e-4 | 14.036 | 16.018 | 0.286 |
| **300** | **1000** | **2e-4** | **11.484** | **16.005** | **0.286** |
| 400 | 1000 | 2e-4 | 12.854 | 15.997 | 0.289 |
| 500 | 1000 | 2e-4 | 13.353 | 16.000 | 0.283 |

---

## Repository Structure

```
Explainability-in-Conditional-Diffusion-for-Vision-AI/
│
├── train.py                # Main training script for the diffusion model
├── test.py                 # Inference and reconstruction script
├── eval_checkpoint.py      # Evaluate saved checkpoints (FID, PSNR, LPIPS)
├── exp.py                  # Explainability visualizations (trajectories, score maps, patches)
│
├── add_noice.py            # Generate Gaussian-noised test images (σ = 0.1)
├── add_white_patches.py    # Generate patch-perturbed test images (5% white patches)
│
├── final_model.pth         # Final trained model weights (Git LFS)
├── latest.pth              # Latest checkpoint weights (Git LFS)
│
└── images/                 # Output visualizations and result figures
```

---

## Setup & Usage

### Requirements

```bash
pip install torch torchvision
pip install numpy pillow matplotlib
pip install pytorch-fid lpips
```

### Training

```bash
python train.py
```

### Generating Test Variants

```bash
# Add Gaussian noise to test images
python add_noice.py

# Add white patch occlusions to test images
python add_white_patches.py
```

### Evaluation

```bash
# Evaluate a checkpoint on FID, PSNR, LPIPS
python eval_checkpoint.py
```

### Explainability Visualizations

```bash
# Generate trajectory, score map, and high-activation patch visualizations
python exp.py
```

### Inference / Testing

```bash
python test.py
```

---

## Model Weights

The trained model weights are stored using **Git LFS**:

| File | Description |
|------|-------------|
| `final_model.pth` | Best performing model checkpoint |
| `latest.pth` | Most recent training checkpoint |

To download via Git LFS:

```bash
git lfs install
git clone https://github.com/surjo0/Explainability-in-Conditional-Diffusion-for-Vision-AI.git
```

---

## Sample Results

### Denoising Trajectories (t = 999 → 0)

Visualization of how the model's reconstruction evolves from pure noise to the final output across all three input conditions. Clean and noisy inputs show smooth, progressive structure formation. Patched inputs show more chaotic intermediate states.

*(See `images/` folder for full trajectory grids)*

### Metric Evolution Across Training

- Clean and noisy conditions converge to similar PSNR (~22.5 dB)
- Patched condition remains ~6.5 dB below throughout training
- Noisy inputs reach best FID earliest (epoch 200), suggesting a regularization benefit from mild input corruption

---

## Mathematical Background

The model follows the **DDPM** framework. The forward process corrupts clean data `X₀` over `T = 1000` timesteps:

```
Xₜ = √ᾱₜ · X₀ + √(1 - ᾱₜ) · ε,    ε ~ N(0, I)
```

The reverse process learns to denoise, with the U-Net predicting the injected noise conditioned on the degraded input `c`:

```
ϵθ([c, xₜ], t) ≈ ε
```

The clean image estimate at each step is:

```
X̂₀⁽ᵗ⁾ = (1/√ᾱₜ) · (Xₜ - √(1 - ᾱₜ) · ϵθ(Xₜ, t))
```

---

## Citation

If you use this work, please cite:

```bibtex
@article{dey2026explainability,
  title={Explainability in Generative Medical Diffusion Models: A Faithfulness-Based Analysis on MRI Synthesis},
  author={Dey, Surjo and Saikia, Pallabi},
  journal={arXiv preprint arXiv:2602.09781},
  year={2026}
}

```

---

## License

This project is submitted as a B.Tech thesis at RGIPT, April 2026, under the supervision of Dr. Pallabi Saikia. 

---

*Department of Computer Science and Engineering, Rajiv Gandhi Institute of Petroleum Technology, Amethi, UP*

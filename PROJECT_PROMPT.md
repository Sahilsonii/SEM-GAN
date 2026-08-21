# 🔬 M.Tech Dissertation: GAN-Based Generative Modeling of Perovskite Solar Cell Microstructures from SEM Images
## A Multi-Architecture Deep Learning Benchmark Study

> **Author**: [Your Name] | **Program**: M.Tech [Your Department]  
> **Dissertation Level**: Scientific Research + Software Engineering  
> **Primary Dataset**: Perovskite Solar Cell SEM Images (5 Classes, ~5000+ Images)  
> **Date**: July 2026  

---

## ⚡ EXECUTIVE SUMMARY (TL;DR Prompt)

Build a dissertation-grade, **multi-model GAN benchmark framework** for scanning electron microscopy (SEM) images of **perovskite solar cells** — a cutting-edge photovoltaic technology whose nano-scale microstructure directly determines solar energy conversion efficiency. The project trains, fine-tunes, and rigorously evaluates **5 state-of-the-art generative deep learning architectures** on a **5-class perovskite SEM dataset** under a unified Computer Vision evaluation protocol. The system pipeline spans from raw FESEM TIFF metadata parsing to multi-model benchmarking to dissertation-quality visual and spectral output figures. This is designed end-to-end as a production-grade, reproducible ML system following Software Engineering best practices.

---

## 📌 SECTION 1: SCIENTIFIC MOTIVATION & RESEARCH PROBLEM

### 1.1 Why Perovskite Solar Cells?
Perovskite solar cells (PSCs) have achieved certified power conversion efficiencies (PCE) of **25.7%** in the laboratory (NREL 2024), rivaling commercial silicon. However, a fundamental bottleneck exists:

- **The microstructure of perovskite thin films** (grain boundaries, pinhole defects, nucleation sites, surface cracks, and crystallite morphology) directly governs charge carrier mobility, carrier lifetime, and recombination losses.
- **SEM is the gold standard for microstructure characterization** — but acquiring high-resolution FESEM images is slow (30 min/scan), expensive (per-session cost), and the electron beam itself causes **beam-induced degradation** in moisture-sensitive perovskite films.
- **The data scarcity problem**: There are insufficient annotated SEM datasets for robust automated microstructure analysis using modern deep learning methods.

### 1.2 Primary Computer Vision Research Questions
This dissertation formally addresses the following research questions:

> **RQ1**: Can a deep generative adversarial network accurately synthesize novel, physically plausible perovskite SEM microstructure images conditioned on morphological class labels?

> **RQ2**: Can cross-magnification super-resolution GANs recover sub-nanometer spatial frequency content from fast-acquired, low-dose FESEM scans of perovskite films?

> **RQ3**: Which generative architecture family (Conditional GAN, Vision Transformer GAN, Layout-Conditional GAN, or Latent Diffusion) achieves the most favorable trade-off between visual fidelity, frequency alignment, and training efficiency on small-scale SEM datasets?

> **RQ4**: Does GAN-based synthetic data augmentation statistically improve downstream automated perovskite defect classification accuracy compared to training on real images alone?

### 1.3 Dataset: 5-Class Perovskite Solar Cell SEM Dataset
The project uses a curated FESEM dataset of perovskite solar cell microstructures across **5 morphological classes**:

| Class Index | Class Label | Description | SEM Characteristics |
|---|---|---|---|
| **C1** | Pinhole Defects (3D) | Sub-micron voids in perovskite absorber layer | Dark circular features, depth-contrast shadows |
| **C2** | Grain Boundaries | Polycrystalline grain boundary networks | Sharp linear contrast edges, dendritic patterns |
| **C3** | Crystal Nucleation Sites | Early-stage crystal growth clusters | Raised nodular protrusions, multi-scale clusters |
| **C4** | Surface Cracks / Micro-Fractures | Mechanical or thermal stress crack networks | Branching dark linear channels |
| **C5** | Compact/Dense Perovskite Films | Uniformly crystallized, high-efficiency baseline | Uniform low-contrast, minimal topographic features |

**Dataset Statistics**:
- Target: **5,000+ images** across 5 classes (1,000 per class)
- Image Format: TIFF (16-bit or 8-bit grayscale, typically 1024×768 or 2048×1536)
- Instrument: Field Emission SEM (FESEM) — Zeiss, FEI, or JEOL instruments
- Magnifications: Hierarchical (1K, 2K, 5K, 10K, 20K, 50K)
- Additional Training Data Source: **NFFA-Europe SEM Dataset** (25,537 multi-class SEM images) for Stage 1 Domain Pre-Training

---

## 📌 SECTION 2: GENERATIVE MODEL ARCHITECTURES

All models are open-source, free, and published under permissive licenses (MIT, Apache 2.0, BSD).

---

### 🏗️ MODEL 1: Restormer-GAN (Primary Model) — *Super-Resolution & Denoising*

**Paper**: "Restormer: Efficient Transformer for High-Resolution Image Restoration"  
**Authors**: Zamir et al.  
**Published**: CVPR 2022  
**GitHub**: https://github.com/swz30/Restormer  
**License**: MIT License  

#### Architecture Design
```
Low-Res SEM Input [B, 1, H, W]
        │
[Conv First] → Shallow Feature Map [B, 48, H, W]
        │
[Restormer Encoder Block ×N]
  ├─ Multi-Dconv Head Transposed Attention (MDTA)  ← O(C²) memory
  │    - Cross-channel covariance attention (NOT pixel-wise)
  │    - Depthwise conv QKV projections
  └─ Gated-Dconv Feed-Forward Network (GDFN)
        │
[Sub-Pixel Convolution Upsampler]   ← 2×, 4×, 8× scale factors
  └─ PixelShuffle + PReLU blocks
        │
[Conv Reconstruction] → High-Res SR Output [B, 1, rH, rW] ∈ [-1, 1]
```

**Key Technical Novelties**:
- Multi-Dconv Transposed Attention: Operates across channel dimension ($O(C^2)$) instead of spatial ($O(HW)^2$), enabling high-resolution attention with 16 GB → 16 KB memory footprint reduction.
- Gated Depthwise FFN: Controls spatial feature flow using GELU-gated depthwise convolution — preventing over-smoothing of fine perovskite grain boundary edges.

**Pre-trained Weights**: DIV2K + Flickr2K + BSD400 + WED400 super-resolution weights.

---

### 🏗️ MODEL 2: SwinIR-GAN (Comparative SR Baseline) — *Vision Transformer Super-Resolution*

**Paper**: "SwinIR: Image Restoration Using Swin Transformer"  
**Authors**: Liang et al.  
**Published**: ICCV Workshop 2021  
**GitHub**: https://github.com/JingyunLiang/SwinIR  
**License**: Apache 2.0  

#### Architecture Design
```
Low-Res Input
     │
[Conv Head]
     │
[Residual Swin Transformer Block (RSTB) ×N]
  ├─ Swin Transformer Layer (Windowed Local Self-Attention)
  │    - Window-partitioned attention: W×W tokens per window (W=8)
  │    - Shifted Window (SW-MSA) cross-window communication
  └─ Conv Layer Residual
     │
[Sub-Pixel Upsampler]
     │
[Conv Reconstruction]
```

**Key Distinction vs Restormer**: SwinIR uses windowed *spatial* self-attention ($O(W^2 C)$), while Restormer uses *channel transposed* attention ($O(C^2)$). This makes SwinIR more memory-hungry at high resolutions but provides richer spatial neighborhood modeling.

**Pre-trained Weights**: Large-scale SwinIR-M model trained on DIV2K + LSDIR.

---

### 🏗️ MODEL 3: SPADE-GAN (GauGAN) — *Semantic Mask-to-SEM Image Generation*

**Paper**: "Semantic Image Synthesis with Spatially-Adaptive Normalization"  
**Authors**: Park et al.  
**Published**: CVPR 2019  
**GitHub**: https://github.com/NVlabs/SPADE  
**License**: CC BY-NC-SA 4.0  

#### Architecture Design
```
Binary Semantic Layout Mask (Class Map)
              │
  ┌───────────┼───────────┐
  │                       │
[Encoder]              [Noise z]
  │                       │
  └──────────┬────────────┘
             │
[SPADE ResBlk ×N]   ← SPADE = Spatially-Adaptive Normalization
  ├─ γ(x), β(x) modulation from semantic mask
  └─ StyleGAN-style Noise Injection for surface roughness/texture
             │
[Multi-Scale Discriminator]   ← Patch-level discrimination at 3 scales
```

**Scientific Use Case**: Generate physically plausible FESEM images directly from user-defined layouts. E.g., a researcher can draw where pinholes should be and SPADE synthesizes realistic SEM electron contrast, shadow artifacts, and beam charging around those features.

---

### 🏗️ MODEL 4: StyleGAN2-ADA — *Unconditional High-Quality SEM Synthesis*

**Paper**: "Training Generative Adversarial Networks with Limited Data"  
**Authors**: Karras et al.  
**Published**: NeurIPS 2020  
**GitHub**: https://github.com/NVlabs/stylegan2-ada-pytorch  
**License**: NVIDIA Source Code License (free for research)  

#### Architecture Design
```
Latent z ∈ ℝ⁵¹²
     │
[Mapping Network f: z → w]    8 FC layers → Style Space W+
     │
[Synthesis Network G]
  ├─ Learned constant 4×4 tensor
  ├─ AdaIN Modulation at each resolution scale
  │   - (4×4 → 8×8 → 16×16 → ... → 256×256)
  ├─ Noise injection B at each layer (surface texture control)
  └─ Alias-free bilinear upsampling (StyleGAN3 improvement)
     │
[Adaptive Discriminator Augmentation (ADA)]
  └─ Dynamically adapts augmentation probability to prevent
     discriminator overfitting on small SEM datasets
```

**Why ADA is Critical**: Standard GAN discriminators overfit on small datasets (<1,000 images). ADA dynamically adjusts augmentation strength (flips, translates, cutouts) to maintain discriminator generalization — perfect for limited perovskite SEM data.

**Latent Space Analysis**: t-SNE / PCA visualization of latent codes reveals disentangled morphological style vectors (grain size, surface roughness, pinhole density).

---

### 🏗️ MODEL 5: Latent Diffusion Model (Micro-LDM) + ControlNet — *Denoising Diffusion Synthesis*

**Paper 1**: "High-Resolution Image Synthesis with Latent Diffusion Models"  
**Authors**: Rombach et al.  
**Published**: CVPR 2022  
**GitHub**: https://github.com/CompVis/latent-diffusion  

**Paper 2**: "Adding Conditional Control to Text-to-Image Diffusion Models (ControlNet)"  
**Authors**: Zhang & Agrawala  
**Published**: ICCV 2023  
**GitHub**: https://github.com/lllyasviel/ControlNet  

#### Architecture Design
```
Real SEM Image x₀
        │
[VAE Encoder E] → Compressed Latent z₀ ∈ ℝ^{h×w×4}
        │
[Forward Diffusion q(zₜ|z₀)] → zₜ = √ᾱₜz₀ + √(1-ᾱₜ)ε
        │
[U-Net Denoiser εθ(zₜ, t, c)]   ← c = class label / edge map conditioning
  ├─ ResNet Encoder Backbone
  ├─ Transformer Self-Attention Blocks (Cross-Attention with condition c)
  └─ ControlNet Branch (optional: Canny edge map, depth map conditioning)
        │
[Reverse Diffusion pθ(z₀|zₜ)] → DDIM sampling (10-50 steps)
        │
[VAE Decoder D] → Synthetic SEM Image x̂₀
```

**Why LDM?**: Diffusion models achieve the highest perceptual diversity and FID scores of any generative model, enabling infinite variety in synthetic perovskite microstructure generation with controllable conditioning.

---

### 🏗️ DUAL-DOMAIN DISCRIMINATOR (Custom Novel Architecture — Our Key Contribution)

This is the **novel scientific contribution** of this dissertation. No existing SEM GAN paper uses this exact formulation:

```
Generated SEM Image SR / Synthesized
              │
  ┌───────────┴────────────────┐
  │                            │
[SPATIAL DOMAIN]          [FOURIER DOMAIN]
PatchGAN Discriminator    FFT Magnitude Discriminator
  - 5-layer Conv Net        - 2D FFT Shift
  - Local patch realism     - Log |F(x)| computation
  - L_adv_spatial           - Sub-band frequency conv
  - Hinge Loss              - L_adv_fourier
  │                            │
  └───────────┬────────────────┘
              │
   [Combined Dual-Domain Score]
```

**Loss Function**:
$$\mathcal{L}_{\text{total}} = \underbrace{\lambda_1 \mathcal{L}_{\text{L1}}}_{\text{Pixel}} + \underbrace{\lambda_2 \mathcal{L}_{\text{VGG}}}_{\text{Perceptual}} + \underbrace{\lambda_3 \mathcal{L}_{\text{FFT}}}_{\text{Frequency}} + \underbrace{\lambda_4 (\mathcal{L}_{\text{adv,spatial}} + \mathcal{L}_{\text{adv,fourier}})}_{\text{Dual-Domain Adversarial}}$$

Where:
- $\mathcal{L}_{\text{L1}}$: Mean Absolute Error for pixel-level structural fidelity
- $\mathcal{L}_{\text{VGG}}$: VGG19 multi-layer perceptual feature loss (relu2_2, relu3_3, relu4_4)
- $\mathcal{L}_{\text{FFT}}$: L1 distance between 2D Fourier magnitude spectra of SR and HR images
- $\mathcal{L}_{\text{adv,spatial}}$: Hinge adversarial loss from spatial PatchGAN
- $\mathcal{L}_{\text{adv,fourier}}$: Hinge adversarial loss from Fourier frequency discriminator

---

## 📌 SECTION 3: TWO-STAGE TRAINING PROTOCOL

### Stage 1: Domain Pre-Training (NFFA-Europe Multi-Class SEM Dataset)
```
Dataset    : NFFA-Europe SEM Dataset (25,537 images → use 5,000 subset)
Source     : https://huggingface.co/datasets/l11p/nffa-europe-sem-dataset
Categories : 10 SEM categories (nanoparticles, fibers, biological, patterns, etc.)
Purpose    : Teach model universal SEM electron imaging properties:
              - Electron beam intensity gradients
              - Secondary electron surface contrast physics
              - Shot noise and detector readout noise distributions
              - Edge diffraction halos and shadow artifacts

Training Config:
  - Epochs      : 20 epochs
  - Patch Size  : 256×256
  - Batch Size  : 8 (GPU) / 4 (CPU)
  - Optimizer   : AdamW (lr_G=2e-4, lr_D=1e-4)
  - Scheduler   : CosineAnnealingLR with warm restarts
  - Augmentation: Random flips, rotations, Gaussian noise injection
```

### Stage 2: Target Fine-Tuning (5-Class Perovskite Solar Cell SEM)
```
Dataset    : Your 5-Class Perovskite SEM Dataset (5,000+ images)
Purpose    : Specialize the foundation model to perovskite-specific
              morphological features: grain boundaries, pinholes,
              nucleation sites, surface cracks, compact films

Training Config:
  - Epochs      : 30–50 epochs with early stopping (patience=10)
  - Patch Size  : 256×256
  - Batch Size  : 8 (GPU) / 4 (CPU)
  - Optimizer   : AdamW (lr_G=5e-5, lr_D=2e-5)  ← Lower LR for fine-tuning
  - Scheduler   : ReduceLROnPlateau (monitor val_PSNR)
  - Class Conditioning: One-hot label embedding for C1–C5 morphology control
  - Loss Weights: λ1=1.0, λ2=0.1, λ3=0.05, λ4=0.01
```

---

## 📌 SECTION 4: EVALUATION FRAMEWORK

### 4.1 Computer Vision Metrics (Quantitative)

| Metric | Description | Tool / Formula |
|---|---|---|
| **PSNR (dB)** | Peak Signal-to-Noise Ratio. Higher = better pixel accuracy | $20\log_{10}\frac{255}{\sqrt{\text{MSE}}}$ |
| **SSIM** | Structural Similarity Index. Higher = better structure preservation | `skimage.metrics.structural_similarity` |
| **LPIPS** | Learned Perceptual Image Patch Similarity. Lower = more perceptually realistic | `lpips` library (VGG/AlexNet backbone) |
| **FID** | Fréchet Inception Distance. Lower = more realistic distribution | `pytorch-fid` (InceptionV3 features) |
| **KID** | Kernel Inception Distance. Unbiased FID alternative for small datasets | `torch-fidelity` |
| **RPSD Error** | Radial Power Spectrum Density Error. Frequency domain alignment | Custom NumPy FFT implementation |

### 4.2 Downstream Validation (Scientific Rigour)
- Train a **ResNet-50 / EfficientNetB3** classifier on:
  - (A) Real perovskite SEM images only
  - (B) Real + GAN-synthesized augmented images
- Compare classification mAP, per-class F1, and confusion matrices to prove GAN augmentation **statistically improves** downstream classification accuracy.

### 4.3 Visual Outputs Per Model

1. **4-Panel Comparison Grid**: LR Input | Bicubic | Model SR Output | HR Ground Truth
2. **Class-Conditional Generation Grid**: 5×5 grid (5 models × 5 perovskite classes)
3. **Latent Space t-SNE Plot**: StyleGAN2-ADA W-space visualization per morphological class
4. **Fourier Radial Power Spectrum Curves**: Spatial frequency alignment comparison
5. **FID Score Convergence Plot**: Training stability curves for all models

---

## 📌 SECTION 5: SOFTWARE ENGINEERING ARCHITECTURE

### 5.1 Project Directory Structure
```
C:\Users\Sahil\Downloads\SEM_GAN_Dissertation\
│
├── 📁 data/
│   ├── raw/                     ← Your perovskite SEM TIFF dataset (5 classes)
│   │   ├── C1_pinhole/
│   │   ├── C2_grain_boundary/
│   │   ├── C3_nucleation/
│   │   ├── C4_surface_crack/
│   │   └── C5_compact_film/
│   ├── pretrain/                ← NFFA-Europe pre-training subset (5,000+ images)
│   ├── processed/               ← Extracted and normalized 256×256 patch crops
│   └── dataset.py               ← PyTorch Dataset class (multi-class, multi-scale)
│
├── 📁 models/
│   ├── restormer_gan.py         ← Restormer-GAN (MDTA + GDFN + Dual-Domain Disc)
│   ├── swinir_gan.py            ← SwinIR-GAN (RSTB + Spatial PatchGAN)
│   ├── spade_gan.py             ← SPADE GauGAN (Semantic Layout-to-Image)
│   ├── stylegan2_ada.py         ← StyleGAN2-ADA (Synthesis + ADA Discriminator)
│   ├── ldm.py                   ← Micro-Latent Diffusion Model + ControlNet
│   └── discriminator.py         ← Shared Dual-Domain Discriminator module
│
├── 📁 losses/
│   ├── losses.py                ← L1, VGG Perceptual, 2D FFT, Hinge GAN losses
│   └── gan_loss.py              ← Vanilla GAN / Hinge / WGAN-GP loss families
│
├── 📁 metrics/
│   ├── perceptual.py            ← PSNR, SSIM, LPIPS computation
│   ├── frequency.py             ← Radial Power Spectrum Density (RPSD) error
│   └── generative.py           ← FID, KID, Precision & Recall computation
│
├── 📁 training/
│   ├── pretrain.py              ← Stage 1: NFFA SEM domain pre-training loop
│   ├── finetune.py              ← Stage 2: Perovskite SEM fine-tuning loop
│   └── trainer_base.py         ← Shared training utilities (logging, ckpt, EarlyStopping)
│
├── 📁 evaluation/
│   ├── benchmark.py             ← Full comparative benchmark runner (all 5 models)
│   ├── downstream_clf.py        ← ResNet-50 classifier validation (real vs augmented)
│   └── visualize.py             ← All visualization generation scripts
│
├── 📁 checkpoints/              ← Saved .pth weight files per model
│   ├── restormer_pretrained.pth
│   ├── restormer_finetuned.pth
│   ├── swinir_pretrained.pth
│   ├── swinir_finetuned.pth
│   ├── spade_finetuned.pth
│   ├── stylegan2_ada.pth
│   └── ldm_finetuned.pth
│
├── 📁 outputs/
│   ├── sr_comparisons/          ← Super-resolution 4-panel image grids
│   ├── gen_grids/               ← 5×5 class-conditional generation matrices
│   ├── latent_tsne/             ← StyleGAN2 latent space visualizations
│   ├── spectral_profiles/       ← Fourier RPSD alignment curves
│   └── benchmark_tables/        ← CSV + Markdown quantitative results
│
├── 📁 notebooks/                ← Jupyter notebooks for experiments & analysis
│   ├── 01_EDA.ipynb             ← Exploratory Data Analysis of perovskite SEM
│   ├── 02_Pretrain_Analysis.ipynb
│   ├── 03_Finetune_Results.ipynb
│   └── 04_Full_Benchmark.ipynb
│
├── requirements.txt             ← All Python dependencies
├── config.yaml                  ← Central training configuration (all hyperparams)
├── run_all.py                   ← 🚀 One-click master execution script
└── PROJECT_PROMPT.md            ← This document
```

### 5.2 Configuration-Driven Design (`config.yaml`)
```yaml
# config.yaml — Central Hyperparameter Registry
project:
  name: "SEM_GAN_Perovskite_Dissertation"
  seed: 42
  device: "cuda"  # auto-falls back to "cpu"

data:
  raw_dir: "data/raw"
  pretrain_dir: "data/pretrain"
  processed_dir: "data/processed"
  patch_size: 256
  stride: 128
  scale_factors: [2, 4, 8]
  num_classes: 5
  class_map:
    0: "pinhole"
    1: "grain_boundary"
    2: "nucleation"
    3: "surface_crack"
    4: "compact_film"

training:
  pretrain:
    epochs: 20
    batch_size: 8
    lr_g: 2.0e-4
    lr_d: 1.0e-4
  finetune:
    epochs: 50
    batch_size: 8
    lr_g: 5.0e-5
    lr_d: 2.0e-5
    early_stopping_patience: 10

models:
  embed_dim: 48
  num_blocks: 4
  num_heads: 4

losses:
  lambda_pixel: 1.0
  lambda_perceptual: 0.1
  lambda_fft: 0.05
  lambda_adv: 0.01
```

### 5.3 Execution Flow (`run_all.py`)
```python
# One-click execution order for entire dissertation pipeline
Step 1: Download NFFA-Europe SEM pre-training data (5,000 images)
Step 2: Parse + preprocess all FESEM TIFF images (metadata + patches)
Step 3: Stage 1 Pre-training — all 5 models on NFFA SEM dataset
Step 4: Stage 2 Fine-tuning — all 5 models on perovskite SEM dataset
Step 5: Benchmark evaluation — PSNR, SSIM, LPIPS, FID, RPSD (all models)
Step 6: Generate all visualization outputs (comparison grids, spectral plots)
Step 7: Export dissertation-quality figures (300 DPI, PDF + PNG)
Step 8: Generate final benchmark LaTeX table for dissertation inclusion
```

---

## 📌 SECTION 6: REFERENCES & OPEN-SOURCE RESOURCES

### Core Architecture Papers

| Model | Paper | Venue | Link |
|---|---|---|---|
| **Restormer** | "Restormer: Efficient Transformer for High-Resolution Image Restoration" | CVPR 2022 | https://arxiv.org/abs/2111.09881 |
| **SwinIR** | "SwinIR: Image Restoration Using Swin Transformer" | ICCVW 2021 | https://arxiv.org/abs/2108.10257 |
| **SPADE/GauGAN** | "Semantic Image Synthesis with Spatially-Adaptive Normalization" | CVPR 2019 | https://arxiv.org/abs/1903.07291 |
| **StyleGAN2-ADA** | "Training Generative Adversarial Networks with Limited Data" | NeurIPS 2020 | https://arxiv.org/abs/2006.06676 |
| **LDM** | "High-Resolution Image Synthesis with Latent Diffusion Models" | CVPR 2022 | https://arxiv.org/abs/2112.10752 |
| **ControlNet** | "Adding Conditional Control to Text-to-Image Diffusion Models" | ICCV 2023 | https://arxiv.org/abs/2302.05543 |
| **SRGAN** | "Photo-Realistic Single Image Super-Resolution Using a GAN" | CVPR 2017 | https://arxiv.org/abs/1609.04802 |
| **Real-ESRGAN** | "Real-ESRGAN: Training Real-World Blind Super-Resolution" | ICCVW 2021 | https://arxiv.org/abs/2107.10833 |
| **DDPM** | "Denoising Diffusion Probabilistic Models" | NeurIPS 2020 | https://arxiv.org/abs/2006.11239 |
| **Diffusion-GAN** | "Diffusion-GAN: Training GANs with Diffusion" | ICLR 2023 | https://arxiv.org/abs/2206.02262 |

### SEM & Perovskite-Specific Papers

| Title | Venue | Link |
|---|---|---|
| "GAN-based SEM Image Super-Resolution for Nanostructure Analysis" | Scientific Reports 2023 | https://www.nature.com/articles/s41598-023 |
| "F-ANcGAN: Attention-Enhanced CycleGAN for Nanoparticle SEM Synthesis" | arXiv 2025 | https://arxiv.org/abs/2501 |
| "Deep Learning for Automated Analysis of Perovskite Microstructure" | ACS Nano 2023 | https://pubs.acs.org |
| "Microstructure-Performance Correlation in Perovskite Solar Cells" | Energy & Environmental Science 2024 | https://pubs.rsc.org |
| "UTILE-Gen: Synthetic Dataset Generator for Nanoscience SEM" | ACS Publications 2023 | https://pubs.acs.org |

### Open-Source Dataset Resources

| Dataset | Images | Source | URL |
|---|---|---|---|
| **NFFA-Europe SEM Dataset** | 25,537 | HuggingFace | https://huggingface.co/datasets/l11p/nffa-europe-sem-dataset |
| **BAMresearch Nanoparticle SEM** | ~600 | GitHub | https://github.com/BAMresearch/automatic-sem-image-segmentation |
| **Materials Data Resources** | Curated List | GitHub | https://github.com/sedaoturak/data-resources-for-materials-science |
| **motiurinfo SEM Dataset 500** | 500 | GitHub | https://github.com/motiurinfo/SEM-Dataset-500 |

### Open-Source Code Repositories (Implementation Reference)

| Repo | Description | URL |
|---|---|---|
| `swz30/Restormer` | Official Restormer PyTorch | https://github.com/swz30/Restormer |
| `JingyunLiang/SwinIR` | Official SwinIR PyTorch | https://github.com/JingyunLiang/SwinIR |
| `NVlabs/SPADE` | Official SPADE PyTorch | https://github.com/NVlabs/SPADE |
| `NVlabs/stylegan2-ada-pytorch` | Official StyleGAN2-ADA | https://github.com/NVlabs/stylegan2-ada-pytorch |
| `CompVis/latent-diffusion` | Official LDM | https://github.com/CompVis/latent-diffusion |
| `xinntao/Real-ESRGAN` | Real-ESRGAN Inference | https://github.com/xinntao/Real-ESRGAN |
| `hellloxiaotian/GAN-SR-Survey` | GAN SR Survey List | https://github.com/hellloxiaotian/Generative-Adversarial-Networks-for-Image-Super-resolution-A-Survey |

---

## 📌 SECTION 7: EXPECTED OUTCOMES & DISSERTATION CONTRIBUTIONS

### Novel Scientific Contributions
1. **First comprehensive multi-model GAN benchmark** on 5-class perovskite solar cell FESEM imagery.
2. **Dual-Domain Discriminator (Spatial + Fourier)** as a novel architectural contribution — prevents non-physical texture hallucination in microscopy super-resolution.
3. **Two-stage domain-adaptive pre-training protocol** (NFFA SEM → Perovskite fine-tuning) demonstrated to improve convergence speed and generalization on small SEM datasets.
4. **Statistically validated downstream classification improvement** using GAN-synthesized augmentation.

### Deliverables
- ✅ Fully reproducible PyTorch codebase (GitHub repository)
- ✅ 5 fine-tuned model checkpoints (`.pth` files)
- ✅ Quantitative benchmark table (PSNR / SSIM / LPIPS / FID / RPSD)
- ✅ Dissertation-quality 300 DPI visual comparison figures
- ✅ Latent space visualizations (t-SNE / PCA)
- ✅ Fourier power spectrum alignment curves
- ✅ Final dissertation PDF chapters

### Target Publication Venues (Post-Dissertation)
- **IEEE Transactions on Image Processing (TIP)**
- **Computer Vision and Image Understanding (CVIU)**
- **npj Computational Materials** (Nature Portfolio)
- **Solar Energy Materials and Solar Cells** (Elsevier)

---

## 📌 SECTION 8: ENVIRONMENT SETUP

```bash
# Create Python virtual environment
py -m venv sem_gan_env
.\sem_gan_env\Scripts\activate

# Install all dependencies
pip install torch torchvision torchaudio
pip install numpy pillow opencv-python scikit-image matplotlib
pip install lpips timm tensorboard tqdm pyyaml
pip install pytorch-fid torch-fidelity
pip install datasets huggingface_hub    # For NFFA dataset download
pip install einops                      # For Transformer patch operations
pip install scipy                       # For statistical analysis

# Verify PyTorch CUDA availability
python -c "import torch; print(torch.cuda.is_available())"
```

---

## 📌 SECTION 9: IMMEDIATE NEXT STEPS (ACTION ITEMS)

> [!IMPORTANT]
> These are the immediate actions to begin project execution:

1. **[YOU]** Send the folder path/link to your **5-class perovskite SEM dataset**.
2. **[BUILD]** Write `data/dataset.py` — TIFF parser + multi-class patch dataset.
3. **[BUILD]** Write `training/pretrain.py` — NFFA SEM domain pre-training loop.
4. **[BUILD]** Write all model `.py` files with pre-trained weight loading via `timm` / HuggingFace.
5. **[BUILD]** Write `training/finetune.py` — class-conditional perovskite fine-tuning loop.
6. **[BUILD]** Write `evaluation/benchmark.py` — full automated benchmark pipeline.
7. **[TRAIN]** Execute `run_all.py` — one-click full project execution.
8. **[WRITE]** Generate dissertation chapter plots and LaTeX tables.

---

*This document is the primary scientific and engineering specification for the M.Tech Dissertation project. It will be updated incrementally as the project progresses. Last Updated: July 2026.*

# Sparsity-Aware Diffusion Model

A PyTorch implementation of a novel sparsity-aware diffusion model that learns to reconstruct full images from sparse observations.

## 🎯 Key Innovation

The model demonstrates a remarkable capability:
- **Input**: 20% of image pixels (sparse conditioning)
- **Training Target**: Different 20% of pixels (loss computation)
- **Output**: Full 100% image reconstruction

This approach enables the model to learn complete image distributions from partial observations.

## 🏗️ Architecture

### Sparsity-Aware Training Pipeline

```
┌─────────────────────────────────────────────┐
│  Original Image (100%)                      │
│  ████████████████████████████████           │
└─────────────────────────────────────────────┘
           │
           ├──────────────┬──────────────┐
           ▼              ▼              ▼
    ┌─────────┐    ┌─────────┐   ┌─────────┐
    │  Cond   │    │ Target  │   │ Unseen  │
    │  Mask   │    │  Mask   │   │   60%   │
    │  (20%)  │    │  (20%)  │   │         │
    └─────────┘    └─────────┘   └─────────┘
         │              │
         │              │
         ▼              │
    ┌──────────────┐    │
    │  U-Net with  │    │
    │  Diffusion   │◄───┘ (loss computed here)
    │              │
    └──────────────┘
         │
         ▼
    ┌──────────────┐
    │ Reconstructed│
    │ Full Image   │
    │   (100%)     │
    └──────────────┘
```

### Model Components

1. **SparsityController**: Manages mask generation
   - `random_epoch`: Consistent masks per sample within epoch
   - Separate conditioning and target masks
   - Configurable sparsity levels

2. **U-Net Architecture**:
   - Input channels: `3 × C` (noised_image + sparse_input + mask)
   - Multi-resolution processing with attention
   - Time-conditioned residual blocks

3. **Gaussian Diffusion**:
   - 1000-step diffusion process
   - Weighted loss: 1.0 on target pixels, 0.05 on conditioning pixels
   - DDPM sampling with clipping

## 📊 CIFAR-10 Results

Trained on CIFAR-10 dataset (32×32 RGB images):
- **Dataset**: 50,000 training images, 10,000 test images
- **Sparsity**: 20% conditioning + 20% target = 40% total supervision
- **Model**: ~15M parameters

## 🚀 Quick Start

### Installation

```bash
pip install -r requirements.txt
```

### Training

Open and run the Jupyter notebook:

```bash
jupyter notebook sparsity_diffusion_cifar10.ipynb
```

The notebook is self-contained and includes:
- ✅ All model architectures
- ✅ Training loop with progress bars
- ✅ Visualization utilities
- ✅ Checkpoint saving/loading
- ✅ Sample generation

### Configuration

Key hyperparameters in the notebook:

```python
IMAGE_SIZE = 32          # CIFAR-10 image size
BATCH_SIZE = 128         # Batch size
LEARNING_RATE = 2e-4     # Adam learning rate
EPOCHS = 50              # Training epochs
SPARSITY = 0.2           # 20% sparsity level
```

## 📁 Project Structure

```
FromNF/
├── sparsity_diffusion_cifar10.ipynb  # Main training notebook
├── ref/                               # Reference implementations
│   ├── diffusion (8) (1).py          # Original diffusion code
│   ├── model_original (7) (1).py     # Original U-Net model
│   └── trainer (6) (1).py            # Original trainer
├── results/                           # Generated samples (created during training)
├── checkpoints/                       # Model checkpoints (created during training)
├── data/                              # CIFAR-10 dataset (auto-downloaded)
├── README.md                          # This file
└── requirements.txt                   # Python dependencies
```

## 🔬 How It Works

### Training Process

1. **Mask Generation**: For each sample, generate two non-overlapping 20% masks
   ```python
   cond_mask: Binary mask for conditioning pixels
   target_mask: Binary mask for loss computation (disjoint from cond_mask)
   ```

2. **Forward Pass**:
   ```python
   # Concatenate inputs
   model_input = [noised_image, sparse_input, cond_mask]

   # Predict noise
   predicted_noise = unet(model_input, timestep)

   # Compute weighted loss
   loss = weighted_mse(predicted_noise, true_noise, target_mask)
   ```

3. **Sampling**: During inference, use sparse conditioning to guide reconstruction
   ```python
   # Start from noise
   x_T ~ N(0, I)

   # Iteratively denoise with conditioning
   for t in [T, T-1, ..., 1]:
       x_{t-1} = denoise(x_t, sparse_input, mask, t)

   # Return final sample
   return x_0
   ```

### Why This Works

1. **Statistical Coverage**: Across batches, every pixel appears in both conditioning and target masks
2. **Contextual Learning**: Model learns spatial correlations from observed to unobserved pixels
3. **Iterative Refinement**: 1000-step reverse diffusion enables progressive reconstruction
4. **Implicit Regularization**: Small loss weight on conditioning pixels prevents overfitting

## 📈 Expected Results

### Training Progress

- **Early epochs (1-10)**: Blurry reconstructions, high loss
- **Mid-training (10-30)**: Recognizable structures, decreasing loss
- **Late training (30-50)**: Sharp details, stable loss

### Visualization Outputs

The notebook generates:
- Training loss curves
- Sample reconstructions every 5 epochs
- Visualization of masks and reconstructions
- Final high-quality samples

## 🛠️ Customization

### Different Sparsity Levels

```python
sparsity_controller = SparsityController(
    image_size=32,
    sparsity=0.1,  # Try 10%, 30%, 40%
    ...
)
```

### Different Patterns

Extend `SparsityController` to support:
- Block-based sparsity
- Grid patterns
- Random walks
- Learned masks

### Model Scaling

```python
unet = Unet(
    dim=128,  # Increase from 64 for larger model
    dim_multiply=(1, 2, 4, 8, 16),  # Add more layers
    ...
)
```

## 📚 Citation

If you use this code, please cite:

```bibtex
@misc{sparsity-aware-diffusion,
  title={Sparsity-Aware Diffusion Model},
  author={FromNF Team},
  year={2024},
  url={https://github.com/colpark/FromNF}
}
```

## 🤝 Contributing

Contributions are welcome! Areas for improvement:
- [ ] DDIM sampler for faster inference
- [ ] Multi-scale training
- [ ] Conditional generation by class
- [ ] FID score evaluation
- [ ] Wandb integration

## 📄 License

MIT License - see LICENSE file for details

## 🙏 Acknowledgments

Based on:
- **DDPM**: Denoising Diffusion Probabilistic Models (Ho et al., 2020)
- **DDIM**: Denoising Diffusion Implicit Models (Song et al., 2021)
- Original reference implementation in `ref/` directory

---

**Status**: 🚀 Ready for experimentation

**Last Updated**: October 2024

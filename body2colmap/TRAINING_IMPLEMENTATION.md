# gsplat Training Implementation Guide

This guide documents how to implement a 3D Gaussian Splat trainer using gsplat with COLMAP input data. It captures key lessons learned and non-obvious gotchas.

## Overview

The trainer reads a standard COLMAP dataset and trains 3D Gaussians using gsplat's `rasterization()` function and `DefaultStrategy` for densification.

```
COLMAP Dataset:
├── sparse/0/
│   ├── cameras.txt    # Camera intrinsics (PINHOLE model)
│   ├── images.txt     # Camera extrinsics (quaternion + translation)
│   └── points3D.txt   # Initial 3D point cloud
└── images/
    └── *.png          # Training images (may have alpha channel)
```

## Core Components

### 1. COLMAP Data Loading

Parse the three COLMAP text files:

```python
# cameras.txt → intrinsic matrix K
# For PINHOLE model: fx, fy, cx, cy
K = [[fx, 0, cx], [0, fy, cy], [0, 0, 1]]

# images.txt → camera-to-world matrices
# COLMAP stores world-to-camera as quaternion (qw,qx,qy,qz) + translation (tx,ty,tz)
R_w2c = quaternion_to_rotation(qw, qx, qy, qz)
t_w2c = [tx, ty, tz]

# Invert to get camera-to-world
R_c2w = R_w2c.T
t_c2w = -R_c2w @ t_w2c
c2w = [[R_c2w, t_c2w], [0, 0, 0, 1]]  # 4x4 matrix

# points3D.txt → initial point positions and colors
points = [(x, y, z), ...]
colors = [(r, g, b), ...]  # 0-255
```

### 2. Gaussian Initialization

Initialize Gaussian parameters from the point cloud:

```python
means = points  # (N, 3)
scales = log(average_knn_distance).expand(N, 3)  # Use K=3 nearest neighbors
quats = [1, 0, 0, 0] for each point  # Identity rotation (wxyz)
opacities = logit(0.1) for each point  # ~-2.2 in logit space
sh0 = (colors/255 - 0.5) / 0.28209  # DC spherical harmonic coefficient
shN = zeros(N, (sh_degree+1)^2 - 1, 3)  # Higher-order SH (start at zero)
```

### 3. Training Loop

```python
for step in range(max_steps):
    # Sample random view
    idx = random.choice(n_images)
    rgb, alpha = load_image(idx)
    viewmat = inverse(c2w[idx])
    K = intrinsics[idx]

    # Rasterize
    renders, render_alphas, info = gsplat.rasterization(
        means=means,
        quats=quats,
        scales=exp(scales),
        opacities=sigmoid(opacities),
        colors=cat([sh0, shN], dim=1),
        viewmats=viewmat.unsqueeze(0),
        Ks=K.unsqueeze(0),
        width=width,
        height=height,
        sh_degree=current_sh_degree,
        backgrounds=bg_color,
    )

    # Loss + backward
    loss = compute_loss(renders, rgb, alpha)
    loss.backward()

    # Densification (DefaultStrategy handles grow/prune)
    strategy.step_pre_backward(...)
    strategy.step_post_backward(...)

    # Optimizer step
    optimizer.step()
```

## Critical Gotchas

### 1. K Matrix Scaling for Resized Images

**Problem**: The K matrix is computed for specific image dimensions. If actual images differ in size (e.g., resized), projection will be wrong.

**Solution**: Scale K matrix proportionally:
```python
if actual_size != colmap_size:
    scale_x = actual_width / colmap_width
    scale_y = actual_height / colmap_height
    K[0, 0] *= scale_x  # fx
    K[0, 2] *= scale_x  # cx
    K[1, 1] *= scale_y  # fy
    K[1, 2] *= scale_y  # cy
```

### 2. Alpha-Weighted Loss Normalization

**Problem**: When masking L1 loss by alpha, naive implementation is 3× too large:
```python
# WRONG - sums over 3 RGB channels but normalizes by 1-channel alpha sum
l1 = (abs(render - gt) * alpha).sum() / alpha.sum()
```

**Solution**: Expand alpha to match RGB channels:
```python
# CORRECT - both numerator and denominator sum over same dimensions
alpha_rgb = alpha.expand_as(render)  # (B, H, W, 1) → (B, H, W, 3)
l1 = (abs(render - gt) * alpha_rgb).sum() / alpha_rgb.sum()
```

### 3. Alpha Compositing for Transparent Images

When training images have transparency (alpha channel):

```python
# Composite GT over background color (same as gsplat does internally)
bg = torch.tensor([0.0, 0.0, 0.0])
gt_composited = rgb * alpha + bg * (1 - alpha)

# Pass same background to gsplat
renders, _, _ = rasterization(..., backgrounds=bg.expand(batch, 3))

# Loss compares composited images, masked by alpha
```

### 4. No Coordinate Transforms Needed

COLMAP data is self-consistent. Do NOT apply OpenGL↔OpenCV rotations when loading - just use:
```python
viewmat = inverse(c2w)  # That's it
```

The cameras and points are already in the same coordinate system.

### 5. Scene Scale for Learning Rates

Compute scene scale from camera positions for learning rate adjustment:
```python
cam_positions = c2w[:, :3, 3]
center = cam_positions.mean(axis=0)
scene_scale = max(norm(cam_positions - center, axis=1))

lr_means = base_lr * scene_scale  # Position LR scales with scene
```

## Recommended Hyperparameters

```python
max_steps = 30_000
batch_size = 1
sh_degree = 3
sh_degree_interval = 1000  # Increase SH degree every N steps

# Learning rates (scaled by sqrt(batch_size) internally)
lr_means = 1.6e-4 * scene_scale
lr_scales = 5e-3
lr_quats = 1e-3
lr_opacities = 5e-2
lr_sh0 = 0.25
lr_shN = lr_sh0 / 20

# Loss
ssim_lambda = 0.2  # L1 * 0.8 + (1-SSIM) * 0.2

# Densification (DefaultStrategy)
refine_start = 500
refine_stop = 15_000
refine_every = 100
grow_grad2d = 0.0002
prune_opacity = 0.005
```

## Output Format

Save trained Gaussians as PLY file compatible with standard 3DGS viewers:
```
x, y, z           # Position
scale_0/1/2       # Log-space scales
rot_0/1/2/3       # Quaternion (wxyz)
opacity           # Logit-space opacity
f_dc_0/1/2        # DC spherical harmonic (RGB)
f_rest_*          # Higher-order SH coefficients
```

## Minimal Debug Logging

Keep these diagnostics (print once at start):
- Number of images and initial points
- Scene scale
- K matrix values and image dimensions (catches size mismatch)
- Alpha statistics if present (catches inverted alpha)

Remove verbose per-step logging in production.

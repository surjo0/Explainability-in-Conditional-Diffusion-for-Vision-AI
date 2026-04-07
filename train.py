#!/usr/bin/env python3
import time
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps

import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from denoising_diffusion_pytorch import Unet, GaussianDiffusion

# -------- CONFIG --------
TRAIN_DIR = Path("/home/user10/USERS/Sujo/medical/Dataset/DIV2K/DIV2K_train_HR")
OUT_DIR   = Path("/home/user10/USERS/Sujo/medical/Dataset/Output/model")

IMAGE_SIZE = 512        
BATCH_SIZE = 2
EPOCHS     = 500
SAVE_EVERY = 30
LR         = 2e-4
TIMESTEPS  = 1000
WAIT_SECS  = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"VRAM total: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# -------- DATASET --------
class TrainDataset(Dataset):
    def __init__(self, root):
        self.files = sorted([
            p for p in Path(root).rglob("*")
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        if len(self.files) == 0:
            raise RuntimeError(f"No images found in {root}")
        print(f"Found {len(self.files)} training images.")

        self.t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        img = Image.open(self.files[i]).convert("RGB")
        img = ImageOps.fit(img, (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        return self.t(img)

# -------- MODEL --------
# 5 levels for 512px: 512→256→128→64→32→16
# dim=64 gives strong capacity without OOM
unet = Unet(
    dim       = 64,
    dim_mults = (1, 2, 4, 8, 16),
    channels  = 3,
)

diffusion = GaussianDiffusion(
    unet,
    image_size = IMAGE_SIZE,
    timesteps  = TIMESTEPS,
).to(DEVICE)

opt       = torch.optim.Adam(diffusion.parameters(), lr=LR)
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=EPOCHS)
scaler    = torch.amp.GradScaler('cuda')

total_params = sum(p.numel() for p in diffusion.parameters()) / 1e6
print(f"Model parameters: {total_params:.1f}M")

# -------- DIRS --------
OUT_DIR.mkdir(parents=True, exist_ok=True)
(OUT_DIR / "checkpoints").mkdir(exist_ok=True)

# -------- DATA --------
dataset = TrainDataset(TRAIN_DIR)
loader  = DataLoader(
    dataset,
    batch_size      = BATCH_SIZE,
    shuffle         = True,
    num_workers     = 4,
    pin_memory      = True,
    prefetch_factor = 2,
)

# -------- RESUME --------
start_epoch = 1
resume_path = OUT_DIR / "latest.pth"
if resume_path.exists():
    ckpt = torch.load(resume_path, map_location=DEVICE)
    diffusion.load_state_dict(ckpt["model"])
    opt.load_state_dict(ckpt["opt"])
    scheduler.load_state_dict(ckpt["scheduler"])
    scaler.load_state_dict(ckpt["scaler"])
    start_epoch = ckpt["epoch"] + 1
    print(f"Resumed from epoch {ckpt['epoch']} | Loss: {ckpt['loss']:.6f}")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, last_epoch=ckpt["epoch"]
    )
else:
    print("Starting fresh training.")

# -------- TRAIN LOOP --------
for ep in range(start_epoch, EPOCHS + 1):
    diffusion.train()
    losses = []
    epoch_start = time.time()

    for batch_idx, images in enumerate(loader):
        images = images.to(DEVICE, non_blocking=True)

        with torch.amp.autocast('cuda'):
            loss = diffusion(images)

        if torch.isnan(loss) or torch.isinf(loss):
            print(f"  Warning: NaN/Inf at epoch {ep} batch {batch_idx+1}, skipping.")
            opt.zero_grad()
            # ✅ also clear cache to prevent fragmentation buildup
            torch.cuda.empty_cache()
            continue

        opt.zero_grad()
        scaler.scale(loss).backward()
        scaler.unscale_(opt)
        torch.nn.utils.clip_grad_norm_(diffusion.parameters(), 1.0)
        scaler.step(opt)
        scaler.update()

        losses.append(loss.item())

        if (batch_idx + 1) % 20 == 0:
            mem = torch.cuda.memory_allocated() / 1e9
            print(f"  Epoch {ep} | Batch {batch_idx+1}/{len(loader)} "
                  f"| Loss: {loss.item():.6f} | VRAM: {mem:.1f}GB")

    # ✅ clear cache at end of every epoch to prevent fragmentation
    torch.cuda.empty_cache()
    scheduler.step()

    avg_loss   = float(np.mean(losses)) if losses else float('nan')
    current_lr = scheduler.get_last_lr()[0]
    elapsed    = time.time() - epoch_start
    print(f"Epoch {ep:03d}/{EPOCHS} | Avg Loss: {avg_loss:.6f} "
          f"| LR: {current_lr:.2e} | Time: {elapsed:.0f}s")

    torch.save({
        "epoch":     ep,
        "model":     diffusion.state_dict(),
        "opt":       opt.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler":    scaler.state_dict(),
        "loss":      avg_loss,
    }, OUT_DIR / "latest.pth")

    if ep % SAVE_EVERY == 0:
        for old in (OUT_DIR / "checkpoints").glob("epoch_*.pth"):
            old.unlink()
        torch.save(
            diffusion.state_dict(),
            OUT_DIR / f"checkpoints/epoch_{ep:03d}.pth"
        )
        print(f"  Checkpoint saved at epoch {ep}")

    if ep < EPOCHS and WAIT_SECS > 0:
        time.sleep(WAIT_SECS)

torch.save(diffusion.state_dict(), OUT_DIR / "final_model.pth")
print("Training complete. Final model saved.")
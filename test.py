#!/usr/bin/env python3
import math, random, json
from pathlib import Path
import numpy as np
from PIL import Image, ImageOps, ImageDraw, ImageFont

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from torchvision import transforms

from denoising_diffusion_pytorch import Unet, GaussianDiffusion
import lpips
from pytorch_fid.fid_score import calculate_fid_given_paths

# Real-ESRGAN for 1024px final output
from realesrgan import RealESRGANer
from basicsr.archs.rrdbnet_arch import RRDBNet

# -------- PATHS --------
ROOT = Path("/home/user10/USERS/Sujo/medical/Dataset")

HQ_DIR    = ROOT / "DIV2K/DIV2K_valid_HR"
NOISE_DIR = ROOT / "5-10% GaussianNoice"
PATCH_DIR = ROOT / "5-10%_WhitePatches"

MODEL_PATH = ROOT / "Output/model/final_model.pth"
OUT_DIR    = ROOT / "Output/test_results"

# -------- CONFIG — must match train.py exactly --------
IMAGE_SIZE    = 512         # ✅ train and reconstruct at 512
TIMESTEPS     = 1000
LR            = 2e-4
EPOCHS        = 300
T_START       = 250
PATCH_SIZE    = 64          # 512 / 8 = 64
EXPLAIN_STEPS = [249, 187, 124, 62, 0]
FINAL_SIZE    = 1024        # ✅ upscale to 1024 after reconstruction

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# -------- LOAD DIFFUSION MODEL --------
unet = Unet(
    dim       = 64,
    dim_mults = (1, 2, 4, 8, 16),   # ✅ must match train.py
    channels  = 3,
)
diffusion = GaussianDiffusion(
    unet,
    image_size = IMAGE_SIZE,
    timesteps  = TIMESTEPS,
).to(DEVICE)

state = torch.load(MODEL_PATH, map_location=DEVICE)
diffusion.load_state_dict(state)
diffusion.eval()
print("Diffusion model loaded.")

# -------- LOAD REAL-ESRGAN UPSCALER --------
# Downloads weights automatically on first run
esrgan_model = RRDBNet(
    num_in_ch=3, num_out_ch=3,
    num_feat=64, num_block=23, num_grow_ch=32,
    scale=2
)
upsampler = RealESRGANer(
    scale      = 2,
    model_path = "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
    model      = esrgan_model,
    tile       = 512,       # ✅ tiled inference — safe on any GPU
    tile_pad   = 10,
    pre_pad    = 0,
    half       = True,      # fp16 for speed
    device     = DEVICE,
)
print("Real-ESRGAN upscaler loaded.")

lpips_net = lpips.LPIPS(net="alex").to(DEVICE)
OUT_DIR.mkdir(parents=True, exist_ok=True)

# -------- DATASET --------
class PairedDataset(Dataset):
    def __init__(self, corrupt_dir, hq_dir):
        self.files = sorted([
            p for p in Path(hq_dir).iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg"}
        ])
        self.corrupt_dir = Path(corrupt_dir)
        self.t = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.5]*3, [0.5]*3)
        ])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        hq      = Image.open(self.files[i]).convert("RGB")
        c_path  = self.corrupt_dir / self.files[i].name
        corrupt = Image.open(c_path).convert("RGB") if c_path.exists() else hq.copy()
        # ✅ resize input to 512 for model, then upscale output to 1024
        hq      = ImageOps.fit(hq,      (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        corrupt = ImageOps.fit(corrupt, (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        return self.t(corrupt), self.t(hq), self.files[i].name

# -------- HELPERS --------
def to_numpy_img(tensor):
    arr = tensor[0].permute(1, 2, 0).cpu().float().numpy()
    arr = ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)
    return arr

def upscale_to_1024(numpy_img_512):
    """
    Takes a 512x512 uint8 numpy array (H,W,3),
    returns a 1024x1024 uint8 numpy array via Real-ESRGAN.
    Falls back to bicubic if ESRGAN fails.
    """
    try:
        # RealESRGANer expects BGR
        import cv2
        bgr = cv2.cvtColor(numpy_img_512, cv2.COLOR_RGB2BGR)
        out_bgr, _ = upsampler.enhance(bgr, outscale=2)
        out_rgb = cv2.cvtColor(out_bgr, cv2.COLOR_BGR2RGB)
        # ensure exactly 1024x1024
        if out_rgb.shape[:2] != (FINAL_SIZE, FINAL_SIZE):
            out_rgb = np.array(
                Image.fromarray(out_rgb).resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS)
            )
        return out_rgb
    except Exception as e:
        print(f"  ESRGAN failed ({e}), falling back to bicubic.")
        return np.array(
            Image.fromarray(numpy_img_512).resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS)
        )

# -------- RECONSTRUCTION (SDEdit) --------
@torch.no_grad()
def reconstruct_from_noisy(cond_img, t_start=T_START):
    t_tensor = torch.tensor([t_start], device=DEVICE)
    noise    = torch.randn_like(cond_img)
    x        = diffusion.q_sample(cond_img, t_tensor, noise)

    for t in reversed(range(t_start)):
        tt = torch.full((1,), t, device=DEVICE, dtype=torch.long)

        with torch.amp.autocast('cuda'):
            pred_noise = diffusion.model(x, tt)

        ab      = diffusion.alphas_cumprod[t]
        ab_prev = diffusion.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=DEVICE)
        beta    = diffusion.betas[t]
        alpha   = 1 - beta

        x0_pred = (x - torch.sqrt(1 - ab) * pred_noise) / torch.sqrt(ab)
        x0_pred = x0_pred.clamp(-1, 1)

        coef1 = torch.sqrt(ab_prev) * beta / (1 - ab)
        coef2 = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab)
        mean  = coef1 * x0_pred + coef2 * x

        if t > 0:
            variance = beta * (1 - ab_prev) / (1 - ab)
            x = mean + torch.sqrt(variance) * torch.randn_like(x)
        else:
            x = mean

    return x.clamp(-1, 1)

# -------- METRICS --------
def evaluate(corrupt_dir, tag):
    dataset      = PairedDataset(corrupt_dir, HQ_DIR)
    recon512_dir = OUT_DIR / f"recon512_{tag}"
    recon1024_dir = OUT_DIR / f"recon1024_{tag}"
    recon512_dir.mkdir(exist_ok=True)
    recon1024_dir.mkdir(exist_ok=True)

    psnr_vals, lpips_vals = [], []

    for idx in range(len(dataset)):
        cond, gt, name = dataset[idx]
        cond = cond[None].to(DEVICE)
        gt   = gt[None].to(DEVICE)

        # Reconstruct at 512px
        rec      = reconstruct_from_noisy(cond, t_start=T_START)
        rec_512  = to_numpy_img(rec)

        # Save 512px reconstruction
        Image.fromarray(rec_512).save(recon512_dir / name)

        # ✅ Upscale to 1024px and save
        rec_1024 = upscale_to_1024(rec_512)
        Image.fromarray(rec_1024).save(recon1024_dir / name)

        # Metrics computed at 512px (same scale as GT)
        mse  = F.mse_loss(rec, gt).item()
        psnr = 10 * math.log10(4.0 / (mse + 1e-10))
        psnr_vals.append(psnr)

        lp = lpips_net(rec, gt).item()
        lpips_vals.append(lp)

        print(f"  [{tag}] {name} | PSNR: {psnr:.2f} dB | LPIPS: {lp:.4f}")

    # FID on 512px reconstructions (stable, no OOM)
    try:
        fid = calculate_fid_given_paths(
            [str(recon512_dir), str(HQ_DIR)],
            batch_size = 8,
            device     = DEVICE,
            dims       = 2048
        )
    except Exception as e:
        print(f"  FID failed: {e}")
        fid = -1.0

    metrics = {
        "PSNR":  float(np.mean(psnr_vals)),
        "LPIPS": float(np.mean(lpips_vals)),
        "FID":   float(fid)
    }
    print(f"\n  [{tag}] PSNR: {metrics['PSNR']:.2f} dB "
          f"| LPIPS: {metrics['LPIPS']:.4f} | FID: {metrics['FID']:.2f}")
    print(f"  512px saved to: {recon512_dir}")
    print(f"  1024px saved to: {recon1024_dir}\n")
    return metrics

# -------- EXPLAINABILITY --------
def save_row(images, labels, filename, extra_text=""):
    n      = len(images)
    margin = 100
    canvas = Image.new("RGB", (IMAGE_SIZE * n + margin, IMAGE_SIZE + 80), (255, 255, 255))
    for i, img in enumerate(images):
        canvas.paste(Image.fromarray(img), (margin + i * IMAGE_SIZE, 0))
    draw = ImageDraw.Draw(canvas)
    try:
        font      = ImageFont.truetype("DejaVuSans.ttf", 20)
        font_side = ImageFont.truetype("DejaVuSans.ttf", 16)
    except:
        font = font_side = ImageFont.load_default()
    for i, t in enumerate(labels):
        draw.text((margin + i * IMAGE_SIZE + IMAGE_SIZE // 3, IMAGE_SIZE + 10),
                  f"t={t}", fill=(0,0,0), font=font)
    if extra_text:
        draw.multiline_text((8, IMAGE_SIZE // 3), extra_text, fill=(0,0,0), font=font_side)
    canvas.save(OUT_DIR / filename)
    print(f"  Saved: {filename}")

@torch.no_grad()
def generate_explainability(dataset, tag):
    cond, gt, name = dataset[random.randint(0, len(dataset) - 1)]
    cond = cond[None].to(DEVICE)

    t_tensor = torch.tensor([T_START], device=DEVICE)
    noise    = torch.randn_like(cond)
    x        = diffusion.q_sample(cond, t_tensor, noise)

    traj, score_maps, patches = [], [], []

    for t in reversed(range(T_START)):
        tt = torch.full((1,), t, device=DEVICE, dtype=torch.long)

        with torch.amp.autocast('cuda'):
            pred_noise = diffusion.model(x, tt)

        ab      = diffusion.alphas_cumprod[t]
        ab_prev = diffusion.alphas_cumprod[t - 1] if t > 0 else torch.tensor(1.0, device=DEVICE)
        beta    = diffusion.betas[t]
        alpha   = 1 - beta

        x0_pred = (x - torch.sqrt(1 - ab) * pred_noise) / torch.sqrt(ab)
        x0_pred = x0_pred.clamp(-1, 1)
        coef1   = torch.sqrt(ab_prev) * beta / (1 - ab)
        coef2   = torch.sqrt(alpha) * (1 - ab_prev) / (1 - ab)
        mean    = coef1 * x0_pred + coef2 * x

        if t > 0:
            variance = beta * (1 - ab_prev) / (1 - ab)
            x = mean + torch.sqrt(variance) * torch.randn_like(x)
        else:
            x = mean

        if t in EXPLAIN_STEPS:
            img = to_numpy_img(x)
            traj.append(img)

            s = pred_noise.abs().mean(1)[0].cpu().float().numpy()
            s = ((s - s.min()) / (s.max() - s.min() + 1e-8) * 255).astype(np.uint8)
            score_maps.append(np.stack([s]*3, axis=-1))

            y, xp = np.unravel_index(np.argmax(s), s.shape)
            y  = min(y,  IMAGE_SIZE - PATCH_SIZE)
            xp = min(xp, IMAGE_SIZE - PATCH_SIZE)
            p  = Image.fromarray(img[y:y+PATCH_SIZE, xp:xp+PATCH_SIZE])
            patches.append(np.array(p.resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)))

    traj.reverse(); score_maps.reverse(); patches.reverse()
    labels = EXPLAIN_STEPS[::-1]
    extra  = f"LR={LR}\nEpochs={EPOCHS}"
    save_row(traj,       labels, f"trajectory_{tag}.png",  extra)
    save_row(score_maps, labels, f"score_maps_{tag}.png",  extra)
    save_row(patches,    labels, f"patches_{tag}.png",     extra)

# -------- RUN ALL 3 CASES --------
results = {}

print("\n===== CASE 1: Clean (HQ → HQ) =====")
results["clean"] = evaluate(HQ_DIR, "clean")
generate_explainability(PairedDataset(HQ_DIR, HQ_DIR), "clean")

print("\n===== CASE 2: Noisy Input =====")
results["noise"] = evaluate(NOISE_DIR, "noise")
generate_explainability(PairedDataset(NOISE_DIR, HQ_DIR), "noise")

print("\n===== CASE 3: Patched/Occluded Input =====")
results["patch"] = evaluate(PATCH_DIR, "patch")
generate_explainability(PairedDataset(PATCH_DIR, HQ_DIR), "patch")

json.dump(results, open(OUT_DIR / "metrics.json", "w"), indent=2)
print("\n===== FINAL RESULTS =====")
print(json.dumps(results, indent=2))
print(f"\nAll outputs saved to: {OUT_DIR}")
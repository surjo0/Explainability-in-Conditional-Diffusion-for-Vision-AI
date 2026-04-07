#!/usr/bin/env python3
import argparse, math, json
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

parser = argparse.ArgumentParser()
parser.add_argument("--checkpoint", type=str, required=True)
args = parser.parse_args()

ROOT      = Path("/home/user10/USERS/Sujo/medical/Dataset")
HQ_DIR    = ROOT / "DIV2K/DIV2K_valid_HR"
NOISE_DIR = ROOT / "5-10% GaussianNoice"
PATCH_DIR = ROOT / "5-10%_WhitePatches"
MODEL_DIR = ROOT / "Output/model"

if args.checkpoint == "latest.pth":
    CKPT_PATH = MODEL_DIR / "latest.pth"
else:
    CKPT_PATH = MODEL_DIR / "checkpoints" / args.checkpoint

epoch_tag = args.checkpoint.replace(".pth", "")
OUT_DIR   = ROOT / f"Output/test_results_{epoch_tag}"
OUT_DIR.mkdir(parents=True, exist_ok=True)

IMAGE_SIZE    = 512
TIMESTEPS     = 1000
T_START       = 250
T_EXPLAIN     = 1000
EXPLAIN_STEPS = [999, 750, 500, 250, 0]
PATCH_SIZE    = 64
FINAL_SIZE    = 1024
NUM_IMAGES    = 5

STEP_LABELS = {
    999: "Pure noise\nt=999",
    750: "Mostly noise\nt=750",
    500: "Half denoised\nt=500",
    250: "Nearly clean\nt=250",
    0:   "Reconstructed\nt=0",
}

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")
print(f"Loading: {CKPT_PATH}")

unet      = Unet(dim=64, dim_mults=(1, 2, 4, 8, 16), channels=3)
diffusion = GaussianDiffusion(unet, image_size=IMAGE_SIZE, timesteps=TIMESTEPS).to(DEVICE)

ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=True)
if "model" in ckpt:
    diffusion.load_state_dict(ckpt["model"])
    print(f"Epoch {ckpt.get('epoch', '?')} | Loss: {ckpt.get('loss', 0):.6f}")
else:
    diffusion.load_state_dict(ckpt)
    print("Model weights loaded.")
diffusion.eval()

lpips_net = lpips.LPIPS(net="alex").to(DEVICE)

try:
    from realesrgan import RealESRGANer
    from basicsr.archs.rrdbnet_arch import RRDBNet
    import cv2
    esrgan_model = RRDBNet(
        num_in_ch=3, num_out_ch=3,
        num_feat=64, num_block=23, num_grow_ch=32, scale=2
    )
    upsampler = RealESRGANer(
        scale=2,
        model_path="https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x2plus.pth",
        model=esrgan_model, tile=512, tile_pad=10,
        pre_pad=0, half=True, device=DEVICE,
    )
    USE_ESRGAN = True
    print("ESRGAN loaded.")
except Exception as e:
    USE_ESRGAN = False
    print(f"ESRGAN unavailable, using bicubic. ({e})")


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
        hq      = ImageOps.fit(hq,      (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        corrupt = ImageOps.fit(corrupt, (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
        return self.t(corrupt), self.t(hq), self.files[i].name


# -------- HELPERS --------
def to_numpy_img(tensor):
    arr = tensor[0].permute(1, 2, 0).cpu().float().numpy()
    return ((arr + 1) / 2 * 255).clip(0, 255).astype(np.uint8)

def upscale_to_1024(img_512):
    if USE_ESRGAN:
        try:
            bgr     = cv2.cvtColor(img_512, cv2.COLOR_RGB2BGR)
            out, _  = upsampler.enhance(bgr, outscale=2)
            out_rgb = cv2.cvtColor(out, cv2.COLOR_BGR2RGB)
            if out_rgb.shape[:2] != (FINAL_SIZE, FINAL_SIZE):
                out_rgb = np.array(
                    Image.fromarray(out_rgb).resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS)
                )
            return out_rgb
        except Exception as e:
            print(f"  ESRGAN failed ({e}), bicubic fallback.")
    return np.array(Image.fromarray(img_512).resize((FINAL_SIZE, FINAL_SIZE), Image.LANCZOS))


# -------- RECONSTRUCTION --------
@torch.no_grad()
def reconstruct(cond_img):
    x = diffusion.q_sample(
        cond_img,
        torch.tensor([T_START], device=DEVICE),
        torch.randn_like(cond_img)
    )
    for t in reversed(range(T_START)):
        tt = torch.full((1,), t, device=DEVICE, dtype=torch.long)
        with torch.amp.autocast('cuda'):
            pred_noise = diffusion.model(x, tt)

        ab      = diffusion.alphas_cumprod[t]
        ab_prev = diffusion.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=DEVICE)
        beta    = diffusion.betas[t]

        x0_pred = ((x - torch.sqrt(1 - ab) * pred_noise) / torch.sqrt(ab)).clamp(-1, 1)
        coef1   = torch.sqrt(ab_prev) * beta / (1 - ab)
        coef2   = torch.sqrt(1 - beta) * (1 - ab_prev) / (1 - ab)
        mean    = coef1 * x0_pred + coef2 * x

        if t > 0:
            x = mean + torch.sqrt(beta * (1 - ab_prev) / (1 - ab)) * torch.randn_like(x)
        else:
            x = mean

    return x.clamp(-1, 1)


# -------- METRICS --------
def evaluate(corrupt_dir, tag):
    dataset       = PairedDataset(corrupt_dir, HQ_DIR)
    recon512_dir  = OUT_DIR / f"recon512_{tag}"
    recon1024_dir = OUT_DIR / f"recon1024_{tag}"
    hq5_dir       = OUT_DIR / f"hq5_{tag}"
    recon512_dir.mkdir(exist_ok=True)
    recon1024_dir.mkdir(exist_ok=True)
    hq5_dir.mkdir(exist_ok=True)

    psnr_vals, lpips_vals = [], []

    for idx in range(min(NUM_IMAGES, len(dataset))):
        cond, gt, name = dataset[idx]
        cond = cond[None].to(DEVICE)
        gt   = gt[None].to(DEVICE)

        rec     = reconstruct(cond)
        rec_512 = to_numpy_img(rec)

        Image.fromarray(rec_512).save(recon512_dir / name)
        Image.fromarray(upscale_to_1024(rec_512)).save(recon1024_dir / name)

        hq_img = Image.open(HQ_DIR / name).convert("RGB")
        ImageOps.fit(hq_img, (IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS).save(hq5_dir / name)

        mse  = F.mse_loss(rec, gt).item()
        psnr = 10 * math.log10(4.0 / (mse + 1e-10))
        psnr_vals.append(psnr)
        lpips_vals.append(lpips_net(rec, gt).item())

        print(f"  [{tag}] {name} | PSNR: {psnr:.2f} dB | LPIPS: {lpips_vals[-1]:.4f}")

    try:
        fid = calculate_fid_given_paths(
            [str(recon512_dir), str(hq5_dir)],
            batch_size=5, device=DEVICE, dims=2048
        )
    except Exception as e:
        print(f"  FID failed: {e}")
        fid = -1.0

    metrics = {
        "PSNR":  float(np.mean(psnr_vals)),
        "LPIPS": float(np.mean(lpips_vals)),
        "FID":   float(fid)
    }
    print(f"  [{tag}] PSNR: {metrics['PSNR']:.2f} | "
          f"LPIPS: {metrics['LPIPS']:.4f} | FID: {metrics['FID']:.2f}")
    return metrics


# -------- EXPLAINABILITY --------
def save_row(images, labels, filename, title=""):
    n       = len(images)
    margin  = 100
    h_text  = 80
    h_title = 40 if title else 0
    canvas  = Image.new(
        "RGB",
        (IMAGE_SIZE * n + margin, IMAGE_SIZE + h_text + h_title),
        (255, 255, 255)
    )
    for i, img in enumerate(images):
        canvas.paste(Image.fromarray(img), (margin + i * IMAGE_SIZE, h_title))

    draw = ImageDraw.Draw(canvas)
    try:
        font_caption = ImageFont.truetype("DejaVuSans.ttf", 18)
        font_title   = ImageFont.truetype("DejaVuSans.ttf", 22)
    except:
        font_caption = ImageFont.load_default()
        font_title   = font_caption

    if title:
        draw.text((margin, 8), title, fill=(0, 0, 0), font=font_title)

    for i, lbl in enumerate(labels):
        caption = STEP_LABELS.get(lbl, f"t={lbl}")
        lines   = caption.split("\n")
        for j, line in enumerate(lines):
            draw.text(
                (margin + i * IMAGE_SIZE + IMAGE_SIZE // 2 - 60,
                 IMAGE_SIZE + h_title + 8 + j * 28),
                line, fill=(0, 0, 0), font=font_caption
            )

    canvas.save(OUT_DIR / filename)
    print(f"  Saved: {filename}")


@torch.no_grad()
def generate_explainability(dataset, tag):
    # ✅ use corrupt image as input (noisy/patched depending on case)
    corrupt, gt, name = dataset[0]
    corrupt = corrupt[None].to(DEVICE)
    print(f"  Explainability on: {name}")

    # ✅ fully noise the corrupt input to t=999
    x = diffusion.q_sample(
        corrupt,
        torch.tensor([T_EXPLAIN - 1], device=DEVICE),
        torch.randn_like(corrupt)
    )

    traj, score_maps, patches = [], [], []

    # ✅ denoise all 1000 steps: 999 → 0
    for t in reversed(range(T_EXPLAIN)):
        tt = torch.full((1,), t, device=DEVICE, dtype=torch.long)
        with torch.amp.autocast('cuda'):
            pred_noise = diffusion.model(x, tt)

        ab      = diffusion.alphas_cumprod[t]
        ab_prev = diffusion.alphas_cumprod[t-1] if t > 0 else torch.tensor(1.0, device=DEVICE)
        beta    = diffusion.betas[t]

        x0_pred = ((x - torch.sqrt(1 - ab) * pred_noise) / torch.sqrt(ab)).clamp(-1, 1)
        coef1   = torch.sqrt(ab_prev) * beta / (1 - ab)
        coef2   = torch.sqrt(1 - beta) * (1 - ab_prev) / (1 - ab)
        mean    = coef1 * x0_pred + coef2 * x

        if t > 0:
            x = mean + torch.sqrt(beta * (1 - ab_prev) / (1 - ab)) * torch.randn_like(x)
        else:
            x = mean

        # ✅ capture snapshot at each explain step
        if t in EXPLAIN_STEPS:
            img = to_numpy_img(x)
            traj.append((t, img))

            s = pred_noise.abs().mean(1)[0].cpu().float().numpy()
            s = ((s - s.min()) / (s.max() - s.min() + 1e-8) * 255).astype(np.uint8)
            score_maps.append((t, np.stack([s]*3, axis=-1)))

            y, xp = np.unravel_index(np.argmax(s), s.shape)
            y  = min(y,  IMAGE_SIZE - PATCH_SIZE)
            xp = min(xp, IMAGE_SIZE - PATCH_SIZE)
            patches.append((t, np.array(
                Image.fromarray(img[y:y+PATCH_SIZE, xp:xp+PATCH_SIZE])
                    .resize((IMAGE_SIZE, IMAGE_SIZE), Image.LANCZOS)
            )))

    # ✅ sort descending: left=999 (noisy) → right=0 (clean)
    traj       = [img for _, img in sorted(traj,       key=lambda x: x[0], reverse=True)]
    score_maps = [img for _, img in sorted(score_maps, key=lambda x: x[0], reverse=True)]
    patches    = [img for _, img in sorted(patches,    key=lambda x: x[0], reverse=True)]
    labels     = sorted(EXPLAIN_STEPS, reverse=True)  # [999, 750, 500, 250, 0]

    save_row(traj,       labels, f"trajectory_{tag}.png",
             title=f"Denoising trajectory — {tag} | {name}")
    save_row(score_maps, labels, f"score_maps_{tag}.png",
             title=f"Score maps (noise prediction) — {tag} | {name}")
    save_row(patches,    labels, f"patches_{tag}.png",
             title=f"High-activation patches — {tag} | {name}")


# -------- RUN --------
results = {}

print("\n--- CASE 1: Clean ---")
results["clean"] = evaluate(HQ_DIR, "clean")
generate_explainability(PairedDataset(HQ_DIR, HQ_DIR), "clean")

print("\n--- CASE 2: Noisy ---")
results["noise"] = evaluate(NOISE_DIR, "noise")
generate_explainability(PairedDataset(NOISE_DIR, HQ_DIR), "noise")

print("\n--- CASE 3: Patched ---")
results["patch"] = evaluate(PATCH_DIR, "patch")
generate_explainability(PairedDataset(PATCH_DIR, HQ_DIR), "patch")

json.dump(results, open(OUT_DIR / "metrics.json", "w"), indent=2)
print(f"\n===== DONE — {OUT_DIR} =====")
print(json.dumps(results, indent=2))
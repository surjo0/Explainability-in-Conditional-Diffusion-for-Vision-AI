import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import random

# ==========================
# Paths (same as before)
# ==========================
input_dir = "/media/surjo/PortableSSD/Dataset/DIV2K/DIV2K_valid_HR"
output_dir = "/media/surjo/PortableSSD/Dataset/5-10% GaussianNoice"

os.makedirs(output_dir, exist_ok=True)

to_tensor = transforms.ToTensor()

# ==========================
# 5% White Patch Function
# ==========================
def add_fixed_white_patches(img_tensor):
    C, H, W = img_tensor.shape

    target_pixels = int(0.05 * H * W)   # EXACT 5%
    occluded = 0

    while occluded < target_pixels:

        # moderate patch sizes
        patch_h = random.randint(H // 25, H // 8)
        patch_w = random.randint(W // 25, W // 8)

        top = random.randint(0, H - patch_h)
        left = random.randint(0, W - patch_w)

        img_tensor[:, top:top+patch_h, left:left+patch_w] = 1.0

        occluded += patch_h * patch_w

    return img_tensor

# ==========================
# Process
# ==========================
image_files = [f for f in os.listdir(input_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(image_files)} images")

for img_name in tqdm(image_files):

    path = os.path.join(input_dir, img_name)

    image = Image.open(path).convert("RGB")
    tensor = to_tensor(image)

    patched = add_fixed_white_patches(tensor.clone())

    save_path = os.path.join(output_dir, img_name)
    save_image(patched, save_path)

print(" Added EXACT 5% white patches to all images")

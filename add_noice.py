import os
import torch
import torchvision.transforms as transforms
from torchvision.utils import save_image
from PIL import Image
from tqdm import tqdm
import random

# ==========================
# Paths
# ==========================
input_dir = "/media/surjo/PortableSSD/Dataset/DIV2K/DIV2K_valid_HR"
output_dir = "/media/surjo/PortableSSD/Dataset/5-10% GaussianNoice"

os.makedirs(output_dir, exist_ok=True)

# ==========================
# Transform (PIL -> Tensor)
# ==========================
to_tensor = transforms.ToTensor()   # converts to [0,1]
to_pil = transforms.ToPILImage()

# ==========================
# Noise Function
# ==========================
def add_gaussian_noise(image_tensor):
    """
    image_tensor: torch tensor in range [0,1]
    """
    noise_percentage = random.uniform(0.05, 0.10)  # random 5-10%
    noise = torch.randn_like(image_tensor) * noise_percentage
    noisy_image = image_tensor + noise
    noisy_image = torch.clamp(noisy_image, 0., 1.)
    return noisy_image

# ==========================
# Process Images
# ==========================
image_files = [f for f in os.listdir(input_dir)
               if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

print(f"Found {len(image_files)} images")

for img_name in tqdm(image_files):
    img_path = os.path.join(input_dir, img_name)

    # Load image
    image = Image.open(img_path).convert("RGB")
    image_tensor = to_tensor(image)

    # Add noise
    noisy_tensor = add_gaussian_noise(image_tensor)

    # Save image
    output_path = os.path.join(output_dir, img_name)
    save_image(noisy_tensor, output_path)

print("Done ✅ All images saved with 5–10% Gaussian noise.")

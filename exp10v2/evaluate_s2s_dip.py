import os
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image

from dataset import prepare_div2k
from utils import compute_psnr, compute_ssim, add_gaussian_noise, tensor_to_numpy, save_comparison_figure

# Import the newly implemented methods
from dip import train_dip
from s2s import train_self2self

# Neighbor2Neighbor evaluation requires loading its pre-trained model. We might not have it.
# We will just focus on BM3D, DIP, and S2S for comparison, or if N2N model exists, we load it.
# Let's focus on BM3D, DIP, Self2Self.
import bm3d

def denoise_bm3d(noisy_np, sigma):
    """
    Apply BM3D denoising.
    noisy_np: [H, W] or [H, W, 3], values in [0, 1]
    sigma: standard deviation in [0, 255]
    """
    sigma_norm = sigma / 255.0
    if noisy_np.ndim == 3:
        # Color
        denoised = bm3d.bm3d(noisy_np, sigma_psd=sigma_norm, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    else:
        # Grayscale
        denoised = bm3d.bm3d(noisy_np, sigma_psd=sigma_norm, stage_arg=bm3d.BM3DStages.ALL_STAGES)
    return np.clip(denoised, 0.0, 1.0)


def load_test_images(img_dir: str, mode: str, max_images: int = 5):
    """Load first few validation images from DIV2K for quick evaluation."""
    images = []
    if not os.path.exists(img_dir):
        print(f"Warning: {img_dir} not found. Ensure dataset is downloaded.")
        return images

    valid_exts = {".png", ".jpg", ".jpeg"}
    files = sorted([f for f in os.listdir(img_dir) if os.path.splitext(f)[1].lower() in valid_exts])
    files = files[:max_images]

    for f in files:
        path = os.path.join(img_dir, f)
        img = Image.open(path)
        if mode == "gray":
            img = img.convert("L")
        else:
            img = img.convert("RGB")

        # Center crop to 256x256 to speed up DIP/S2S evaluation
        # S2S and DIP are very slow for full HR images.
        w, h = img.size
        crop_size = 256
        left = (w - crop_size) // 2
        top = (h - crop_size) // 2
        img = img.crop((left, top, left + crop_size, top + crop_size))

        img_np = np.array(img).astype(np.float32) / 255.0
        images.append((f, img_np))

    return images

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default="gray", choices=["gray", "color"])
    parser.add_argument("--sigmas", type=int, nargs="+", default=[15, 25, 35, 50])
    parser.add_argument("--data_root", type=str, default="/tmp/my_experiment/DIV2K_valid_HR")
    parser.add_argument("--max_images", type=int, default=3, help="Number of images to evaluate")
    parser.add_argument("--num_iter", type=int, default=1500, help="Iterations for S2S/DIP")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # For speed, if DIV2K_valid_HR is not there, we'll try to find it
    img_dir = args.data_root
    if not os.path.exists(img_dir):
        prepare_div2k("/tmp/my_experiment") # Will download to /tmp/my_experiment/DIV2K_valid_HR
        img_dir = "/tmp/my_experiment/DIV2K_valid_HR"

    test_images = load_test_images(img_dir, args.mode, args.max_images)
    if not test_images:
        print("No test images found. Exiting.")
        return

    out_dir = "figures/exp10"
    os.makedirs(out_dir, exist_ok=True)

    results = {
        "BM3D": {s: {"psnr": [], "ssim": []} for s in args.sigmas},
        "DIP": {s: {"psnr": [], "ssim": []} for s in args.sigmas},
        "Self2Self": {s: {"psnr": [], "ssim": []} for s in args.sigmas},
    }

    print("\nStarting Evaluation...")
    print("-" * 50)

    for sigma in args.sigmas:
        print(f"\nEvaluating Noise Level: σ = {sigma}")

        for name, clean_np in test_images:
            # 1. Add noise
            noisy_np = add_gaussian_noise(clean_np, sigma)

            # Convert to tensor for PyTorch methods
            if args.mode == "gray":
                noisy_tensor = torch.from_numpy(noisy_np).unsqueeze(0).unsqueeze(0).to(device) # [1, 1, H, W]
            else:
                noisy_tensor = torch.from_numpy(noisy_np).permute(2, 0, 1).unsqueeze(0).to(device) # [1, 3, H, W]

            print(f"  Image: {name}")

            # 2. BM3D
            t0 = time.time()
            bm3d_np = denoise_bm3d(noisy_np, sigma)
            bm3d_psnr = compute_psnr(clean_np, bm3d_np)
            bm3d_ssim = compute_ssim(clean_np, bm3d_np)
            results["BM3D"][sigma]["psnr"].append(bm3d_psnr)
            results["BM3D"][sigma]["ssim"].append(bm3d_ssim)
            print(f"    [BM3D]      PSNR: {bm3d_psnr:.2f} dB, SSIM: {bm3d_ssim:.4f}  (Time: {time.time()-t0:.2f}s)")

            # 3. DIP
            t0 = time.time()
            dip_tensor = train_dip(noisy_tensor, sigma, num_iter=args.num_iter, device=device)
            dip_np = tensor_to_numpy(dip_tensor)
            dip_psnr = compute_psnr(clean_np, dip_np)
            dip_ssim = compute_ssim(clean_np, dip_np)
            results["DIP"][sigma]["psnr"].append(dip_psnr)
            results["DIP"][sigma]["ssim"].append(dip_ssim)
            print(f"    [DIP]       PSNR: {dip_psnr:.2f} dB, SSIM: {dip_ssim:.4f}  (Time: {time.time()-t0:.2f}s)")

            # 4. Self2Self
            t0 = time.time()
            s2s_tensor = train_self2self(noisy_tensor, sigma, num_iter=args.num_iter, device=device)
            s2s_np = tensor_to_numpy(s2s_tensor)
            s2s_psnr = compute_psnr(clean_np, s2s_np)
            s2s_ssim = compute_ssim(clean_np, s2s_np)
            results["Self2Self"][sigma]["psnr"].append(s2s_psnr)
            results["Self2Self"][sigma]["ssim"].append(s2s_ssim)
            print(f"    [Self2Self] PSNR: {s2s_psnr:.2f} dB, SSIM: {s2s_ssim:.4f}  (Time: {time.time()-t0:.2f}s)")

            # Save visual comparison for the first image of each sigma
            if name == test_images[0][0]:
                fig_path = os.path.join(out_dir, f"compare_sigma{sigma}_{name}")

                plt.figure(figsize=(20, 5))
                titles = ["Clean", f"Noisy (σ={sigma})", f"BM3D\n{bm3d_psnr:.2f}dB",
                          f"DIP\n{dip_psnr:.2f}dB", f"Self2Self\n{s2s_psnr:.2f}dB"]
                imgs = [clean_np, noisy_np, bm3d_np, dip_np, s2s_np]

                for i, (title, img) in enumerate(zip(titles, imgs)):
                    plt.subplot(1, 5, i+1)
                    if args.mode == "gray":
                        plt.imshow(img, cmap="gray")
                    else:
                        plt.imshow(img)
                    plt.title(title)
                    plt.axis("off")
                plt.tight_layout()
                plt.savefig(fig_path)
                plt.close()

    print("\n" + "=" * 50)
    print("Summary Average Results (PSNR / SSIM)")
    print("=" * 50)
    print(f"{'Sigma':<10} | {'BM3D':<15} | {'DIP':<15} | {'Self2Self':<15}")
    print("-" * 50)

    for sigma in args.sigmas:
        bm3d_p = np.mean(results["BM3D"][sigma]["psnr"])
        bm3d_s = np.mean(results["BM3D"][sigma]["ssim"])

        dip_p = np.mean(results["DIP"][sigma]["psnr"])
        dip_s = np.mean(results["DIP"][sigma]["ssim"])

        s2s_p = np.mean(results["Self2Self"][sigma]["psnr"])
        s2s_s = np.mean(results["Self2Self"][sigma]["ssim"])

        print(f"{sigma:<10} | {bm3d_p:.2f}/{bm3d_s:.4f} | {dip_p:.2f}/{dip_s:.4f} | {s2s_p:.2f}/{s2s_s:.4f}")

    print("=" * 50)

if __name__ == "__main__":
    main()

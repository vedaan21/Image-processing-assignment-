# ==========================================
# Name: Ritik Sharma
# Roll No: 2301010342
# Course: Image Processing & Computer Vision
# Unit: Image Restoration
# Assignment: Noise Modeling and Image Restoration
# Date:
# ==========================================

print("Welcome to Image Restoration System")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
os.makedirs("outputs", exist_ok=True)

# Image list (3 images)
image_list = ['tiger.jpg', 'forest.jpg', 'river.jpg']


# ----------- Functions -----------

def add_gaussian_noise(image):
    mean = 0
    std = 25
    noise = np.random.normal(mean, std, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(image, prob=0.02):
    noisy = image.copy()

    # Salt
    salt = np.random.rand(*image.shape) < prob
    noisy[salt] = 255

    # Pepper
    pepper = np.random.rand(*image.shape) < prob
    noisy[pepper] = 0

    return noisy


def mse(original, restored):
    return np.mean((original - restored) ** 2)


def psnr(original, restored):
    mse_val = mse(original, restored)
    if mse_val == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(mse_val))


# ----------- Main Processing -----------

for img_name in image_list:

    print(f"\nProcessing: {img_name}")

    image = cv2.imread(f'images/{img_name}')

    if image is None:
        print(f"❌ {img_name} not found")
        continue

    # Resize and grayscale
    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ----------- Noise Addition -----------
    gaussian_noise = add_gaussian_noise(gray)
    sp_noise = add_salt_pepper(gray)

    # ----------- Filters -----------
    mean_filtered = cv2.blur(gaussian_noise, (5, 5))
    median_filtered = cv2.medianBlur(sp_noise, 5)
    gaussian_filtered = cv2.GaussianBlur(gaussian_noise, (5, 5), 0)

    # ----------- Performance Metrics -----------
    print("\n--- Performance Metrics ---")

    print(f"{img_name} Mean Filter MSE:", mse(gray, mean_filtered))
    print(f"{img_name} Mean Filter PSNR:", psnr(gray, mean_filtered))

    print(f"{img_name} Median Filter MSE:", mse(gray, median_filtered))
    print(f"{img_name} Median Filter PSNR:", psnr(gray, median_filtered))

    print(f"{img_name} Gaussian Filter MSE:", mse(gray, gaussian_filtered))
    print(f"{img_name} Gaussian Filter PSNR:", psnr(gray, gaussian_filtered))

    # ----------- Save Outputs -----------
    cv2.imwrite(f"outputs/{img_name}_original.jpg", gray)
    cv2.imwrite(f"outputs/{img_name}_gaussian_noise.jpg", gaussian_noise)
    cv2.imwrite(f"outputs/{img_name}_sp_noise.jpg", sp_noise)

    cv2.imwrite(f"outputs/{img_name}_mean.jpg", mean_filtered)
    cv2.imwrite(f"outputs/{img_name}_median.jpg", median_filtered)
    cv2.imwrite(f"outputs/{img_name}_gaussian.jpg", gaussian_filtered)

    # ----------- Display Comparison -----------

    plt.figure(figsize=(12, 8))

    titles = [
        "Original", "Gaussian Noise", "Salt & Pepper",
        "Mean Filter", "Median Filter", "Gaussian Filter"
    ]

    images = [
        gray, gaussian_noise, sp_noise,
        mean_filtered, median_filtered, gaussian_filtered
    ]

    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')

    plt.suptitle(f"Results for {img_name}")
    plt.tight_layout()
    plt.savefig(f"outputs/{img_name}_comparison.png")
    plt.show()


# ----------- Final Analysis -----------

print("\n--- Overall Analysis ---")

print("1. Gaussian noise affects smooth regions more (visible in river image).")
print("2. Salt & pepper noise introduces sharp black-white pixels.")
print("3. Median filter performs best for salt & pepper noise.")
print("4. Gaussian filter works best for Gaussian noise.")
print("5. Mean filter smooths image but reduces edge sharpness.")
print("6. Forest image retains texture better after filtering.")
print("7. River image shows more visible degradation due to smooth background.")
print("8. High PSNR indicates better image restoration quality.")

print("\n✅ Processing Completed Successfully!")
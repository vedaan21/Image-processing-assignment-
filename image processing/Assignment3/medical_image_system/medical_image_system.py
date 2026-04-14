# ==========================================
# Name: Ritik Sharma
# Roll No: 2301010342
# Course: Image Processing & Computer Vision
# Assignment: Medical Image Compression & Segmentation
# ==========================================

print("Welcome to Medical Image Compression & Segmentation System")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create outputs folder (absolute path safe)
output_dir = os.path.join(os.getcwd(), "outputs")
os.makedirs(output_dir, exist_ok=True)

# ----------- RLE Compression -----------

def rle_encode(image):
    pixels = image.flatten()
    encoding = []

    prev = pixels[0]
    count = 1

    for pixel in pixels[1:]:
        if pixel == prev:
            count += 1
        else:
            encoding.append((prev, count))
            prev = pixel
            count = 1

    encoding.append((prev, count))
    return encoding


def compression_ratio(original, encoded):
    original_size = len(original.flatten())
    compressed_size = len(encoded) * 2
    return original_size / compressed_size


# ----------- Image List -----------

image_list = ['xray.jpg', 'mri.jpg', 'ct.jpg']


# ----------- MAIN LOOP -----------

for img_name in image_list:

    print(f"\nProcessing: {img_name}")

    # Load image
    image_path = os.path.join("images", img_name)
    image = cv2.imread(image_path, 0)

    if image is None:
        print(f"❌ ERROR: {img_name} not found")
        continue
    else:
        print(f"✅ Loaded: {img_name}")

    # Resize
    image = cv2.resize(image, (512, 512))

    # ----------- COMPRESSION -----------
    encoded = rle_encode(image)
    ratio = compression_ratio(image, encoded)

    print(f"Compression Ratio: {ratio:.3f}")
    print(f"Storage Saving: {(1 - 1/ratio)*100:.2f}%")

    # ----------- SEGMENTATION -----------

    # Global Threshold
    _, thresh_global = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)

    # Otsu Threshold
    _, thresh_otsu = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
    )

    # ----------- MORPHOLOGY -----------

    kernel = np.ones((3, 3), np.uint8)

    dilation = cv2.dilate(thresh_otsu, kernel, iterations=1)
    erosion = cv2.erode(thresh_otsu, kernel, iterations=1)

    # ----------- SAVE OUTPUTS -----------

    print("Saving outputs...")

    save_items = [
        (f"{img_name}_original.jpg", image),
        (f"{img_name}_global.jpg", thresh_global),
        (f"{img_name}_otsu.jpg", thresh_otsu),
        (f"{img_name}_dilation.jpg", dilation),
        (f"{img_name}_erosion.jpg", erosion),
    ]

    for filename, img in save_items:
        path = os.path.join(output_dir, filename)
        success = cv2.imwrite(path, img)

        if success:
            print(f"✅ Saved: {filename}")
        else:
            print(f"❌ Failed to save: {filename}")

    # ----------- DISPLAY -----------

    plt.figure(figsize=(10, 8))

    titles = ["Original", "Global", "Otsu", "Dilation", "Erosion"]
    images = [image, thresh_global, thresh_otsu, dilation, erosion]

    for i in range(len(images)):
        plt.subplot(2, 3, i + 1)
        plt.title(titles[i])
        plt.imshow(images[i], cmap='gray')
        plt.axis('off')

    plt.suptitle(img_name)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{img_name}_comparison.png"))
    plt.show()


# ----------- FINAL ANALYSIS -----------

print("\n--- Analysis ---")

print("1. RLE compresses repetitive pixel values efficiently.")
print("2. Higher compression occurs in smoother regions.")
print("3. Otsu thresholding performs better than global thresholding.")
print("4. Global threshold may fail for varying intensity images.")
print("5. Dilation fills gaps in segmented regions.")
print("6. Erosion removes noise but reduces region size.")
print("7. Morphological operations improve segmentation accuracy.")

print("\n✅ All Processing Completed Successfully!")
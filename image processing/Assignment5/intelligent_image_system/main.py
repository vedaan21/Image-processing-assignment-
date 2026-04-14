# ==========================================
# Name: Ritik Sharma
# Roll No: 2301010342
# Course: Image Processing & Computer Vision
# Assignment: Intelligent Image Enhancement System
# ==========================================

print("Intelligent Image Processing System Started")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
from skimage.metrics import structural_similarity as ssim

# Create output folder
output_dir = os.path.join(os.getcwd(), "outputs")
os.makedirs(output_dir, exist_ok=True)

# Image list
image_list = ['img1.jpg', 'img2.jpg', 'img3.jpg']


# ----------- FUNCTIONS -----------

def add_gaussian_noise(image):
    noise = np.random.normal(0, 25, image.shape)
    noisy = image + noise
    return np.clip(noisy, 0, 255).astype(np.uint8)


def add_salt_pepper(image, prob=0.02):
    noisy = image.copy()
    salt = np.random.rand(*image.shape) < prob
    pepper = np.random.rand(*image.shape) < prob
    noisy[salt] = 255
    noisy[pepper] = 0
    return noisy


def mse(original, restored):
    return np.mean((original - restored) ** 2)


def psnr(original, restored):
    m = mse(original, restored)
    if m == 0:
        return float('inf')
    return 20 * np.log10(255.0 / np.sqrt(m))


# ----------- MAIN LOOP -----------

for img_name in image_list:

    print(f"\nProcessing: {img_name}")

    path = os.path.join("images", img_name)
    image = cv2.imread(path)

    if image is None:
        print(f"❌ {img_name} not found")
        continue

    image = cv2.resize(image, (512,512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ----------- NOISE -----------

    gaussian_noise = add_gaussian_noise(gray)
    sp_noise = add_salt_pepper(gray)

    # ----------- RESTORATION -----------

    mean = cv2.blur(gaussian_noise, (5,5))
    median = cv2.medianBlur(sp_noise, 5)
    gaussian = cv2.GaussianBlur(gaussian_noise, (5,5), 0)

    # ----------- ENHANCEMENT -----------

    enhanced = cv2.equalizeHist(gray)

    # ----------- SEGMENTATION -----------

    _, global_th = cv2.threshold(enhanced, 127, 255, cv2.THRESH_BINARY)
    _, otsu = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    # Morphology
    kernel = np.ones((3,3), np.uint8)
    dilation = cv2.dilate(otsu, kernel)
    erosion = cv2.erode(otsu, kernel)

    # ----------- EDGE -----------

    sobel = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel = cv2.convertScaleAbs(sobel)
    canny = cv2.Canny(gray, 100, 200)

    # ----------- CONTOURS -----------

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contour_img = image.copy()

    for cnt in contours:
        if cv2.contourArea(cnt) > 500:
            x,y,w,h = cv2.boundingRect(cnt)
            cv2.rectangle(contour_img, (x,y), (x+w,y+h), (0,255,0), 2)

    # ----------- FEATURES -----------

    orb = cv2.ORB_create()
    kp, des = orb.detectAndCompute(gray, None)
    orb_img = cv2.drawKeypoints(image, kp, None, color=(0,255,0))

    print(f"Keypoints: {len(kp)}")

    # ----------- METRICS -----------

    print("MSE:", mse(gray, gaussian))
    print("PSNR:", psnr(gray, gaussian))
    print("SSIM:", ssim(gray, gaussian))

    # ----------- SAVE -----------

    cv2.imwrite(os.path.join(output_dir, f"{img_name}_enhanced.jpg"), enhanced)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_segmented.jpg"), otsu)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_features.jpg"), orb_img)

    # ----------- FINAL DISPLAY -----------

    plt.figure(figsize=(12,8))

    titles = ["Original","Noisy","Restored","Enhanced",
              "Segmented","Features"]

    images = [gray, gaussian_noise, gaussian,
              enhanced, otsu, orb_img]

    for i in range(len(images)):
        plt.subplot(2,3,i+1)
        plt.title(titles[i])
        if len(images[i].shape)==2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{img_name}_pipeline.png"))
    plt.show()


# ----------- FINAL ANALYSIS -----------

print("\n--- Conclusion ---")

print("1. Noise degrades image quality significantly.")
print("2. Filters restore images but may blur details.")
print("3. Histogram equalization improves contrast.")
print("4. Otsu gives better segmentation.")
print("5. ORB detects useful features.")
print("6. Metrics confirm quality improvement.")

print("\n✅ System Completed Successfully!")
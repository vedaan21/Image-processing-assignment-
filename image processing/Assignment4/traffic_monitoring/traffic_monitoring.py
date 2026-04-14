# ==========================================
# Name: Ritik Sharma
# Roll No: 2301010342
# Course: Image Processing & Computer Vision
# Assignment: Traffic Monitoring System
# ==========================================

print("Traffic Monitoring System - Feature Extraction")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# Create output folder
output_dir = os.path.join(os.getcwd(), "outputs")
os.makedirs(output_dir, exist_ok=True)

# Use 3 traffic images
image_list = ['road1.jpg', 'road2.jpg', 'road3.jpg']


for img_name in image_list:

    print(f"\nProcessing: {img_name}")

    path = os.path.join("images", img_name)
    image = cv2.imread(path)

    if image is None:
        print(f"❌ {img_name} not found")
        continue

    image = cv2.resize(image, (512,512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ----------- EDGE DETECTION -----------

    # Sobel
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0)
    sobel = cv2.convertScaleAbs(sobelx)

    # Canny
    canny = cv2.Canny(gray, 100, 200)

    # ----------- CONTOURS -----------

    contours, _ = cv2.findContours(canny, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    contour_img = image.copy()

    for cnt in contours:
        area = cv2.contourArea(cnt)
        perimeter = cv2.arcLength(cnt, True)

        if area > 500:  # ignore small noise
            x, y, w, h = cv2.boundingRect(cnt)
            cv2.rectangle(contour_img, (x,y), (x+w, y+h), (0,255,0), 2)

            print(f"Object Area: {area:.2f}, Perimeter: {perimeter:.2f}")

    # ----------- FEATURE EXTRACTION (ORB) -----------

    orb = cv2.ORB_create()
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    orb_img = cv2.drawKeypoints(image, keypoints, None, color=(0,255,0))

    print(f"Keypoints detected: {len(keypoints)}")

    # ----------- SAVE OUTPUTS -----------

    cv2.imwrite(os.path.join(output_dir, f"{img_name}_sobel.jpg"), sobel)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_canny.jpg"), canny)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_contours.jpg"), contour_img)
    cv2.imwrite(os.path.join(output_dir, f"{img_name}_orb.jpg"), orb_img)

    # ----------- DISPLAY -----------

    plt.figure(figsize=(10,8))

    titles = ["Original", "Sobel", "Canny", "Contours", "ORB"]
    images = [image, sobel, canny, contour_img, orb_img]

    for i in range(len(images)):
        plt.subplot(2,3,i+1)
        plt.title(titles[i])
        if len(images[i].shape) == 2:
            plt.imshow(images[i], cmap='gray')
        else:
            plt.imshow(cv2.cvtColor(images[i], cv2.COLOR_BGR2RGB))
        plt.axis('off')

    plt.suptitle(img_name)
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{img_name}_comparison.png"))
    plt.show()


# ----------- FINAL ANALYSIS -----------

print("\n--- Analysis ---")

print("1. Canny edge detector provides clearer edges than Sobel.")
print("2. Sobel detects gradients but is more sensitive to noise.")
print("3. Contours help in identifying object boundaries.")
print("4. Bounding boxes represent detected objects.")
print("5. ORB detects key features useful for tracking vehicles.")
print("6. Feature extraction is important for traffic monitoring systems.")

print("\n✅ Assignment 4 Completed!")
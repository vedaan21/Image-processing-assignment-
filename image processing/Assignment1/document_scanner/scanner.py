# ==========================================
# Name: Ritik Sharma
# Roll No: 2301010342
# Course: Image Processing & Computer Vision
# Unit: Image Acquisition & Processing
# Assignment: Smart Document Scanner & Quality Analysis System
# Date: 
# ==========================================

print("Welcome to Smart Document Scanner & Quality Analysis System")
print("This system analyzes how resolution and quantization affect document quality.\n")

import cv2
import numpy as np
import matplotlib.pyplot as plt
import os
os.makedirs("outputs", exist_ok=True)

image = cv2.imread('images/lions.jpg')
if image is None:
    print("❌ Error: Image not found. Check file path!")
    exit()

image_resized = cv2.resize(image,(512,512))

gray = cv2.cvtColor(image_resized, cv2.COLOR_BGR2GRAY)
# Display the original and processed images
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(cv2.cvtColor(image_resized, cv2.COLOR_BGR2RGB))
plt.axis('off')
plt.subplot(1, 2, 2)
plt.title('Grayscale Image')
plt.imshow(gray, cmap='gray')
plt.axis('off')
plt.show()



high = gray #512 * 512
medium = cv2.resize(gray, (256, 256)) #256 * 256
low = cv2.resize(gray, (128, 128)) #128 * 128


medium_up = cv2.resize(medium, (512, 512))
low_up = cv2.resize(low, (512, 512))
# Display the original and processed images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('High Resolution (512x512)')
plt.imshow(high, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title('Medium Resolution (256x256)')
plt.imshow(medium_up, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('Low Resolution (128x128)')
plt.imshow(low_up, cmap='gray')
plt.axis('off')
plt.show()

# Quantization
def quantize(image, levels):
    factor = 256 // levels
    quantized = (image // factor) * factor
    return quantized

q8 = quantize(gray, 256)
q4 = quantize(gray, 16)
q2 = quantize(gray, 4)

# Display the original and quantized images
plt.figure(figsize=(15, 5))
plt.subplot(1, 3, 1)
plt.title('8-bit Quantization (256 levels)')
plt.imshow(q8, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 2)
plt.title('4-bit Quantization (16 levels)')
plt.imshow(q4, cmap='gray')
plt.axis('off')
plt.subplot(1, 3, 3)
plt.title('2-bit Quantization (4 levels)')
plt.imshow(q2, cmap='gray')
plt.axis('off')
plt.show()


cv2.imwrite('outputs/high_resolution.jpg', high)
cv2.imwrite('outputs/medium_resolution.jpg', medium_up)
cv2.imwrite('outputs/low_resolution.jpg', low_up)
cv2.imwrite('outputs/quantized_8bit.jpg', q8)
cv2.imwrite('outputs/quantized_4bit.jpg', q4)
cv2.imwrite('outputs/quantized_2bit.jpg', q2)

plt.figure(figsize=(15, 5))
titles = [
    "Original","512x512","256x256","128x128","8-bit Quantization","4-bit Quantization","2-bit Quantization"
]
images = [gray, high, medium_up, low_up, q8, q4, q2]
for i in range(len(images)):
    plt.subplot(2,4,i+1)
    plt.title(titles[i])
    plt.imshow(images[i], cmap='gray')
    plt.axis('off')
plt.tight_layout()
plt.savefig('outputs/comparison.png')
plt.show()


print("\n--- Observations ---")

print("1. As resolution decreases, fine text details are lost.")
print("2. Low resolution images show blurred edges and poor readability.")
print("3. High resolution maintains sharp text edges.")
print("4. Quantization reduces gray levels causing banding effect.")
print("5. 2-bit images lose most details and are unsuitable for OCR.")
print("6. 8-bit images preserve maximum information and are best for OCR.")
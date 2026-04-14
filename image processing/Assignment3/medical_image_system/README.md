# Medical Image Compression & Segmentation

## Objective

To compress medical images using RLE and segment regions using thresholding and morphology.

---

## Tools Used

* Python
* OpenCV
* NumPy
* Matplotlib

---

## Steps Performed

1. Applied Run Length Encoding (RLE)
2. Calculated compression ratio
3. Applied segmentation:

   * Global Threshold
   * Otsu Threshold
4. Applied morphological operations:

   * Dilation
   * Erosion

---

## Outputs

* Original image
* Segmented images
* Morphological results
* Comparison images

(All saved in `outputs/` folder)

---

## Observations

* RLE compresses repetitive data
* Otsu gives better segmentation
* Dilation fills gaps
* Erosion removes noise

---

## How to Run

```bash
python3 medical_image_system.py
```

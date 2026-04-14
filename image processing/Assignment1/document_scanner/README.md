
# Smart Document Scanner & Quality Analysis

## Objective

To analyze how image resolution (sampling) and gray levels (quantization) affect document quality and readability.

---

## Tools Used

* Python
* OpenCV
* NumPy
* Matplotlib

---

## Steps Performed

1. Loaded image and converted to grayscale
2. Applied sampling:

   * 512×512 (High)
   * 256×256 (Medium)
   * 128×128 (Low)
3. Applied quantization:

   * 8-bit (256 levels)
   * 4-bit (16 levels)
   * 2-bit (4 levels)
4. Compared all results visually

---

## Outputs

* Original image
* Sampled images
* Quantized images
* Comparison figure

(All saved in `outputs/` folder)

---

## Observations

* High resolution gives better clarity
* Low resolution causes blur
* Less gray levels reduce image quality
* 8-bit images are best for OCR

---

## Conclusion

Better resolution and higher gray levels improve document readability and scanning quality.

---

## How to Run

```bash
python3 scanner.py
```

# Traffic Monitoring System (Feature Extraction)

## Objective

To detect edges, contours, and features in traffic images.

---

## Tools Used

* Python
* OpenCV
* NumPy
* Matplotlib

---

## Steps Performed

1. Edge Detection:

   * Sobel
   * Canny
2. Object Representation:

   * Contours
   * Bounding Boxes
3. Feature Extraction:

   * ORB

---

## Outputs

* Edge detection images
* Contours with bounding boxes
* Feature keypoints
* Comparison images

(All saved in `outputs/` folder)

---

## Observations

* Canny gives better edges
* Contours detect objects
* ORB finds keypoints
* Useful for traffic monitoring

---

## How to Run

```bash
python3 traffic_monitoring.py
```

# 🧠 Layout-Based Handwriting Analysis for Behavior Prediction

This project analyzes scanned or digital images of handwritten pages to infer psychological traits based on margin spacing and line orientation.

---

## 📌 Features

- ✅ Detects written content using contour and morphological analysis  
- ✅ Computes and evaluates margins (Left, Right, Top, Bottom)  
- ✅ Detects line orientations: Straight, Sloped, Curved  
- ✅ Infers basic personality traits based on spatial layout  
- ✅ Visualizes bounding box around handwritten area  
- ✅ Works under varying lighting conditions using adaptive thresholding  

---

## 🛠️ Tech Stack

- **Python 3.x**  
- **OpenCV**  
- **NumPy**  
- **Matplotlib**  

---

## 🖼️ Input

- A scanned or digital image of a handwritten page (preferably A4 size)  
- Format: `.jpg`, `.png`, etc.

---

## 📤 Output

- A list of boolean values:
  ```
  [left_margin_good, right_margin_good, top_margin_good, bottom_margin_good,
   is_line_straight, is_line_sloped, is_line_curved]
  ```

- A psychological/personality assessment generated based on the result.

---

## 📂 Directory Structure

```
.
├── handwriting_analysis.py     # Main code file  
├── a01-049u.png                # Sample image (replace with your own)  
└── README.md                   # This file  
```

---

## 🧪 How It Works

1. **Preprocessing**: Grayscale → Gaussian Blur → Adaptive Thresholding  
2. **Dilation**: Merge fragmented contours  
3. **Contour Detection**: Extract meaningful bounding boxes  
4. **Margin Analysis**: Measure spacing between text and page edges  
5. **Line Orientation Analysis**: Use HoughLinesP for angle detection  
6. **Personality Mapping**: Simple interpretation based on spatial traits  

---

## 🚀 Usage

```bash
# Run the script
python handwriting_analysis.py
```

- Replace `a01-049u.png` with the path to your own scanned handwritten image.

---

## 🧠 Sample Personality Insights

| Trait              | Description                                 |
|--------------------|---------------------------------------------|
| Left Margin Small  | Emotionally attached to family and roots    |
| Right Margin Small | Risk-taker, impulsive, adventurous          |
| Sloped Lines       | Optimistic and energetic                    |
| Bottom Margin Good | Visionary and productive                    |
| ...                | ...                                         |

---

## 📌 Example Output

```
Results: [True, False, True, True, False, True, False]
Personality assessment: Socially conscious, blind risk taker, adventurous, positive energy, productive...
```

---

## 📎 Notes

- Adjust `margin_threshold` based on image resolution if needed.  
- For best results, use a clean scanned image with minimal noise.  

---


## 👤 Author

- Nishant Kadlak
- Vivek Janbandhu
- Prathamesh Khokaralkar
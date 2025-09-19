# Captcha Recognition Solution

## 1. Problem Statement
The website uses **5-character captchas** (A–Z, 0–9).  
Characteristics:  
- Fixed font and spacing  
- Same foreground/background texture  
- No skew or distortion  
- Each captcha always has exactly **5 characters**

We are given a sample dataset of **25 captchas** (`input00.jpg`–`input24.jpg`) and their ground-truth labels (`output00.txt`–`output24.txt`).  
The task: **Design an algorithm to recognize unseen captchas (e.g. input100.jpg).**

---

## 1.5. Initial Thoughts and Approach Shift
At first, we considered a **deep learning approach** with CNNs. The intuition was straightforward: feed the raw captcha images into a CNN and let it learn directly.  
But the dataset is tiny — only 25 labeled samples. Even with data augmentation, training a CNN (we actually tried a **tiny CNN + augmentation**) did not generalize well and accuracy was poor.  

So we decided to pivot towards **classical computer vision + machine learning methods**, which are more data-efficient. This led us to the step-by-step approach described next.

---

## 2. Step-by-Step Solution Approach

### (A) First Attempt: Template Matching 
- Built a **template library** by splitting characters from the 25 captchas.   
- Recognition = compare new character image vs. templates using pixel MSE.  
- Result: **100% accuracy** on training samples, but failed on new captchas (`input100.jpg`) due to limited generalization.

### (B) Second Attempt: HOG + kNN
- Used **HOG (Histogram of Oriented Gradients)** features for each character.  
- Classifier: **k-Nearest Neighbors (kNN)**.  
- Advantage: better generalization, `input100.jpg` recognized correctly.  
- Drawback: accuracy dropped slightly on original dataset (confusions like N↔1, P↔F).

### (C) Final Solution: HOG + Data Augmentation + SVM
1. **Segmentation**  
   - Used **connected component analysis** to split characters.  
   - Robust against spacing/background variation.  

2. **Feature Extraction**  
   - Each character resized to `28x28`.  
   - Extracted **HOG features** (cell=7×7, block=14×14).  

3. **Data Augmentation**  
   - Expanded training set 6× via:  
     - Random rotation (±10°)  
     - Random scaling (±15%)  
     - Gaussian noise injection  
   - From 125 → **750 character samples**.  

4. **Classifier**  
   - Used **SVM (RBF kernel)** for robust classification.  
   - More stable than kNN, avoids noise sensitivity.  

---

## 3. Code Evolution Timeline

| Version | Method | Highlights | Result |
|---------|--------|------------|--------|
| **v1** (`captcha_pre_yh_v1.py`) | Template Matching | Split fixed-width chars, MSE against template lib | 100% on training, failed on new captcha |
| **v2** (`captcha_pre_yh_v2.py`) | HOG + kNN | HOG features + kNN classifier | Better generalization, input100 works, some errors on training set |
| **v3** (`captcha_pre_yh_final.py`) | HOG + Augmentation + SVM | Connected components, HOG(28x28), augmentation, SVM | 100% accuracy on both training + new captcha |

---

## 4. Results

| Dataset        | Accuracy |
|----------------|----------|
| SampleCaptchas (25 images) | 100% |
| New image (input100.jpg)   | Correct |
| **Final accuracy** | **26/26 = 100%** |

Console output:

```
input00.jpg -> pred=EGYK4, gt=EGYK4 ✓
...
input24.jpg -> pred=UHVFO, gt=UHVFO ✓
input100.jpg -> pred=YMB1Q, gt=YMB1Q ✓
Final accuracy = 26/26 = 1.000
```

---

## 5. File Structure

```
captcha_pre_yh_package/
├── sampleCaptchas/
│   ├── input/         # input00.jpg ... input24.jpg
│   ├── output/        # output00.txt ... output24.txt
├── input100.jpg       # unseen captcha for testing
├── test.py            # main code (final solution)
├── captcha_pre_yh_v1.py      # (A) template matching
├── captcha_pre_yh_v2.py      # (B) HOG + kNN
├── captcha_pre_yh_final.py   # (C) HOG + SVM + augmentation
└── README.md
```

---

## 6. How to Run
1. Install dependencies:
   ```bash
   pip install opencv-python scikit-learn numpy pillow
   ```
2. Run:
   ```bash
   python test.py
   ```
3. Output will show predictions vs. ground truth and final accuracy.

---

## 7. Key Takeaways
- **Template matching** is simple but lacks generalization.  
- **HOG + ML classifiers** improve robustness.  
- **Data augmentation** is critical when training data is tiny.  
- With these steps, we achieved **100% accuracy on both training and new captchas**.

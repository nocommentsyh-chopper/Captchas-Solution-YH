import os, glob
import numpy as np
import cv2
from sklearn.neighbors import KNeighborsClassifier

# ===== (B) Second Attempt: HOG + kNN =====
# - Extract HOG features for each segmented char
# - Train kNN classifier
# - Generalizes better, input100.jpg works
# - Accuracy drops a bit on training set (confusions like N vs 1)

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ============ Segmentation (split chars) ============
def split_chars_cc(img, n_chars=5):
    # segment by connected components
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes = sorted(boxes, key=lambda b: b[0])
    chars = []
    for (x,y,w,h) in boxes:
        if w*h < 50: continue
        chars.append(gray[y:y+h, x:x+w])
    if len(chars) != n_chars:
        W = gray.shape[1] // n_chars
        chars = [gray[:, i*W:(i+1)*W] for i in range(n_chars)]
    return chars

#=========== HOG Feature ============
def extract_hog(img, size=(20,20)):
    # resize + compute HOG features
    winSize = size
    blockSize = (10,10)
    blockStride = (5,5)
    cellSize = (5,5)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    img_resized = cv2.resize(img, size)
    return hog.compute(img_resized).flatten()


 #========== build training dataset ============
def build_trainset(root="sampleCaptchas"):
    in_dir = os.path.join(root,"input")
    out_dir = os.path.join(root,"output")
    X, y = [], []
    for i in range(25):
        img_path = os.path.join(in_dir, f"input{i:02d}.jpg")
        lbl_path = os.path.join(out_dir, f"output{i:02d}.txt")
        if not os.path.exists(img_path): continue
        label = open(lbl_path).read().strip()
        img = cv2.imread(img_path)
        chars = split_chars_cc(img, len(label))
        for cimg, ch in zip(chars, label):
            feat = extract_hog(cimg)
            X.append(feat); y.append(ch)
    return np.array(X), np.array(y)

# ========== knn + preddiction ============
def train_knn(X, y, k=3):
    clf = KNeighborsClassifier(n_neighbors=k)
    clf.fit(X, y)
    return clf

def predict(img_path, clf, n_chars=5):
    img = cv2.imread(img_path)
    chars = split_chars_cc(img, n_chars)
    result = ""
    for cimg in chars:
        feat = extract_hog(cimg)
        pred = clf.predict([feat])[0]
        result += pred
    return result

# ======== MAIN =======
if __name__ == "__main__":
    X, y = build_trainset("sampleCaptchas")
    print(f"[INFO] training size: {len(y)} samples")
    clf = train_knn(X, y, k=3)

    # test sampleCaptchas
    in_dir = "sampleCaptchas/input"
    out_dir = "sampleCaptchas/output"

    total, correct = 0, 0   # <<< add counters

    for ipath in sorted(glob.glob(os.path.join(in_dir,"input*.jpg"))):
        base = os.path.basename(ipath)
        pred = predict(ipath, clf)
        opath = os.path.join(out_dir, base.replace("input","output").replace(".jpg",".txt"))
        if os.path.exists(opath):
            gt = open(opath).read().strip()
            mark = "✓" if pred == gt else "×"
            if pred == gt:
                correct += 1
            total += 1
            print(f"{base} -> pred={pred}, gt={gt} {mark}")
        else:
            print(f"{base} -> pred={pred}, gt=?")

    # print total accuracy at the end
    if total > 0:
        print(f"\nFinal accuracy = {correct}/{total} = {correct/total:.3f}")

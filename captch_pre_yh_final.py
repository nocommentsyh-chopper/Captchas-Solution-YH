import os, glob, random
import numpy as np
import cv2
from sklearn.svm import SVC

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

# ============ HOG 特征 ============
def extract_hog(img, size=(28,28)):
    """输入灰度图 -> HOG 特征向量"""
    winSize = size
    blockSize = (14,14)
    blockStride = (7,7)
    cellSize = (7,7)
    nbins = 9
    hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
    img_resized = cv2.resize(img, size)
    return hog.compute(img_resized).flatten()

# ============ 连通域分割 ============
def split_chars_cc(img, n_chars=5):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
    _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)
    
    cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    boxes = [cv2.boundingRect(c) for c in cnts]
    boxes = sorted(boxes, key=lambda b: b[0])  # 按 x 排序
    
    chars = []
    for (x,y,w,h) in boxes:
        if w*h < 50:  # 去噪
            continue
        char_img = gray[y:y+h, x:x+w]
        chars.append(char_img)
    # 如果数量不对，兜底等宽切
    if len(chars) != n_chars:
        W = gray.shape[1] // n_chars
        chars = [gray[:, i*W:(i+1)*W] for i in range(n_chars)]
    return chars

# ============ 数据增强 ============
def augment_char(img, n_aug=5, size=(28,28)):
    h, w = img.shape[:2]
    results = []
    for _ in range(n_aug):
        # 1. 随机旋转
        angle = random.uniform(-10, 10)
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        rot = cv2.warpAffine(img, M, (w, h), borderValue=255)

        # 2. 随机缩放
        scale = random.uniform(0.85, 1.15)
        nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
        scaled = cv2.resize(rot, (nw, nh))

        # 3. 放入固定大小的画布，带边界检查
        canvas = np.ones((h,w), dtype=np.uint8)*255
        x_off = max(0,(w-nw)//2); y_off = max(0,(h-nh)//2)
        x_end = min(w, x_off+nw); y_end = min(h, y_off+nh)
        canvas[y_off:y_end, x_off:x_end] = scaled[0:(y_end-y_off), 0:(x_end-x_off)]

        # 4. 加高斯噪声
        noise = np.random.normal(0, 10, canvas.shape).astype(np.int16)
        noisy = np.clip(canvas.astype(np.int16)+noise,0,255).astype(np.uint8)

        norm = cv2.resize(noisy, size)
        results.append(norm)
    return results

# ============ 构建训练集 ============
def build_trainset(root="sampleCaptchas", aug=True):
    in_dir = os.path.join(root,"input")
    out_dir = os.path.join(root,"output")
    X, y = [], []

    for i in range(25):  # input00–input24
        img_path = os.path.join(in_dir, f"input{i:02d}.jpg")
        lbl_path = os.path.join(out_dir, f"output{i:02d}.txt")
        if not os.path.exists(img_path) or not os.path.exists(lbl_path):
            continue
        label = open(lbl_path).read().strip()
        img   = cv2.imread(img_path)
        chars = split_chars_cc(img, len(label))
        for cimg, ch in zip(chars, label):
            # 原始样本
            feat = extract_hog(cimg)
            X.append(feat); y.append(ch)
            # 数据增强
            if aug:
                for aug_img in augment_char(cimg, n_aug=5):
                    feat = extract_hog(aug_img)
                    X.append(feat); y.append(ch)
    return np.array(X), np.array(y)

# ============ 训练 + 预测 ============
def train_svm(X, y):
    clf = SVC(kernel="rbf", probability=False, C=5, gamma="scale")
    clf.fit(X, y)
    return clf

def predict(img_path, clf, n_chars=5):
    img   = cv2.imread(img_path)
    chars = split_chars_cc(img, n_chars)
    result = ""
    for cimg in chars:
        feat = extract_hog(cimg)
        pred = clf.predict([feat])[0]
        result += pred
    return result

# ============ MAIN ============
if __name__ == "__main__":
    X, y = build_trainset("sampleCaptchas", aug=True)
    print(f"[INFO] 训练集大小: {len(y)} 个字符样本")
    clf = train_svm(X, y)

    in_dir = "sampleCaptchas/input"
    out_dir = "sampleCaptchas/output"
    total, correct = 0, 0

    # 测试 sampleCaptchas + input100
    test_imgs = sorted(glob.glob(os.path.join(in_dir,"input*.jpg")))
    if os.path.exists("input100.jpg"):
        test_imgs.append("input100.jpg")

    for ipath in test_imgs:
        base = os.path.basename(ipath)
        pred = predict(ipath, clf)
        opath = os.path.join(out_dir, base.replace("input","output").replace(".jpg",".txt"))
        if os.path.exists(opath):
            gt = open(opath).read().strip()
            mark = "✓" if pred == gt else "×"
            print(f"{base} -> pred={pred}, gt={gt} {mark}")
            total += 1
            if pred == gt:
                correct += 1
        else:
            print(f"{base} -> pred={pred}, gt=?")
    if total > 0:
        print(f"\nFinal accuracy = {correct}/{total} = {correct/total:.3f}")

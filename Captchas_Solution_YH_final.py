import os, glob, random
import numpy as np
import cv2
from sklearn.svm import SVC

# ===== (C) Final Solution: HOG + Augmentation + SVM =====
# Highlights:
# 1) Segmentation: connected components -> robust to spacing/bg wiggles
# 2) Feature extraction: HOG on 28x28 chars (a bit more detail than 20x20)
# 3) Data augmentation: rotate/scale/noise, ~x6 data (125 -> ~750 chars)
# 4) Classifier: SVM (RBF), tends to be stabler than kNN on tiny data

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

class Captcha(object):
    def __init__(self, train_root="sampleCaptchas", n_chars=5, do_augment=True, aug_times=5):
        """
        Train once in ctor so inference (__call__) is just: load image -> predict -> save.
        Small dataset, so training here is quick enough.
        """
        self.n_chars = n_chars
        self.do_augment = do_augment
        self.aug_times = aug_times

        # build dataset from the 25 labeled captchas (input00–input24)
        X, y = self.build_trainset(train_root, aug=self.do_augment, n_aug=self.aug_times)
        print(f"[INFO] training size: {len(y)} samples {'(after augmentation)' if self.do_augment else ''}")

        # train SVM (RBF). Pretty safe default for small HOG features.
        self.clf = self.train_svm(X, y)

    # ============ Segmentation (split chars) ============
    def split_chars_cc(self, img, n_chars=5):
        """
        Step 1 = segmentation (yep, we split first).
        Try connected components first; if count != expected, fall back to equal-width slicing.
        """
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape)==3 else img
        _, bin_ = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)

        cnts, _ = cv2.findContours(bin_, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        boxes = [cv2.boundingRect(c) for c in cnts]
        boxes = sorted(boxes, key=lambda b: b[0])  # left -> right

        chars = []
        for (x,y,w,h) in boxes:
            if w*h < 50:      # tiny crumbs… toss them
                continue
            chars.append(gray[y:y+h, x:x+w])

        # fallback: when CC fails (merged/split), just equal-slice by width
        if len(chars) != n_chars:
            W = gray.shape[1] // n_chars
            chars = [gray[:, i*W:(i+1)*W] for i in range(n_chars)]
        return chars

    # ============ HOG Feature ============
    def extract_hog(self, img, size=(28,28)):
        """
        Keep HOG simple & consistent. 28x28 gives a tad more detail than 20x20.
        """
        winSize = size
        blockSize = (14,14)
        blockStride = (7,7)
        cellSize = (7,7)
        nbins = 9
        hog = cv2.HOGDescriptor(winSize, blockSize, blockStride, cellSize, nbins)
        img_resized = cv2.resize(img, size)
        return hog.compute(img_resized).flatten()

    # ============ Data Augmentation ============
    def augment_char(self, img, n_aug=5, size=(28,28)):
        """
        Light-touch augmentations (rotate/scale/noise). Nothing fancy,
        just enough to make SVM less brittle to tiny visual shifts.
        """
        h, w = img.shape[:2]
        results = []
        for _ in range(n_aug):
            # random rotate
            angle = random.uniform(-10, 10)
            M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
            rot = cv2.warpAffine(img, M, (w, h), borderValue=255)

            # random scale
            scale = random.uniform(0.85, 1.15)
            nh, nw = max(1, int(h*scale)), max(1, int(w*scale))
            scaled = cv2.resize(rot, (nw, nh))

            # paste onto a clean canvas with boundary checks (avoid numpy broadcast errors)
            canvas = np.ones((h,w), dtype=np.uint8)*255
            x_off = max(0,(w-nw)//2); y_off = max(0,(h-nh)//2)
            x_end = min(w, x_off+nw);  y_end = min(h, y_off+nh)
            canvas[y_off:y_end, x_off:x_end] = scaled[0:(y_end-y_off), 0:(x_end-x_off)]

            # sprinkle some gaussian noise
            noise = np.random.normal(0, 10, canvas.shape).astype(np.int16)
            noisy = np.clip(canvas.astype(np.int16)+noise,0,255).astype(np.uint8)

            norm = cv2.resize(noisy, size)
            results.append(norm)
        return results

    # ============ build training dataset ============
    def build_trainset(self, root="sampleCaptchas", aug=True, n_aug=5):
        """
        Build (X, y) by segmenting each labeled captcha into 5 chars,
        extracting HOG features, and (optionally) augmenting each char.
        """
        in_dir = os.path.join(root,"input")
        out_dir = os.path.join(root,"output")
        X, y = [], []

        # by spec we have input00–input24 (25 images -> 125 chars)
        for i in range(25):
            img_path = os.path.join(in_dir, f"input{i:02d}.jpg")
            lbl_path = os.path.join(out_dir, f"output{i:02d}.txt")
            if not os.path.exists(img_path) or not os.path.exists(lbl_path):
                # if something's missing, just skip (or you can raise)
                continue

            label = open(lbl_path, "r", encoding="utf-8").read().strip()
            img = cv2.imread(img_path)
            chars = self.split_chars_cc(img, len(label))

            for cimg, ch in zip(chars, label):
                # original char sample
                feat = self.extract_hog(cimg)
                X.append(feat); y.append(ch)

                # (optional) augmentation expands data ~x(n_aug+1) per char
                if aug:
                    for aug_img in self.augment_char(cimg, n_aug=n_aug):
                        feat = self.extract_hog(aug_img)
                        X.append(feat); y.append(ch)

        return np.array(X), np.array(y)

    # ============ training ============
    def train_svm(self, X, y):
        """
        SVM with RBF kernel: a good fit for small-ish HOG features.
        """
        clf = SVC(kernel="rbf", probability=False, C=5, gamma="scale")
        clf.fit(X, y)
        return clf

    # ============ inference ============
    def predict_one(self, img_path):
        """
        Predict the 5-char string from a single .jpg path.
        """
        img = cv2.imread(img_path)
        chars = self.split_chars_cc(img, self.n_chars)
        result = []
        for cimg in chars:
            feat = self.extract_hog(cimg)
            pred = self.clf.predict([feat])[0]
            result.append(pred)
        return "".join(result)

    def __call__(self, im_path, save_path):
        """
        Algo for inference
        args:
            im_path: .jpg image path to load and infer
            save_path: output file path to save the one-line outcome
        """
        pred = self.predict_one(im_path)
        os.makedirs(os.path.dirname(save_path) or ".", exist_ok=True)
        with open(save_path, "w", encoding="utf-8") as f:
            f.write(pred + "\n")
        # also print for quick eyeballing
        print(f"[PRED] {os.path.basename(im_path)} -> {pred} (saved to {save_path})")


# ============ (Optional) quick CLI demo ============
if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--train_root", type=str, default="sampleCaptchas",
                    help="where inputXX.jpg/outputXX.txt live")
    ap.add_argument("--im", type=str, required=False, default="input100.jpg",
                    help="a .jpg image to infer")
    ap.add_argument("--out", type=str, required=False, default="pred.txt",
                    help="where to save the one-line prediction")
    ap.add_argument("--no_aug", action="store_true",
                    help="turn off data augmentation (not recommended)")
    ap.add_argument("--n_aug", type=int, default=5,
                    help="how many augmented samples per char")

    args = ap.parse_args()

    # train the model (small data… this is fast)
    cap = Captcha(train_root=args.train_root,
                  n_chars=5,
                  do_augment=not args.no_aug,
                  aug_times=args.n_aug)

    # run one example
    if args.im:
        cap(args.im, args.out)

    # if you want to quickly check dataset accuracy (incl. input100 if present)
    in_dir = os.path.join(args.train_root, "input")
    out_dir = os.path.join(args.train_root, "output")
    test_imgs = sorted(glob.glob(os.path.join(in_dir, "input*.jpg")))
    if os.path.exists("input100.jpg"):
        test_imgs.append("input100.jpg")

    total = correct = 0
    for ipath in test_imgs:
        base = os.path.basename(ipath)
        pred = cap.predict_one(ipath)
        # only compare when we have a ground-truth
        opath = os.path.join(out_dir, base.replace("input","output").replace(".jpg",".txt"))
        if os.path.exists(opath):
            gt = open(opath, "r", encoding="utf-8").read().strip()
            mark = "✓" if pred == gt else "×"
            print(f"{base} -> pred={pred}, gt={gt} {mark}")
            total += 1
            if pred == gt:
                correct += 1
        else:
            print(f"{base} -> pred={pred}, gt=? (no label)")
    if total > 0:
        print(f"\nFinal accuracy = {correct}/{total} = {correct/total:.3f}")

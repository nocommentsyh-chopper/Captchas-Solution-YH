import os, glob
import numpy as np
from PIL import Image

# ===== (A) First Attempt: Template Matching =====
# - Build a simple template library by splitting 25 training captchas
# - Recognition = pixel-level mean squared error (MSE)
# - Works perfectly on training set, but fails to generalize (e.g. input100.jpg)

CHARS = "ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"
TRAIN_IDS = list(range(0,25))   # training images: input00–input24
IMG_SIZE = (60,30)              # resize captcha to fixed size (w=60,h=30)
CHAR_W = IMG_SIZE[0] // 5       # 5 chars => each ~12px wide

def load_txt_label(path):
    return open(path, "r", encoding="utf-8").read().strip()

def img_to_arr(img):
    # convert to grayscale array normalized [0,1]
    return np.array(img.convert("L"), dtype=np.float32) / 255.0

def mse(a, b):
    return ((a - b) ** 2).mean()  # simple MSE

# ===== Build template library =====
def build_templates(root="sampleCaptchas"):
    in_dir = os.path.join(root,"input")
    out_dir = os.path.join(root,"output")
    templates = {c:[] for c in CHARS}
    for i in TRAIN_IDS:
        img_path = os.path.join(in_dir, f"input{i:02d}.jpg")
        lbl_path = os.path.join(out_dir, f"output{i:02d}.txt")
        if not os.path.exists(img_path): continue
        label = load_txt_label(lbl_path)
        img = Image.open(img_path).resize(IMG_SIZE)
        arr = img_to_arr(img)
        # naive split into 5 equal-width pieces
        for j, ch in enumerate(label):
            x0, x1 = j*CHAR_W, (j+1)*CHAR_W
            char_img = arr[:, x0:x1]
            templates[ch].append(char_img)
    return templates

# ===== Predict with template matching =====
def predict(img_path, templates):
    img = Image.open(img_path).resize(IMG_SIZE)
    arr = img_to_arr(img)
    result = []
    for j in range(5):
        x0, x1 = j*CHAR_W, (j+1)*CHAR_W
        char_img = arr[:, x0:x1]
        best_c, best_score = None, 1e9
        for c, tlist in templates.items():
            for temp in tlist:
                score = mse(char_img, temp)
                if score < best_score:
                    best_c, best_score = c, score
        result.append(best_c if best_c else "?")
    return "".join(result)

# ===== MAIN =====
if __name__ == "__main__":
    sample_root = "sampleCaptchas"
    in_dir = os.path.join(sample_root,"input")
    out_dir = os.path.join(sample_root,"output")
    templates = build_templates(sample_root)
    all_imgs = sorted(glob.glob(os.path.join(in_dir,"input*.jpg")))
    total, correct = 0, 0
    for ipath in all_imgs:
        base = os.path.basename(ipath)
        pred = predict(ipath, templates)
        opath = os.path.join(out_dir, base.replace("input","output").replace(".jpg",".txt"))
        if os.path.exists(opath):
            gt = load_txt_label(opath)
            mark = "✔" if pred == gt else "✘"
            print(f"{base} -> pred={pred}, gt={gt} {mark}")
            total += 1
            if pred == gt: correct += 1
    if total > 0:
        print(f"Final accuracy = {correct}/{total} = {correct/total:.3f}")

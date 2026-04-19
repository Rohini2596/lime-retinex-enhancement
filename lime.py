import os
import cv2
import numpy as np
from utils import *
LOW_DIR = "LOLDataset/eval15/low"
HIGH_DIR = "LOLDataset/eval15/high"
OUT_DIR = "output"
os.makedirs(OUT_DIR, exist_ok=True)
for img_name in os.listdir(LOW_DIR):
    print("Processing:", img_name)
    low_path = os.path.join(LOW_DIR, img_name)
    high_path = os.path.join(HIGH_DIR, img_name)
    img = read_image(low_path)
    T_hat = initial_illumination(img)
    T = refine_illumination(T_hat)
    enhanced = enhance(img, T)
    den = denoise(enhanced)
    final = recomposition(enhanced, den, T)
    save_image(os.path.join(OUT_DIR, "enhanced_"+img_name), enhanced)
    save_image(os.path.join(OUT_DIR, "final_"+img_name), final)
    if os.path.exists(high_path):
        gt = read_image(high_path)
        combined = np.hstack((img, enhanced, final, gt))
        save_image(os.path.join(OUT_DIR, "compare_"+img_name), combined)
print("DONE!!!!")
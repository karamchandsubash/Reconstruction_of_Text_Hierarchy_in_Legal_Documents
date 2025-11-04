#!/usr/bin/env python3
"""
utils.py - small helpers shared by scripts
"""
import json
from pathlib import Path
import cv2
from pytesseract import Output, image_to_data
from PIL import Image
import numpy as np

def load_config(path="config.json"):
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with open(p, "r", encoding="utf-8") as f:
        return json.load(f)

def ensure_dirs(*paths):
    for p in paths:
        Path(p).mkdir(parents=True, exist_ok=True)

def read_results_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def read_image(path):
    img = cv2.imread(str(path))
    if img is None:
        pil = Image.open(path).convert("RGB")
        img = np.array(pil)[:, :, ::-1].copy()
    return img

def ocr_words_from_image(img_or_path, lang="deu+frk", tess_config="--psm 6 --oem 3"):
    # Accepts cv2 image or path
    if isinstance(img_or_path, str) or isinstance(img_or_path, Path):
        img = read_image(img_or_path)
    else:
        img = img_or_path
    data = image_to_data(img, lang=lang, config=tess_config, output_type=Output.DICT)
    words = []
    H, W = img.shape[:2]
    for i, txt in enumerate(data.get("text", [])):
        t = str(txt).strip()
        if not t:
            continue
        try:
            conf = float(data["conf"][i])
        except:
            conf = -1.0
        x,y,w,h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append({"text": t, "conf": conf, "bbox":[x,y,w,h], "cx": x + w/2, "cy": y + h/2})
    return words, W, H

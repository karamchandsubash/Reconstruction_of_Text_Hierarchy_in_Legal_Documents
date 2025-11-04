#!/usr/bin/env python3
"""
token_labeler.py
- Reads Label Studio prediction JSONs (labelstudio_preds_dir)
- Runs word-level OCR to get tokens
- Aligns tokens to rectangles and writes token-level BIO JSONs
"""
import json
from pathlib import Path
from utils import load_config
from pytesseract import Output, image_to_data
from PIL import Image
import cv2

cfg = load_config()
LS_DIR = Path(cfg.get("labelstudio_preds_dir"))
OUT_DIR = Path(cfg.get("output_dir")) / "token_samples"
OUT_DIR.mkdir(parents=True, exist_ok=True)
TESS_CONFIG = cfg.get("tesseract_config", "--psm 6 --oem 3")
LANG = cfg.get("lang", "deu+frk")

def safe_imread(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        pil = Image.open(path).convert("RGB")
        import numpy as np
        img = np.array(pil)[:, :, ::-1].copy()
    return img

def ocr_words(img_path: str):
    img = safe_imread(Path(img_path))
    data = image_to_data(img, lang=LANG, config=TESS_CONFIG, output_type=Output.DICT)
    words = []
    H, W = img.shape[:2]
    for i, t in enumerate(data.get("text", [])):
        txt = str(t).strip()
        if not txt:
            continue
        try:
            conf = float(data["conf"][i])
        except:
            conf = -1.0
        x,y,w,h = int(data["left"][i]), int(data["top"][i]), int(data["width"][i]), int(data["height"][i])
        words.append({"text": txt, "conf": conf, "bbox":[x,y,w,h]})
    return words, W, H

def token_labels_for_page(pred_json_path: Path):
    with open(pred_json_path, "r", encoding="utf-8") as f:
        pred = json.load(f)
    image_path = pred["data"]["image"]
    results = pred.get("predictions", [{}])[0].get("result", [])
    words, W, H = ocr_words(image_path)
    tokens = [w["text"] for w in words]
    centers = [((w["bbox"][0] + w["bbox"][2]) / 2, (w["bbox"][1] + w["bbox"][3]) / 2) for w in words]
    token_labels = ["O"] * len(tokens)
    for r in results:
        val = r["value"]
        lab = val["rectanglelabels"][0]
        x = val["x"] / 100 * W
        y = val["y"] / 100 * H
        w = val["width"] / 100 * W
        h = val["height"] / 100 * H
        x0, y0, x1, y1 = x, y, x + w, y + h
        inside_idx = [i for i, (cx, cy) in enumerate(centers) if cx >= x0 and cx <= x1 and cy >= y0 and cy <= y1]
        if not inside_idx:
            continue
        for j, idx in enumerate(sorted(inside_idx)):
            token_labels[idx] = ("B-" if j == 0 else "I-") + lab.replace(" ", "_").upper()
    boxes = [[int(w["bbox"][0]), int(w["bbox"][1]), int(w["bbox"][0] + w["bbox"][2]), int(w["bbox"][1] + w["bbox"][3])] for w in words]
    return {"image": image_path, "tokens": tokens, "labels": token_labels, "boxes": boxes}

def run_all():
    preds = sorted(LS_DIR.glob("*.json"))
    if not preds:
        print("No predictions found in", LS_DIR)
        return
    for p in preds:
        try:
            sample = token_labels_for_page(p)
            out = OUT_DIR / (Path(sample["image"]).stem + ".json")
            with open(out, "w", encoding="utf-8") as fo:
                json.dump(sample, fo, ensure_ascii=False, indent=2)
            print("Wrote", out.name)
        except Exception as e:
            print("Error on", p.name, e)

if __name__ == "__main__":
    run_all()

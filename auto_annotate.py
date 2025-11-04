#!/usr/bin/env python3
"""
auto_annotate.py
- Runs Tesseract OCR (deu+frk by default)
- Clusters words into blocks
- Heuristically labels blocks (Title, Paragraph, Section-Heading, List-Item, Page-Footer, Page-Number, Picture, Caption)
- Writes Label Studio prediction JSONs (one per image)
- Writes TEI XML (one per image)
"""
import json
import re
from pathlib import Path
from collections import defaultdict
import numpy as np
import cv2
from pytesseract import Output
import pytesseract
from lxml import etree
from PIL import Image
from sklearn.cluster import DBSCAN

# Load config
from utils import load_config
cfg = load_config()

IMAGES_DIR = Path(cfg.get("images_dir"))
OUT_DIR = Path(cfg.get("output_dir"))
LS_DIR = Path(cfg.get("labelstudio_preds_dir"))
TEI_DIR = Path(cfg.get("tei_dir"))
LANG = cfg.get("lang", "deu+frk")
TESS_CONFIG = cfg.get("tesseract_config", "--psm 6 --oem 3")
DBSCAN_EPS = cfg.get("dbscan_eps", 12)

for p in [OUT_DIR, LS_DIR, TEI_DIR]:
    p.mkdir(parents=True, exist_ok=True)

IMAGE_EXTS = [".jpg", ".jpeg", ".png", ".tif", ".tiff"]

def safe_imread(path: Path):
    img = cv2.imread(str(path))
    if img is None:
        pil = Image.open(path).convert("RGB")
        img = np.array(pil)[:, :, ::-1].copy()
    return img

def ocr_words_from_image_path(img_path: Path, lang=LANG, tess_config=TESS_CONFIG):
    img = safe_imread(img_path)
    data = pytesseract.image_to_data(img, lang=lang, config=tess_config, output_type=Output.DICT)
    words = []
    H, W = img.shape[:2]
    for i, txt in enumerate(data.get('text', [])):
        t = str(txt).strip()
        if not t:
            continue
        try:
            conf = float(data['conf'][i])
        except Exception:
            conf = -1.0
        x,y,w,h = int(data['left'][i]), int(data['top'][i]), int(data['width'][i]), int(data['height'][i])
        words.append({"text": t, "conf": conf, "bbox":[x,y,w,h], "cx": x + w/2, "cy": y + h/2})
    return words, W, H

def cluster_words(words, eps=DBSCAN_EPS):
    if not words:
        return []
    pts = np.array([[w['cy'], w['cx']] for w in words])
    if len(pts) == 1:
        labels = np.array([0])
    else:
        labels = DBSCAN(eps=eps, min_samples=1).fit_predict(pts)
    blocks = defaultdict(list)
    for i, lab in enumerate(labels):
        blocks[int(lab)].append(words[i])
    out = []
    for lab, ws in blocks.items():
        xs = [w['bbox'][0] for w in ws]; ys = [w['bbox'][1] for w in ws]
        wsizes = [w['bbox'][2] for w in ws]; hs = [w['bbox'][3] for w in ws]
        x0 = min(xs); y0 = min(ys)
        x1 = max([x + ww for x,ww in zip(xs, wsizes)]); y1 = max([y + hh for y,hh in zip(ys, hs)])
        text = " ".join([w['text'] for w in sorted(ws, key=lambda z: z['cx'])])
        out.append({"words": ws, "bbox": [int(x0), int(y0), int(x1-x0), int(y1-y0)], "text": text})
    return sorted(out, key=lambda b: b['bbox'][1])

def decide_label(block, H):
    x,y,w,h = block['bbox']
    cy = y + h/2
    text = block['text'].strip()
    nwords = len(text.split())
    # Title heuristic: near top, short
    if cy / H < 0.18 and 1 <= nwords <= 15:
        return "Title"
    # Footer heuristics
    if cy / H > 0.92:
        if re.fullmatch(r"[\dIViv\.\- ]{1,10}", text):
            return "Page-Number"
        return "Page-Footer"
    # List bullets / numbered
    if re.match(r"^(\d+[\.\)]|[-•\u2022])\s+", text):
        return "List-Item"
    # Short lines with capitals -> headings
    if 1 <= nwords <= 8:
        capratio = sum(1 for w in text.split() if w[:1].isupper()) / max(1, nwords)
        if capratio > 0.6 or text.isupper():
            return "Section-Heading"
    # Default sensible split
    if nwords < 4:
        return "Paragraph"
    return "Paragraph"

def percent_bbox_from_pixels(bbox, W, H):
    x,y,w,h = bbox
    return {"x": x/W*100, "y": y/H*100, "width": w/W*100, "height": h/H*100}

def build_tei(results, tei_path):
    TEI = etree.Element("TEI", nsmap={None:"http://www.tei-c.org/ns/1.0"})
    text_el = etree.SubElement(TEI, "text")
    body = etree.SubElement(text_el, "body")
    for r in sorted(results, key=lambda r: r['value']['y']):
        lab = r['value']['rectanglelabels'][0]
        content = r['value'].get('text', "").strip()
        if not content:
            continue
        if lab == "Title":
            div = etree.SubElement(body, "div", type="title")
            etree.SubElement(div, "head").text = content
        elif lab == "Paragraph":
            etree.SubElement(body, "p").text = content
        elif lab.startswith("Section"):
            div = etree.SubElement(body, "div", type="section")
            etree.SubElement(div, "head").text = content
        elif lab == "List-Item":
            lst = etree.SubElement(body, "list")
            etree.SubElement(lst, "item").text = content
        elif lab in ("Page-Footer","Page-Header","Page-Number"):
            etree.SubElement(body, "note", type=lab.lower()).text = content
        elif lab in ("Picture","Caption"):
            fig = etree.SubElement(body, "figure")
            etree.SubElement(fig, "figDesc").text = content
        else:
            etree.SubElement(body, "p").text = content
    with open(tei_path, "wb") as fo:
        fo.write(etree.tostring(TEI, pretty_print=True, encoding="utf-8", xml_declaration=True))

def main():
    imgs = []
    for ext in IMAGE_EXTS:
        imgs.extend(sorted(IMAGES_DIR.rglob(f"*{ext}")))
    if not imgs:
        print("No images found in", IMAGES_DIR)
        return
    for idx, img_path in enumerate(imgs):
        try:
            words, W, H = ocr_words_from_image_path(img_path, lang=LANG)
            blocks = cluster_words(words)
            blocks = [b for b in blocks if len(b['text'].split()) >= 1]
            results = []
            for i, blk in enumerate(blocks):
                b = blk['bbox']
                p = percent_bbox_from_pixels(b, W, H)
                label = decide_label(blk, H)
                value = {
                    "rotation": 0,
                    "x": round(p['x'],4),
                    "y": round(p['y'],4),
                    "width": round(p['width'],4),
                    "height": round(p['height'],4),
                    "rectanglelabels": [label],
                    "text": blk['text']
                }
                res = {
                    "id": f"result_{idx}_{i}",
                    "type": "rectanglelabels",
                    "from_name": "label",
                    "to_name": "image",
                    "original_width": float(W),
                    "original_height": float(H),
                    "image_rotation": 0,
                    "value": value
                }
                results.append(res)
            pred = {"data": {"image": str(img_path)}, "predictions": [{"result": results, "model_version": "auto-frk-v1", "score": 0.8}]}
            out_pred = LS_DIR / (Path(img_path).stem + "_pred.json")
            with open(out_pred, "w", encoding="utf-8") as fo:
                json.dump(pred, fo, ensure_ascii=False, indent=2)
            tei_path = TEI_DIR / (Path(img_path).stem + ".xml")
            build_tei(results, tei_path)
            print(f"[OK] {img_path.name} -> {len(results)} blocks")
        except Exception as e:
            print(f"[ERR] {img_path.name}: {e}")

if __name__ == "__main__":
    main()

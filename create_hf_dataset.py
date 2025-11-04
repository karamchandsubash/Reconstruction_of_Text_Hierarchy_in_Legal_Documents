#!/usr/bin/env python3
"""
create_hf_dataset.py
- Loads token JSONs (output_dir/token_samples)
- Uses LayoutLMv3Processor to encode images+tokens+boxes
- Aligns word-level labels -> token-level labels using word_ids()
- Saves dataset to disk (hf_dataset) and writes label2id.json
"""
import json
from pathlib import Path
from transformers import LayoutLMv3Processor
from datasets import Dataset
from PIL import Image
import numpy as np
from utils import load_config

cfg = load_config()
TOKEN_DIR = Path(cfg.get("output_dir")) / "token_samples"
HF_DIR = Path(cfg.get("output_dir")) / "hf_dataset"
HF_DIR.mkdir(parents=True, exist_ok=True)
MODEL_NAME = "microsoft/layoutlmv3-base"
MAX_LENGTH = cfg.get("max_length", 512)

processor = LayoutLMv3Processor.from_pretrained(MODEL_NAME, apply_ocr = False)

def normalize_boxes(boxes, W, H):
    # LayoutLM expects bbox in 0-1000
    out = []
    for (x0,y0,x1,y1) in boxes:
        out.append([int(round(x0 / W * 1000)), int(round(y0 / H * 1000)), int(round(x1 / W * 1000)), int(round(y1 / H * 1000))])
    return out

def load_samples():
    files = sorted(TOKEN_DIR.glob("*.json"))
    samples = []
    for p in files:
        with open(p, "r", encoding="utf-8") as f:
            samples.append(json.load(f))
    return samples

def build_label_map(samples):
    labels = []
    for s in samples:
        for l in s["labels"]:
            if l not in labels:
                labels.append(l)
    labels = sorted(labels)
    label2id = {l:i for i,l in enumerate(labels)}
    id2label = {i:l for l,i in label2id.items()}
    return labels, label2id, id2label

def encode_sample(sample, label2id):
    image = Image.open(sample["image"]).convert("RGB")
    W, H = image.size
    words = sample["tokens"]
    boxes = sample["boxes"]
    normalized_boxes = normalize_boxes(boxes, W, H)
    encoding = processor(image, words, boxes=normalized_boxes, padding="max_length", truncation=True, max_length=MAX_LENGTH)
    # map word-level labels to token-level via word_ids
    word_ids = encoding.word_ids()
    labels = sample["labels"]
    label_ids = []
    for wid in word_ids:
        if wid is None:
            label_ids.append(-100)
        else:
            # some samples shorter/longer - protect
            if wid >= len(labels):
                label_ids.append(label2id.get("O", -100))
            else:
                label_ids.append(label2id.get(labels[wid], label2id.get("O", -100)))
    item = {
        "input_ids": encoding["input_ids"],
        "attention_mask": encoding["attention_mask"],
        "bbox": encoding["bbox"],
        "labels": label_ids,
        "image_path": sample["image"]
    }
    return item

def main():
    samples = load_samples()
    if not samples:
        raise SystemExit("No token samples found. Run token_labeler.py first.")
    labels, label2id, id2label = build_label_map(samples)
    with open(HF_DIR / "label2id.json", "w", encoding="utf-8") as fo:
        json.dump(label2id, fo, ensure_ascii=False, indent=2)
    encodings = []
    for s in samples:
        enc = encode_sample(s, label2id)
        encodings.append(enc)
    ds = Dataset.from_list(encodings)
    ds.save_to_disk(str(HF_DIR))
    print(f"Saved HF dataset to {HF_DIR}. Labels: {labels}")

if __name__ == "__main__":
    main()


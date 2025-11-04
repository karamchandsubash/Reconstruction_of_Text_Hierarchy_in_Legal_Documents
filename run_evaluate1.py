

import os
import numpy as np
import torch
from datasets import load_from_disk
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    DataCollatorForTokenClassification,
)
from seqeval.metrics import (
    precision_score, recall_score, f1_score,
    classification_report
)
import evaluate
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns


DATASET_PATH = "/home/labthomas/auto_output/hf_dataset_split"
MODEL_DIR = "/home/labthomas/text-structure-project/text-structure-project/model_output_1"

label_list = [
    "O", "Paragraph", "List-Item", "Section-Heading", "Page-Footer", "Page-Header",
    "Page-Number", "Table", "Table-of-contents", "Title",
    "2nd-Level Section-Heading", "3rd-Level Section-Heading", "Lower-Level Section-Heading",
    "Picture", "Caption", "Heading-with-List-Item", "Salutation", "Reference"
]
id2label = {i: label for i, label in enumerate(label_list)}
label2id = {label: i for i, label in id2label.items()}


print(" Loading dataset...")
dataset = load_from_disk(DATASET_PATH)
print(f" Dataset splits: {list(dataset.keys())}")

# if no validation split, use test
if "validation" in dataset:
    eval_dataset = dataset["validation"]
else:
    eval_dataset = dataset["test"]


print(" Loading model...")
processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)

model = LayoutLMv3ForTokenClassification.from_pretrained(
    MODEL_DIR,
    id2label=id2label,
    label2id=label2id
)

data_collator = DataCollatorForTokenClassification(tokenizer=processor.tokenizer)

wer_metric = evaluate.load("wer")
cer_metric = evaluate.load("cer")

all_true = []
all_pred = []

def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []

    for true_seq, pred_seq in zip(labels, predictions):
        true_seq_labels = []
        pred_seq_labels = []
        for t, p_ in zip(true_seq, pred_seq):
            if t != -100:
                true_seq_labels.append(id2label[t])
                pred_seq_labels.append(id2label[p_])
        true_labels.append(true_seq_labels)
        pred_labels.append(pred_seq_labels)

        all_true.extend(true_seq_labels)
        all_pred.extend(pred_seq_labels)

    # Seqeval metrics
    precision = precision_score(true_labels, pred_labels)
    recall = recall_score(true_labels, pred_labels)
    f1 = f1_score(true_labels, pred_labels)

    # CER/WER
    cer_scores, wer_scores = [], []
    for t_seq, p_seq in zip(true_labels, pred_labels):
        t_str, p_str = " ".join(t_seq), " ".join(p_seq)
        try:
            cer_scores.append(cer_metric.compute(predictions=[p_str], references=[t_str]))
            wer_scores.append(wer_metric.compute(predictions=[p_str], references=[t_str]))
        except Exception:
            continue
    cer = float(np.mean(cer_scores)) if cer_scores else 1.0
    wer = float(np.mean(wer_scores)) if wer_scores else 1.0

    # Per-class report
    class_report = classification_report(true_labels, pred_labels, output_dict=True)

    metrics = {
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cer": cer,
        "wer": wer,
    }

    # Add per-class F1
    for label in class_report.keys():
        if label in ["micro avg", "macro avg", "weighted avg"]:
            continue
        metrics[f"{label}_f1"] = class_report[label]["f1-score"]

    return metrics

trainer = Trainer(
    model=model,
    args=None,  # No training, only evaluation
    eval_dataset=eval_dataset,
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

print(" Running evaluation...")
results = trainer.evaluate()
print(" Evaluation complete!")

# Print metrics
print("\n Final Results:")
for k, v in results.items():
    print(f"{k}: {v}")


print("\n Generating confusion matrix...")
cm = confusion_matrix(all_true, all_pred, labels=label_list)

plt.figure(figsize=(14, 12))
sns.heatmap(cm, annot=False, xticklabels=label_list, yticklabels=label_list, cmap="Blues", fmt="d")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.title("Confusion Matrix")
cm_path = os.path.join(MODEL_DIR, "confusion_matrix.png")
plt.savefig(cm_path)
plt.close()

output_file = os.path.join(MODEL_DIR, "evaluation_report.txt")
with open(output_file, "w", encoding="utf-8") as f:
    for k, v in results.items():
        f.write(f"{k}: {v}\n")

print(f" Saved evaluation report to {output_file}")
print(f" Confusion matrix saved to {cm_path}")

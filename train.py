
import os
import json
import torch
import numpy as np
from datasets import load_from_disk
from transformers import (
    LayoutLMv3ForTokenClassification,
    LayoutLMv3Processor,
    Trainer,
    DataCollatorForTokenClassification,
    TrainingArguments,
    EarlyStoppingCallback,
    __version__ as hf_version,
)
from seqeval.metrics import precision_score, recall_score, f1_score


DATASET_PATH = "/home/labthomas/auto_output/hf_dataset_split"
MODEL_DIR = "./model_output_1"


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

if "train" not in dataset:
    raise ValueError(" No training split found in dataset. Did you run create_hf_dataset.py?")

if "validation" not in dataset and "test" in dataset:
    dataset["validation"] = dataset["test"]

print(f" Dataset splits: {list(dataset.keys())}")


processor = LayoutLMv3Processor.from_pretrained("microsoft/layoutlmv3-base", apply_ocr=False)
model = LayoutLMv3ForTokenClassification.from_pretrained(
    "microsoft/layoutlmv3-base",
    num_labels=len(label_list),
    id2label=id2label,
    label2id=label2id,
)

data_collator = DataCollatorForTokenClassification(tokenizer=processor.tokenizer)


print(f"Transformers version: {hf_version}")

args = {
    "output_dir": MODEL_DIR,
    "overwrite_output_dir": True,
    "learning_rate": 2e-5,               
    "per_device_train_batch_size": 8,    
    "per_device_eval_batch_size": 8,
    "num_train_epochs": 250,             
    "weight_decay": 0.01,
    "gradient_accumulation_steps": 2,    
    "save_total_limit": 3,
    "logging_dir": "./logs",
    "logging_steps": 200,                
    "save_steps": 1000,
    "eval_steps": 1000,
    "fp16": torch.cuda.is_available(),
    "load_best_model_at_end": True,
    "metric_for_best_model": "f1",
    "greater_is_better": True,
    "warmup_ratio": 0.05,                
    "lr_scheduler_type": "linear",
}


try:
    training_args = TrainingArguments(
        **args,
        evaluation_strategy="steps",
        save_strategy="steps",
    )
except TypeError:
    training_args = TrainingArguments(
        **args,
        eval_strategy="steps",
        save_strategy="steps",
    )


def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_labels = []
    pred_labels = []
    for true_seq, pred_seq in zip(labels, predictions):
        for t, p_ in zip(true_seq, pred_seq):
            if t != -100:
                true_labels.append(id2label[t])
                pred_labels.append(id2label[p_])

    return {
        "precision": precision_score([true_labels], [pred_labels]),
        "recall": recall_score([true_labels], [pred_labels]),
        "f1": f1_score([true_labels], [pred_labels]),
    }


trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["validation"],
    tokenizer=processor.feature_extractor,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=3)],  
)


print(" Training on GPU" if torch.cuda.is_available() else " Training on CPU")
trainer.train()
trainer.save_model(MODEL_DIR)
train_logs=trainer.train()
trainer.save_model(MODEL_DIR)
print(" Training complete! Best model saved to", MODEL_DIR)



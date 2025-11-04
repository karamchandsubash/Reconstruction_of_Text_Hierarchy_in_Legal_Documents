from transformers import LayoutLMv3ForTokenClassification

def build_model(num_labels, model_name='microsoft/layoutlmv3-base', id2label=None, label2id=None):
    model = LayoutLMv3ForTokenClassification.from_pretrained(model_name, num_labels=num_labels, id2label=id2label, label2id=label2id)
    return model

import torch
import torch.nn as nn
from transformers import DistilBertModel, DistilBertTokenizerFast


class FakeNewsClassifier(nn.Module):
    def __init__(self, num_labels=2, dropout=0.3):
        super().__init__()
        self.bert = DistilBertModel.from_pretrained("distilbert-base-uncased")
        self.drop = nn.Dropout(dropout)
        self.classifier = nn.Linear(768, num_labels)

    def forward(self, input_ids, attention_mask):
        out = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        pooled = out.last_hidden_state[:, 0, :]
        return self.classifier(self.drop(pooled))


class MisinfoDetector:
    LABEL_MAP = {0: "TRUE", 1: "FALSE"}

    def __init__(self, model_path=None, device=None):
        self.device = device or (
            'cuda' if torch.cuda.is_available() else 'cpu'
        )
        self.tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-uncased"
        )
        self.model = FakeNewsClassifier()

        if model_path:
            self.model.load_state_dict(
                torch.load(model_path, map_location=self.device)
            )
        self.model.eval().to(self.device)

    def predict(self, text: str, max_len=256) -> dict:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=max_len,
            return_tensors='pt'
        )
        ids  = enc['input_ids'].to(self.device)
        mask = enc['attention_mask'].to(self.device)

        with torch.no_grad():
            logits = self.model(ids, mask)
            probs  = torch.softmax(logits, dim=1)[0]

        label_idx = probs.argmax().item()
        return {
            "verdict":    self.LABEL_MAP[label_idx],
            "confidence": round(probs[label_idx].item(), 4),
            "scores": {
                "true":  round(probs[0].item(), 4),
                "false": round(probs[1].item(), 4)
            }
        }
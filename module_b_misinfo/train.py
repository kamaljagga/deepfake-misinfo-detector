import os
import sys
import torch
import pandas as pd
import torch.utils.data
from torch.utils.data import Dataset, DataLoader
from torch.optim import AdamW
from transformers import DistilBertTokenizerFast
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from module_b_misinfo.model import FakeNewsClassifier

# ── Label mapping ──────────────────────────────────────────────
LABEL_MAP = {
    "true":        0,
    "mostly-true": 0,
    "half-true":   0,
    "barely-true": 1,
    "false":       1,
    "pants-fire":  1
}

# ── Dataset ────────────────────────────────────────────────────
class LIARDataset(Dataset):
    def __init__(self, tsv_path, tokenizer, max_len=256):
        df = pd.read_csv(
            tsv_path, sep='\t', header=None,
            usecols=[1, 2], names=['label', 'statement']
        )
        df = df[df['label'].isin(LABEL_MAP)]
        self.texts  = df['statement'].tolist()
        self.labels = [LABEL_MAP[l] for l in df['label']]
        self.tokenizer = tokenizer
        self.max_len   = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        enc = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding='max_length',
            max_length=self.max_len,
            return_tensors='pt'
        )
        return {
            'input_ids':      enc['input_ids'].squeeze(),
            'attention_mask': enc['attention_mask'].squeeze(),
            'labels':         torch.tensor(self.labels[idx])
        }


# ── Training ───────────────────────────────────────────────────
def train():
    device    = 'cuda' if torch.cuda.is_available() else 'cpu'
    tokenizer = DistilBertTokenizerFast.from_pretrained(
        "distilbert-base-uncased"
    )

    BASE     = os.path.join(os.path.dirname(__file__), '..', 'data', 'raw', 'liar_dataset')
    train_ds = LIARDataset(os.path.join(BASE, 'train.tsv'), tokenizer)
    val_ds   = LIARDataset(os.path.join(BASE, 'valid.tsv'), tokenizer)

    # ── Use only 3000 samples for fast CPU training ──
    train_ds = torch.utils.data.Subset(train_ds, range(3000))
    val_ds   = torch.utils.data.Subset(val_ds,   range(500))

    print(f"Train samples : {len(train_ds)}")
    print(f"Val   samples : {len(val_ds)}")
    print(f"Device        : {device}")

    train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=64)

    model     = FakeNewsClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    # REPLACE WITH THIS — adds class weights to fix bias:
    class_weights = torch.tensor([1.0, 1.5]).to(device)
    criterion      = torch.nn.CrossEntropyLoss(weight=class_weights)

    best_val_acc = 0.0


    for epoch in range(2):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/4 [Train]")
        for batch in loop:
            ids   = batch['input_ids'].to(device)
            mask  = batch['attention_mask'].to(device)
            labs  = batch['labels'].to(device)

            optimizer.zero_grad()
            logits = model(ids, mask)
            loss   = criterion(logits, labs)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds       = logits.argmax(dim=1)
            correct    += (preds == labs).sum().item()
            total      += labs.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = correct / total * 100

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for batch in val_loader:
                ids  = batch['input_ids'].to(device)
                mask = batch['attention_mask'].to(device)
                labs = batch['labels'].to(device)
                out  = model(ids, mask)
                val_correct += (out.argmax(1) == labs).sum().item()
                val_total   += labs.size(0)

        val_acc = val_correct / val_total * 100
        print(f"\nEpoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Train Acc: {train_acc:.1f}% | Val Acc: {val_acc:.1f}%")

        # ── Save best model ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = os.path.join(
                os.path.dirname(__file__), '..', 'models', 'nlp', 'distilbert_finetuned.pth'
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Best model saved — Val Acc: {val_acc:.1f}%")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.1f}%")


if __name__ == "__main__":
    train()
import os
import sys
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))


# ── Dataset ────────────────────────────────────────────────────
class FaceDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_samples=5000):
        """
        root_dir should contain two subfolders: real/ and fake/
        """
        self.samples   = []
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )
        ])

        # Load real images → label 0
        real_dir = os.path.join(root_dir, 'real')
        if os.path.exists(real_dir):
            real_imgs = [
                (os.path.join(real_dir, f), 0)
                for f in os.listdir(real_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            self.samples.extend(real_imgs[:max_samples // 2])

        # Load fake images → label 1
        fake_dir = os.path.join(root_dir, 'fake')
        if os.path.exists(fake_dir):
            fake_imgs = [
                (os.path.join(fake_dir, f), 1)
                for f in os.listdir(fake_dir)
                if f.lower().endswith(('.jpg', '.jpeg', '.png'))
            ]
            self.samples.extend(fake_imgs[:max_samples // 2])

        print(f"  Loaded {len(self.samples)} images from {root_dir}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), torch.tensor(label)
        except Exception:
            # Return black image if file is corrupt
            return torch.zeros(3, 224, 224), torch.tensor(label)


# ── Training ───────────────────────────────────────────────────
def train():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device : {device}")

    # ── Dataset paths ──
    BASE       = os.path.join(
        os.path.dirname(__file__), '..',
        'data', 'raw', 'faceforensics', 'real_vs_fake', 'real-vs-fake'
    )
    train_path = os.path.join(BASE, 'train')
    val_path   = os.path.join(BASE, 'valid')

    print("\nLoading datasets...")
    train_ds = FaceDataset(train_path, max_samples=4000)
    val_ds   = FaceDataset(val_path,   max_samples=1000)

    if len(train_ds) == 0:
        print("ERROR: No training images found!")
        print(f"Expected images at: {train_path}")
        return

    train_loader = DataLoader(
        train_ds, batch_size=32,
        shuffle=True, num_workers=0
    )
    val_loader = DataLoader(
        val_ds, batch_size=64,
        shuffle=False, num_workers=0
    )

    # ── Model ──
    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )
    model.classifier[1] = nn.Linear(1280, 2)
    model = model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=1, gamma=0.5
    )

    best_val_acc = 0.0

    for epoch in range(3):
        # ── Train ──
        model.train()
        total_loss, correct, total = 0, 0, 0

        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/3 [Train]")
        for imgs, labels in loop:
            imgs   = imgs.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(imgs)
            loss    = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            preds       = outputs.argmax(dim=1)
            correct    += (preds == labels).sum().item()
            total      += labels.size(0)
            loop.set_postfix(loss=f"{loss.item():.4f}")

        train_acc = correct / total * 100

        # ── Validate ──
        model.eval()
        val_correct, val_total = 0, 0
        with torch.no_grad():
            for imgs, labels in val_loader:
                imgs   = imgs.to(device)
                labels = labels.to(device)
                out    = model(imgs)
                val_correct += (out.argmax(1) == labels).sum().item()
                val_total   += labels.size(0)

        val_acc = val_correct / val_total * 100
        scheduler.step()

        print(f"\nEpoch {epoch+1} | Loss: {total_loss/len(train_loader):.4f} | Train: {train_acc:.1f}% | Val: {val_acc:.1f}%")

        # ── Save best ──
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path    = os.path.join(
                os.path.dirname(__file__), '..',
                'models', 'deepfake', 'efficientnet_b0.pth'
            )
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            torch.save(model.state_dict(), save_path)
            print(f"  ✅ Best model saved — Val Acc: {val_acc:.1f}%")

    print(f"\nTraining complete. Best Val Accuracy: {best_val_acc:.1f}%")


if __name__ == "__main__":
    train()
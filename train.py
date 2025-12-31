
import os
import asyncio
import threading
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import models, transforms
from PIL import Image
from tqdm import tqdm


try:
    import ray
    RAY = True
except:
    RAY = False


class DeepfakeDataset(Dataset):
    def __init__(self, root_dir, augment=False):
        self.samples = []
        root = Path(root_dir)

        for label, folder in enumerate(["real", "fake"]):
            img_dir = root / folder
            if not img_dir.exists():
                raise Exception(f"Folder missing: {img_dir}")
            for img_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
                self.samples.append((str(img_path), label))

        if augment:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(8),
                transforms.ColorJitter(0.2, 0.2, 0.2),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])
            ])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        img_path, label = self.samples[idx]
        img = Image.open(img_path).convert("RGB")
        return self.transform(img), label


def create_model(num_classes=2):
    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    num_ftrs = model.fc.in_features
    model.fc = nn.Linear(num_ftrs, num_classes)
    return model


def threaded_log(msg):
    threading.Thread(target=lambda: print(f"[THREAD] {msg}"), daemon=True).start()


async def async_save_model(model, path):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, torch.save, model.state_dict(), path)
    print(f"[ASYNC] Checkpoint saved at {path}")


if RAY:
    @ray.remote
    def ray_validate(model_state, batch_images, batch_labels):
        model = create_model()
        model.load_state_dict(model_state)
        model.eval()
        imgs = torch.stack(batch_images)
        labels = torch.tensor(batch_labels)
        with torch.no_grad():
            out = model(imgs)
            preds = out.argmax(1)
            correct = (preds == labels).sum().item()
        return correct, len(labels)

def train(model, loader, criterion, optimizer, device):
    model.train()
    total, correct, loss_sum = 0, 0, 0
    for inputs, labels in tqdm(loader, desc="Training"):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        loss_sum += loss.item() * inputs.size(0)
        preds = outputs.argmax(1)
        correct += (preds == labels).sum().item()
        total += inputs.size(0)
    return loss_sum / total, 100 * correct / total


def validate(model, loader, device, use_ray=False):
    model.eval()
    if use_ray and RAY:
        state = model.state_dict()
        futures = []
        for x, y in loader:
            imgs = [img.cpu() for img in x]
            labs = [int(l) for l in y]
            futures.append(ray_validate.remote(state, imgs, labs))
        results = ray.get(futures)
        correct = sum(r[0] for r in results)
        total = sum(r[1] for r in results)
        return 100 * correct / total

    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Validation"):
            inputs, labels = inputs.to(device), labels.to(device)
            out = model(inputs)
            preds = out.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    return 100 * correct / total


def main():
    dataset_root = "training/datasets/deepfake_mini"  # à changer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    full_set = DeepfakeDataset(dataset_root, augment=True)

    subset_size = min(20000, len(full_set))
    full_set, _ = random_split(full_set, [subset_size, len(full_set)-subset_size])

    val_size = int(0.2 * len(full_set))
    train_size = len(full_set) - val_size
    train_set, val_set = random_split(full_set, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=32, shuffle=False, num_workers=2)

    model = create_model().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)

    use_ray = True if RAY else False
    if use_ray:
        ray.init(num_cpus=4)
        print("[RAY] Distributed validation enabled.")

    best_acc = 0
    os.makedirs("training/checkpoints", exist_ok=True)

    # Training loop
    for epoch in range(1, 11):
        print(f"\n===== Epoch {epoch} =====")
        threaded_log(f"Epoch {epoch} started")

        train_loss, train_acc = train(model, train_loader, criterion, optimizer, device)
        val_acc = validate(model, val_loader, device, use_ray=use_ray)

        print(f"[EPOCH {epoch}] Loss={train_loss:.4f} | TrainAcc={train_acc:.2f}% | ValAcc={val_acc:.2f}%")

        if val_acc > best_acc:
            best_acc = val_acc
            ckpt = f"training/checkpoints/best_epoch{epoch}.pth"
            torch.save(model.state_dict(), ckpt)
            asyncio.run(async_save_model(model, ckpt + ".async"))

    print(f"\n Training completed. Best Accuracy = {best_acc:.2f}%")

if __name__ == "__main__":
    main()

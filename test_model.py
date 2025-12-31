import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, models
from PIL import Image
from pathlib import Path
from tqdm import tqdm



class TestDataset(Dataset):
    def __init__(self, root_dir):
        self.samples = []
        root = Path(root_dir)
        for label, folder in enumerate(["real", "fake"]):
            img_dir = root / folder
            for img_path in list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png")):
                self.samples.append((str(img_path), label))

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
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
    model.fc = torch.nn.Linear(num_ftrs, num_classes)
    return model



def evaluate(model, loader, device):
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for inputs, labels in tqdm(loader, desc="Testing"):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            preds = outputs.argmax(1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)
    acc = 100 * correct / total
    print(f"\n[Test Accuracy] = {acc:.2f}%")



def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[INFO] Using device: {device}")

    test_dataset = TestDataset("training/datasets/dataset_test")
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)

    model = create_model().to(device)
    checkpoint_path = "training/checkpoints/best_epoch1.pth"
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    print(f"[INFO] Model loaded from {checkpoint_path}")

    evaluate(model, test_loader, device)


if __name__ == "__main__":
    main()

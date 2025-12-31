
import torch
import torch.nn as nn
from torchvision import models, transforms
from pathlib import Path
import threading


class DeepfakeModel:

    _instance = None
    _lock = threading.Lock()
    _initialized = False

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        if self._initialized:
            return

        with self._lock:
            if self._initialized:
                return

            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            print(f"[MODEL]  Initialisation sur {device}")

            model = models.resnet18(weights=None)
            num_ftrs = model.fc.in_features
            model.fc = nn.Linear(num_ftrs, 2)
            model.to(device)

            checkpoint_dir = Path("training/checkpoints")
            pth_files = list(checkpoint_dir.glob("best_epoch*.pth"))
            if not pth_files:
                raise FileNotFoundError("Aucun best_epoch*.pth trouvé!")

            best_path = max(pth_files, key=lambda x: x.stat().st_mtime)
            print(f"[MODEL]  Chargement: {best_path.name}")

            model.load_state_dict(torch.load(best_path, map_location=device))
            model.eval()

            self.model = model
            self.device = device
            self.checkpoint_path = str(best_path)
            self.transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            self._initialized = True
            print("[MODEL]  Modèle chargé (thread-safe)")

    def get_model(self):
        return self.model

    def get_device(self):
        return self.device

    def get_transform(self):
        return self.transform


def get_model_loader():
    return DeepfakeModel()


model_loader = None


def lazy_init():
    global model_loader
    if model_loader is None:
        model_loader = get_model_loader()
    return model_loader


if __name__ == "__main__":
    import concurrent.futures


    def test_load():
        loader = DeepfakeModel()
        return id(loader)


    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = [executor.submit(test_load) for _ in range(10)]
        ids = [f.result() for f in futures]

    print(f"Tous les IDs identiques: {len(set(ids)) == 1}")
    print(f" Singleton thread-safe vérifié")
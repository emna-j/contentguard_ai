import torch
import torch.nn.functional as F
import numpy as np
import cv2
from PIL import Image
from pathlib import Path
import logging
from torchvision import transforms

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("Explainability")


class GradCAM:

    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.gradients = None
        self.activations = None

        self.target_layer.register_forward_hook(self._save_activation)
        self.target_layer.register_backward_hook(self._save_gradient)

    def _save_activation(self, module, input, output):
        self.activations = output.detach()

    def _save_gradient(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx=None):

        self.model.eval()
        output = self.model(input_tensor)

        if class_idx is None:
            class_idx = output.argmax(dim=1).item()

        self.model.zero_grad()
        class_score = output[0, class_idx]
        class_score.backward()

        gradients = self.gradients[0]
        activations = self.activations[0]

        weights = torch.mean(gradients, dim=(1, 2))

        cam = torch.zeros(activations.shape[1:], dtype=torch.float32)
        for i, w in enumerate(weights):
            cam += w * activations[i]

        cam = F.relu(cam)
        cam = cam - cam.min()
        cam = cam / (cam.max() + 1e-8)

        return cam.cpu().numpy()


class DeepfakeExplainer:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.heatmap_dir = Path("heatmaps")
        self.heatmap_dir.mkdir(exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def generate_heatmap(self, image_path: str, save: bool = True) -> str:

        try:
            original_img = Image.open(image_path).convert("RGB")
            original_array = np.array(original_img)

            img_tensor = self.transform(original_img).unsqueeze(0).to(self.device)

            target_layer = self._get_target_layer()
            gradcam = GradCAM(self.model, target_layer)

            cam = gradcam.generate(img_tensor)

            cam_resized = cv2.resize(cam, (original_array.shape[1], original_array.shape[0]))

            heatmap = cv2.applyColorMap(np.uint8(255 * cam_resized), cv2.COLORMAP_JET)
            heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)

            superimposed = heatmap * 0.4 + original_array * 0.6
            superimposed = np.uint8(superimposed)

            if save:
                filename = Path(image_path).stem + "_heatmap.jpg"
                save_path = self.heatmap_dir / filename
                Image.fromarray(superimposed).save(save_path, quality=95)
                logger.info(f"Heatmap saved: {filename}")
                return str(save_path)

            return superimposed

        except Exception as e:
            logger.error(f"Heatmap generation error: {e}")
            return None

    def _get_target_layer(self):

        if hasattr(self.model, 'layer4'):
            return self.model.layer4[-1]
        elif hasattr(self.model, 'features'):
            return self.model.features[-1]
        else:
            layers = list(self.model.children())
            return layers[-2] if len(layers) > 1 else layers[-1]

    def generate_batch_heatmaps(self, image_paths: list) -> list:

        logger.info(f"Generating heatmaps for {len(image_paths)} images")
        heatmap_paths = []

        for path in image_paths:
            heatmap_path = self.generate_heatmap(path)
            heatmap_paths.append(heatmap_path)

        return heatmap_paths


class SaliencyMapGenerator:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.saliency_dir = Path("saliency_maps")
        self.saliency_dir.mkdir(exist_ok=True)

        self.transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def generate_saliency_map(self, image_path: str) -> str:

        try:
            original_img = Image.open(image_path).convert("RGB")

            img_tensor = self.transform(original_img).unsqueeze(0).to(self.device)
            img_tensor.requires_grad = True

            self.model.eval()
            output = self.model(img_tensor)

            target_class = output.argmax(dim=1)

            self.model.zero_grad()
            output[0, target_class].backward()

            saliency = img_tensor.grad.data.abs()
            saliency = saliency.max(dim=1)[0]
            saliency = saliency.squeeze().cpu().numpy()

            saliency = (saliency - saliency.min()) / (saliency.max() - saliency.min() + 1e-8)

            filename = Path(image_path).stem + "_saliency.jpg"
            save_path = self.saliency_dir / filename

            saliency_img = np.uint8(255 * saliency)
            Image.fromarray(saliency_img).save(save_path)

            logger.info(f"Saliency map saved: {filename}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Saliency map generation error: {e}")
            return None


class AttentionVisualizer:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.attention_dir = Path("attention_maps")
        self.attention_dir.mkdir(exist_ok=True)

    def visualize_attention(self, image_path: str, layer_name: str = None) -> str:

        try:
            activations = {}

            def hook_fn(name):
                def hook(module, input, output):
                    activations[name] = output.detach()

                return hook

            if layer_name:
                target_layer = dict(self.model.named_modules())[layer_name]
                target_layer.register_forward_hook(hook_fn(layer_name))
            else:
                for name, module in self.model.named_modules():
                    if isinstance(module, torch.nn.Conv2d):
                        module.register_forward_hook(hook_fn(name))

            transform = transforms.Compose([
                transforms.Resize((128, 128)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])

            img = Image.open(image_path).convert("RGB")
            img_tensor = transform(img).unsqueeze(0).to(self.device)

            self.model.eval()
            with torch.no_grad():
                _ = self.model(img_tensor)

            if layer_name and layer_name in activations:
                attention = activations[layer_name]
            else:
                last_key = list(activations.keys())[-1]
                attention = activations[last_key]

            attention_map = attention[0].mean(dim=0).cpu().numpy()
            attention_map = (attention_map - attention_map.min()) / (attention_map.max() - attention_map.min() + 1e-8)

            filename = Path(image_path).stem + "_attention.jpg"
            save_path = self.attention_dir / filename

            attention_img = np.uint8(255 * attention_map)
            attention_resized = cv2.resize(attention_img, (img.width, img.height))
            Image.fromarray(attention_resized).save(save_path)

            logger.info(f"Attention map saved: {filename}")
            return str(save_path)

        except Exception as e:
            logger.error(f"Attention visualization error: {e}")
            return None


class ExplainabilityEngine:

    def __init__(self, model, device):
        self.model = model
        self.device = device
        self.gradcam = DeepfakeExplainer(model, device)
        self.saliency = SaliencyMapGenerator(model, device)
        self.attention = AttentionVisualizer(model, device)

    def explain_all(self, image_path: str) -> dict:

        logger.info(f"Generating all explanations for: {Path(image_path).name}")

        results = {
            "heatmap": None,
            "saliency": None,
            "attention": None
        }

        try:
            results["heatmap"] = self.gradcam.generate_heatmap(image_path)
        except Exception as e:
            logger.error(f"Heatmap error: {e}")

        try:
            results["saliency"] = self.saliency.generate_saliency_map(image_path)
        except Exception as e:
            logger.error(f"Saliency error: {e}")

        try:
            results["attention"] = self.attention.visualize_attention(image_path)
        except Exception as e:
            logger.error(f"Attention error: {e}")

        return results


if __name__ == "__main__":
    print("Explainability module loaded")
    print("Available visualizations:")
    print("  - GradCAM (heatmaps)")
    print("  - Saliency maps")
    print("  - Attention visualization")
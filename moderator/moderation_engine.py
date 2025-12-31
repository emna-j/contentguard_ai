import ray
import torch
import logging
import asyncio
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModerationEngine")

try:
    ray.init(
        ignore_reinit_error=True,
        log_to_driver=False,
        num_cpus=None,
        num_gpus=None
    )
    logger.info("Ray initialized for distributed computing")
except Exception as e:
    logger.warning(f"Ray not available: {e}")


@ray.remote(num_cpus=1, num_gpus=0.25 if torch.cuda.is_available() else 0)
def _ray_predict(image_path: str, checkpoint_path: str):
    import torch
    from PIL import Image
    from torchvision import transforms, models
    import torch.nn as nn
    import sys
    import os
    from pathlib import Path
    import time

    sys.path.insert(0, str(Path(__file__).parent.parent))

    worker_id = ray.get_runtime_context().get_worker_id()[:8]
    start_time = time.time()

    try:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = models.resnet18(weights=None)
        num_ftrs = model.fc.in_features
        model.fc = nn.Linear(num_ftrs, 2)
        model.load_state_dict(torch.load(checkpoint_path, map_location=device))
        model.to(device)
        model.eval()

        transform = transforms.Compose([
            transforms.Resize((128, 128)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

        img = Image.open(image_path).convert("RGB")
        img_t = transform(img).unsqueeze(0).to(device)

        with torch.no_grad():
            outputs = model(img_t)
            probabilities = torch.softmax(outputs, dim=1)[0]
            prob_real = probabilities[0].item()
            prob_fake = probabilities[1].item()

        is_fake = prob_fake > 0.5
        confidence = max(prob_real, prob_fake)
        filename = Path(image_path).name

        heatmap_filename = None
        try:
            import sys
            from pathlib import Path as PathLib
            sys.path.insert(0, str(PathLib(__file__).parent))

            from explainability import DeepfakeExplainer
            explainer = DeepfakeExplainer(model, device)
            heatmap_path = explainer.generate_heatmap(image_path)
            if heatmap_path and os.path.exists(heatmap_path):
                heatmap_filename = os.path.basename(heatmap_path)
        except Exception as e:
            pass

        metrics = {}
        try:
            import sys
            from pathlib import Path as PathLib
            sys.path.insert(0, str(PathLib(__file__).parent))

            from metrics import DetailedMetrics
            metrics = DetailedMetrics.calculate_detailed_analysis(
                prob_real, prob_fake, image_path
            )
        except Exception as e:
            pass

        total_time = time.time() - start_time

        return {
            "success": True,
            "is_fake": is_fake,
            "confidence": round(confidence * 100, 2),
            "prob_real": round(prob_real * 100, 2),
            "prob_fake": round(prob_fake * 100, 2),
            "filename": filename,
            "heatmap": heatmap_filename,
            "metrics": metrics,
            "processing_time_ms": round(total_time * 1000, 2),
            "worker_id": worker_id
        }

    except Exception as e:
        logger.error(f"Worker error: {e}")
        import traceback
        traceback.print_exc()
        return {"success": False, "error": str(e)}


class ModerationEngine:

    def __init__(self):
        checkpoint_dir = Path("training/checkpoints")
        pth_files = list(checkpoint_dir.glob("best_epoch*.pth"))
        if not pth_files:
            raise FileNotFoundError("No best_epoch*.pth found!")

        self.checkpoint_path = str(max(pth_files, key=lambda x: x.stat().st_mtime))
        self._thread_pool = ThreadPoolExecutor(max_workers=4, thread_name_prefix="Engine")

        logger.info(f"ModerationEngine v2.0 initialized")
        logger.info(f"Model: {Path(self.checkpoint_path).name}")
        logger.info(f"Ray workers: {ray.cluster_resources().get('CPU', 0)} CPUs")
        if ray.cluster_resources().get('GPU', 0) > 0:
            logger.info(f"GPU available: {ray.cluster_resources()['GPU']}")

    def submit_image(self, image_path: str, priority: int = 1):
        logger.info(f"Submitting: {Path(image_path).name}")

        future = _ray_predict.remote(image_path, self.checkpoint_path)
        result = ray.get(future)

        return result

    async def submit_image_async(self, image_path: str, priority: int = 1):
        logger.info(f"Submitting async: {Path(image_path).name}")

        future = _ray_predict.remote(image_path, self.checkpoint_path)

        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            self._thread_pool,
            ray.get,
            future
        )

        return result

    def submit_batch(self, image_paths: list):
        logger.info(f"Batch of {len(image_paths)} images")

        start_time = time.time()

        futures = [
            _ray_predict.remote(path, self.checkpoint_path)
            for path in image_paths
        ]

        results = ray.get(futures)

        elapsed = time.time() - start_time
        successful = sum(1 for r in results if r.get("success"))

        logger.info(f"Batch completed: {successful}/{len(image_paths)} in {elapsed:.2f}s")
        logger.info(f"Throughput: {len(image_paths) / elapsed:.2f} images/s")

        return results

    async def submit_batch_async(self, image_paths: list):
        logger.info(f"Batch async of {len(image_paths)} images")

        tasks = [
            self.submit_image_async(path)
            for path in image_paths
        ]

        results = await asyncio.gather(*tasks)

        return results

    def shutdown(self):
        logger.info("Shutting down engine")
        self._thread_pool.shutdown(wait=True)
        logger.info("Engine stopped")


engine = ModerationEngine()

if __name__ == "__main__":
    print("=" * 80)
    print("ModerationEngine v2.0 - Concurrent Programming")
    print("=" * 80)
    print(f"Ray: Distributed computing")
    print(f"AsyncIO: Asynchronous operations")
    print(f"Threading: Parallel I/O")
    print(f"Multiprocessing: CPU-intensive computations")
    print("=" * 80)
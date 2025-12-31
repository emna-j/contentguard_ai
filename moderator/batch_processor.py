import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from dataclasses import dataclass
from datetime import datetime
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("BatchProcessor")


@dataclass
class BatchResult:
    filename: str
    success: bool
    is_fake: bool = False
    confidence: float = 0.0
    prob_real: float = 0.0
    prob_fake: float = 0.0
    processing_time_ms: float = 0.0
    error: str = None
    metrics: Dict[str, Any] = None
    heatmap: str = None
    worker_id: str = None


class BatchProcessor:

    def __init__(self, max_workers: int = 4, use_multiprocessing: bool = True):
        self.max_workers = max_workers
        self.use_multiprocessing = use_multiprocessing

        if use_multiprocessing:
            self.executor = ProcessPoolExecutor(max_workers=max_workers)
            logger.info(f"Initialized with ProcessPoolExecutor ({max_workers} processes)")
        else:
            self.executor = ThreadPoolExecutor(max_workers=max_workers)
            logger.info(f"Initialized with ThreadPoolExecutor ({max_workers} threads)")

    async def process_batch_async(self, image_paths: List[str]) -> List[BatchResult]:
        logger.info(f"Processing batch of {len(image_paths)} images asynchronously")
        start_time = time.time()

        loop = asyncio.get_event_loop()
        tasks = []

        for path in image_paths:
            task = loop.run_in_executor(
                self.executor,
                self._process_single_image,
                path
            )
            tasks.append(task)

        results = await asyncio.gather(*tasks, return_exceptions=True)

        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if isinstance(r, BatchResult) and r.success)

        logger.info(f"Batch completed: {success_count}/{len(image_paths)} in {elapsed:.2f}s")
        logger.info(f"Throughput: {len(image_paths) / elapsed:.2f} images/sec")

        return results

    def process_batch_sync(self, image_paths: List[str]) -> List[BatchResult]:
        logger.info(f"Processing batch of {len(image_paths)} images synchronously")
        start_time = time.time()

        results = list(self.executor.map(self._process_single_image, image_paths))

        elapsed = time.time() - start_time
        success_count = sum(1 for r in results if r.success)

        logger.info(f"Batch completed: {success_count}/{len(image_paths)} in {elapsed:.2f}s")
        logger.info(f"Throughput: {len(image_paths) / elapsed:.2f} images/sec")

        return results

    def _process_single_image(self, image_path: str) -> BatchResult:
        try:
            import sys
            sys.path.insert(0, str(Path(__file__).parent.parent))

            from moderator.model_loader import get_model_loader
            from moderator.metrics import DetailedMetrics
            from moderator.explainability import DeepfakeExplainer
            from PIL import Image
            import torch

            loader = get_model_loader()
            model = loader.get_model()
            device = loader.get_device()
            transform = loader.get_transform()

            img = Image.open(image_path).convert("RGB")
            img_t = transform(img).unsqueeze(0).to(device)

            with torch.no_grad():
                outputs = model(img_t)
                probs = torch.softmax(outputs, dim=1)[0]
                prob_real = probs[0].item()
                prob_fake = probs[1].item()

            is_fake = prob_fake > 0.5
            confidence = max(prob_real, prob_fake)

            metrics = DetailedMetrics.calculate_detailed_analysis(
                prob_real, prob_fake, image_path
            )

            heatmap_filename = None
            try:
                explainer = DeepfakeExplainer(model, device)
                heatmap_path = explainer.generate_heatmap(image_path)
                if heatmap_path:
                    heatmap_filename = Path(heatmap_path).name
            except:
                pass

            return BatchResult(
                filename=Path(image_path).name,
                success=True,
                is_fake=is_fake,
                confidence=round(confidence * 100, 2),
                prob_real=round(prob_real * 100, 2),
                prob_fake=round(prob_fake * 100, 2),
                processing_time_ms=metrics.get("processing_time_ms", 0),
                metrics=metrics,
                heatmap=heatmap_filename
            )

        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")
            return BatchResult(
                filename=Path(image_path).name,
                success=False,
                error=str(e)
            )

    def shutdown(self):
        logger.info("Shutting down batch processor")
        self.executor.shutdown(wait=True)


class PriorityBatchProcessor(BatchProcessor):

    def __init__(self, max_workers: int = 4):
        super().__init__(max_workers, use_multiprocessing=False)
        self.priority_queue = asyncio.PriorityQueue()
        self.results_map = {}

    async def add_batch_with_priority(self, image_paths: List[str], priority: int = 5):
        batch_id = datetime.now().isoformat()

        for path in image_paths:
            await self.priority_queue.put((priority, batch_id, path))

        self.results_map[batch_id] = {
            "total": len(image_paths),
            "completed": 0,
            "results": []
        }

        return batch_id

    async def process_priority_queue(self):
        logger.info("Starting priority queue processor")

        while True:
            try:
                priority, batch_id, image_path = await asyncio.wait_for(
                    self.priority_queue.get(),
                    timeout=1.0
                )

                result = await asyncio.get_event_loop().run_in_executor(
                    self.executor,
                    self._process_single_image,
                    image_path
                )

                self.results_map[batch_id]["results"].append(result)
                self.results_map[batch_id]["completed"] += 1

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Queue processing error: {e}")

    def get_batch_status(self, batch_id: str) -> Dict[str, Any]:
        if batch_id not in self.results_map:
            return None

        batch_info = self.results_map[batch_id]
        progress = (batch_info["completed"] / batch_info["total"]) * 100

        return {
            "batch_id": batch_id,
            "total": batch_info["total"],
            "completed": batch_info["completed"],
            "progress": round(progress, 2),
            "results": batch_info["results"]
        }


def create_processor(processor_type: str = "standard", **kwargs) -> BatchProcessor:
    if processor_type == "priority":
        return PriorityBatchProcessor(**kwargs)
    else:
        return BatchProcessor(**kwargs)

import asyncio
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("ModerationCoordinator")


@dataclass
class ModerationResult:
    filename: str
    filepath: str
    is_fake: bool
    confidence: float
    prob_real: float
    prob_fake: float
    status: str
    timestamp: str
    processing_time_ms: float
    heatmap_path: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None
    risk_assessment: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    worker_id: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ModerationCoordinator:

    def __init__(self, max_concurrent: int = 8):
        self.max_concurrent = max_concurrent
        self.thread_pool = ThreadPoolExecutor(
            max_workers=max_concurrent,
            thread_name_prefix="Coordinator"
        )
        try:
            from moderator.moderation_engine import engine as ray_engine
            self.ray_engine = ray_engine
            self.use_ray = True
            logger.info(" Ray engine loaded for distributed processing")
        except ImportError as e:
            logger.warning(f" Ray engine not available: {e}")
            self.use_ray = False

        # Import multiprocessing metrics
        try:
            from moderator.metrics import DetailedMetrics
            self.metrics_calculator = DetailedMetrics
            logger.info(" Metrics calculator loaded (multiprocessing)")
        except ImportError:
            self.metrics_calculator = None
            logger.warning(" Metrics calculator not available")

        logger.info(f" ModerationCoordinator initialized (max_concurrent={max_concurrent})")

    async def analyze_single_image(
            self,
            image_path: str,
            priority: int = 5
    ) -> ModerationResult:

        start_time = time.time()
        filepath = Path(image_path)

        original_filename = self._extract_original_filename(filepath.name)

        logger.info(f" Analyzing: {original_filename}")

        try:
            if self.use_ray:
                loop = asyncio.get_event_loop()
                result = await loop.run_in_executor(
                    self.thread_pool,
                    self.ray_engine.submit_image,
                    str(image_path),
                    priority
                )

                if result.get("success"):
                    prob_real = result["prob_real"]
                    prob_fake = result["prob_fake"]
                    risk = self._assess_risk(prob_real, prob_fake, result.get("metrics", {}))

                    processing_time = (time.time() - start_time) * 1000

                    is_fake = result["is_fake"]
                    status = "rejected" if is_fake else "approved"

                    return ModerationResult(
                        filename=original_filename,
                        filepath=str(filepath),
                        is_fake=is_fake,
                        confidence=result["confidence"],
                        prob_real=prob_real,
                        prob_fake=prob_fake,
                        status=status,
                        timestamp=datetime.utcnow().isoformat(),
                        processing_time_ms=processing_time,
                        heatmap_path=result.get("heatmap"),
                        metrics=result.get("metrics", {}),
                        risk_assessment=risk,
                        worker_id=result.get("worker_id")
                    )
                else:
                    raise Exception(result.get("error", "Unknown error"))
            else:
                raise Exception("Ray engine not available")

        except Exception as e:
            logger.error(f" Error analyzing {original_filename}: {e}")
            return ModerationResult(
                filename=original_filename,
                filepath=str(filepath),
                is_fake=False,
                confidence=0.0,
                prob_real=0.0,
                prob_fake=0.0,
                status="error",
                timestamp=datetime.utcnow().isoformat(),
                processing_time_ms=(time.time() - start_time) * 1000,
                error=str(e)
            )

    def _extract_original_filename(self, filename: str) -> str:
        parts = filename.split('_', 1)
        if len(parts) > 1:
            return parts[1]
        return filename

    async def analyze_batch(
            self,
            image_paths: List[str],
            progress_callback: Optional[callable] = None
    ) -> List[ModerationResult]:

        logger.info(f" Starting batch analysis: {len(image_paths)} images")
        start_time = time.time()

        semaphore = asyncio.Semaphore(self.max_concurrent)

        async def analyze_with_semaphore(path: str, index: int):
            async with semaphore:
                result = await self.analyze_single_image(path)

                if progress_callback:
                    await progress_callback(index + 1, len(image_paths), result)

                return result

        tasks = [
            analyze_with_semaphore(path, i)
            for i, path in enumerate(image_paths)
        ]

        results = await asyncio.gather(*tasks, return_exceptions=True)

        valid_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f" Task {i} failed: {result}")
                original_filename = self._extract_original_filename(Path(image_paths[i]).name)
                valid_results.append(ModerationResult(
                    filename=original_filename,
                    filepath=image_paths[i],
                    is_fake=False,
                    confidence=0.0,
                    prob_real=0.0,
                    prob_fake=0.0,
                    status="error",
                    timestamp=datetime.utcnow().isoformat(),
                    processing_time_ms=0,
                    error=str(result)
                ))
            else:
                valid_results.append(result)

        elapsed = time.time() - start_time
        success_count = sum(1 for r in valid_results if r.status != "error")

        logger.info(f" Batch completed: {success_count}/{len(image_paths)} in {elapsed:.2f}s")
        logger.info(f"⚡ Throughput: {len(image_paths) / elapsed:.2f} images/sec")

        return valid_results

    def _assess_risk(self, prob_real, prob_fake, metrics):

        confidence_level = metrics.get("confidence_level", "UNKNOWN")

        if prob_fake >= 85:
            severity = "CRITICAL"
            description = "Very high probability of manipulation detected"
            recommendation = "REJECT - Strong evidence of manipulation"
        elif prob_fake >= 70:
            severity = "HIGH"
            description = "High probability of manipulation detected"
            recommendation = "REJECT - Manual verification recommended"
        elif prob_fake >= 50:
            severity = "MEDIUM"
            description = "Moderate probability of manipulation detected"
            recommendation = "REVIEW - Additional verification recommended"
        elif prob_real >= 85:
            severity = "MINIMAL"
            description = "Authentic image with high confidence"
            recommendation = "ACCEPT - Minimal risk detected"
        elif prob_real >= 70:
            severity = "LOW"
            description = "Probably authentic image"
            recommendation = "ACCEPT - Low risk detected"
        else:
            severity = "MEDIUM"
            description = "Uncertainty in analysis"
            recommendation = "REVIEW - Additional analysis recommended"

        return {
            "severity": severity,
            "description": description,
            "confidence_level": confidence_level,
            "recommendation": recommendation
        }

    async def get_batch_statistics(
            self,
            results: List[ModerationResult]
    ) -> Dict[str, Any]:
        """Calculate batch statistics"""
        total = len(results)
        approved = sum(1 for r in results if r.status == "approved")
        rejected = sum(1 for r in results if r.status == "rejected")
        review = sum(1 for r in results if r.status == "review")
        errors = sum(1 for r in results if r.status == "error")

        valid_results = [r for r in results if r.status != "error"]

        avg_confidence = sum(r.confidence for r in valid_results) / max(len(valid_results), 1)
        avg_processing_time = sum(r.processing_time_ms for r in results) / max(total, 1)

        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "review": review,
            "errors": errors,
            "avg_confidence": round(avg_confidence, 2),
            "avg_processing_time_ms": round(avg_processing_time, 2),
            "success_rate": round((total - errors) / total * 100, 2) if total > 0 else 0
        }

    def shutdown(self):
        """Clean shutdown of coordinator"""
        logger.info(" Shutting down coordinator...")
        self.thread_pool.shutdown(wait=True)
        if self.use_ray and hasattr(self.ray_engine, 'shutdown'):
            self.ray_engine.shutdown()
        logger.info(" Coordinator shutdown complete")


_coordinator = None


def get_coordinator(max_concurrent: int = 8) -> ModerationCoordinator:
    """Factory to get coordinator instance"""
    global _coordinator
    if _coordinator is None:
        _coordinator = ModerationCoordinator(max_concurrent=max_concurrent)
    return _coordinator


if __name__ == "__main__":
    print("=" * 80)
    print("Moderation Coordinator - Test Module")
    print("=" * 80)
    print("Features:")
    print("   AsyncIO for async orchestration")
    print("   ThreadPoolExecutor for concurrent I/O")
    print("   Ray for distributed computing")
    print("   Multiprocessing for metrics (via DetailedMetrics)")
    print("=" * 80)
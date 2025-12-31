import ray
import logging
from .api_client import HuggingFaceClient
from .models import GenerationTask

try:
    ray.init(ignore_reinit_error=True, log_to_driver=False, num_cpus=None)
    logging.info("Ray initialized successfully")
except Exception as e:
    logging.warning(f"Ray initialization warning: {e}")


@ray.remote(num_cpus=0.5)
class DistributedWorker:

    def __init__(self):
        self.client = HuggingFaceClient()
        self.logger = logging.getLogger(f"Worker-{id(self)}")

    def generate(self, task: GenerationTask) -> bytes:

        try:
            self.logger.info(f"Worker generating image for task {task.task_id}")
            self.logger.info(f"Parameters: {task.width}x{task.height}, steps={task.num_steps}, seed={task.seed}")

            image_buffer = self.client.generate_image(
                prompt=task.prompt,
                negative_prompt=task.negative_prompt,
                num_steps=task.num_steps,
                guidance_scale=task.guidance_scale,
                width=task.width,
                height=task.height,
                seed=task.seed
            )

            result = image_buffer.getvalue()
            self.logger.info(f"Worker completed task {task.task_id} ({len(result)} bytes)")
            return result

        except Exception as e:
            self.logger.error(f"Worker failed for task {task.task_id}: {e}")
            raise

import uuid
import logging
from pathlib import Path
from typing import Optional, Dict, List
from concurrent.futures import ThreadPoolExecutor, Future
from queue import Queue, Empty
import threading

from .models import GenerationTask, TaskStatus
from .prompt_enhancer import PromptEnhancer
from .api_client import HuggingFaceClient

from database.mongo import mongodb

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class TaskQueue:
    def __init__(self, maxsize: int = 100):
        self._queue = Queue(maxsize=maxsize)

    def put(self, task: GenerationTask) -> bool:
        try:
            self._queue.put_nowait(task)
            return True
        except:
            return False

    def get(self, timeout: float = 1.0) -> Optional[GenerationTask]:
        try:
            return self._queue.get(timeout=timeout)
        except Empty:
            return None

    def size(self) -> int:
        return self._queue.qsize()


class TaskStorage:
    def __init__(self):
        self._tasks: Dict[str, GenerationTask] = {}
        self._lock = threading.RLock()

    def add(self, task: GenerationTask):
        with self._lock:
            self._tasks[task.task_id] = task

    def get(self, task_id: str) -> Optional[GenerationTask]:
        with self._lock:
            return self._tasks.get(task_id)

    def update(self, task_id: str, **kwargs):
        with self._lock:
            task = self._tasks.get(task_id)
            if task:
                for k, v in kwargs.items():
                    setattr(task, k, v)


class WorkerManager:
    def __init__(self, num_workers: int = 1):
        self.pool = ThreadPoolExecutor(
            max_workers=num_workers,
            thread_name_prefix="GenWorker"
        )
        logger.info(f"WorkerManager démarré avec {num_workers} worker(s)")

    def submit(self, fn, *args, **kwargs) -> Future:
        return self.pool.submit(fn, *args, **kwargs)

    def shutdown(self, wait: bool = True):
        logger.info("Arrêt des workers...")
        self.pool.shutdown(wait=wait)


class TaskDispatcher:
    def __init__(self, queue: TaskQueue, storage: TaskStorage, worker_manager: WorkerManager):
        self.queue = queue
        self.storage = storage
        self.worker_manager = worker_manager

    def dispatch(self, task: GenerationTask):
        logger.info(f"Envoi de la tâche {task.task_id} au worker")
        future = self.worker_manager.submit(self.process, task.task_id)
        future.add_done_callback(self._handle_completion)

    def _handle_completion(self, future: Future):
        try:
            future.result()
        except Exception as e:
            logger.error(f"Worker a échoué : {e}")

    def process(self, task_id: str):
        task = self.storage.get(task_id)
        if not task:
            return

        try:
            self.storage.update(task_id, status=TaskStatus.PROCESSING, progress=10)

            enhanced_prompt = PromptEnhancer.enhance(task.prompt)
            task.prompt = enhanced_prompt
            self.storage.update(task_id, progress=30)

            logger.info(f"Génération en cours pour task {task_id}...")
            client = HuggingFaceClient()
            image_buffer = client.generate_image(enhanced_prompt)
            self.storage.update(task_id, progress=80)

            filename = f"{uuid.uuid4()}.png"
            path = Path(__file__).parent.parent / "generated" / filename
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(image_buffer.getvalue())

            mongodb.save_generation_result(
                task_id=task.task_id,
                prompt=task.prompt,
                filename=filename
            )

            self.storage.update(
                task_id,
                filename=filename,
                status=TaskStatus.COMPLETED,
                progress=100.0
            )
            logger.info(f"Tâche {task_id} terminée → {filename}")

        except Exception as e:
            self.storage.update(task_id, status=TaskStatus.FAILED, error=str(e))
            logger.error(f"Échec tâche {task_id} : {e}")


class GeneratorEngine:
    def __init__(self, num_workers: int = 4):  # 4 workers pour bien paralléliser
        self.queue = TaskQueue()
        self.storage = TaskStorage()
        self.worker_manager = WorkerManager(num_workers=num_workers)
        self.dispatcher = TaskDispatcher(self.queue, self.storage, self.worker_manager)

        self.running = True
        self.dispatcher_thread = threading.Thread(
            target=self._dispatch_loop,
            daemon=True,
            name="TaskDispatcher"
        )
        self.dispatcher_thread.start()
        logger.info("GeneratorEngine prêt et en écoute")

    def _dispatch_loop(self):
        while self.running:
            task = self.queue.get(timeout=1.0)
            if task:
                self.dispatcher.dispatch(task)

    def submit(self, prompt: str) -> str:
        task_id = str(uuid.uuid4())
        task = GenerationTask(task_id=task_id, prompt=prompt)
        self.storage.add(task)

        if self.queue.put(task):
            logger.info(f"Tâche soumise : {task_id}")
        else:
            self.storage.update(task_id, status=TaskStatus.FAILED, error="File pleine")
            logger.error(f"Tâche {task_id} refusée – file pleine")

        return task_id

    def submit_batch(self, prompt: str, count: int = 4) -> List[str]:
        task_ids = []
        for _ in range(count):
            task_id = self.submit(prompt)
            task_ids.append(task_id)
        logger.info(f"Batch de {count} images soumis pour le prompt")
        return task_ids

    def get_status(self, task_id: str) -> Optional[dict]:
        task = self.storage.get(task_id)
        return task.to_dict() if task else None

    def shutdown(self):
        self.running = False
        self.worker_manager.shutdown()
        logger.info("GeneratorEngine arrêté proprement")


generator = GeneratorEngine(num_workers=4)
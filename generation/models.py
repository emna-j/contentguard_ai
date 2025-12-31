
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Optional, Dict, Any


class TaskStatus(Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


@dataclass
class GenerationTask:
    task_id: str
    prompt: str
    status: TaskStatus = TaskStatus.PENDING
    progress: float = 0.0
    filename: Optional[str] = None
    error: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convertit en dictionnaire pour JSON"""
        data = asdict(self)
        data["status"] = self.status.value
        data["progress"] = round(self.progress, 1)
        return data
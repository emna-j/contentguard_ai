from dataclasses import dataclass
from typing import Dict
import os

@dataclass
class GenerationConfig:
    HF_TOKEN: str = os.getenv("HF_TOKEN")
    MODEL_ID: str = "stabilityai/stable-diffusion-xl-base-1.0"

    DEFAULT_STEPS: int = 50
    DEFAULT_GUIDANCE: float = 7.5
    DEFAULT_WIDTH: int = 512
    DEFAULT_HEIGHT: int = 512

    STYLES: Dict[str, str] = None
    QUALITIES: Dict[str, str] = None

    def __post_init__(self):
        self.STYLES = {
            "realistic": "photorealistic, natural lighting, highly detailed, sharp focus",
            "cinematic": "cinematic, dramatic lighting, film grain, epic composition",
            "artistic": "painting, oil on canvas, brush strokes, artistic style",
            "anime": "anime style, vibrant colors, detailed eyes, clean lines",
            "photorealistic": "8k, ultra realistic, photorealistic, intricate details"
        }

        self.QUALITIES = {
            "best": "masterpiece, best quality, ultra detailed, 8k",
            "high": "high quality, detailed, sharp",
            "normal": "good quality"
        }

CONFIG = GenerationConfig()

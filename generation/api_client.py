from huggingface_hub import InferenceClient
from io import BytesIO
from typing import Optional
import random

from .config import CONFIG


class HuggingFaceClient:
    def __init__(self):
        self.client = InferenceClient(model=CONFIG.MODEL_ID, token=CONFIG.HF_TOKEN)

    def generate_image(
            self,
            prompt: str,
            negative_prompt: str = "",
            num_steps: int = 30,
            guidance_scale: float = 7.5,
            width: int = 512,
            height: int = 512,
            seed: Optional[int] = None
    ) -> BytesIO:

        params = {
            "prompt": prompt,
            "num_inference_steps": num_steps,
            "guidance_scale": guidance_scale,
            "width": width,
            "height": height
        }

        if negative_prompt:
            params["negative_prompt"] = negative_prompt

        if seed is None:
            seed = random.randint(0, 2**32 - 1)

        params["seed"] = seed

        try:
            image = self.client.text_to_image(**params)
        except Exception as e:
            raise RuntimeError(f"Erreur lors de la génération de l'image: {e}")

        buffer = BytesIO()
        image.save(buffer, format="PNG")
        buffer.seek(0)
        return buffer
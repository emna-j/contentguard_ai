
import re
from typing import List, Optional
from .config import CONFIG


class PromptValidator:

    @staticmethod
    def clean(prompt: str) -> str:
        prompt = re.sub(r'\s+', ' ', prompt)
        prompt = re.sub(r'[<>{}[\]\\]', '', prompt)
        prompt = prompt.strip()
        return prompt

    @staticmethod
    def is_valid(prompt: str) -> bool:
        """Vérifie si un prompt est valide"""
        if not prompt or len(prompt) < 3:
            return False
        if len(prompt) > 1000:
            return False
        return True


class PromptBuilder:

    def __init__(self):
        self.validator = PromptValidator()

    def build(self, base_prompt: str, style: Optional[str] = None,
              quality: Optional[str] = None,
              additional_tags: Optional[List[str]] = None) -> str:
        base_prompt = self.validator.clean(base_prompt)

        if not self.validator.is_valid(base_prompt):
            raise ValueError("Invalid base prompt")

        parts = [base_prompt]

        if style and style in CONFIG.STYLES:
            parts.append(CONFIG.STYLES[style])

        if quality and quality in CONFIG.QUALITIES:
            parts.append(CONFIG.QUALITIES[quality])

        if additional_tags:
            parts.extend(additional_tags)

        default_tags = ["professional", "detailed"]
        parts.extend(default_tags)

        enhanced_prompt = ", ".join(parts)
        enhanced_prompt = self.validator.clean(enhanced_prompt)

        return enhanced_prompt


class PromptEnhancer:

    def __init__(self):
        self.builder = PromptBuilder()

    @staticmethod
    def enhance(prompt: str, style: str = "realistic",
                quality: str = "high",
                additional_tags: Optional[List[str]] = None) -> str:
        builder = PromptBuilder()
        return builder.build(prompt, style, quality, additional_tags)
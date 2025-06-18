"""
Translation Model Module

"""

import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any

from ._translation_rag_model import TranslationRagModel
from ._loader_model import LoaderModel
from ..config import config


class TranslationModel(ABC):
    """번역 모델 클래스"""

    def __init__(self, model_info: Dict[str, Dict[str, str]]):
        """
        Args:
            model_info: Hugging Face 모델 {"name": str, "transformer": str}
        """
        self.model_info = model_info
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()

        max_length, num_beams = self._get_config()
        self.max_length = max_length
        self.num_beams = num_beams

    def load_model(self, auth_token: Optional[str] = None, **kwargs) -> None:
        # Load the model and tokenizer from Hugging Face
        auto_model = LoaderModel(self.model_info, auth_token)
        auto_model.load_model(**kwargs)
        self.model_name = auto_model.model_name
        self.tokenizer = auto_model.tokenizer
        self.model = auto_model.model

        # Set the model to evaluation mode
        self.rag_model = TranslationRagModel()
        self.rag_model.load_terminology_db()

    def vaidate_model(self) -> None:
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

    def vaidate_support_lang(
        self,
        source_lang: str,
        target_lang: str,
    ) -> tuple[str, str]:
        if source_lang not in config.LANGUAGE_CODES:
            raise ValueError(f"Unsupported source language: {source_lang}")
        if target_lang not in config.LANGUAGE_CODES:
            raise ValueError(f"Unsupported target language: {target_lang}")

        source_code = self.lang_code_to_id(source_lang)
        target_code = self.lang_code_to_id(target_lang)
        print(f"✓ Translating from '{source_code}' to '{target_code}'...")
        return source_code, target_code

    def move_inputs_to_device(self, inputs) -> Any:
        if self.device != "cpu":
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        return inputs

    def get_model_info(self) -> Dict[str, Any]:
        return {
            "model_name": self.model_info["name"],
            "device": self.device,
            "is_loaded": self.model is not None,
            "supported_languages": list(config.LANGUAGE_CODES),
        }

    def print_generation_config(self) -> None:
        gen_config = self.model.generation_config
        print(f"✓ Generation config: {gen_config}")

    @abstractmethod
    def lang_code_to_id(self, lang: str) -> str:
        """
        Convert language code to model-specific ID.

        Args:
            lang (str): Language code (e.g., "korean", "japanese", "english")

        Returns:
            str: Model-specific language ID
        """
        pass

    @abstractmethod
    def vaidate_lang(
        self,
        source_lang: str,
        target_lang: str,
    ) -> None:
        pass

    @abstractmethod
    def translate(
        self,
        text,
        source_lang,
        target_lang,
        **generate_kwargs,
    ):
        """
        Translate text from source language to target language.

        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            str: Translated text
        """

        self.vaidate_model()

        self.vaidate_lang(source_lang, target_lang)

    # Private method

    def _get_device(self) -> str:
        """최적의 디바이스 선택"""
        if torch.backends.mps.is_available():
            return "mps"  # M1/M2/M3 맥북
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def _get_config(self) -> tuple[int, int]:
        max_length = config.MAX_LENGTH
        num_beams = config.NUM_BEAMS
        return max_length, num_beams

"""
Translation Model Module

"""

import os
import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from transformers import AutoTokenizer
from huggingface_hub import login
from ..config import config


class TranslationModel(ABC):
    """번역 모델 클래스"""

    def __init__(self, model_name: str, auth_token: Optional[str] = None):
        """
        Args:
            model_name: Hugging Face 모델 이름
            auth_token: Hugging Face 인증 토큰 (선택사항)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()

        # 인증이 필요한 경우 로그인
        if auth_token:
            login(token=auth_token)
        elif config.get_huggingface_token():
            login(token=config.get_huggingface_token())

    def load_model_seq2seqlm(self, **kwargs) -> None:
        """모델 로드"""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")

        try:
            from transformers import AutoModelForSeq2SeqLM

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("✓ Tokenizer loaded")

            # 모델 로드
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                **kwargs,
            }

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, **model_kwargs
            )

            # 디바이스로 이동
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            print("✓ Model loaded successfully!")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def load_model_causallm(self, **kwargs) -> None:
        """모델 로드"""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")

        try:
            from transformers import AutoModelForCausalLM

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("✓ Tokenizer loaded")

            # CausalLM 모델 로드
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                **kwargs,
            }

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )

            # 디바이스로 이동
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            print("✓ Model loaded successfully!")

        except Exception as e:
            print(f"✗ Error loading HyperCLOVAX model: {e}")
            raise

    @abstractmethod
    def translate(self, text, source_lang, target_lang):
        """
        Translate text from source language to target language.

        Args:
            text (str): Text to translate
            source_lang (str): Source language code
            target_lang (str): Target language code

        Returns:
            str: Translated text
        """
        pass

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.model is not None,
            "supported_languages": list(config.LANGUAGE_CODES.keys()),
        }

    def _get_device(self) -> str:
        """최적의 디바이스 선택"""
        if torch.backends.mps.is_available():
            return "mps"  # M1/M2/M3 맥북
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

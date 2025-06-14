"""
Translation Model Module

"""

import os
import torch
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
from huggingface_hub import login

from ._loader_model import LoaderModel
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

    def load_model(self, auth_token: Optional[str] = None, **kwargs) -> None:
        auto_model = LoaderModel(self.model_name, auth_token)
        auto_model.load_model(**kwargs)
        self.model_name = auto_model.model_name
        self.tokenizer = auto_model.tokenizer
        self.model = auto_model.model

    def lang_code_to_id(self, lang: str) -> str:
        if any(keyword in self.model_name.lower() for keyword in ["mbart"]):
            return {
                "korean": "ko_KR",
                "japanese": "ja_XX",
                "english": "en_XX",
            }.get(lang, lang)
        elif any(keyword in self.model_name.lower() for keyword in ["m2m"]):
            return {
                "korean": "ko",
                "japanese": "ja",
                "english": "en",
            }.get(lang, lang)
        elif any(keyword in self.model_name.lower() for keyword in ["hyperclova"]):
            return {
                "korean": "한국어",
                "japanese": "일본어",
                "english": "영어",
            }.get(lang, lang)
        else:
            return {
                "korean": "kor_Hang",
                "japanese": "jpn_Jpan",
                "english": "eng_Latn",
            }.get(lang, lang)

    def show_support_lang(self) -> None:
        if any(keyword in self.model_name.lower() for keyword in ["mbart"]):
            print(self.tokenizer.lang_code_to_id)
        else:
            print(None)

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
            "supported_languages": list(config.LANGUAGE_CODES),
        }

    def _get_device(self) -> str:
        """최적의 디바이스 선택"""
        if torch.backends.mps.is_available():
            return "mps"  # M1/M2/M3 맥북
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

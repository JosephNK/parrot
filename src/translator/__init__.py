"""
Korean-Japanese Translation Package

Hugging Face 모델을 사용한 한국어-일본어 번역 패키지
"""

from .translator import KoreanJapaneseTranslator
from .models import (
    TranslationModel,
    NLLBTranslationModel,
    OpusTranslationModel,
    HyperCLOVAXTranslationModel,
)
from .config import config

__version__ = "0.1.0"
__author__ = "Your Name"

__all__ = [
    "KoreanJapaneseTranslator",
    "TranslationModel",
    "NLLBTranslationModel",
    "OpusTranslationModel",
    "HyperCLOVAXTranslationModel",
    "config",
]

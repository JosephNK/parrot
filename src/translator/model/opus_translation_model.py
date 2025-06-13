"""
Opus-MT Model Module

"""

import os
import torch
from typing import Optional
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
from ._translation_model import TranslationModel
from ..config import config


class OpusTranslationModel(TranslationModel):
    """Opus-MT 모델 전용 클래스"""

    def __init__(self, model_name: str, auth_token: Optional[str] = None):
        """Opus-MT 모델 초기화"""
        super().__init__(model_name, auth_token)

    def load_model(self, **kwargs) -> None:
        """모델 로드"""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")

        try:
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

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        **generate_kwargs,
    ) -> str:
        """Opus-MT 모델 전용 번역 함수"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        # 기본값 설정
        if max_length is None:
            max_length = config.MAX_LENGTH
        if num_beams is None:
            num_beams = config.NUM_BEAMS

        try:
            # Opus 모델은 단순하게 입력 처리
            inputs = self.tokenizer(text, return_tensors="pt")

            # 디바이스로 이동
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 번역 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    max_length=max_length,
                    num_beams=num_beams,
                    early_stopping=True,
                    **generate_kwargs,
                )

            # 디코딩
            translated_text = self.tokenizer.decode(
                outputs[0], skip_special_tokens=True
            )

            return translated_text

        except Exception as e:
            print(f"Opus Translation error: {e}")
            return f"[Error: {str(e)}]"

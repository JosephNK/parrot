"""
NLLB Model Module

"""

import os
import torch
from typing import Optional
from huggingface_hub import login
from ._translation_model import TranslationModel
from ..config import config


class NLLBTranslationModel(TranslationModel):
    """NLLB 모델 전용 클래스"""

    def __init__(self, model_name: str, auth_token: Optional[str] = None):
        super().__init__(model_name, auth_token)

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        **generate_kwargs,
    ) -> str:
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        # 기본값 설정
        if max_length is None:
            max_length = config.MAX_LENGTH
        if num_beams is None:
            num_beams = config.NUM_BEAMS

        # 언어 코드 확인
        if source_lang not in config.LANGUAGE_CODES:
            raise ValueError(f"Unsupported source language: {source_lang}")
        if target_lang not in config.LANGUAGE_CODES:
            raise ValueError(f"Unsupported target language: {target_lang}")

        source_code = self.lang_code_to_id(source_lang)
        target_code = self.lang_code_to_id(target_lang)

        print(f"✓ Translating from '{source_code}' to '{target_code}'...")

        try:
            # NLLB 모델은 src_lang을 토크나이저 속성으로 설정
            self.tokenizer.src_lang = source_code
            inputs = self.tokenizer(text, return_tensors="pt")

            # 디바이스로 이동
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 번역 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                        target_code
                    ),
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
            print(f"Translation error: {e}")
            return f"[Error: {str(e)}]"

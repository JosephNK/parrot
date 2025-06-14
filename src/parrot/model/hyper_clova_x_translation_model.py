"""
HyperCLOVAX Model Module

"""

import os
import re
import time
import torch
from typing import Optional
from transformers import AutoTokenizer
from huggingface_hub import login
from ._translation_model import TranslationModel
from ..config import config


class HyperCLOVAXTranslationModel(TranslationModel):
    """HyperCLOVAX 모델 전용 클래스"""

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
            max_length = min(config.MAX_LENGTH * 2, 1024)

        # 언어 코드 확인
        if source_lang not in config.LANGUAGE_CODES:
            raise ValueError(f"Unsupported source language: {source_lang}")
        if target_lang not in config.LANGUAGE_CODES:
            raise ValueError(f"Unsupported target language: {target_lang}")

        source_code = self.lang_code_to_id(source_lang)
        target_code = self.lang_code_to_id(target_lang)

        print(f"✓ Translating from '{source_code}' to '{target_code}'...")

        try:
            # Chat template 구성
            chat = [
                {
                    "role": "system",
                    "content": f'- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 당신은 전문 번역가입니다. 주어진 텍스트를 정확하고 자연스럽게 번역해주세요.',
                },
                {
                    "role": "user",
                    "content": f"다음 {source_code} 텍스트를 {target_code}로 번역해주세요.\n\n{text}",
                },
            ]

            # Chat template 적용
            inputs = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )

            # 디바이스로 이동
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 번역 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    do_sample=False,  # 탐욕적 디코딩으로 가장 확률 높은 토큰 선택
                    repetition_penalty=1.1,  # 반복 방지
                    stop_strings=["<|endofturn|>", "<|stop|>"],
                    tokenizer=self.tokenizer,
                    pad_token_id=self.tokenizer.eos_token_id,
                    **generate_kwargs,
                )

            # 디코딩 (입력 부분 제거)
            generated_tokens = outputs[0][len(inputs["input_ids"][0]) :]
            translated_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            print(f"✓ Translation completed: {translated_text}")

            # 불필요한 부분 정리
            if "<|endofturn|>" in translated_text:
                translated_text = translated_text.split("<|endofturn|>")[0].strip()

            # 백틱과 불필요한 줄바꿈 제거
            translated_text = re.sub(r"```[\r\n]*|[\r\n]*```", "", translated_text)

            return translated_text

        except Exception as e:
            print(f"Translation error: {e}")
            return f"[Error: {str(e)}]"

"""
HyperCLOVAX Model Module

"""

import re
import torch
from typing import Any, Optional

from ..exception.exception import TranslationError, TranslationErrorCode
from ._translation_model import TranslationModel
from ..config import config


class HyperCLOVAXTranslationModel(TranslationModel):
    """HyperCLOVAX 모델 전용 클래스"""

    def __init__(self, model_name: str):
        super().__init__(model_name)

        # 이 특정 모델에 맞게 max_length 조정
        self.max_length = min(self.max_length * 2, 1024)

    def lang_code_to_id(self, lang: str) -> str:
        return {
            "korean": "한국어",
            "japanese": "일본어",
            "english": "영어",
        }.get(lang, lang)

    def vaidate_lang(
        self,
        source_lang: str,
        target_lang: str,
    ) -> None:
        self.source_code, self.target_code = self.vaidate_support_lang(
            source_lang, target_lang
        )

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        **generate_kwargs,
    ) -> str:
        try:
            super().translate(text, source_lang, target_lang, **generate_kwargs)

            # Chat template 구성
            chat = [
                {
                    "role": "system",
                    "content": f'- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 당신은 전문 번역가입니다. 주어진 텍스트를 정확하고 자연스럽게 번역해주세요.',
                },
                {
                    "role": "user",
                    "content": f"다음 {self.source_code} 텍스트를 {self.target_code}로 번역해주세요.\n\n{text}",
                },
            ]

            # Chat template 적용
            inputs = self.tokenizer.apply_chat_template(
                chat, add_generation_prompt=True, return_dict=True, return_tensors="pt"
            )

            # 디바이스로 이동
            inputs = self.move_inputs_to_device(inputs)

            # 번역 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_new_tokens=self.max_length,
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
            raise TranslationError(
                message=str(e), error_code=TranslationErrorCode.TRANSLATION_ERROR
            )

"""
Qwen Model Module

"""

import re
import torch

from ..exception.exception import TranslationError, TranslationErrorCode
from ._translation_model import TranslationModel
from ..config import config


class QwenTranslationModel(TranslationModel):
    """Qwen 모델 전용 클래스"""

    def __init__(self, model_name: str):
        super().__init__(model_name)

        # 이 특정 모델에 맞게 max_length 조정
        self.max_length = min(self.max_length * 2, 1024)

    def lang_code_to_id(self, lang: str) -> str:
        return {
            "korean": "korean",
            "japanese": "japanese",
            "english": "english",
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

            # 텍스트 전처리
            terminology_hint = self.rag_model.retrieve_text_with_domain(
                text=text,
                domain="ko2ja",
            )

            # Chat template 구성
            chat = [
                {
                    "role": "system",
                    "content": "You are Qwen, created by Alibaba Cloud. You are a professional translator. Translate the given text accurately and naturally.",
                },
                {
                    "role": "user",
                    "content": f"""Translate {self.source_code} to {self.target_code}:

{text}

[Terms] 
{terminology_hint}

Output {self.target_code} translation only:""",
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
                    do_sample=True,
                    temperature=0.3,
                    top_p=0.8,
                    pad_token_id=self.tokenizer.eos_token_id,
                    tokenizer=self.tokenizer,
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

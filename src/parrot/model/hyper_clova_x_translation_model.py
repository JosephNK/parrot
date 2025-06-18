"""
HyperCLOVAX Model Module

"""

import re
from typing import Dict
import torch

from ..exception.exception import TranslationError, TranslationErrorCode
from ._translation_model import TranslationModel


class HyperCLOVAXTranslationModel(TranslationModel):
    """HyperCLOVAX 모델 전용 클래스"""

    def __init__(self, model_info: Dict[str, Dict[str, str]]):
        super().__init__(model_info)

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

            # RAG 모델을 사용하여 용어 검색
            terminology_hint = self.rag_model.retrieve_text_with_domain(
                text=text,
                domain=self.rag_model.get_domain_from_lang(
                    source_lang,
                    target_lang,
                ),
            )

            # Chat template 구성
            template = [
                {"role": "tool_list", "content": ""},
                {
                    "role": "system",
                    "content": f"""
당신은 {self.source_code}을 {self.target_code}로 번역하는 전문 번역가입니다. 
""".strip(),
                },
                {
                    "role": "user",
                    "content": f"""
1번만 번역하세요:

1. {text}

2. 참고용어: {terminology_hint}

1번을 {self.target_code}로 번역
""".strip(),
                },
            ]

            # Chat template 적용
            inputs = self.tokenizer.apply_chat_template(
                template,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            # 디바이스로 이동
            inputs = self.move_inputs_to_device(inputs)

            # 번역 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=self.max_length,
                    stop_strings=[
                        "<|endofturn|>",
                        "<|stop|>",
                    ],
                    **generate_kwargs,
                )

            # 디코딩 (입력 부분 제거)
            generated_tokens = outputs[0][len(inputs["input_ids"][0]) :]
            translated_text = self.tokenizer.decode(
                generated_tokens, skip_special_tokens=True
            ).strip()

            # 불필요한 부분 정리
            if "<|endofturn|>" in translated_text:
                translated_text = translated_text.split("<|endofturn|>")[0].strip()

            # 백틱과 불필요한 줄바꿈 제거
            translated_text = re.sub(r"```[\r\n]*|[\r\n]*```", "", translated_text)

            print(f"✓ Translation completed: {translated_text}")

            return translated_text

        except Exception as e:
            print(f"Translation error: {e}")
            raise TranslationError(
                message=str(e), error_code=TranslationErrorCode.TRANSLATION_ERROR
            )

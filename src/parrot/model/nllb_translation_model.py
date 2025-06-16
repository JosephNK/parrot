"""
NLLB Model Module

"""

import torch
from typing import Optional

from ..exception.exception import TranslationError, TranslationErrorCode
from ._translation_model import TranslationModel
from ..config import config


class NLLBTranslationModel(TranslationModel):
    """NLLB 모델 전용 클래스"""

    def __init__(self, model_name: str):
        super().__init__(model_name)

    def lang_code_to_id(self, lang: str) -> str:
        return {
            "korean": "kor_Hang",
            "japanese": "jpn_Jpan",
            "english": "eng_Latn",
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

            # NLLB 모델은 src_lang을 토크나이저 속성으로 설정
            self.tokenizer.src_lang = self.source_code
            inputs = self.tokenizer(text, return_tensors="pt")

            # 디바이스로 이동
            inputs = self.move_inputs_to_device(inputs)

            # 번역 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    forced_bos_token_id=self.tokenizer.convert_tokens_to_ids(
                        self.target_code
                    ),
                    max_length=self.max_length,
                    num_beams=self.num_beams,
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
            raise TranslationError(
                message=str(e), error_code=TranslationErrorCode.TRANSLATION_ERROR
            )

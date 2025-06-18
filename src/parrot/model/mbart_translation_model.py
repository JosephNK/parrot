"""
MBart Model Module

"""

import torch
from typing import Dict

from ..exception.exception import TranslationError, TranslationErrorCode
from ._translation_model import TranslationModel


class MBartTranslationModel(TranslationModel):
    """MBart 모델 전용 클래스"""

    def __init__(self, model_info: Dict[str, Dict[str, str]]):
        super().__init__(model_info)

    def lang_code_to_id(self, lang: str) -> str:
        return {
            "korean": "ko_KR",
            "japanese": "ja_XX",
            "english": "en_XX",
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
            text = self.rag_model.retrieve_replace_text_with_domain(
                text=text,
                domain=self.rag_model.get_domain_from_lang(
                    source_lang,
                    target_lang,
                    use_replacement=True,
                ),
            )

            # MBart 모델은 src_lang을 토크나이저 속성으로 설정
            self.tokenizer.src_lang = self.source_code
            inputs = self.tokenizer(text, return_tensors="pt")

            # 디바이스로 이동
            inputs = self.move_inputs_to_device(inputs)

            # 번역 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
                    forced_bos_token_id=self.tokenizer.lang_code_to_id[
                        self.target_code
                    ],
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

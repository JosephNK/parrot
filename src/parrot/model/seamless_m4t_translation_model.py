"""
SeamlessM4t Model Module

"""

import torch
from typing import Dict

from ..exception.exception import TranslationError, TranslationErrorCode
from ._translation_model import TranslationModel


class SeamlessM4tTranslationModel(TranslationModel):
    """SeamlessM4t 모델 전용 클래스"""

    def __init__(self, model_info: Dict[str, Dict[str, str]]):
        super().__init__(model_info)

    def lang_code_to_id(self, lang: str) -> str:
        return {
            "korean": "kor",
            "japanese": "jpn",
            "english": "eng",
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

            inputs = self.tokenizer(
                text, src_lang=self.source_code, return_tensors="pt"
            )

            # 번역 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    tgt_lang=self.target_code,
                    generate_speech=False,
                    **generate_kwargs,
                )

            # 결과 처리
            if hasattr(outputs, "sequences"):
                # GenerateEncoderDecoderOutput의 경우
                token_ids = outputs.sequences[0]
            else:
                # 직접 텐서인 경우
                token_ids = outputs[0]
            if hasattr(token_ids, "tolist"):
                # 텐서를 리스트로 변환 (필요한 경우)
                token_ids = token_ids.tolist()

            translated_text = self.tokenizer.decode(
                token_ids,
                skip_special_tokens=True,
            )

            return translated_text

        except Exception as e:
            print(f"Translation error: {e}")
            raise TranslationError(
                message=str(e), error_code=TranslationErrorCode.TRANSLATION_ERROR
            )

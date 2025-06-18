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
        self.max_length = min(self.max_length * 3, 1024)

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
                {
                    "role": "system",
                    "content": f"""
You are Qwen, created by Alibaba Cloud. You are a professional {self.source_code}-{self.target_code} translator. 
Translate the given text accurately and naturally.

**Output Format:**
Provide only the translation result without additional explanations.
""".strip(),
                },
                {
                    "role": "user",
                    "content": f"""
Translate {self.source_code} to {self.target_code}:
{text}

**Terms:**
{terminology_hint}

Output {self.target_code} translation only.
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
                    max_new_tokens=self.max_length,  # 번역문이 원문보다 길 수 있으므로 충분한 길이
                    min_new_tokens=10,  # 너무 짧은 번역 방지
                    do_sample=False,  # 번역은 일관성이 중요하므로 deterministic (temperature, top_p 등 사용 안함)
                    num_beams=self.num_beams,  # beam search로 더 나은 번역 품질
                    early_stopping=True,  # beam search에서 조기 종료 활성화
                    repetition_penalty=1.1,  # 반복되는 구문 방지
                    no_repeat_ngram_size=3,  # 3-gram 반복 방지
                    length_penalty=1.0,  # 길이에 대한 페널티 (번역에서는 중립적으로)
                    bad_words_ids=None,  # 필요시 특정 단어 제외
                    use_cache=False,  # 캐시 사용
                    return_dict_in_generate=False,  # 단순한 출력 형태
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

            # # 백틱과 불필요한 줄바꿈 제거
            # translated_text = re.sub(r"```[\r\n]*|[\r\n]*```", "", translated_text)

            print(f"✓ Translation completed: {translated_text}")

            return translated_text

        except Exception as e:
            print(f"Translation error: {e}")
            raise TranslationError(
                message=str(e), error_code=TranslationErrorCode.TRANSLATION_ERROR
            )

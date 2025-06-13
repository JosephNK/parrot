"""
HyperCLOVAX Model Module

"""

import os
import torch
from typing import Optional
from transformers import AutoTokenizer
from huggingface_hub import login
from ._translation_model import TranslationModel
from ..config import config


class HyperCLOVAXTranslationModel(TranslationModel):
    """HyperCLOVAX 모델 전용 클래스"""

    def __init__(self, model_name: str, auth_token: Optional[str] = None):
        """HyperCLOVAX 모델 초기화"""
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
        """HyperCLOVAX 모델을 사용한 번역"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        # 기본값 설정
        if max_length is None:
            max_length = config.MAX_LENGTH * 2  # HyperCLOVAX는 더 긴 출력 가능

        # 언어 매핑
        lang_map = {"korean": "한국어", "japanese": "일본어", "english": "영어"}

        source_lang_ko = lang_map.get(source_lang, source_lang)
        target_lang_ko = lang_map.get(target_lang, target_lang)

        try:
            # Chat template 구성
            chat = [
                {"role": "tool_list", "content": ""},
                {
                    "role": "system",
                    "content": f'- AI 언어모델의 이름은 "CLOVA X" 이며 네이버에서 만들었다.\n- 당신은 전문 번역가입니다. 주어진 텍스트를 정확하고 자연스럽게 번역해주세요.',
                },
                {
                    "role": "user",
                    "content": f"다음 {source_lang_ko} 텍스트를 {target_lang_ko}로 번역해주세요:\n\n{text}\n\n번역 결과만 출력하고 다른 설명은 하지 마세요.",
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
                    do_sample=True,
                    top_p=0.9,
                    temperature=0.7,
                    stop_strings=["<|endofturn|>", "<|stop|>"],
                    tokenizer=self.tokenizer,
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

            return translated_text

        except Exception as e:
            print(f"HyperCLOVAX Translation error: {e}")
            return f"[Error: {str(e)}]"

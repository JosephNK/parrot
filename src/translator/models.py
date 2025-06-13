"""
Translation Models Module

Hugging Face 모델 로딩 및 관리 클래스
"""

import os
import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from huggingface_hub import login
from .config import config


class TranslationModel:
    """번역 모델 클래스"""

    # 지원하는 언어 코드
    LANGUAGE_CODES = {
        "korean": "kor_Hang",
        "japanese": "jpn_Jpan",
        "english": "eng_Latn",
    }

    def __init__(self, model_name: str, auth_token: Optional[str] = None):
        """
        Args:
            model_name: Hugging Face 모델 이름
            auth_token: Hugging Face 인증 토큰 (선택사항)
        """
        self.model_name = model_name
        self.tokenizer = None
        self.model = None
        self.device = self._get_device()

        # 인증이 필요한 경우 로그인
        if auth_token:
            login(token=auth_token)
        elif config.get_huggingface_token():
            login(token=config.get_huggingface_token())

    def _get_device(self) -> str:
        """최적의 디바이스 선택"""
        if torch.backends.mps.is_available():
            return "mps"  # M1/M2/M3 맥북
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

    def load_model(self, **kwargs) -> None:
        """모델 로드"""
        print(f"Loading model: {self.model_name}")
        print(f"Using device: {self.device}")

        try:
            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("✓ Tokenizer loaded")

            # 모델 로드
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                **kwargs,
            }

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name, **model_kwargs
            )

            # 디바이스로 이동
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            print("✓ Model loaded successfully!")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        **generate_kwargs,
    ) -> str:
        """
        텍스트 번역

        Args:
            text: 번역할 텍스트
            source_lang: 소스 언어 (korean, japanese, english)
            target_lang: 타겟 언어 (korean, japanese, english)
            max_length: 최대 생성 길이 (기본값: config에서 읽음)
            num_beams: 빔 서치 수 (기본값: config에서 읽음)
            **generate_kwargs: 추가 생성 옵션

        Returns:
            번역된 텍스트
        """
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        # 기본값 설정
        if max_length is None:
            max_length = config.MAX_LENGTH
        if num_beams is None:
            num_beams = config.NUM_BEAMS

        # 언어 코드 확인
        if source_lang not in self.LANGUAGE_CODES:
            raise ValueError(f"Unsupported source language: {source_lang}")
        if target_lang not in self.LANGUAGE_CODES:
            raise ValueError(f"Unsupported target language: {target_lang}")

        source_code = self.LANGUAGE_CODES[source_lang]
        target_code = self.LANGUAGE_CODES[target_lang]

        try:
            # NLLB 모델의 경우 특별한 처리 필요
            if "nllb" in self.model_name.lower():
                # NLLB 모델의 경우 src_lang을 토크나이저에 설정
                self.tokenizer.src_lang = source_code
                inputs = self.tokenizer(text, return_tensors="pt")
            else:
                # 다른 모델들의 경우 기존 방식 사용
                inputs = self.tokenizer(text, return_tensors="pt", src_lang=source_code)

            # 디바이스로 이동
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 번역 생성
            with torch.no_grad():
                if "nllb" in self.model_name.lower():
                    # NLLB 모델의 경우
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
                else:
                    # 다른 모델들의 경우
                    outputs = self.model.generate(
                        inputs["input_ids"],
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


class HyperCLOVAXTranslationModel(TranslationModel):
    """HyperCLOVAX 모델 전용 클래스"""

    def __init__(self, model_name: str, auth_token: Optional[str] = None):
        """HyperCLOVAX 모델 초기화"""
        super().__init__(model_name, auth_token)
        self.model_type = "causal_lm"  # CausalLM 타입

    def load_model(self, **kwargs) -> None:
        """HyperCLOVAX 모델 로드"""
        print(f"Loading HyperCLOVAX model: {self.model_name}")
        print(f"Using device: {self.device}")

        try:
            from transformers import AutoModelForCausalLM

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            print("✓ Tokenizer loaded")

            # CausalLM 모델 로드
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                **kwargs,
            }

            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name, **model_kwargs
            )

            # 디바이스로 이동
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            print("✓ HyperCLOVAX model loaded successfully!")

        except Exception as e:
            print(f"✗ Error loading HyperCLOVAX model: {e}")
            raise

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

    def save_model(self, save_path: str) -> None:
        """모델을 로컬에 저장"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded")

        os.makedirs(save_path, exist_ok=True)
        self.tokenizer.save_pretrained(save_path)
        self.model.save_pretrained(save_path)
        print(f"Model saved to: {save_path}")

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""


class NLLBTranslationModel(TranslationModel):
    """NLLB 모델 전용 클래스"""

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        **generate_kwargs,
    ) -> str:
        """NLLB 모델 전용 번역 함수"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        # 기본값 설정
        if max_length is None:
            max_length = config.MAX_LENGTH
        if num_beams is None:
            num_beams = config.NUM_BEAMS

        # 언어 코드 확인
        if source_lang not in self.LANGUAGE_CODES:
            raise ValueError(f"Unsupported source language: {source_lang}")
        if target_lang not in self.LANGUAGE_CODES:
            raise ValueError(f"Unsupported target language: {target_lang}")

        source_code = self.LANGUAGE_CODES[source_lang]
        target_code = self.LANGUAGE_CODES[target_lang]

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
            print(f"NLLB Translation error: {e}")
            return f"[Error: {str(e)}]"


class OpusTranslationModel(TranslationModel):
    """Opus-MT 모델 전용 클래스"""

    def translate(
        self,
        text: str,
        source_lang: str,
        target_lang: str,
        max_length: Optional[int] = None,
        num_beams: Optional[int] = None,
        **generate_kwargs,
    ) -> str:
        """Opus-MT 모델 전용 번역 함수"""
        if not self.model or not self.tokenizer:
            raise ValueError("Model not loaded. Call load_model() first.")

        # 기본값 설정
        if max_length is None:
            max_length = config.MAX_LENGTH
        if num_beams is None:
            num_beams = config.NUM_BEAMS

        try:
            # Opus 모델은 단순하게 입력 처리
            inputs = self.tokenizer(text, return_tensors="pt")

            # 디바이스로 이동
            if self.device != "cpu":
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # 번역 생성
            with torch.no_grad():
                outputs = self.model.generate(
                    inputs["input_ids"],
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
            print(f"Opus Translation error: {e}")
            return f"[Error: {str(e)}]"

    def get_model_info(self) -> Dict[str, Any]:
        """모델 정보 반환"""
        return {
            "model_name": self.model_name,
            "device": self.device,
            "is_loaded": self.model is not None,
            "supported_languages": list(self.LANGUAGE_CODES.keys()),
        }

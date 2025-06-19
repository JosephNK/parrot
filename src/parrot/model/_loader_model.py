import os
import torch
from typing import Optional, Dict, Any
from transformers import AutoTokenizer
from huggingface_hub import login
from ..config import config


class LoaderModel:
    """모델 Loader 클래스"""

    def __init__(
        self,
        model_info: Dict[str, Dict[str, str]],
        auth_token: Optional[str] = None,
    ):
        """
        Args:
            model_name: Hugging Face 모델 이름
            auth_token: Hugging Face 인증 토큰 (선택사항)
        """
        self.model_info = model_info
        self.tokenizer = None
        self.model = None
        self.device = self.__get_device()

        self.model_name = self.model_info["name"]
        self.tokenizer_name = self.model_info["tokenizer"]
        self.transformer = self.model_info["transformer"]

        # 인증이 필요한 경우 로그인
        if auth_token:
            login(token=auth_token)
        elif config.get_huggingface_token():
            login(token=config.get_huggingface_token())

    def load_model(self, **kwargs) -> None:
        if self.transformer == "seq2seqlm":
            # Seq2SeqLM
            self.__load_model_seq2seqlm(**kwargs)
        elif self.transformer == "causallm":
            # CausalLM
            self.__load_model_causallm(**kwargs)
        elif self.transformer == "ctranslate2":
            # ctranslate2
            self.__load_model_ctranslate2(**kwargs)

    def __load_model_seq2seqlm(self, **kwargs) -> None:
        """Seq2SeqLM 모델 로드"""
        print(f"Loading model (seq2seqlm): {self.model_name}")
        print(f"Using device: {self.device}")

        try:
            from transformers import AutoModelForSeq2SeqLM

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            print("✓ Tokenizer loaded")

            # 모델 로드
            model_kwargs = {
                "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
                **kwargs,
            }

            self.model = AutoModelForSeq2SeqLM.from_pretrained(
                self.model_name,
                **model_kwargs,
            )

            # 디바이스로 이동
            if self.device != "cpu":
                self.model = self.model.to(self.device)

            print("✓ Model loaded successfully.")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def __load_model_causallm(self, **kwargs) -> None:
        """CausalLM 모델 로드"""
        print(f"Loading model (causallm): {self.model_name}")
        print(f"Using device: {self.device}")

        try:
            from transformers import AutoModelForCausalLM

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
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

            print("✓ Model loaded successfully.")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def __load_model_ctranslate2(self, **kwargs) -> None:
        """Ctranslate2 모델 로드"""
        print(f"Loading model (ctranslate2): {self.model_name}")
        DEVICE_MAPPING = {
            "cuda": {"device": "cuda", "compute_type": "int8_float16"},
            "cpu": {"device": "cpu", "compute_type": "int8"},
            "mps": {"device": "cpu", "compute_type": "int8"},  # MPS 폴백
        }
        config = DEVICE_MAPPING.get(
            self.device, {"device": "cpu", "compute_type": "int8"}
        )
        self.device = config["device"]
        print(f"Using device: {self.device}")

        try:
            from hf_hub_ctranslate2 import MultiLingualTranslatorCT2fromHfHub

            # 토크나이저 로드
            self.tokenizer = AutoTokenizer.from_pretrained(self.tokenizer_name)
            print("✓ Tokenizer loaded")

            self.model = MultiLingualTranslatorCT2fromHfHub(
                model_name_or_path=self.model_name,
                device=config["device"],
                compute_type=config["compute_type"],
                tokenizer=self.tokenizer,
            )

            print("✓ Model loaded successfully.")

        except Exception as e:
            print(f"✗ Error loading model: {e}")
            raise

    def __get_device(self) -> str:
        """최적의 디바이스 선택"""
        if torch.backends.mps.is_available():
            return "mps"  # M1/M2/M3 맥북
        elif torch.cuda.is_available():
            return "cuda"
        else:
            return "cpu"

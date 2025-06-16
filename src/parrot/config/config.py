"""
Configuration Module

환경변수 및 설정 관리
"""

import os
from typing import Optional
from dotenv import load_dotenv

# .env 파일 로드
load_dotenv()


class Config:
    """설정 클래스"""

    # Hugging Face 설정
    HUGGINGFACE_HUB_TOKEN: Optional[str] = os.getenv("HUGGINGFACE_HUB_TOKEN")

    # 모델 설정
    DEFAULT_MODEL: str = os.getenv("DEFAULT_MODEL", "facebook/nllb-200-distilled-600M")
    CACHE_DIR: str = os.getenv("CACHE_DIR", "./models_cache")

    # 번역 설정
    MAX_LENGTH: int = int(os.getenv("MAX_LENGTH", "128"))
    NUM_BEAMS: int = int(os.getenv("NUM_BEAMS", "5"))

    # 지원 모델 목록
    SUPPORTED_MODELS = {
        "nllb-200": "facebook/nllb-200-distilled-600M",
        "m2m-100-1.2b": "facebook/m2m100_1.2B",
        "mbart-50": "facebook/mbart-large-50-many-to-many-mmt",
        "opus-ko-ja": "Helsinki-NLP/opus-mt-ko-jap",
        "opus-ja-ko": "Helsinki-NLP/opus-mt-jap-ko",
        "hyperclova-0.5b": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
        "hyperclova-1.5b": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
    }

    # 지원하는 언어 코드
    LANGUAGE_CODES = ["korean", "japanese", "english"]

    # Radis 설정
    REDIS_HOST: str = os.getenv("REDIS_HOST", "localhost")
    REDIS_PORT: int = int(os.getenv("REDIS_PORT", "6379"))
    REDIS_CACHE_TTL: int = int(os.getenv("REDIS_CACHE_TTL", "86400"))

    @classmethod
    def get_huggingface_token(cls) -> Optional[str]:
        """Hugging Face 토큰 반환"""
        return cls.HUGGINGFACE_HUB_TOKEN

    @classmethod
    def validate_config(cls) -> bool:
        """설정 유효성 검사"""
        if not cls.DEFAULT_MODEL:
            return False

        if cls.HUGGINGFACE_HUB_TOKEN:
            if not cls.HUGGINGFACE_HUB_TOKEN.startswith("hf_"):
                print("⚠️  Warning: Hugging Face token format seems incorrect")

        return True

    @classmethod
    def print_config(cls) -> None:
        """현재 설정 출력"""
        print("⚙️  Current Configuration:")
        print(f"  Default Model: {cls.DEFAULT_MODEL}")
        print(f"  Cache Directory: {cls.CACHE_DIR}")
        print(f"  Max Length: {cls.MAX_LENGTH}")
        print(f"  Num Beams: {cls.NUM_BEAMS}")

        token_status = "✅ Set" if cls.HUGGINGFACE_HUB_TOKEN else "❌ Not set"
        print(f"  HF Token: {token_status}")


# 전역 설정 인스턴스
config = Config()

# 설정 유효성 검사
if not config.validate_config():
    print("❌ Configuration validation failed!")
    exit(1)

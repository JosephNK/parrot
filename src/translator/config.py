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

    # 디바이스 설정
    FORCE_CPU: bool = os.getenv("FORCE_CPU", "false").lower() == "true"

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
        print(f"  Force CPU: {cls.FORCE_CPU}")

        token_status = "✅ Set" if cls.HUGGINGFACE_HUB_TOKEN else "❌ Not set"
        print(f"  HF Token: {token_status}")


# 전역 설정 인스턴스
config = Config()

# 설정 유효성 검사
if not config.validate_config():
    print("❌ Configuration validation failed!")
    exit(1)

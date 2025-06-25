"""
Configuration Module

환경변수 및 설정 관리 (Pydantic Settings)
"""

from typing import Optional, Dict, Any, List
from pydantic import BaseModel, Field, field_validator, model_validator
from pydantic_settings import BaseSettings


class ModelConfig(BaseModel):
    """모델 설정 스키마"""

    name: str
    tokenizer: str
    transformer: str


class Config(BaseSettings):
    """설정 클래스"""

    # Hugging Face 설정
    HUGGINGFACE_HUB_TOKEN: Optional[str] = Field(
        default=None, description="Hugging Face Hub 토큰"
    )

    # 모델 설정
    DEFAULT_MODEL: str = Field(
        default="facebook/nllb-200-distilled-600M", description="기본 모델"
    )
    CACHE_DIR: str = Field(default="./models_cache", description="모델 캐시 디렉토리")

    # 번역 설정
    MAX_LENGTH: int = Field(default=128, ge=1, le=2048, description="최대 토큰 길이")
    NUM_BEAMS: int = Field(default=5, ge=1, le=20, description="빔 서치 개수")

    # Redis 설정
    REDIS_HOST: str = Field(default="", description="Redis 호스트")
    REDIS_PORT: int = Field(default=0, ge=0, le=65535, description="Redis 포트")
    REDIS_CACHE_TTL: int = Field(
        default=86400, ge=60, description="Redis 캐시 TTL (초)"
    )

    # 지원하는 언어 코드
    LANGUAGE_CODES: List[str] = Field(default=["korean", "japanese", "english"])

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        # case_sensitive = True  # 환경변수 대소문자 구분

    @field_validator("HUGGINGFACE_HUB_TOKEN")
    @classmethod
    def validate_hf_token(cls, v: Optional[str]) -> Optional[str]:
        """Hugging Face 토큰 형식 검증"""
        if v and not v.startswith("hf_"):
            print("⚠️  Warning: Hugging Face token format seems incorrect")
        return v

    @model_validator(mode="after")
    def validate_redis_config(self) -> "Config":
        """Redis 설정 전체 검증"""
        if self.REDIS_HOST and self.REDIS_PORT == 0:
            raise ValueError("Redis host is set but port is 0")
        return self

    @property
    def SUPPORTED_MODELS(self) -> Dict[str, Dict[str, str]]:
        """지원 모델 목록"""
        return {
            "nllb-200": {
                "name": "facebook/nllb-200-distilled-600M",
                "tokenizer": "facebook/nllb-200-distilled-600M",
                "transformer": "seq2seqlm",
            },
            "m2m-100-1.2b": {
                "name": "facebook/m2m100_1.2B",
                "tokenizer": "facebook/m2m100_1.2B",
                "transformer": "seq2seqlm",
            },
            "ct2fast-m2m-100_1.2b": {
                "name": "michaelfeil/ct2fast-m2m100_1.2B",
                "tokenizer": "facebook/m2m100_1.2B",
                "transformer": "ctranslate2",
            },
            "mbart-50": {
                "name": "facebook/mbart-large-50-many-to-many-mmt",
                "tokenizer": "facebook/mbart-large-50-many-to-many-mmt",
                "transformer": "seq2seqlm",
            },
            "qwen2.5-1.5b": {
                "name": "Qwen/Qwen2.5-1.5B-Instruct",
                "tokenizer": "Qwen/Qwen2.5-1.5B-Instruct",
                "transformer": "causallm",
            },
            "seamless-m4t-v2-large": {
                "name": "facebook/seamless-m4t-v2-large",
                "tokenizer": "facebook/seamless-m4t-v2-large",
                "transformer": "seamlessM4Tv2",
            },
            "hyperclova-1.5b": {
                "name": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
                "tokenizer": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
                "transformer": "causallm",
            },
            "varco-8b": {
                "name": "NCSOFT/Llama-VARCO-8B-Instruct",
                "tokenizer": "NCSOFT/Llama-VARCO-8B-Instruct",
                "transformer": "causallm",
            },
        }

    def get_huggingface_token(self) -> Optional[str]:
        """Hugging Face 토큰 반환"""
        return self.HUGGINGFACE_HUB_TOKEN

    def get_model_config(self, model_key: str) -> Optional[ModelConfig]:
        """모델 설정 반환"""
        model_data = self.SUPPORTED_MODELS.get(model_key)
        if model_data:
            return ModelConfig(**model_data)
        return None

    def is_redis_enabled(self) -> bool:
        """Redis 사용 가능 여부 확인"""
        return bool(self.REDIS_HOST and self.REDIS_PORT > 0)

    def validate_config(self) -> bool:
        """설정 유효성 검사 (호환성을 위해 유지)"""
        try:
            # Pydantic이 이미 검증했으므로 항상 True
            return True
        except Exception:
            return False

    def print_config(self) -> None:
        """현재 설정 출력"""
        print("⚙️  Current Configuration:")
        print(f"  Default Model: {self.DEFAULT_MODEL}")
        print(f"  Cache Directory: {self.CACHE_DIR}")
        print(f"  Max Length: {self.MAX_LENGTH}")
        print(f"  Num Beams: {self.NUM_BEAMS}")

        token_status = "✅ Set" if self.HUGGINGFACE_HUB_TOKEN else "❌ Not set"
        print(f"  HF Token: {token_status}")

        redis_status = "✅ Enabled" if self.is_redis_enabled() else "❌ Disabled"
        print(f"  Redis: {redis_status}")


# 전역 설정 인스턴스
try:
    config = Config()
    print("✅ Configuration loaded successfully!")
except Exception as e:
    print(f"❌ Configuration validation failed: {e}")
    exit(1)

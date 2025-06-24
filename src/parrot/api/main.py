import time
from typing import Optional
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

from ..cache.translation_cache import TranslationCache
from ..exception.exception import TranslationError
from ..translator import KoreanJapaneseTranslator
from ..config import config

# FastAPI 앱 생성
app = FastAPI()

# 설정 로드
redis_host = config.REDIS_HOST
redis_port = config.REDIS_PORT
redis_cache_ttl = config.REDIS_CACHE_TTL

# 한국어-일본어 번역기 인스턴스 생성
translator: KoreanJapaneseTranslator = None

# 캐시 시스템 초기화
translation_cache: TranslationCache = None
if redis_host != "" and redis_port != 0:
    translation_cache = TranslationCache(
        host=redis_host,
        port=redis_port,
        expire_time=redis_cache_ttl,
    )


# 요청 모델 정의
class TranslationRequest(BaseModel):
    text: str
    model: str

    class Config:
        json_schema_extra = {
            "example": {
                "text": "안녕하세요. 오늘 날씨가 정말 좋네요",
                "model": "nllb-200",
            }
        }


class TranslationResponse(BaseModel):
    original: str
    translated: str
    translate_time: str
    cached_at: Optional[str] = None


# 커스텀 예외 핸들러
@app.exception_handler(TranslationError)
async def custom_exception_handler(request: Request, exc: TranslationError):
    return JSONResponse(
        status_code=exc.code,
        content={
            "error": {
                "message": exc.message,
                "code": exc.error_code,
            }
        },
    )


# 기본 루트 경로
@app.get("/")
def read_root():
    return {"message": "Hello, this is the Korean-Japanese translation API!"}


# 헬스체크
@app.get("/health")
def health_check():
    try:
        # 기본적인 상태 확인
        status = {
            "status": "healthy",
            "timestamp": time.time(),
        }

        # 캐시 연결 상태 확인 (선택사항)
        try:
            cache_status = (
                translation_cache.ping()
                if hasattr(translation_cache, "ping")
                else "unknown"
            )
            status["cache"] = cache_status
        except Exception:
            status["cache"] = "disconnected"

        return status
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "timestamp": time.time(), "error": str(e)},
        )


# 번역기 초기화/로드 공통 함수
def ensure_translator_loaded(model_name: str):
    """번역기가 로드되어 있지 않거나 다른 모델이면 로드합니다."""
    global translator

    if translator is None:
        translator = KoreanJapaneseTranslator(auto_load=False)
    if translator.model_name != model_name:
        translator.load_model(model_name=model_name, auto_load=True)


# 캐시 조회 공통 함수
def get_cached_translation(text: str):
    """캐시에서 번역 결과를 조회합니다."""
    if translation_cache is not None:
        return translation_cache.get_translation(text)
    return None


# 캐시 저장 공통 함수
def save_to_cache(text: str, translated_text: dict, translate_time: str):
    """번역 결과를 캐시에 저장합니다."""
    if translation_cache is not None:
        translation_cache.save_translation(text, translated_text, translate_time)


# 한국어 -> 일본어 번역 API (POST 방식)
@app.post("/translate/ko2ja", response_model=TranslationResponse)
def translate_ko2ja(request: TranslationRequest):
    """
    한국어를 일본어로 번역합니다.

    - **text**: 번역할 한국어 텍스트
    - **model**: 사용할 번역 모델명
    """
    # 번역기 초기화/로드
    ensure_translator_loaded(request.model)

    # 캐시에서 번역 결과 조회
    cache_result = get_cached_translation(request.text)
    if cache_result is not None:
        return cache_result

    # 번역
    translate_start = time.time()
    result = translator.ko2ja(request.text)
    translate_time = time.time() - translate_start

    response = {
        "original": request.text,
        "translated": result,
        "translate_time": f"{translate_time:.2f}s",
    }

    # 캐시에 저장
    save_to_cache(
        response["original"],
        response["translated"],
        response["translate_time"],
    )

    return response


# 일본어 -> 한국어 번역 API (POST 방식)
@app.post("/translate/ja2ko", response_model=TranslationResponse)
def translate_ja2ko(request: TranslationRequest):
    """
    일본어를 한국어로 번역합니다.

    - **text**: 번역할 일본어 텍스트
    - **model**: 사용할 번역 모델명
    """
    # 번역기 초기화/로드
    ensure_translator_loaded(request.model)

    # 캐시에서 번역 결과 조회
    cache_result = get_cached_translation(request.text)
    if cache_result is not None:
        return cache_result

    # 번역
    translate_start = time.time()
    result = translator.ja2ko(request.text)
    translate_time = time.time() - translate_start

    response = {
        "original": request.text,
        "translated": result,
        "translate_time": f"{translate_time:.2f}s",
    }

    # 캐시에 저장
    save_to_cache(
        response["original"],
        response["translated"],
        response["translate_time"],
    )

    return response


# 테스트
@app.get("/test")
def hello_name():
    ensure_translator_loaded("nllb-200")
    text = "안녕하세요. 오늘 날씨가 정말 좋네요"
    translate_start = time.time()
    result = translator.ko2ja(text)
    translate_time = time.time() - translate_start
    return {
        "original": text,
        "translated": result,
        "translate_time": f"{translate_time:.2f}s",
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)

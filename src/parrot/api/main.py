import time
from fastapi import FastAPI, Query, Request
from fastapi.responses import JSONResponse

from ..cache.translation_cache import TranslationCache
from ..exception.exception import TranslationError
from ..translator import KoreanJapaneseTranslator
from ..config import config

# FastAPI 앱 생성
app = FastAPI()

# 한국어-일본어 번역기 인스턴스 생성
translator: KoreanJapaneseTranslator = None

redis_host = config.REDIS_HOST
redis_port = config.REDIS_PORT
redis_cache_ttl = config.REDIS_CACHE_TTL

translation_cache: TranslationCache = None
if redis_host is None or redis_port is None:
    # 캐시 시스템 초기화
    translation_cache = TranslationCache(
        host=redis_host,
        port=redis_port,
        expire_time=redis_cache_ttl,
    )


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
            cache_status = cache.ping() if hasattr(cache, "ping") else "unknown"
            status["cache"] = cache_status
        except Exception:
            status["cache"] = "disconnected"

        return status
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "timestamp": time.time(), "error": str(e)},
        )


# 한국어 -> 일본어 번역 API (Query 파라미터)
@app.get("/translate/ko2ja")
def translate_ko2ja(
    text: str = Query(..., description="번역할 한국어 텍스트"),
    model: str = Query(..., description="모델"),
):
    """
    한국어를 일본어로 번역합니다.

    - **text**: 번역할 한국어 텍스트
    """
    translator = KoreanJapaneseTranslator(auto_load=False)
    if translator.model_name != model:
        translator.load_model(model_name=model, auto_load=True)

    if translation_cache is not None:
        # 캐시에서 번역 결과 조회
        cache_result = translation_cache.get_translation(text)
        if cache_result is not None:
            return cache_result

    # 번역
    translate_start = time.time()
    result = translator.ko2ja(text)
    translate_time = time.time() - translate_start

    if translation_cache is not None:
        # 캐시에 저장
        translation_cache.save_translation(text, result, translate_time)

    return {
        "original": text,
        "translated": result,
        "translate_time": f"{translate_time:.2f}s",
    }


# 테스트
@app.get("/test")
def hello_name():
    if translator is None:
        translator = KoreanJapaneseTranslator(model_name="nllb-200", auto_load=True)
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

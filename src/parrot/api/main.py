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

# 캐시 시스템 초기화
cache = TranslationCache(
    host=config.RADIS_HOST,
    port=config.REDIS_PORT,
    expire_time=config.REDIS_CACHE_TTL,
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

    # 캐시에서 번역 결과 조회
    cache_result = cache.get_translation(text)
    if cache_result is not None:
        return cache_result

    # 번역
    translate_start = time.time()
    result = translator.ko2ja(text)
    translate_time = time.time() - translate_start

    # 캐시에 저장
    cache.save_translation(text, result, translate_time)

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

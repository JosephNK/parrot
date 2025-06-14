import time
from fastapi import FastAPI, Query

from parrot.translator import KoreanJapaneseTranslator

# FastAPI 앱 생성
app = FastAPI()

# 한국어-일본어 번역기 인스턴스 생성
translator: KoreanJapaneseTranslator = None


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

    translate_start = time.time()
    result = translator.ko2ja(text)
    translate_time = time.time() - translate_start
    return {
        "original": text,
        "translated": result,
        "translate_time": f"{translate_time:.2f}s",
    }


# 테스트
@app.get("/test")
def hello_name():
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

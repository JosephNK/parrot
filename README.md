# Parrot

This project provides **automatic translation between Korean and Japanese** using Facebook's NLLB or Opus models.

## Key Features

- Supports bidirectional translation: Korean → Japanese and Japanese → Korean
- Built on the Hugging Face Transformers library
- Utilizes large pre-trained AI translation models (NLLB, Opus, etc.)

### Support Model

```
SUPPORTED_MODELS = {
    "nllb-200": "facebook/nllb-200-distilled-600M",
    "mbart-50": "facebook/mbart-large-50-many-to-many-mmt", // not work :(
    "opus-ko-ja": "Helsinki-NLP/opus-mt-ko-jap",
    "opus-ja-ko": "Helsinki-NLP/opus-mt-jap-ko",
    "hyperclova-0.5b": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B",
    "hyperclova-1.5b": "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B",
}
```

### Download Model

```
poetry run python scripts/download_models.py --model nllb-200
poetry run python scripts/download_models.py --model hyperclova-1.5b
```

### Translate

```
poetry run python scripts/test.py --translate "안녕하세요. 오늘 날씨가 정말 좋네요." ko2ja --model nllb-200
poetry run python scripts/test.py --translate "안녕하세요. 오늘 날씨가 정말 좋네요." ko2ja --model hyperclova-1.5b
```

### Info

```
poetry run python scripts/test.py --info
```

### API

```
poe dev - 개발 서버 (자동 재시작)
poe start - 일반 서버
poe prod - 프로덕션 서버 (멀티 워커)
```

#### Example Curl
```
curl "http://localhost:8000/translate/ko2ja?text=Hello" | jq
curl -G "http://localhost:8000/translate/ko2ja" --data-urlencode "text=안녕하세요. 오늘 날씨가 정말 좋네요" | jq
```

### Testing

Hardware: Macbook M3

Model: hyperclova-1.5b

```
curl -G "http://localhost:8000/translate/ko2ja" --data-urlencode "text=안녕하세요. 오늘 날씨 가 정말 좋네요" | jq
{
  "original": "안녕하세요. 오늘 날씨가 정말 좋네요",
  "translated": "こんにちは。今日の天気が本当に好いです。",
  "translate_time": "0.89s"
}
```
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
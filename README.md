# Parrot

This project provides **automatic translation between Korean and Japanese** using Facebook's NLLB or Opus models.

## Key Features

- Supports bidirectional translation: Korean → Japanese and Japanese → Korean
- Built on the Hugging Face Transformers library
- Utilizes large pre-trained AI translation models (NLLB, Opus, etc.)

### Download Model

```
poetry run python scripts/download_models.py --model nllb-200
poetry run python scripts/download_models.py --model m2m-100-1.2b
poetry run python scripts/download_models.py --model hyperclova-1.5b
```

### Translate

```
poetry run python scripts/test.py --translate "안녕하세요. 오늘 날씨가 정말 좋네요." ko2ja --model nllb-200
poetry run python scripts/test.py --translate "안녕하세요. 오늘 날씨가 정말 좋네요." ko2ja --model m2m-100-1.2b
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

### Testing

### Environment Information
- **Hardware**: Macbook M3
- **Model**: m2m-100-1.2b
- **Description**: `Text2Text Generation` A task that takes input text and transforms it into a different form of text.

Request:
```bash
curl -G "http://localhost:8000/translate/ko2ja" \
  --data-urlencode "text=안녕하세요. 오늘 날씨가 정말 좋네요" \
  --data-urlencode "model=m2m-100-1.2b" | jq
```

Response:
```json
{
  "original": "안녕하세요. 오늘 날씨가 정말 좋네요",
  "translated": "こんにちは! 今日の天気はとても良いです。",
  "translate_time": "0.98s"
}
```

Request:
```bash
curl -G "http://localhost:8000/translate/ko2ja" \
  --data-urlencode "text=재이미샵 포카 굿즈 일괄 원가 양도 판매합니다."\
  --data-urlencode "model=m2m-100-1.2b" | jq
```

Response:
```json
{
  "original": "재이미샵 포카 굿즈 일괄 원가 양도 판매합니다.",
  "translated": "ジェイミショップポカ・グッズ一括価格も販売しています。",
  "translate_time": "1.64s"
}
```

### Environment Information
- **Hardware**: Macbook M3
- **Model**: hyperclova-1.5b (Safetensors)
- **Description**: `Safetensors` A new file format for storing machine learning model weights.

Request:
```bash
curl -G "http://localhost:8000/translate/ko2ja" \
  --data-urlencode "text=안녕하세요. 오늘 날씨 가 정말 좋네요" \
  --data-urlencode "model=hyperclova-1.5b" | jq
```

Response:
```json
{
  "original": "안녕하세요. 오늘 날씨가 정말 좋네요",
  "translated": "こんにちは。今日の天気が本当に好いです。",
  "translate_time": "0.89s"
}
```

Request:
```bash
curl -G "http://localhost:8000/translate/ko2ja" \
  --data-urlencode "text=재이미샵 포카 굿즈 일괄 원가 양도 판매합니다."\
  --data-urlencode "model=hyperclova-1.5b" | jq
```

Response:
```json
{
  "original": "재이미샵 포카 굿즈 일괄 원가 양도 판매합니다.",
  "translated": "ジャイミシーのポカグッド一緒に 원가를 양도합니다。",
  "translate_time": "2.94s"
}
```
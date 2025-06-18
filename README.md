# Parrot

This project provides **automatic translation between Korean and Japanese** using Facebook's M2M or NLLB models.

## Key Features

- Supports bidirectional translation: Korean → Japanese and Japanese → Korean
- Built on the Hugging Face Transformers library
- Utilizes large pre-trained AI translation models (M2M, NLLB, etc.)
- Smart terminology processing:
  - Retrieval-Augmented Generation (RAG) for CausalLM models
  - Preprocessing-based replacement for Seq2SeqLM models
- Redis caching for optimized response times

### Setup

Getting a Hugging Face Access Token (Read)

https://huggingface.co/docs/hub/security-tokens

```
.env

HUGGINGFACE_HUB_TOKEN=value
```

### Install

```bash
poetry install
```

### Download Model

```bash
poetry run python scripts/download_models.py --model nllb-200
poetry run python scripts/download_models.py --model m2m-100-1.2b
poetry run python scripts/download_models.py --model qwen2.5-1.5b
poetry run python scripts/download_models.py --model hyperclova-1.5b
poetry run python scripts/download_models.py --model varco-8b
```

### Translate

```bash
poetry run python scripts/test.py --translate "안녕하세요. 오늘 날씨가 정말 좋네요." ko2ja --model nllb-200
poetry run python scripts/test.py --translate "안녕하세요. 오늘 날씨가 정말 좋네요." ko2ja --model m2m-100-1.2b
poetry run python scripts/test.py --translate "안녕하세요. 오늘 날씨가 정말 좋네요." ko2ja --model qwen2.5-1.5b
poetry run python scripts/test.py --translate "안녕하세요. 오늘 날씨가 정말 좋네요." ko2ja --model hyperclova-1.5b
poetry run python scripts/test.py --translate "안녕하세요. 오늘 날씨가 정말 좋네요." ko2ja --model varco-8b
poetry run python scripts/test.py --translate "재이미샵 포카 굿즈 일괄 원가 양도 판매합니다." ko2ja --model nllb-200
poetry run python scripts/test.py --translate "재이미샵 포카 굿즈 일괄 원가 양도 판매합니다." ko2ja --model m2m-100-1.2b
poetry run python scripts/test.py --translate "재이미샵 포카 굿즈 일괄 원가 양도 판매합니다." ko2ja --model qwen2.5-1.5b
poetry run python scripts/test.py --translate "재이미샵 포카 굿즈 일괄 원가 양도 판매합니다." ko2ja --model hyperclova-1.5b
poetry run python scripts/test.py --translate "재이미샵 포카 굿즈 일괄 원가 양도 판매합니다." ko2ja --model varco-8b
```

### Model Performance

```bash
poetry run python scripts/test.py --performance --model nllb-200
poetry run python scripts/test.py --performance --model m2m-100-1.2b
```

Example
```
🚀 Performance Test
==================================================
📥 Testing model: m2m-100-1.2b
Loading model (seq2seqlm): facebook/m2m100_1.2B
Using device: mps
✓ Tokenizer loaded
✓ Model loaded successfully.
✓ FAISS loaded successfully.
✓ Terminology database loaded: 12 terms

📊 Performance Results:
------------------------------
✓ Translating from 'ko' to 'ja'...
🎯 짧은 문장:
  Input: 안녕
  Output: おはよう
  Time: 0.800s
  Speed: 2.5 chars/sec

✓ Translating from 'ko' to 'ja'...
🎯 중간 문장:
  Input: 오늘 날씨가 정말 좋네요
  Output: 今日は天気いいね。
  Time: 1.096s
  Speed: 11.9 chars/sec

✓ Translating from 'ko' to 'ja'...
🎯 긴 문장:
  Input: 파이썬을 사용해서 허깅페이스 모델로 한국어와 일본어 번역기를 만들고 있습니다
  Output: Pythonを使用して、ハッキングフェイスモデルとして韓国語と日本語の翻訳機を作っています
  Time: 1.791s
  Speed: 23.5 chars/sec

✓ Translating from 'ko' to 'ja'...
🎯 복잡한 문장:
  Input: 저는 서울에 살고 있는 대학생이며, 인공지능과 자연어 처리에 관심이 많습니다
  Output: 私はソウルに住む大学生で、人工知能と自然言語処理に興味があります。
  Time: 1.442s
  Speed: 29.1 chars/sec

📈 Overall Performance:
  Total time: 5.128s
  Total characters: 99
  Average speed: 19.3 chars/sec
```

### Model Info

```bash
poetry run python scripts/test.py --info --model nllb-200
poetry run python scripts/test.py --info --model m2m-100-1.2b
```

Example
```
Loading model (seq2seqlm): facebook/m2m100_1.2B
Using device: mps
✓ Tokenizer loaded
✓ Model loaded successfully.
✓ FAISS loaded successfully.
✓ Terminology database loaded: 12 terms
ℹ️  Translation Model Information
==================================================
Model: facebook/m2m100_1.2B
Device: mps
Languages: korean, japanese, english
Directions: Korean → Japanese, Japanese → Korean
==================================================

📋 Available Models:
nllb-200: {'name': 'facebook/nllb-200-distilled-600M', 'transformer': 'seq2seqlm'}
m2m-100-1.2b: {'name': 'facebook/m2m100_1.2B', 'transformer': 'seq2seqlm'}
...
```

### Benchmark Models

```bash
poetry run python scripts/test.py --benchmark
```

Example
```
⚡ Model Benchmark
==================================================

🔍 Testing: nllb-200
----------------------------------------
Loading model (seq2seqlm): facebook/nllb-200-distilled-600M
Using device: mps
✓ Tokenizer loaded
✓ Model loaded successfully.
✓ FAISS loaded successfully.
✓ Terminology database loaded: 12 terms
✓ Translating from 'kor_Hang' to 'jpn_Jpan'...
🇰🇷 Korean: 안녕하세요. 오늘 날씨가 좋네요.
✓ Translation completed: こんにちは. 今日は良い天気です
⏱️  Load: 7.97s, Translate: 1.31s

✓ Translating from 'jpn_Jpan' to 'kor_Hang'...
🇯🇵 Japanese: こんにちは。今日はいい天気ですね。
✓ Translation completed: 안녕, 안녕. 오늘 좋은 날씨입니다.
⏱️  Load: 7.97s, Translate: 0.63s


🔍 Testing: m2m-100-1.2b
----------------------------------------
Loading model (seq2seqlm): facebook/m2m100_1.2B
Using device: mps
✓ Tokenizer loaded
✓ Model loaded successfully.
✓ Terminology database loaded: 12 terms
✓ Translating from 'ko' to 'ja'...
🇰🇷 Korean: 안녕하세요. 오늘 날씨가 좋네요.
✓ Translation completed: こんにちは!今日はいい天気です。
⏱️  Load: 8.29s, Translate: 0.99s

✓ Translating from 'ja' to 'ko'...
🇯🇵 Japanese: こんにちは。今日はいい天気ですね。
✓ Translation completed: 안녕하세요 오늘은 좋은 날씨입니다.
⏱️  Load: 8.29s, Translate: 0.65s


📊 Benchmark Summary:
==================================================
nllb-200 (ko2ja): Load 7.97s, Translate 1.31s
nllb-200 (ja2ko): Load 7.97s, Translate 0.63s
m2m-100-1.2b (ko2ja): Load 8.29s, Translate 0.99s
m2m-100-1.2b (ja2ko): Load 8.29s, Translate 0.65s
...
```

### API

```bash
poe dev - 개발 서버 (자동 재시작)
poe start - 일반 서버
poe prod - 프로덕션 서버 (멀티 워커)
```

### Testing

#### Environment Information
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

#### Environment Information
- **Hardware**: Macbook M3
- **Model**: hyperclova-1.5b
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
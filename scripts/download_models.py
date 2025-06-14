#!/usr/bin/env python3
"""
Model Download Script

번역 모델들을 미리 다운로드하는 스크립트
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Optional
from parrot.model import LoaderModel

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from parrot.translator import KoreanJapaneseTranslator
from parrot.config import config


def download_model(
    model_name: str, save_path: Optional[str] = None, auth_token: Optional[str] = None
) -> bool:
    """
    모델 다운로드

    Args:
        model_name: 다운로드할 모델 이름
        save_path: 저장할 경로 (선택사항)
        auth_token: 인증 토큰 (선택사항)

    Returns:
        다운로드 성공 여부
    """
    try:
        print(f"\n📥 Downloading model: {model_name}")
        print("-" * 50)

        model = LoaderModel(model_name, auth_token)
        model.load_model()

        # 로컬 저장 (선택사항)
        if save_path:
            print(f"💾 Saving to local path: {save_path}")
            model.save_model(save_path)

        print(f"✅ Successfully downloaded: {model_name}")
        return True

    except Exception as e:
        print(f"❌ Failed to download {model_name}: {e}")
        return False


def download_supported_models(
    save_dir: Optional[str] = None, auth_token: Optional[str] = None
) -> None:
    """지원 모델들 일괄 다운로드"""

    models = KoreanJapaneseTranslator.list_models()
    success_count = 0

    print("🚀 Downloading supported translation models...")
    print(f"📋 Models to download: {len(models)}")

    for model_key, model_name in models.items():
        # 로컬 저장 경로 설정
        local_path = None
        if save_dir:
            local_path = os.path.join(save_dir, model_key)

        # 다운로드 시도
        if download_model(model_name, local_path, auth_token):
            success_count += 1

    print(f"\n📊 Download Summary:")
    print(f"✅ Success: {success_count}/{len(models)}")
    print(f"❌ Failed: {len(models) - success_count}/{len(models)}")


def test_model(model_name: str, auth_token: Optional[str] = None) -> None:
    """모델 테스트"""
    try:
        print(f"\n🧪 Testing model: {model_name}")
        print("-" * 50)

        # 번역기 초기화 (모델 타입에 따라 자동 선택)
        if model_name in KoreanJapaneseTranslator.list_models():
            translator = KoreanJapaneseTranslator(
                model_name=model_name, auth_token=auth_token
            )
        else:
            translator = KoreanJapaneseTranslator(
                model_name=model_name, auth_token=auth_token
            )

        # 테스트 문장들
        test_cases = [
            ("안녕하세요", "ko2ja"),
            ("오늘 날씨가 좋네요", "ko2ja"),
        ]

        # HyperCLOVAX 모델의 경우 일본어→한국어도 테스트
        if "hyperclova" not in model_name.lower():
            test_cases.extend(
                [
                    ("こんにちは", "ja2ko"),
                    ("今日はいい天気ですね", "ja2ko"),
                ]
            )

        for text, direction in test_cases:
            try:
                if direction == "ko2ja":
                    result = translator.ko2ja(text)
                    print(f"🇰🇷 → 🇯🇵: {text} → {result}")
                else:
                    result = translator.ja2ko(text)
                    print(f"🇯🇵 → 🇰🇷: {text} → {result}")
            except Exception as e:
                print(f"❌ Translation error: {e}")

        print("✅ Model test completed")

    except Exception as e:
        print(f"❌ Model test failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(description="Download and test translation models")

    parser.add_argument(
        "--model",
        type=str,
        help="Specific model to download (e.g., nllb-200, mbart-50)",
    )

    parser.add_argument(
        "--all", action="store_true", help="Download all recommended models"
    )

    parser.add_argument("--save-dir", type=str, help="Directory to save models locally")

    parser.add_argument(
        "--test", action="store_true", help="Test the model after download"
    )

    parser.add_argument(
        "--auth-token", type=str, help="Hugging Face authentication token"
    )

    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    # 환경변수에서 토큰 읽기
    auth_token = args.auth_token or config.get_huggingface_token()

    # 사용 가능한 모델 목록 출력
    if args.list:
        models = KoreanJapaneseTranslator.list_models()
        print("📋 Available models:")
        for key, name in models.items():
            print(f"  {key}: {name}")
        return

    # 모든 모델 다운로드
    if args.all:
        download_supported_models(args.save_dir, auth_token)
        return

    # 특정 모델 다운로드
    if args.model:
        models = KoreanJapaneseTranslator.list_models()

        if args.model in models:
            model_name = models[args.model]
        else:
            model_name = args.model

        # 저장 경로 설정
        save_path = None
        if args.save_dir:
            save_path = os.path.join(args.save_dir, args.model)

        # 다운로드
        success = download_model(model_name, save_path, auth_token)

        # 테스트
        if success and args.test:
            test_model(model_name, auth_token)

        return

    # 기본 동작: NLLB 모델 다운로드
    print("🎯 Downloading default model (NLLB-200)...")
    model_name = "facebook/nllb-200-distilled-600M"
    save_path = None

    if args.save_dir:
        save_path = os.path.join(args.save_dir, "nllb-200")

    success = download_model(model_name, save_path, auth_token)

    if success and args.test:
        test_model(model_name, auth_token)


if __name__ == "__main__":
    main()

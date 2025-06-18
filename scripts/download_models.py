#!/usr/bin/env python3
"""
Model Download Script

번역 모델들을 미리 다운로드하는 스크립트
"""

import os
import sys
import argparse
from pathlib import Path
from typing import Dict, Optional
from parrot.model import LoaderModel

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from parrot.translator import KoreanJapaneseTranslator
from parrot.config import config


def download_model(
    model_info: Dict[str, Dict[str, str]],
    save_path: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> bool:
    """모델 다운로드"""
    try:
        print(f"\n📥 Downloading model: {model_info}")
        print("-" * 50)

        model = LoaderModel(model_info, auth_token)
        model.load_model()

        # 로컬 저장 (선택사항)
        if save_path:
            print(f"💾 Saving to local path: {save_path}")
            model.save_model(save_path)

        print(f"✅ Successfully downloaded: {model_info}")
        return True

    except Exception as e:
        print(f"❌ Failed to download {model_info}: {e}")
        return False


def download_supported_models(
    save_dir: Optional[str] = None,
    auth_token: Optional[str] = None,
) -> None:
    """지원 모델들 일괄 다운로드"""

    models = KoreanJapaneseTranslator.list_models()
    success_count = 0

    print("🚀 Downloading supported translation models...")
    print(f"📋 Models to download: {len(models)}")

    for model_key, model_info in models.items():
        # 로컬 저장 경로 설정
        local_path = None
        if save_dir:
            local_path = os.path.join(save_dir, model_key)

        # 다운로드 시도
        if download_model(model_info, local_path, auth_token):
            success_count += 1

    print(f"\n📊 Download Summary:")
    print(f"✅ Success: {success_count}/{len(models)}")
    print(f"❌ Failed: {len(models) - success_count}/{len(models)}")


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

    parser.add_argument("--list", action="store_true", help="List available models")

    args = parser.parse_args()

    # 환경변수에서 토큰 읽기
    auth_token = config.get_huggingface_token()

    # 사용 가능한 모델 목록 출력
    if args.list:
        models = KoreanJapaneseTranslator.list_models()
        print("📋 Available models:")
        for key, info in models.items():
            print(f"  {key}: {info}")
        return

    # 모든 모델 다운로드
    if args.all:
        download_supported_models(args.save_dir, auth_token)
        return

    # 특정 모델 다운로드
    if args.model:
        models = KoreanJapaneseTranslator.list_models()

        if args.model in models:
            model_info = models[args.model]
        else:
            model_info = models["nllb-200"]  # 기본 모델 설정

        # 저장 경로 설정
        save_path = None
        if args.save_dir:
            save_path = os.path.join(args.save_dir, args.model)

        # 다운로드
        download_model(model_info, save_path, auth_token)
        return


if __name__ == "__main__":
    main()

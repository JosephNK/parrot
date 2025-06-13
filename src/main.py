#!/usr/bin/env python3
"""
Main Translation Script

번역 기능을 테스트하고 사용하는 메인 스크립트
"""

import os
import sys
import argparse
import time
from pathlib import Path
from typing import List, Dict, Any

# 프로젝트 루트를 Python path에 추가
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "src"))

from translator import KoreanJapaneseTranslator


def demo_translation() -> None:
    """번역 데모"""
    print("🚀 Translation Demo")
    print("=" * 50)

    try:
        # 번역기 초기화
        print("📥 Loading translation model...")
        translator = KoreanJapaneseTranslator()

        # 테스트 문장들
        test_sentences = {
            "korean": [
                "안녕하세요!",
                "오늘 날씨가 정말 좋네요.",
                "한국어를 일본어로 번역해주세요.",
                "이 번역기가 잘 작동하는지 확인해보겠습니다.",
                "Python과 AI를 사용한 번역 프로젝트입니다.",
            ],
            "japanese": [
                "こんにちは！",
                "今日はとても良い天気ですね。",
                "日本語を韓国語に翻訳してください。",
                "この翻訳機がうまく動作するか確認してみます。",
                "PythonとAIを使った翻訳プロジェクトです。",
            ],
        }

        # 한국어 → 일본어
        print("\n🇰🇷 → 🇯🇵 Korean to Japanese:")
        print("-" * 30)
        for ko_text in test_sentences["korean"]:
            start_time = time.time()
            ja_result = translator.ko2ja(ko_text)
            translate_time = time.time() - start_time

            print(f"KO: {ko_text}")
            print(f"JA: {ja_result}")
            print(f"⏱️  Time: {translate_time:.2f}s")
            print()

        # 일본어 → 한국어
        print("\n🇯🇵 → 🇰🇷 Japanese to Korean:")
        print("-" * 30)
        for ja_text in test_sentences["japanese"]:
            start_time = time.time()
            ko_result = translator.ja2ko(ja_text)
            translate_time = time.time() - start_time

            print(f"JA: {ja_text}")
            print(f"KO: {ko_result}")
            print(f"⏱️  Time: {translate_time:.2f}s")
            print()

        # 배치 번역 테스트
        print("\n📦 Batch Translation Test:")
        print("-" * 30)
        batch_ko = ["안녕", "감사합니다", "잘 지내세요"]

        start_time = time.time()
        batch_ja_results = translator.ko2ja_batch(batch_ko)
        batch_time = time.time() - start_time

        for ko, ja in zip(batch_ko, batch_ja_results):
            print(f"KO: {ko} → JA: {ja}")
        print(f"⏱️  Batch time: {batch_time:.2f}s")

        print("\n✅ Demo completed successfully!")

    except Exception as e:
        print(f"❌ Error during demo: {e}")
        import traceback

        traceback.print_exc()


def interactive_mode() -> None:
    """대화형 번역 모드"""
    import sys

    # 인코딩 강제 설정
    if hasattr(sys.stdin, "reconfigure"):
        try:
            sys.stdin.reconfigure(encoding="utf-8")
            sys.stdout.reconfigure(encoding="utf-8")
        except Exception:
            pass

    try:
        print("🎯 Interactive Translation Mode")
        print("=" * 50)

        translator = KoreanJapaneseTranslator()

        # 커스텀 대화형 번역
        print("=== 한국어-일본어 번역기 ===")
        print("명령어:")
        print("  'ko': 한국어 → 일본어")
        print("  'ja': 일본어 → 한국어")
        print("  'quit' 또는 Ctrl+C: 종료")
        print("-" * 30)

        while True:
            try:
                # 안전한 입력 처리
                sys.stdout.write("\n번역 방향 (ko/ja/quit): ")
                sys.stdout.flush()

                try:
                    command = sys.stdin.readline().strip().lower()
                except UnicodeDecodeError as e:
                    print(f"❌ 입력 인코딩 오류: {e}")
                    print("영어로 명령어를 입력해보세요 (ko/ja/quit)")
                    continue

                if command in ["quit", "exit", "q"]:
                    print("번역기를 종료합니다.")
                    break
                elif command not in ["ko", "ja"]:
                    print("올바른 명령어를 입력하세요 (ko/ja/quit)")
                    continue

                # 텍스트 입력
                if command == "ko":
                    sys.stdout.write("한국어 텍스트 입력: ")
                else:
                    sys.stdout.write("일본어 텍스트 입력: ")
                sys.stdout.flush()

                try:
                    text = sys.stdin.readline().strip()
                    if not text:
                        print("텍스트를 입력해주세요.")
                        continue
                except UnicodeDecodeError as e:
                    print(f"❌ 텍스트 인코딩 오류: {e}")
                    print("다시 시도해주세요.")
                    continue

                print("번역 중...")
                start_time = time.time()

                try:
                    if command == "ko":
                        result = translator.ko2ja(text)
                        print(f"한국어: {text}")
                        print(f"일본어: {result}")
                    else:
                        result = translator.ja2ko(text)
                        print(f"일본어: {text}")
                        print(f"한국어: {result}")

                    translate_time = time.time() - start_time
                    print(f"⏱️  번역 시간: {translate_time:.2f}초")

                except Exception as e:
                    print(f"❌ 번역 오류: {e}")

            except KeyboardInterrupt:
                print("\n번역기를 종료합니다.")
                break
            except EOFError:
                print("\n번역기를 종료합니다.")
                break
            except Exception as e:
                print(f"❌ 예상치 못한 오류: {e}")

    except Exception as e:
        print(f"❌ Error in interactive mode: {e}")
        import traceback

        traceback.print_exc()


def benchmark_models() -> None:
    """여러 모델 성능 비교"""
    print("⚡ Model Benchmark")
    print("=" * 50)

    # 테스트할 모델들
    models_to_test = [
        ("NLLB-200", "facebook/nllb-200-distilled-600M"),
        ("Opus KO-JA", "Helsinki-NLP/opus-mt-ko-jap"),
        ("Opus JA-KO", "Helsinki-NLP/opus-mt-jap-ko"),
        ("HyperCLOVA-0.5B", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-0.5B"),
        ("HyperCLOVA-1.5B", "naver-hyperclovax/HyperCLOVAX-SEED-Text-Instruct-1.5B"),
    ]

    test_sentences = [
        ("한국어", "안녕하세요. 오늘 날씨가 좋네요.", "ko2ja"),
        ("일본어", "こんにちは。今日はいい天気ですね。", "ja2ko"),
    ]

    results = []

    for model_name, model_path in models_to_test:
        print(f"\n🔍 Testing: {model_name}")
        print("-" * 40)

        try:
            # 모델 로딩 시간 측정
            load_start = time.time()

            if "opus-mt-ko-jap" in model_path:
                # 한국어 → 일본어만 테스트
                translator = KoreanJapaneseTranslator(model_name=model_path)
                load_time = time.time() - load_start

                for lang, sentence, direction in test_sentences:
                    if direction == "ko2ja":
                        start_time = time.time()
                        result = translator.ko2ja(sentence)
                        translate_time = time.time() - start_time

                        print(f"{lang}: {sentence}")
                        print(f"Result: {result}")
                        print(
                            f"⏱️  Load: {load_time:.2f}s, Translate: {translate_time:.2f}s"
                        )

                        results.append(
                            {
                                "model": model_name,
                                "direction": direction,
                                "load_time": load_time,
                                "translate_time": translate_time,
                                "input": sentence,
                                "output": result,
                            }
                        )
                        break

            elif "opus-mt-jap-ko" in model_path:
                # 일본어 → 한국어만 테스트
                translator = KoreanJapaneseTranslator(model_name=model_path)
                load_time = time.time() - load_start

                for lang, sentence, direction in test_sentences:
                    if direction == "ja2ko":
                        start_time = time.time()
                        result = translator.ja2ko(sentence)
                        translate_time = time.time() - start_time

                        print(f"{lang}: {sentence}")
                        print(f"Result: {result}")
                        print(
                            f"⏱️  Load: {load_time:.2f}s, Translate: {translate_time:.2f}s"
                        )

                        results.append(
                            {
                                "model": model_name,
                                "direction": direction,
                                "load_time": load_time,
                                "translate_time": translate_time,
                                "input": sentence,
                                "output": result,
                            }
                        )
                        break

            else:
                # 양방향 번역 가능
                translator = KoreanJapaneseTranslator(model_name=model_path)
                load_time = time.time() - load_start

                for lang, sentence, direction in test_sentences:
                    start_time = time.time()

                    if direction == "ko2ja":
                        result = translator.ko2ja(sentence)
                    else:
                        result = translator.ja2ko(sentence)

                    translate_time = time.time() - start_time

                    print(f"{lang}: {sentence}")
                    print(f"Result: {result}")
                    print(
                        f"⏱️  Load: {load_time:.2f}s, Translate: {translate_time:.2f}s"
                    )
                    print()

                    results.append(
                        {
                            "model": model_name,
                            "direction": direction,
                            "load_time": load_time,
                            "translate_time": translate_time,
                            "input": sentence,
                            "output": result,
                        }
                    )

        except Exception as e:
            print(f"❌ Error testing {model_name}: {e}")

    # 결과 요약
    print(f"\n📊 Benchmark Summary:")
    print("=" * 50)
    for result in results:
        print(
            f"{result['model']} ({result['direction']}): "
            f"Load {result['load_time']:.2f}s, "
            f"Translate {result['translate_time']:.2f}s"
        )


def custom_translation(text: str, direction: str, model: str = None) -> None:
    """사용자 지정 번역"""
    try:
        print(f"🔄 Translating with {model or 'default model'}")
        print("-" * 50)

        start_time = time.time()

        if model:
            translator = KoreanJapaneseTranslator(model_name=model)
        else:
            translator = KoreanJapaneseTranslator()

        load_time = time.time() - start_time

        translate_start = time.time()

        if direction.lower() == "ko2ja":
            result = translator.ko2ja(text)
            print(f"🇰🇷 Korean: {text}")
            print(f"🇯🇵 Japanese: {result}")
        elif direction.lower() == "ja2ko":
            result = translator.ja2ko(text)
            print(f"🇯🇵 Japanese: {text}")
            print(f"🇰🇷 Korean: {result}")
        else:
            print("❌ Invalid direction. Use 'ko2ja' or 'ja2ko'")
            return

        translate_time = time.time() - translate_start
        total_time = time.time() - start_time

        print(f"\n⏱️  Performance:")
        print(f"  Model load: {load_time:.2f}s")
        print(f"  Translation: {translate_time:.2f}s")
        print(f"  Total: {total_time:.2f}s")

    except Exception as e:
        print(f"❌ Translation error: {e}")
        import traceback

        traceback.print_exc()


def show_model_info() -> None:
    """모델 정보 표시"""
    try:
        translator = KoreanJapaneseTranslator()
        info = translator.get_info()

        print("ℹ️  Translation Model Information")
        print("=" * 50)
        print(f"Model: {info['model_name']}")
        print(f"Device: {info['device']}")
        print(f"Loaded: {info['is_loaded']}")
        print(f"Languages: {', '.join(info['supported_languages'])}")
        print(f"Directions: {', '.join(info['supported_directions'])}")

        print("\n📋 Available Models:")
        for key, model_name in info["supported_models"].items():
            print(f"  {key}: {model_name}")

    except Exception as e:
        print(f"❌ Error getting model info: {e}")


def performance_test(model_name: str = None) -> None:
    """성능 테스트"""
    print("🚀 Performance Test")
    print("=" * 50)

    try:
        if model_name:
            print(f"📥 Testing model: {model_name}")
            translator = KoreanJapaneseTranslator(model_name=model_name)
        else:
            print("📥 Testing default model...")
            translator = KoreanJapaneseTranslator()

        # 다양한 길이의 문장 테스트
        test_cases = [
            ("짧은 문장", "안녕"),
            ("중간 문장", "오늘 날씨가 정말 좋네요"),
            (
                "긴 문장",
                "파이썬을 사용해서 허깅페이스 모델로 한국어와 일본어 번역기를 만들고 있습니다",
            ),
            (
                "복잡한 문장",
                "저는 서울에 살고 있는 대학생이며, 인공지능과 자연어 처리에 관심이 많습니다",
            ),
        ]

        print(f"\n📊 Performance Results:")
        print("-" * 30)

        total_time = 0
        total_chars = 0

        for desc, text in test_cases:
            start_time = time.time()
            result = translator.ko2ja(text)
            end_time = time.time()

            duration = end_time - start_time
            chars_per_sec = len(text) / duration if duration > 0 else 0

            total_time += duration
            total_chars += len(text)

            print(f"{desc}:")
            print(f"  Input: {text}")
            print(f"  Output: {result}")
            print(f"  Time: {duration:.3f}s")
            print(f"  Speed: {chars_per_sec:.1f} chars/sec")
            print()

        avg_speed = total_chars / total_time if total_time > 0 else 0
        print(f"📈 Overall Performance:")
        print(f"  Total time: {total_time:.3f}s")
        print(f"  Total characters: {total_chars}")
        print(f"  Average speed: {avg_speed:.1f} chars/sec")

    except Exception as e:
        print(f"❌ Performance test failed: {e}")
        import traceback

        traceback.print_exc()


def main():
    """메인 함수"""
    parser = argparse.ArgumentParser(
        description="Korean-Japanese Translation Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --demo                           # Run translation demo
  %(prog)s --interactive                    # Interactive translation mode
  %(prog)s --translate "안녕하세요" ko2ja      # Translate specific text
  %(prog)s --benchmark                      # Compare different models
  %(prog)s --info                           # Show model information
  %(prog)s --performance                    # Run performance test
        """,
    )

    parser.add_argument(
        "--demo", action="store_true", help="Run translation demo with sample sentences"
    )

    parser.add_argument(
        "--interactive", action="store_true", help="Start interactive translation mode"
    )

    parser.add_argument(
        "--translate",
        nargs=2,
        metavar=("TEXT", "DIRECTION"),
        help="Translate text (direction: ko2ja or ja2ko)",
    )

    parser.add_argument(
        "--model", type=str, help="Specify model to use for translation"
    )

    parser.add_argument(
        "--benchmark",
        action="store_true",
        help="Benchmark different translation models",
    )

    parser.add_argument("--info", action="store_true", help="Show model information")

    parser.add_argument(
        "--performance", action="store_true", help="Run performance test"
    )

    args = parser.parse_args()

    # 명령어 처리
    if args.demo:
        demo_translation()
    elif args.interactive:
        interactive_mode()
    elif args.translate:
        text, direction = args.translate
        custom_translation(text, direction, args.model)
    elif args.benchmark:
        benchmark_models()
    elif args.info:
        show_model_info()
    elif args.performance:
        performance_test(args.model)
    else:
        # 기본 동작: 데모 실행
        print("No specific command provided. Running demo...")
        demo_translation()

        print("\n💡 Try other options:")
        print("  --demo         : Translation demo")
        print("  --interactive  : Interactive mode")
        print("  --benchmark    : Model comparison")
        print("  --info         : Model information")
        print("  --performance  : Performance test")
        print("  --help         : Full help")


if __name__ == "__main__":
    main()

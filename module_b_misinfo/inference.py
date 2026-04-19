import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from module_b_misinfo.model import MisinfoDetector


def run_inference(text: str, model_path: str = None) -> dict:
    if model_path is None:
        model_path = os.path.join(
            os.path.dirname(__file__), '..',
            'models', 'nlp', 'distilbert_finetuned.pth'
        )

    if not os.path.exists(model_path):
        return {
            "verdict":    "MODEL_NOT_TRAINED",
            "confidence": 0.0,
            "message":    "Please run module_b_misinfo/train.py first"
        }

    detector = MisinfoDetector(model_path=model_path)
    return detector.predict(text)


if __name__ == "__main__":
    model_path = os.path.join(
        os.path.dirname(__file__), '..',
        'models', 'nlp', 'distilbert_finetuned.pth'
    )

    # ── Load model ONCE ──
    print("\nLoading model once...")
    detector = MisinfoDetector(model_path=model_path)

    test_texts = [
        ("Scientists discover water on Mars",           "TRUE"),
        ("Government puts microchips in vaccines",      "FALSE"),
        ("Local council approves new housing project",  "TRUE"),
        ("You won't believe what this celebrity did",   "FALSE"),
        ("New study links coffee to longer life",       "TRUE"),
        ("Aliens landed in New York last night",        "FALSE"),
    ]

    print("\n" + "="*55)
    print("  MISINFORMATION DETECTOR — TEST RUN")
    print("="*55)

    correct = 0
    for text, expected in test_texts:
        result = detector.predict(text)
        emoji  = "🔴" if result["verdict"] == "FALSE" else "🟢"
        match  = "✅" if result["verdict"] == expected else "❌"
        print(f"\n{emoji} {result['verdict']} ({result['confidence']*100:.1f}%) {match}")
        print(f"   Text     : {text}")
        print(f"   Expected : {expected}")
        if result["verdict"] == expected:
            correct += 1

    print(f"\n{'='*55}")
    print(f"  Test Accuracy: {correct}/{len(test_texts)} = {correct/len(test_texts)*100:.0f}%")
    print("="*55)